//! Portable direct-arena applications.
//!
//! A direct application is compiled once from a normal Symbolica application,
//! but reads point-contiguous state planes and immutable scalar pointer
//! descriptors. Outputs may alias input planes, so accumulated destinations are
//! updated by the generated code without an output buffer or scatter pass.

use std::collections::{BTreeSet, HashSet};
use std::ffi::c_void;
use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use anyhow::{anyhow, bail, Context, Result};

use crate::applet::Applet;
use crate::config::Config;
use crate::generator::Generator;
use crate::mir::{Instruction, Mir};
use crate::runnable::{Application, CompilerType};
use crate::serializer::MirWriter;
use crate::symbol::Loc;
use crate::utils::{Compiled, Reg, Storage};

pub const DIRECT_APPLICATION_STORAGE_ABI: &str = "symjit-direct-application-storage-v1";
pub const DIRECT_NO_ALIAS: u32 = u32::MAX;
pub const DIRECT_COMPLEX_SCALE_REAL_SCALAR: u32 = 0;
pub const DIRECT_COMPLEX_SCALE_IMAG_SCALAR: u32 = 1;
pub const DIRECT_COMPLEX_SCALE_SCALAR_COUNT: u32 = 2;

const DIRECT_MAGIC: [u8; 8] = *b"SJDA0001";
const DIRECT_VERSION: u32 = 1;

pub const DIRECT_STATUS_OK: i32 = 0;
pub const DIRECT_STATUS_INVALID_CONTEXT: i32 = 1;
pub const DIRECT_STATUS_INVALID_ARGUMENT: i32 = 2;
pub const DIRECT_STATUS_EXECUTION_FAILED: i32 = 3;

/// Store operation applied when a direct kernel produces one output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DirectDestinationOperation {
    Overwrite = 0,
    Accumulate = 1,
}

impl DirectDestinationOperation {
    fn accumulates(self) -> bool {
        matches!(self, Self::Accumulate)
    }
}

impl TryFrom<u8> for DirectDestinationOperation {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Overwrite),
            1 => Ok(Self::Accumulate),
            _ => bail!("unknown direct destination operation {value}"),
        }
    }
}

/// Whether generated code reads input planes live or snapshots them before
/// the first destination write.
///
/// Snapshotting is useful when outputs alias inputs and a later output still
/// depends on a value overwritten by an earlier output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DirectInputSnapshot {
    Live = 0,
    BeforeWrite = 1,
}

impl TryFrom<u8> for DirectInputSnapshot {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Live),
            1 => Ok(Self::BeforeWrite),
            _ => bail!("unknown direct input snapshot mode {value}"),
        }
    }
}

/// Scaling applied before a generated direct kernel stores one complex output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DirectOutputScale {
    ComplexScalar = 0,
    Identity = 1,
}

impl DirectOutputScale {
    fn requires_complex_scalar(self) -> bool {
        matches!(self, Self::ComplexScalar)
    }

    fn required_optimization_level(self) -> u8 {
        match self {
            Self::ComplexScalar => 2,
            Self::Identity => 3,
        }
    }

    fn accepts_source_optimization_level(self, optimization_level: u8) -> bool {
        match self {
            Self::ComplexScalar => optimization_level == 2,
            Self::Identity => optimization_level <= 3,
        }
    }

    fn source_optimization_requirement(self) -> &'static str {
        match self {
            Self::ComplexScalar => "O2",
            Self::Identity => "O0, O1, O2, or O3",
        }
    }
}

impl TryFrom<u8> for DirectOutputScale {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::ComplexScalar),
            1 => Ok(Self::Identity),
            _ => bail!("unknown direct output scale {value}"),
        }
    }
}

/// Maps one scalar input of the original Symbolica application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectInputBinding {
    /// Read a point-dependent value from this input plane.
    Plane(u32),
    /// Read a point-independent value through this scalar pointer descriptor.
    Scalar(u32),
}

/// Shape and alias metadata stored beside portable MIR.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectApplicationMetadata {
    pub destination_operation: DirectDestinationOperation,
    pub input_snapshot: DirectInputSnapshot,
    pub output_scale: DirectOutputScale,
    pub state_plane_indices: Vec<u32>,
    pub parameter_bindings: Vec<DirectInputBinding>,
    pub input_plane_count: u32,
    pub scalar_input_count: u32,
    pub output_alias_inputs: Vec<u32>,
}

impl DirectApplicationMetadata {
    pub fn new(
        destination_operation: DirectDestinationOperation,
        input_snapshot: DirectInputSnapshot,
        output_scale: DirectOutputScale,
        state_plane_indices: Vec<u32>,
        parameter_bindings: Vec<DirectInputBinding>,
        input_plane_count: u32,
        scalar_input_count: u32,
        output_alias_inputs: Vec<u32>,
    ) -> Result<Self> {
        let metadata = Self {
            destination_operation,
            input_snapshot,
            output_scale,
            state_plane_indices,
            parameter_bindings,
            input_plane_count,
            scalar_input_count,
            output_alias_inputs,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<()> {
        if self.output_alias_inputs.is_empty() {
            bail!("direct application must expose at least one output plane");
        }
        if self.output_scale.requires_complex_scalar()
            && self.scalar_input_count < DIRECT_COMPLEX_SCALE_SCALAR_COUNT
        {
            bail!(
                "direct application must reserve scalar slots 0 and 1 for its complex output scale"
            );
        }

        let mut planes = BTreeSet::new();
        for &plane in &self.state_plane_indices {
            if plane >= self.input_plane_count {
                bail!("direct state-plane binding {plane} is out of bounds");
            }
            planes.insert(plane);
        }
        let mut scalars = BTreeSet::new();
        if self.output_scale.requires_complex_scalar() {
            scalars.extend([
                DIRECT_COMPLEX_SCALE_REAL_SCALAR,
                DIRECT_COMPLEX_SCALE_IMAG_SCALAR,
            ]);
        }
        for binding in &self.parameter_bindings {
            match *binding {
                DirectInputBinding::Plane(plane) => {
                    if plane >= self.input_plane_count {
                        bail!("direct parameter-plane binding {plane} is out of bounds");
                    }
                    planes.insert(plane);
                }
                DirectInputBinding::Scalar(scalar) => {
                    if self.output_scale.requires_complex_scalar()
                        && scalar < DIRECT_COMPLEX_SCALE_SCALAR_COUNT
                    {
                        bail!(
                            "direct source parameter scalar {scalar} overlaps reserved complex-scale slots 0 and 1"
                        );
                    }
                    if scalar >= self.scalar_input_count {
                        bail!("direct scalar binding {scalar} is out of bounds");
                    }
                    scalars.insert(scalar);
                }
            }
        }

        let mut output_aliases = BTreeSet::new();
        for &alias in &self.output_alias_inputs {
            if alias == DIRECT_NO_ALIAS {
                continue;
            }
            if alias >= self.input_plane_count {
                bail!("direct output alias {alias} is out of bounds");
            }
            if !output_aliases.insert(alias) {
                bail!("direct output aliases must identify distinct destination planes");
            }
            planes.insert(alias);
        }
        require_dense("input plane", self.input_plane_count, &planes)?;
        require_dense("scalar input", self.scalar_input_count, &scalars)?;
        Ok(())
    }

    fn validate_source(&self, source: &Application) -> Result<()> {
        self.validate()?;
        if source.count_states != self.state_plane_indices.len() {
            bail!(
                "direct state binding count {} does not match source state count {}",
                self.state_plane_indices.len(),
                source.count_states
            );
        }
        if source.count_params != self.parameter_bindings.len() {
            bail!(
                "direct parameter binding count {} does not match source parameter count {}",
                self.parameter_bindings.len(),
                source.count_params
            );
        }
        if source.count_obs != self.output_alias_inputs.len() {
            bail!(
                "direct output alias count {} does not match source output count {}",
                self.output_alias_inputs.len(),
                source.count_obs
            );
        }
        if source.count_diffs != 0 {
            bail!("direct applications do not support differential outputs");
        }
        if !matches!(source.config.ty, CompilerType::Native)
            && !(cfg!(target_arch = "x86_64") && matches!(source.config.ty, CompilerType::AmdAVX))
            || !self
                .output_scale
                .accepts_source_optimization_level(source.config.opt_level())
            || !source.config.symbolica()
            || !source.config.is_complex()
            || source.config.direct_arena()
            || source.config.direct_arena_identity_output()
            || source.config.direct_arena_operation() != 0
        {
            bail!(
                "direct {:?} applications require a portable native Symbolica complex {} source",
                self.output_scale,
                self.output_scale.source_optimization_requirement()
            );
        }
        if !source.prog.builder.ft.is_empty() {
            bail!("direct applications cannot retain external function calls");
        }
        Ok(())
    }

    fn validate_compiled(&self, application: &Application) -> Result<()> {
        self.validate()?;
        if application.count_states != self.input_plane_count as usize
            || application.count_params != self.scalar_input_count as usize
            || application.count_obs != self.output_alias_inputs.len()
            || application.count_diffs != 0
        {
            bail!("stored direct application shape does not match its metadata");
        }
        let required_optimization_level = self.output_scale.required_optimization_level();
        let identity_output = self.output_scale == DirectOutputScale::Identity;
        if !matches!(application.config.ty, CompilerType::Native)
            && !(cfg!(target_arch = "x86_64")
                && matches!(application.config.ty, CompilerType::AmdAVX))
            || application.config.opt_level() != required_optimization_level
            || application.config.symbolica()
            || application.config.is_complex() != identity_output
            || !application.config.direct_arena()
            || application.config.direct_arena_operation() != self.destination_operation as u8
            || application.config.direct_arena_identity_output() != identity_output
        {
            bail!(
                "stored application does not use the portable direct-arena O{} ABI",
                required_optimization_level
            );
        }
        if application.compiled.is_none() {
            bail!("stored direct application has no native scalar callable");
        }
        Ok(())
    }

    fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&[self.destination_operation as u8])?;
        stream.write_all(&[self.input_snapshot as u8])?;
        stream.write_all(&[self.output_scale as u8])?;
        write_u32(stream, self.input_plane_count)?;
        write_u32(stream, self.scalar_input_count)?;
        write_u32_len(stream, self.state_plane_indices.len())?;
        for &value in &self.state_plane_indices {
            write_u32(stream, value)?;
        }
        write_u32_len(stream, self.parameter_bindings.len())?;
        for binding in &self.parameter_bindings {
            match *binding {
                DirectInputBinding::Plane(value) => {
                    stream.write_all(&[0])?;
                    write_u32(stream, value)?;
                }
                DirectInputBinding::Scalar(value) => {
                    stream.write_all(&[1])?;
                    write_u32(stream, value)?;
                }
            }
        }
        write_u32_len(stream, self.output_alias_inputs.len())?;
        for &value in &self.output_alias_inputs {
            write_u32(stream, value)?;
        }
        Ok(())
    }

    fn load_for_source(stream: &mut impl Read, source: &Application) -> Result<Self> {
        let mut operation = [0_u8; 1];
        stream.read_exact(&mut operation)?;
        let destination_operation = DirectDestinationOperation::try_from(operation[0])?;
        let mut snapshot = [0_u8; 1];
        stream.read_exact(&mut snapshot)?;
        let input_snapshot = DirectInputSnapshot::try_from(snapshot[0])?;
        let mut output_scale = [0_u8; 1];
        stream.read_exact(&mut output_scale)?;
        let output_scale = DirectOutputScale::try_from(output_scale[0])?;
        let input_plane_count = read_u32(stream)?;
        let scalar_input_count = read_u32(stream)?;
        let state_plane_indices =
            read_source_sized_u32_vec(stream, source.count_states, "state binding")?;
        let parameter_count =
            read_source_sized_count(stream, source.count_params, "parameter binding")?;
        let mut parameter_bindings = Vec::new();
        parameter_bindings
            .try_reserve_exact(parameter_count)
            .context("cannot reserve direct parameter bindings")?;
        for _ in 0..parameter_count {
            let mut kind = [0_u8; 1];
            stream.read_exact(&mut kind)?;
            let value = read_u32(stream)?;
            parameter_bindings.push(match kind[0] {
                0 => DirectInputBinding::Plane(value),
                1 => DirectInputBinding::Scalar(value),
                other => bail!("unknown direct input binding kind {other}"),
            });
        }
        let output_alias_inputs =
            read_source_sized_u32_vec(stream, source.count_obs, "output alias")?;
        Self::new(
            destination_operation,
            input_snapshot,
            output_scale,
            state_plane_indices,
            parameter_bindings,
            input_plane_count,
            scalar_input_count,
            output_alias_inputs,
        )
    }
}

fn require_dense(name: &str, count: u32, values: &BTreeSet<u32>) -> Result<()> {
    if values.len() != count as usize
        || values
            .iter()
            .copied()
            .enumerate()
            .any(|(expected, actual)| u32::try_from(expected).ok() != Some(actual))
    {
        bail!("direct {name} bindings must be dense zero-based");
    }
    Ok(())
}

/// ABI-stable plane descriptor consumed by generated indirect code.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DirectPlane {
    pub values: *mut f64,
    pub len: usize,
}

impl DirectPlane {
    /// # Safety
    ///
    /// The caller must keep `values[0..len]` alive for every call using this
    /// descriptor.
    pub const unsafe fn from_raw_parts(values: *mut f64, len: usize) -> Self {
        Self { values, len }
    }
}

/// One immutable scalar binding. Generated SIMD code broadcasts `*value`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DirectScalar {
    pub value: *const f64,
}

impl DirectScalar {
    /// # Safety
    ///
    /// The caller must keep `value` alive for every call using this descriptor.
    pub const unsafe fn from_raw(value: *const f64) -> Self {
        Self { value }
    }
}

/// Portable direct application. Saving stores MIR and direct binding metadata,
/// never host machine code.
pub struct DirectApplication {
    metadata: DirectApplicationMetadata,
    /// Portable source retained by the storage representation.
    source: Application,
    application: Application,
}

impl DirectApplication {
    pub fn new(source: Application, metadata: DirectApplicationMetadata) -> Result<Self> {
        metadata.validate_source(&source)?;
        ensure_supported_host()?;
        let application = lower_direct_application(&source, &metadata)?;
        metadata.validate_compiled(&application)?;
        Ok(Self {
            metadata,
            source,
            application,
        })
    }

    /// Build a direct application from an existing portable optimized application
    /// payload. This lets a prepared-model bundle share `application.symjit`
    /// with its ordinary evaluator instead of storing a duplicate payload.
    pub fn from_source_storage(
        stream: &mut impl Read,
        config: &Config,
        metadata: DirectApplicationMetadata,
    ) -> Result<Self> {
        match catch_unwind(AssertUnwindSafe(|| {
            let source = Application::load(stream, config)
                .context("cannot load direct application portable source")?;
            Self::new(source, metadata).context("cannot lower stored direct application source")
        })) {
            Ok(result) => result,
            Err(payload) => {
                drop(payload);
                bail!("direct application source loader panicked")
            }
        }
    }

    pub fn metadata(&self) -> &DirectApplicationMetadata {
        &self.metadata
    }

    /// Optimization level declared by the retained portable source.
    pub fn source_optimization_level(&self) -> u8 {
        self.source.config.opt_level()
    }

    pub fn prepare_simd(&mut self) {
        self.application.prepare_simd();
    }

    pub fn seal(self) -> Result<DirectApplet> {
        self.metadata.validate_compiled(&self.application)?;
        let applet = self.application.seal()?;
        Ok(DirectApplet {
            metadata: self.metadata,
            applet,
        })
    }
}

impl Storage for DirectApplication {
    fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&DIRECT_MAGIC)?;
        write_u32(stream, DIRECT_VERSION)?;
        self.source.save(stream)?;
        self.metadata.save(stream)
    }

    fn load(stream: &mut impl Read, config: &Config) -> Result<Self> {
        match catch_unwind(AssertUnwindSafe(|| Self::load_inner(stream, config))) {
            Ok(result) => result,
            Err(payload) => {
                drop(payload);
                bail!("direct application storage loader panicked")
            }
        }
    }
}

impl DirectApplication {
    fn load_inner(stream: &mut impl Read, config: &Config) -> Result<Self> {
        let mut magic = [0_u8; 8];
        stream.read_exact(&mut magic)?;
        let version = read_u32(stream)?;
        match (magic, version) {
            (DIRECT_MAGIC, DIRECT_VERSION) => Self::load_v1(stream, config),
            (DIRECT_MAGIC, other) => {
                bail!("unsupported direct application storage version {other}")
            }
            _ => bail!("invalid direct application magic"),
        }
    }

    fn load_v1(stream: &mut impl Read, config: &Config) -> Result<Self> {
        ensure_supported_host()?;
        let source = Application::load(stream, config)
            .context("cannot load direct application portable source")?;
        let metadata = DirectApplicationMetadata::load_for_source(stream, &source)
            .context("cannot load direct application metadata")?;
        Self::new(source, metadata).context("cannot lower stored direct application source")
    }
}

/// Loaded direct application context.
pub struct DirectApplet {
    metadata: DirectApplicationMetadata,
    applet: Applet,
}

impl DirectApplet {
    pub fn metadata(&self) -> &DirectApplicationMetadata {
        &self.metadata
    }

    pub fn simd_lane_width(&self) -> usize {
        self.applet
            .compiled_simd
            .as_ref()
            .map_or(1, Compiled::count_lanes)
    }

    /// Evaluate directly against persistent plane and scalar descriptors.
    ///
    /// # Safety
    ///
    /// Every descriptor must remain valid and obey the alias contract for the
    /// duration of the call. Output aliases are the only writable overlaps
    /// accepted by this ABI.
    pub unsafe fn evaluate_planes(
        &self,
        planes: &[DirectPlane],
        scalars: &[DirectScalar],
        point_start: usize,
        point_count: usize,
    ) -> Result<()> {
        validate_call(&self.metadata, planes, scalars, point_start, point_count)?;
        unsafe { self.evaluate_planes_unchecked(planes, scalars, point_start, point_count) }
    }

    /// Evaluate descriptors whose complete contract was authenticated by an
    /// owning runtime when the application was loaded.
    ///
    /// # Safety
    ///
    /// The plane/scalar counts, pointer ranges, output aliases, and requested
    /// point range must satisfy [`DirectApplicationMetadata`]. This entry
    /// point is intended for persistent table-driven runtimes that construct
    /// descriptors from already validated fixed projections.
    pub unsafe fn evaluate_planes_unchecked(
        &self,
        planes: &[DirectPlane],
        scalars: &[DirectScalar],
        point_start: usize,
        point_count: usize,
    ) -> Result<()> {
        let scalar = self
            .applet
            .compiled
            .as_ref()
            .ok_or_else(|| anyhow!("direct scalar callable is absent"))?
            .func();
        let lane_width = self.simd_lane_width();
        let point_end = point_start
            .checked_add(point_count)
            .ok_or_else(|| anyhow!("direct point range overflows"))?;
        let mut point = point_start;

        while point < point_end && !point.is_multiple_of(lane_width) {
            call_machine(scalar, planes, scalars, point)?;
            point += 1;
        }
        if lane_width > 1 {
            if let Some(simd) = &self.applet.compiled_simd {
                let function = simd.func();
                while point_end - point >= lane_width {
                    let block = point / lane_width;
                    if call_machine_status(function, planes, scalars, block) != 0 {
                        for offset in 0..lane_width {
                            call_machine(scalar, planes, scalars, point + offset)?;
                        }
                    }
                    point += lane_width;
                }
            }
        }
        while point < point_end {
            call_machine(scalar, planes, scalars, point)?;
            point += 1;
        }
        Ok(())
    }

    /// Pin this application behind a context-aware direct callable.
    pub fn into_callable(self) -> DirectCallable {
        DirectCallable {
            context: Box::new(self),
        }
    }
}

pub type DirectCallFunction = unsafe extern "C" fn(
    *const c_void,
    *const DirectPlane,
    u32,
    *const DirectScalar,
    u32,
    u32,
    u32,
) -> i32;

/// Borrowed function/context pair suitable for an authenticated executor catalog.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DirectCallableHandle {
    pub call: DirectCallFunction,
    pub context: *const c_void,
}

impl DirectCallableHandle {
    /// # Safety
    ///
    /// The owning [`DirectCallable`] and all descriptors must outlive the call.
    pub unsafe fn invoke(
        self,
        planes: &[DirectPlane],
        scalars: &[DirectScalar],
        point_start: u32,
        point_count: u32,
    ) -> i32 {
        let Ok(plane_count) = u32::try_from(planes.len()) else {
            return DIRECT_STATUS_INVALID_ARGUMENT;
        };
        let Ok(scalar_count) = u32::try_from(scalars.len()) else {
            return DIRECT_STATUS_INVALID_ARGUMENT;
        };
        unsafe {
            (self.call)(
                self.context,
                planes.as_ptr(),
                plane_count,
                scalars.as_ptr(),
                scalar_count,
                point_start,
                point_count,
            )
        }
    }
}

/// Owns the immutable context referenced by [`DirectCallableHandle`].
pub struct DirectCallable {
    context: Box<DirectApplet>,
}

impl DirectCallable {
    pub fn handle(&self) -> DirectCallableHandle {
        DirectCallableHandle {
            call: direct_call_trampoline,
            context: ptr::from_ref(self.context.as_ref()).cast(),
        }
    }

    pub fn metadata(&self) -> &DirectApplicationMetadata {
        self.context.metadata()
    }

    /// Invoke a callable using descriptors authenticated by the owning
    /// table-driven runtime.
    ///
    /// # Safety
    ///
    /// The arguments must satisfy
    /// [`DirectApplet::evaluate_planes_unchecked`].
    pub unsafe fn invoke_unchecked(
        &self,
        planes: &[DirectPlane],
        scalars: &[DirectScalar],
        point_start: usize,
        point_count: usize,
    ) -> i32 {
        match unsafe {
            self.context
                .evaluate_planes_unchecked(planes, scalars, point_start, point_count)
        } {
            Ok(()) => DIRECT_STATUS_OK,
            Err(error) => {
                if std::env::var_os("SYMJIT_DIRECT_DEBUG").is_some() {
                    eprintln!("SymJIT Direct-Arena unchecked call failed: {error:#}");
                }
                DIRECT_STATUS_EXECUTION_FAILED
            }
        }
    }
}

unsafe extern "C" fn direct_call_trampoline(
    context: *const c_void,
    planes: *const DirectPlane,
    plane_count: u32,
    scalars: *const DirectScalar,
    scalar_count: u32,
    point_start: u32,
    point_count: u32,
) -> i32 {
    if context.is_null() {
        return DIRECT_STATUS_INVALID_CONTEXT;
    }
    if (plane_count != 0 && planes.is_null()) || (scalar_count != 0 && scalars.is_null()) {
        return DIRECT_STATUS_INVALID_ARGUMENT;
    }
    let applet = unsafe { &*context.cast::<DirectApplet>() };
    let planes = unsafe { std::slice::from_raw_parts(planes, plane_count as usize) };
    let scalars = unsafe { std::slice::from_raw_parts(scalars, scalar_count as usize) };
    match unsafe {
        applet.evaluate_planes(planes, scalars, point_start as usize, point_count as usize)
    } {
        Ok(()) => DIRECT_STATUS_OK,
        Err(error) => {
            if std::env::var_os("SYMJIT_DIRECT_DEBUG").is_some() {
                eprintln!("SymJIT Direct-Arena call failed: {error:#}");
            }
            DIRECT_STATUS_EXECUTION_FAILED
        }
    }
}

fn lower_direct_application(
    source: &Application,
    metadata: &DirectApplicationMetadata,
) -> Result<Application> {
    let mut config = source.config.clone();
    config.set_symbolica(false);
    let identity_output = metadata.output_scale == DirectOutputScale::Identity;
    config.set_complex(identity_output);
    config.set_fast_complex(false);
    config.set_dicect(false);
    config.set_direct_arena(true);
    config.set_direct_arena_operation(metadata.destination_operation as u8);
    config.set_direct_arena_identity_output(identity_output);
    config.set_threads(false);
    config.set_opt_level(metadata.output_scale.required_optimization_level());

    let mut program = source.prog.clone();
    program.builder.config = config.clone();
    program.builder.primary_block.config = config.clone();
    program.count_states = metadata.input_plane_count as usize;
    program.count_params = metadata.scalar_input_count as usize;
    program.count_obs = metadata.output_alias_inputs.len();
    program.count_diffs = 0;
    program.count_loops = 0;

    let source_mir = source.bytecode.mir.as_ref();
    let mut mir = remap_mir(
        source_mir,
        &config,
        source.count_states,
        source.count_obs,
        metadata,
    )?;
    if metadata.input_snapshot == DirectInputSnapshot::BeforeWrite {
        let snapshot_start = u32::try_from(program.builder.stack_size())
            .context("direct input snapshot stack size exceeds u32")?;
        let snapshot_stop = snapshot_start
            .checked_add(metadata.input_plane_count)
            .ok_or_else(|| anyhow!("direct input snapshot range exceeds u32"))?;
        mir = snapshot_inputs(mir, snapshot_start, metadata.input_plane_count)?;
        program.builder.count_stack = Some(snapshot_stop as usize);
    }
    let mut application = if identity_output {
        Application::with_precomplexified_mir(program, HashSet::new(), mir)?
    } else {
        Application::with_mir(program, HashSet::new(), mir)?
    };
    application.prepare_simd();
    Ok(application)
}

/// Snapshot all input planes before the first destination write.
///
/// When output planes alias inputs, emitting one component before later
/// expressions have read every input can make evaluation order observable.
/// Keeping the snapshot in the generated function's fixed stack frame
/// preserves in-place execution without caller-side scratch storage.
fn snapshot_inputs(mut mir: Mir, snapshot_start: u32, input_plane_count: u32) -> Result<Mir> {
    let mut code = MirWriter::new();
    for plane in 0..input_plane_count {
        let stack = snapshot_start
            .checked_add(plane)
            .ok_or_else(|| anyhow!("direct input snapshot index exceeds u32"))?;
        code.push(&Instruction::Load {
            dst: Reg::Ret,
            loc: Loc::Mem(plane),
        });
        code.push(&Instruction::Save {
            src: Reg::Ret,
            loc: Loc::Stack(stack),
        });
    }
    for instruction in mir.code.iter() {
        code.push(&snapshot_instruction(
            instruction,
            snapshot_start,
            input_plane_count,
        )?);
    }
    mir.code = code;
    mir.populate_labels();
    Ok(mir)
}

fn snapshot_instruction(
    instruction: Instruction,
    snapshot_start: u32,
    input_plane_count: u32,
) -> Result<Instruction> {
    let snapshot = |location| snapshot_location(location, snapshot_start, input_plane_count);
    Ok(match instruction {
        Instruction::Load { dst, loc } => Instruction::Load {
            dst,
            loc: snapshot(loc)?,
        },
        Instruction::LoadComplex { xd, yd, loc } => Instruction::LoadComplex {
            xd,
            yd,
            loc: snapshot(loc)?,
        },
        Instruction::LoadMath { op, dst, s1, loc } => Instruction::LoadMath {
            op,
            dst,
            s1,
            loc: snapshot(loc)?,
        },
        Instruction::IfElse {
            dst,
            true_val,
            false_val,
            cond,
        } => Instruction::IfElse {
            dst,
            true_val,
            false_val,
            cond: snapshot(cond)?,
        },
        other => other,
    })
}

fn snapshot_location(location: Loc, snapshot_start: u32, input_plane_count: u32) -> Result<Loc> {
    match location {
        Loc::Mem(index) if index < input_plane_count => {
            Ok(Loc::Stack(snapshot_start.checked_add(index).ok_or_else(
                || anyhow!("direct input snapshot index exceeds u32"),
            )?))
        }
        other => Ok(other),
    }
}

fn remap_mir(
    source: &Mir,
    config: &Config,
    source_state_count: usize,
    source_output_count: usize,
    metadata: &DirectApplicationMetadata,
) -> Result<Mir> {
    let mut code = MirWriter::new();
    for instruction in source.code.iter() {
        append_remapped(
            &mut code,
            instruction,
            source_state_count,
            source_output_count,
            metadata,
        )?;
    }
    let mut mir = Mir {
        code,
        consts: source.consts.clone(),
        labels: source.labels.clone(),
        config: config.clone(),
    };
    mir.populate_labels();
    Ok(mir)
}

fn append_remapped(
    code: &mut MirWriter,
    instruction: Instruction,
    source_state_count: usize,
    source_output_count: usize,
    metadata: &DirectApplicationMetadata,
) -> Result<()> {
    match instruction {
        Instruction::LoadComplex { xd, yd, loc } => {
            code.push(&Instruction::Load {
                dst: xd,
                loc: remap_loc(loc, source_state_count, source_output_count, metadata)?,
            });
            code.push(&Instruction::Load {
                dst: yd,
                loc: remap_loc(
                    loc.imag(),
                    source_state_count,
                    source_output_count,
                    metadata,
                )?,
            });
        }
        Instruction::SaveComplex { xs, ys, loc } => {
            if !matches!(xs, Reg::Gen(_)) || !matches!(ys, Reg::Gen(_)) {
                bail!("direct complex output must be held in allocated general registers");
            }
            let mapped = remap_loc(loc, source_state_count, source_output_count, metadata)?;
            let mapped_imag = remap_loc(
                loc.imag(),
                source_state_count,
                source_output_count,
                metadata,
            )?;
            if mapped.imag() != mapped_imag {
                bail!("direct complex output planes must remain adjacent");
            }
            code.push(&Instruction::SaveComplex {
                xs,
                ys,
                loc: mapped,
            });
        }
        Instruction::Load { dst, loc } => code.push(&Instruction::Load {
            dst,
            loc: remap_loc(loc, source_state_count, source_output_count, metadata)?,
        }),
        Instruction::Save { src, loc } => {
            if matches!(loc, Loc::Mem(index) if index as usize >= source_state_count) {
                bail!("direct complex applications require paired complex output stores");
            }
            code.push(&Instruction::Save {
                src,
                loc: remap_loc(loc, source_state_count, source_output_count, metadata)?,
            });
        }
        Instruction::LoadMath { op, dst, s1, loc } => {
            code.push(&Instruction::LoadMath {
                op,
                dst,
                s1,
                loc: remap_loc(loc, source_state_count, source_output_count, metadata)?,
            });
        }
        Instruction::IfElse {
            dst,
            true_val,
            false_val,
            cond,
        } => code.push(&Instruction::IfElse {
            dst,
            true_val,
            false_val,
            cond: remap_loc(cond, source_state_count, source_output_count, metadata)?,
        }),
        other => code.push(&other),
    }
    Ok(())
}

/// Emit scaling and destination semantics at the direct complex-output
/// boundary. The source application neither reads the destination nor
/// consumes the optional output scale.
pub(crate) fn emit_direct_complex_destination(
    generator: &mut (impl Generator + ?Sized),
    xs: Reg,
    ys: Reg,
    destination: u32,
    operation: u8,
    identity_output: bool,
) -> Result<()> {
    let operation = DirectDestinationOperation::try_from(operation)?;
    if !matches!(xs, Reg::Gen(_)) || !matches!(ys, Reg::Gen(_)) {
        bail!("direct complex destination operands must use general registers");
    }

    if !identity_output {
        // Keep the original x/y values live while forming
        // (x + i y) * (scale_re + i scale_im).
        generator.load_param(Reg::Ret, DIRECT_COMPLEX_SCALE_REAL_SCALAR);
        generator.times(Reg::Ret, xs, Reg::Ret);
        generator.load_param(Reg::Temp, DIRECT_COMPLEX_SCALE_IMAG_SCALAR);
        generator.times(Reg::Temp, ys, Reg::Temp);
        generator.minus(Reg::Temp, Reg::Ret, Reg::Temp);

        generator.load_param(Reg::Ret, DIRECT_COMPLEX_SCALE_IMAG_SCALAR);
        generator.times(Reg::Ret, xs, Reg::Ret);
        generator.load_param(xs, DIRECT_COMPLEX_SCALE_REAL_SCALAR);
        generator.times(xs, ys, xs);
        generator.plus(ys, Reg::Ret, xs);
        generator.fmov(xs, Reg::Temp);
    }

    if operation.accumulates() {
        generator.load_mem(Reg::Ret, destination);
        generator.plus(xs, xs, Reg::Ret);
        generator.load_mem(Reg::Ret, destination + 1);
        generator.plus(ys, ys, Reg::Ret);
    }
    generator.save_mem_complex(xs, ys, destination);
    Ok(())
}

fn remap_loc(
    location: Loc,
    source_state_count: usize,
    source_output_count: usize,
    metadata: &DirectApplicationMetadata,
) -> Result<Loc> {
    match location {
        Loc::Stack(index) => Ok(Loc::Stack(index)),
        Loc::Param(index) => {
            let binding = metadata
                .parameter_bindings
                .get(index as usize)
                .with_context(|| format!("source parameter location {index} is out of bounds"))?;
            Ok(match *binding {
                DirectInputBinding::Plane(plane) => Loc::Mem(plane),
                DirectInputBinding::Scalar(scalar) => Loc::Param(scalar),
            })
        }
        Loc::Mem(index) => {
            let index = index as usize;
            if index < source_state_count {
                return Ok(Loc::Mem(metadata.state_plane_indices[index]));
            }
            let output = index
                .checked_sub(source_state_count)
                .ok_or_else(|| anyhow!("source memory location underflows the output range"))?;
            if output >= source_output_count {
                bail!("source memory location {index} is outside states and outputs");
            }
            let output = u32::try_from(output).context("direct output index exceeds u32")?;
            Ok(Loc::Mem(metadata.input_plane_count + output))
        }
    }
}

fn validate_call(
    metadata: &DirectApplicationMetadata,
    planes: &[DirectPlane],
    scalars: &[DirectScalar],
    point_start: usize,
    point_count: usize,
) -> Result<()> {
    let expected_planes = metadata.input_plane_count as usize + metadata.output_alias_inputs.len();
    if planes.len() != expected_planes {
        bail!(
            "direct call has {} planes, expected {expected_planes}",
            planes.len()
        );
    }
    if scalars.len() != metadata.scalar_input_count as usize {
        bail!(
            "direct call has {} scalar descriptors, expected {}",
            scalars.len(),
            metadata.scalar_input_count
        );
    }
    if point_count == 0 {
        bail!("direct call point count must be positive");
    }
    let point_end = point_start
        .checked_add(point_count)
        .ok_or_else(|| anyhow!("direct call point range overflows"))?;
    for (index, plane) in planes.iter().enumerate() {
        if plane.values.is_null() || plane.len < point_end {
            bail!("direct plane {index} does not cover the point tile");
        }
    }
    for (index, scalar) in scalars.iter().enumerate() {
        if scalar.value.is_null() {
            bail!("direct scalar descriptor {index} is null");
        }
    }
    for (output, &alias) in metadata.output_alias_inputs.iter().enumerate() {
        let output_plane = &planes[metadata.input_plane_count as usize + output];
        if alias == DIRECT_NO_ALIAS {
            for input in &planes[..metadata.input_plane_count as usize] {
                if plane_ranges_overlap(input, output_plane)? {
                    bail!("identity-scaled direct output {output} overlaps an input plane");
                }
            }
            for (scalar, descriptor) in scalars.iter().enumerate() {
                if scalar_plane_ranges_overlap(descriptor, output_plane)? {
                    bail!(
                        "identity-scaled direct output {output} overlaps scalar descriptor {scalar}"
                    );
                }
            }
            for previous in &planes
                [metadata.input_plane_count as usize..metadata.input_plane_count as usize + output]
            {
                if plane_ranges_overlap(previous, output_plane)? {
                    bail!("identity-scaled direct outputs overlap each other");
                }
            }
            continue;
        }
        let input_plane = &planes[alias as usize];
        if output_plane.values != input_plane.values || output_plane.len != input_plane.len {
            bail!("direct output {output} does not satisfy alias input {alias}");
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DirectAddressRange {
    start: usize,
    end: usize,
}

impl DirectAddressRange {
    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

fn checked_direct_address_range(
    name: &str,
    pointer: *const u8,
    count: usize,
    element_bytes: usize,
) -> Result<DirectAddressRange> {
    let bytes = count
        .checked_mul(element_bytes)
        .ok_or_else(|| anyhow!("direct {name} byte range overflows"))?;
    if bytes > isize::MAX as usize {
        bail!("direct {name} byte range exceeds isize");
    }
    let start = pointer as usize;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("direct {name} address range overflows"))?;
    Ok(DirectAddressRange { start, end })
}

fn plane_address_range(plane: &DirectPlane) -> Result<DirectAddressRange> {
    checked_direct_address_range(
        "plane",
        plane.values.cast(),
        plane.len,
        std::mem::size_of::<f64>(),
    )
}

fn plane_ranges_overlap(left: &DirectPlane, right: &DirectPlane) -> Result<bool> {
    Ok(plane_address_range(left)?.overlaps(plane_address_range(right)?))
}

fn scalar_plane_ranges_overlap(scalar: &DirectScalar, plane: &DirectPlane) -> Result<bool> {
    let scalar = checked_direct_address_range(
        "scalar descriptor",
        scalar.value.cast(),
        1,
        std::mem::size_of::<f64>(),
    )?;
    Ok(scalar.overlaps(plane_address_range(plane)?))
}

fn call_machine(
    function: crate::utils::CompiledFunc<f64>,
    planes: &[DirectPlane],
    scalars: &[DirectScalar],
    index: usize,
) -> Result<()> {
    let status = call_machine_status(function, planes, scalars, index);
    if status != 0 {
        bail!("direct machine call returned status {status}");
    }
    Ok(())
}

fn call_machine_status(
    function: crate::utils::CompiledFunc<f64>,
    planes: &[DirectPlane],
    scalars: &[DirectScalar],
    index: usize,
) -> i32 {
    function(
        ptr::null(),
        planes.as_ptr().cast(),
        index,
        scalars.as_ptr().cast(),
    )
}

fn ensure_supported_host() -> Result<()> {
    if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        Ok(())
    } else {
        bail!("direct arena applications currently require x86-64 or arm64")
    }
}

fn write_u32(stream: &mut impl Write, value: u32) -> Result<()> {
    stream.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u32_len(stream: &mut impl Write, value: usize) -> Result<()> {
    write_u32(
        stream,
        u32::try_from(value).context("direct metadata length exceeds u32")?,
    )
}

fn read_u32(stream: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    stream.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_source_sized_count(stream: &mut impl Read, expected: usize, label: &str) -> Result<usize> {
    let encoded = read_u32(stream)?;
    let expected_u32 =
        u32::try_from(expected).with_context(|| format!("direct {label} count exceeds u32"))?;
    if encoded != expected_u32 {
        bail!("direct {label} count {encoded} does not match source count {expected_u32}");
    }
    Ok(expected)
}

fn read_source_sized_u32_vec(
    stream: &mut impl Read,
    expected: usize,
    label: &str,
) -> Result<Vec<u32>> {
    let count = read_source_sized_count(stream, expected, label)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .with_context(|| format!("cannot reserve direct {label} values"))?;
    for _ in 0..count {
        result.push(read_u32(stream)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    use crate::{Compiler, Config};

    fn source_application(opt_level: u8) -> Application {
        let mut config = Config::default();
        config.set_opt_level(opt_level);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let x = Expr::var("x");
        let y = Expr::var("y");
        Compiler::with_config(config)
            .compile_params(&[], &[&x * &y], &[x, y])
            .unwrap()
    }

    fn metadata(operation: DirectDestinationOperation) -> DirectApplicationMetadata {
        DirectApplicationMetadata::new(
            operation,
            DirectInputSnapshot::Live,
            DirectOutputScale::ComplexScalar,
            vec![],
            vec![
                DirectInputBinding::Plane(0),
                DirectInputBinding::Plane(1),
                DirectInputBinding::Plane(2),
                DirectInputBinding::Plane(3),
            ],
            6,
            2,
            vec![4, 5],
        )
        .unwrap()
    }

    fn compiled_source_application_at(optimization_level: u8) -> Application {
        let mut config = Config::default();
        config.set_opt_level(optimization_level);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        config.set_compress(true);
        let instructions = r#"[[{"Mul":[{"Out":0},[{"Param":0},{"Param":1}],0]},{"Add":[{"Out":1},[{"Param":0},{"Param":1}],0]}],2,[]]"#;
        Compiler::with_config(config)
            .translate(instructions.to_owned(), 2)
            .unwrap()
    }

    fn compiled_source_application() -> Application {
        compiled_source_application_at(3)
    }

    fn compiled_metadata(source: &Application) -> DirectApplicationMetadata {
        assert_eq!(source.count_states, 0);
        DirectApplicationMetadata::new(
            DirectDestinationOperation::Overwrite,
            DirectInputSnapshot::Live,
            DirectOutputScale::Identity,
            vec![],
            (0..source.count_params as u32)
                .map(DirectInputBinding::Plane)
                .collect(),
            source.count_params as u32,
            0,
            vec![DIRECT_NO_ALIAS; source.count_obs],
        )
        .unwrap()
    }

    fn machine_code_contains_call(application: &Application) -> bool {
        let bytes = application
            .compiled
            .as_ref()
            .expect("compiled fixture must have scalar machine code")
            .dumps();
        #[cfg(target_arch = "aarch64")]
        {
            bytes.chunks_exact(4).any(|instruction| {
                let word = u32::from_le_bytes(instruction.try_into().unwrap());
                word & 0xfc00_0000 == 0x9400_0000
            })
        }
        #[cfg(target_arch = "x86_64")]
        {
            bytes.contains(&0xe8)
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            false
        }
    }

    fn assert_storage_load_fails_without_panic(bytes: &[u8], expected: &str) {
        let mut stored = bytes;
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            DirectApplication::load(&mut stored, &Config::default())
        }));
        let result = attempt.expect("malformed direct storage must not unwind");
        let error = result
            .err()
            .expect("malformed direct storage must fail closed");
        let message = format!("{error:#}");
        assert!(
            message.contains(expected),
            "unexpected direct storage error: {message}"
        );
    }

    fn stored_metadata_offset(direct: &DirectApplication) -> usize {
        let mut source = Vec::new();
        direct.source.save(&mut source).unwrap();
        DIRECT_MAGIC.len() + std::mem::size_of::<u32>() + source.len()
    }

    #[test]
    fn complex_scaled_calls_accumulate_into_aliased_outputs() {
        let metadata = metadata(DirectDestinationOperation::Accumulate);
        let mut application =
            DirectApplication::new(source_application(2), metadata.clone()).unwrap();
        application.prepare_simd();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        let loaded = DirectApplication::load(&mut bytes.as_slice(), &Config::default()).unwrap();
        assert_eq!(loaded.metadata(), &metadata);
        let callable = loaded.seal().unwrap().into_callable();

        let mut x_re = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut x_im = vec![0.5, -0.5, 1.0, -1.0, 0.25];
        let mut y_re = vec![2.0, 1.0, -1.0, 0.5, 4.0];
        let mut y_im = vec![0.0, 1.0, 0.5, -0.5, -1.0];
        let mut destination_re = vec![10.0; 5];
        let mut destination_im = vec![-3.0; 5];
        let original_re = destination_re.clone();
        let original_im = destination_im.clone();
        let factors = [(0.25, -0.5), (-1.25, 0.75)];

        let mut planes = vec![
            unsafe { DirectPlane::from_raw_parts(x_re.as_mut_ptr(), x_re.len()) },
            unsafe { DirectPlane::from_raw_parts(x_im.as_mut_ptr(), x_im.len()) },
            unsafe { DirectPlane::from_raw_parts(y_re.as_mut_ptr(), y_re.len()) },
            unsafe { DirectPlane::from_raw_parts(y_im.as_mut_ptr(), y_im.len()) },
            unsafe {
                DirectPlane::from_raw_parts(destination_re.as_mut_ptr(), destination_re.len())
            },
            unsafe {
                DirectPlane::from_raw_parts(destination_im.as_mut_ptr(), destination_im.len())
            },
            unsafe {
                DirectPlane::from_raw_parts(destination_re.as_mut_ptr(), destination_re.len())
            },
            unsafe {
                DirectPlane::from_raw_parts(destination_im.as_mut_ptr(), destination_im.len())
            },
        ];
        for (index, (factor_re, factor_im)) in factors.iter().enumerate() {
            let scalars = [
                unsafe { DirectScalar::from_raw(ptr::from_ref(factor_re)) },
                unsafe { DirectScalar::from_raw(ptr::from_ref(factor_im)) },
            ];
            let status = if index == 0 {
                unsafe { callable.handle().invoke(&planes, &scalars, 0, 5) }
            } else {
                unsafe { callable.invoke_unchecked(&planes, &scalars, 0, 5) }
            };
            assert_eq!(status, DIRECT_STATUS_OK);
        }

        let factor_re: f64 = factors.iter().map(|factor| factor.0).sum();
        let factor_im: f64 = factors.iter().map(|factor| factor.1).sum();
        for point in 0..5 {
            let product_re = x_re[point] * y_re[point] - x_im[point] * y_im[point];
            let product_im = x_re[point] * y_im[point] + x_im[point] * y_re[point];
            let expected_re = original_re[point] + factor_re * product_re - factor_im * product_im;
            let expected_im = original_im[point] + factor_re * product_im + factor_im * product_re;
            assert!((destination_re[point] - expected_re).abs() < 1.0e-12);
            assert!((destination_im[point] - expected_im).abs() < 1.0e-12);
        }

        planes.clear();
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn overwrite_replaces_destination_after_complex_scaling() {
        let metadata = metadata(DirectDestinationOperation::Overwrite);
        let callable = DirectApplication::new(source_application(2), metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let mut storage = [
            vec![2.0, 3.0],
            vec![1.0, -1.0],
            vec![4.0, 0.5],
            vec![-2.0, 1.0],
            vec![999.0; 2],
            vec![-999.0; 2],
        ];
        let mut planes = storage
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        planes.push(planes[4]);
        planes.push(planes[5]);
        let factor_re = -0.5;
        let factor_im = 0.25;
        let scalars = [
            unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_re)) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_im)) },
        ];
        let status = unsafe { callable.handle().invoke(&planes, &scalars, 0, 2) };
        assert_eq!(status, DIRECT_STATUS_OK);
        for point in 0..2 {
            let product_re =
                storage[0][point] * storage[2][point] - storage[1][point] * storage[3][point];
            let product_im =
                storage[0][point] * storage[3][point] + storage[1][point] * storage[2][point];
            assert!(
                (storage[4][point] - (factor_re * product_re - factor_im * product_im)).abs()
                    < 1.0e-12
            );
            assert!(
                (storage[5][point] - (factor_re * product_im + factor_im * product_re)).abs()
                    < 1.0e-12
            );
        }
    }

    #[test]
    #[allow(clippy::cloned_ref_to_slice_refs)]
    fn snapshot_overwrite_transforms_aliased_inputs_with_complex_scale() {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_fast_complex(false);
        let current = Expr::var("current");
        let source = Compiler::with_config(config)
            .compile_params(&[], &[current.clone()], &[current])
            .unwrap();
        let metadata = DirectApplicationMetadata::new(
            DirectDestinationOperation::Overwrite,
            DirectInputSnapshot::BeforeWrite,
            DirectOutputScale::ComplexScalar,
            vec![],
            vec![DirectInputBinding::Plane(0), DirectInputBinding::Plane(1)],
            2,
            2,
            vec![0, 1],
        )
        .unwrap();
        let callable = DirectApplication::new(source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let mut current_re = vec![1.0, -2.0];
        let mut current_im = vec![3.0, 0.5];
        let mut planes = vec![
            unsafe { DirectPlane::from_raw_parts(current_re.as_mut_ptr(), current_re.len()) },
            unsafe { DirectPlane::from_raw_parts(current_im.as_mut_ptr(), current_im.len()) },
        ];
        planes.push(planes[0]);
        planes.push(planes[1]);
        let factor_re = -0.25;
        let factor_im = 2.0;
        let scalars = [
            unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_re)) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_im)) },
        ];
        let status = unsafe { callable.handle().invoke(&planes, &scalars, 0, 2) };
        assert_eq!(status, DIRECT_STATUS_OK);
        assert!((current_re[0] - -6.25).abs() < 1.0e-12);
        assert!((current_im[0] - 1.25).abs() < 1.0e-12);
        assert!((current_re[1] - -0.5).abs() < 1.0e-12);
        assert!((current_im[1] - -4.125).abs() < 1.0e-12);
    }

    #[test]
    fn snapshot_mode_reads_all_inputs_before_aliased_component_writes() {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let names = ["current_0", "current_1", "current_2", "current_3"];
        let current = names.map(Expr::var);
        let source = Compiler::with_config(config)
            .compile_params(
                &[],
                &[
                    current[2].clone(),
                    current[3].clone(),
                    current[0].clone(),
                    current[1].clone(),
                ],
                &current,
            )
            .unwrap();
        let metadata = DirectApplicationMetadata::new(
            DirectDestinationOperation::Overwrite,
            DirectInputSnapshot::BeforeWrite,
            DirectOutputScale::ComplexScalar,
            vec![],
            (0..8).map(DirectInputBinding::Plane).collect(),
            16,
            2,
            (8..16).collect(),
        )
        .unwrap();
        let callable = DirectApplication::new(source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();

        let mut storage = [
            vec![1.0, 2.0],
            vec![-1.0, -2.0],
            vec![3.0, 4.0],
            vec![-3.0, -4.0],
            vec![5.0, 6.0],
            vec![-5.0, -6.0],
            vec![7.0, 8.0],
            vec![-7.0, -8.0],
        ];
        let original = storage.clone();
        let mut planes = storage
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        for input in 0..8 {
            planes.push(planes[input]);
        }
        for destination in 8..16 {
            planes.push(planes[destination]);
        }
        let factor_re = 1.0;
        let factor_im = 0.0;
        let scalars = [
            unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_re)) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_im)) },
        ];

        let status = unsafe { callable.handle().invoke(&planes, &scalars, 0, 2) };
        assert_eq!(status, DIRECT_STATUS_OK);
        assert_eq!(storage[0], original[4]);
        assert_eq!(storage[1], original[5]);
        assert_eq!(storage[2], original[6]);
        assert_eq!(storage[3], original[7]);
        assert_eq!(storage[4], original[0]);
        assert_eq!(storage[5], original[1]);
        assert_eq!(storage[6], original[2]);
        assert_eq!(storage[7], original[3]);
    }

    #[test]
    fn portable_source_payload_can_be_transformed_without_duplicate_storage() {
        let source = source_application(2);
        let mut source_bytes = Vec::new();
        source.save(&mut source_bytes).unwrap();
        let direct = DirectApplication::from_source_storage(
            &mut source_bytes.as_slice(),
            &Config::default(),
            metadata(DirectDestinationOperation::Accumulate),
        )
        .unwrap();
        assert_eq!(
            direct.metadata(),
            &metadata(DirectDestinationOperation::Accumulate)
        );
    }

    #[test]
    fn compiled_portable_source_payload_can_be_transformed_without_duplicate_storage() {
        let source = compiled_source_application();
        let metadata = compiled_metadata(&source);
        let mut source_bytes = Vec::new();
        source.save(&mut source_bytes).unwrap();
        let direct = DirectApplication::from_source_storage(
            &mut source_bytes.as_slice(),
            &Config::default(),
            metadata.clone(),
        )
        .unwrap();
        assert_eq!(direct.metadata(), &metadata);
        assert_eq!(direct.source.config.opt_level(), 3);
        assert!(direct.source.config.compress());
        assert!(machine_code_contains_call(&direct.application));
    }

    #[test]
    fn compiled_identity_overwrite_accepts_o0_through_o3_sources_and_lowers_to_o3() {
        for optimization_level in 0..=3 {
            let source = compiled_source_application_at(optimization_level);
            assert_eq!(source.config.opt_level(), optimization_level);
            let metadata = compiled_metadata(&source);
            let direct = DirectApplication::new(source, metadata).unwrap();
            assert_eq!(direct.source_optimization_level(), optimization_level);
            assert_eq!(direct.application.config.opt_level(), 3);
            assert!(direct.application.config.direct_arena());
            assert!(direct.application.config.direct_arena_identity_output());
        }
    }

    #[test]
    fn direct_application_rejects_unoptimized_source_level() {
        let error = DirectApplication::new(
            source_application(1),
            metadata(DirectDestinationOperation::Accumulate),
        )
        .err()
        .expect("O1 must be rejected");
        assert!(error.to_string().contains("O2 source"));
    }

    #[test]
    fn complex_scaled_direct_application_rejects_o3_source_level() {
        let error = DirectApplication::new(
            source_application(3),
            metadata(DirectDestinationOperation::Accumulate),
        )
        .err()
        .expect("complex-scaled applications must remain O2");
        assert!(error.to_string().contains("O2 source"));
    }

    #[test]
    fn aliased_outputs_require_distinct_destination_planes() {
        let error = DirectApplicationMetadata::new(
            DirectDestinationOperation::Accumulate,
            DirectInputSnapshot::Live,
            DirectOutputScale::ComplexScalar,
            vec![],
            vec![
                DirectInputBinding::Plane(0),
                DirectInputBinding::Plane(1),
                DirectInputBinding::Plane(2),
                DirectInputBinding::Plane(3),
            ],
            5,
            2,
            vec![4, 4],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("distinct destination planes"),
            "unexpected duplicate-alias error: {error:#}"
        );
    }

    #[test]
    fn compiled_identity_overwrite_accepts_real_o3_funclets_and_preserves_odd_tails() {
        let source = compiled_source_application();
        assert_eq!(source.config.opt_level(), 3);
        assert!(source.config.compress());
        assert!(
            source
                .bytecode
                .mir
                .code
                .iter()
                .any(|instruction| matches!(instruction, Instruction::ComplexBi { .. })),
            "the compiled Direct-Arena fixture must contain compressible complex multiplication"
        );
        assert!(
            machine_code_contains_call(&source),
            "the O3 source must emit an out-of-line compressed funclet call"
        );
        let metadata = compiled_metadata(&source);
        let mut direct = DirectApplication::new(source, metadata.clone()).unwrap();
        assert_eq!(direct.application.config.opt_level(), 3);
        assert!(direct.application.config.compress());
        assert!(direct.application.config.is_complex());
        assert!(
            machine_code_contains_call(&direct.application),
            "direct lowering must preserve compressed O3 funclet generation"
        );
        let mut storage = Vec::new();
        direct.save(&mut storage).unwrap();
        let mut stored = storage.as_slice();
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            DirectApplication::load(&mut stored, &Config::default())
        }))
        .expect("valid v1 identity/O3 storage must not unwind")
        .expect("valid v1 identity/O3 storage must load");
        assert_eq!(loaded.metadata(), &metadata);
        let loaded_source = &loaded.source;
        assert_eq!(loaded_source.config.opt_level(), 3);
        assert!(loaded_source.config.symbolica());
        assert!(loaded_source.config.is_complex());
        assert!(loaded_source.config.compress());
        assert!(!loaded_source.config.direct_arena());
        assert_eq!(loaded.application.config.opt_level(), 3);
        assert!(loaded.application.config.direct_arena());
        assert!(loaded.application.config.direct_arena_identity_output());
        assert!(machine_code_contains_call(&loaded.application));
        direct = loaded;
        direct.prepare_simd();
        let callable = direct.seal().unwrap().into_callable();

        for point_count in [1_usize, 2, 3, 127, 128, 129] {
            let mut dense_source = compiled_source_application();
            dense_source.prepare_simd();
            let dense = dense_source.seal().unwrap();
            let mut row_major = vec![0.0; point_count * dense.count_params];
            let mut input_planes = vec![vec![0.0; point_count]; dense.count_params];
            for point in 0..point_count {
                let values = [
                    0.25 + point as f64 * 0.03125,
                    if point % 2 == 0 { -0.0 } else { 0.5 },
                    1.5 - point as f64 * 0.0078125,
                    if point % 3 == 0 { 0.0 } else { -0.25 },
                ];
                for (parameter, value) in values.into_iter().enumerate() {
                    row_major[point * dense.count_params + parameter] = value;
                    input_planes[parameter][point] = value;
                }
            }
            let mut expected = vec![f64::NAN; point_count * dense.count_obs];
            dense.evaluate_matrix(&row_major, &mut expected, point_count);

            let mut output_planes = vec![vec![f64::NAN; point_count]; dense.count_obs];
            let mut planes = input_planes
                .iter_mut()
                .chain(output_planes.iter_mut())
                .map(|values| unsafe {
                    DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
                })
                .collect::<Vec<_>>();
            let status = unsafe {
                callable
                    .handle()
                    .invoke(&planes, &[], 0, point_count as u32)
            };
            assert_eq!(status, DIRECT_STATUS_OK);
            for point in 0..point_count {
                for output in 0..dense.count_obs {
                    assert_eq!(
                        output_planes[output][point].to_bits(),
                        expected[point * dense.count_obs + output].to_bits(),
                        "compiled Direct-Arena mismatch at point {point}, output {output}, count {point_count}"
                    );
                }
            }
            planes.clear();
        }
    }

    #[test]
    fn compiled_identity_overwrite_is_factor_free_and_preserves_output_bits() {
        let mut config = Config::default();
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let instructions = r#"[[{"Assign":[{"Out":0},{"Param":0}]}],1,[]]"#;
        let source = Compiler::with_config(config)
            .translate(instructions.to_owned(), 1)
            .unwrap();
        let metadata = compiled_metadata(&source);
        assert_eq!(metadata.scalar_input_count, 0);
        let callable = DirectApplication::new(source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let edge_bits = [
            0_u64,
            (-0.0_f64).to_bits(),
            1_u64,
            (f64::MIN_POSITIVE).to_bits(),
            0x7ff8_0000_0000_1234,
            0xfff8_0000_0000_5678,
        ];
        let mut input_re = edge_bits.map(f64::from_bits).to_vec();
        let mut input_im = edge_bits
            .into_iter()
            .rev()
            .map(f64::from_bits)
            .collect::<Vec<_>>();
        let mut output_re = vec![1.0; input_re.len()];
        let mut output_im = vec![1.0; input_re.len()];
        let mut planes = vec![
            unsafe { DirectPlane::from_raw_parts(input_re.as_mut_ptr(), input_re.len()) },
            unsafe { DirectPlane::from_raw_parts(input_im.as_mut_ptr(), input_im.len()) },
            unsafe { DirectPlane::from_raw_parts(output_re.as_mut_ptr(), output_re.len()) },
            unsafe { DirectPlane::from_raw_parts(output_im.as_mut_ptr(), output_im.len()) },
        ];
        assert_eq!(
            unsafe {
                callable
                    .handle()
                    .invoke(&planes, &[], 0, input_re.len() as u32)
            },
            DIRECT_STATUS_OK
        );
        assert_eq!(
            output_re
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            input_re
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            output_im
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            input_im
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        planes.clear();
    }

    #[test]
    fn identity_scale_can_accumulate_into_distinct_output_planes() {
        let mut config = Config::default();
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let instructions = r#"[[{"Assign":[{"Out":0},{"Param":0}]}],1,[]]"#;
        let source = Compiler::with_config(config)
            .translate(instructions.to_owned(), 1)
            .unwrap();
        let mut metadata = compiled_metadata(&source);
        metadata.destination_operation = DirectDestinationOperation::Accumulate;
        let callable = DirectApplication::new(source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();

        let mut input_re = vec![1.0, -2.0, 3.5];
        let mut input_im = vec![4.0, 5.0, -6.0];
        let mut output_re = vec![10.0, 20.0, 30.0];
        let mut output_im = vec![-10.0, -20.0, -30.0];
        let planes = vec![
            unsafe { DirectPlane::from_raw_parts(input_re.as_mut_ptr(), input_re.len()) },
            unsafe { DirectPlane::from_raw_parts(input_im.as_mut_ptr(), input_im.len()) },
            unsafe { DirectPlane::from_raw_parts(output_re.as_mut_ptr(), output_re.len()) },
            unsafe { DirectPlane::from_raw_parts(output_im.as_mut_ptr(), output_im.len()) },
        ];
        assert_eq!(
            unsafe { callable.handle().invoke(&planes, &[], 0, 3) },
            DIRECT_STATUS_OK
        );
        assert_eq!(output_re, vec![11.0, 18.0, 33.5]);
        assert_eq!(output_im, vec![-6.0, -15.0, -36.0]);
    }

    #[test]
    fn compiled_identity_overwrite_rejects_input_output_aliasing() {
        let source = compiled_source_application();
        let metadata = compiled_metadata(&source);
        let callable = DirectApplication::new(source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let mut storage = vec![0.0; 8];
        let mut planes = (0..8)
            .map(|_| unsafe { DirectPlane::from_raw_parts(storage.as_mut_ptr(), storage.len()) })
            .collect::<Vec<_>>();
        assert_eq!(
            unsafe { callable.handle().invoke(&planes, &[], 0, 1) },
            DIRECT_STATUS_EXECUTION_FAILED
        );
        planes.clear();

        let mut inputs = vec![vec![0.0; 8]; 4];
        let mut distinct_outputs = vec![vec![0.0; 8]; 3];
        let mut partially_overlapping = inputs
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        partially_overlapping.push(unsafe {
            DirectPlane::from_raw_parts(inputs[0].as_mut_ptr().add(1), inputs[0].len() - 1)
        });
        partially_overlapping.extend(distinct_outputs.iter_mut().map(|values| unsafe {
            DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
        }));
        assert_eq!(
            unsafe { callable.handle().invoke(&partially_overlapping, &[], 0, 4) },
            DIRECT_STATUS_EXECUTION_FAILED
        );
        partially_overlapping.clear();

        // The active [0, 4) tiles do not overlap, but output 0 points into the
        // future [8, 16) portion of input 0's declared plane. A later tile
        // would otherwise read values already overwritten by this call.
        let mut future_input = vec![0.0; 16];
        let mut other_inputs = vec![vec![0.0; 16]; 3];
        let mut future_outputs = vec![vec![0.0; 16]; 3];
        let mut future_overlap = vec![unsafe {
            DirectPlane::from_raw_parts(future_input.as_mut_ptr(), future_input.len())
        }];
        future_overlap.extend(other_inputs.iter_mut().map(|values| unsafe {
            DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
        }));
        future_overlap
            .push(unsafe { DirectPlane::from_raw_parts(future_input.as_mut_ptr().add(8), 8) });
        future_overlap.extend(future_outputs.iter_mut().map(|values| unsafe {
            DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
        }));
        assert_eq!(
            unsafe { callable.handle().invoke(&future_overlap, &[], 0, 4) },
            DIRECT_STATUS_EXECUTION_FAILED
        );
        future_overlap.clear();

        // Likewise, output planes must not overlap anywhere in their declared
        // ranges even when the active tiles happen to be disjoint.
        let mut output_overlap_inputs = vec![vec![0.0; 16]; 4];
        let mut output_backing = vec![0.0; 24];
        let mut output_tail = vec![vec![0.0; 16]; 2];
        let mut output_overlap = output_overlap_inputs
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        output_overlap
            .push(unsafe { DirectPlane::from_raw_parts(output_backing.as_mut_ptr(), 16) });
        output_overlap
            .push(unsafe { DirectPlane::from_raw_parts(output_backing.as_mut_ptr().add(8), 16) });
        output_overlap.extend(output_tail.iter_mut().map(|values| unsafe {
            DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
        }));
        assert_eq!(
            unsafe { callable.handle().invoke(&output_overlap, &[], 0, 4) },
            DIRECT_STATUS_EXECUTION_FAILED
        );
        output_overlap.clear();
    }

    #[test]
    fn compiled_identity_overwrite_rejects_scalar_output_aliasing() {
        let mut config = Config::default();
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let instructions = r#"[[{"Assign":[{"Out":0},{"Param":0}]}],1,[]]"#;
        let source = Compiler::with_config(config)
            .translate(instructions.to_owned(), 1)
            .unwrap();
        assert_eq!(source.count_params, 2);
        assert_eq!(source.count_obs, 2);
        let metadata = DirectApplicationMetadata::new(
            DirectDestinationOperation::Overwrite,
            DirectInputSnapshot::Live,
            DirectOutputScale::Identity,
            vec![],
            vec![DirectInputBinding::Scalar(0), DirectInputBinding::Scalar(1)],
            0,
            2,
            vec![DIRECT_NO_ALIAS, DIRECT_NO_ALIAS],
        )
        .unwrap();
        let callable = DirectApplication::new(source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();

        let mut output_re = vec![-7.0; 16];
        let mut output_im = vec![11.0; 16];
        let planes = vec![
            unsafe { DirectPlane::from_raw_parts(output_re.as_mut_ptr(), output_re.len()) },
            unsafe { DirectPlane::from_raw_parts(output_im.as_mut_ptr(), output_im.len()) },
        ];
        let input_re = 0.25;
        let input_im = -0.5;
        let valid_scalars = [
            unsafe { DirectScalar::from_raw(ptr::from_ref(&input_re)) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&input_im)) },
        ];
        assert_eq!(
            unsafe { callable.handle().invoke(&planes, &valid_scalars, 0, 4) },
            DIRECT_STATUS_OK
        );

        // Pointer equality at the start of an output plane must fail closed.
        let exact_overlap = [
            unsafe { DirectScalar::from_raw(output_re.as_ptr()) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&input_im)) },
        ];
        assert_eq!(
            unsafe { callable.handle().invoke(&planes, &exact_overlap, 0, 4) },
            DIRECT_STATUS_EXECUTION_FAILED
        );

        // The active tile is [0, 4), but scalar 1 points into the future tail
        // of output 1's complete declared [0, 16) range.
        let future_partial_overlap = [
            unsafe { DirectScalar::from_raw(ptr::from_ref(&input_re)) },
            unsafe { DirectScalar::from_raw(output_im.as_ptr().add(8)) },
        ];
        assert_eq!(
            unsafe {
                callable
                    .handle()
                    .invoke(&planes, &future_partial_overlap, 0, 4)
            },
            DIRECT_STATUS_EXECUTION_FAILED
        );
    }

    #[test]
    fn direct_storage_v1_rejects_malformed_and_nonportable_sources_without_unwinding() {
        let source = compiled_source_application();
        let metadata = compiled_metadata(&source);
        let mut direct = DirectApplication::new(source, metadata).unwrap();
        let metadata_offset = stored_metadata_offset(&direct);
        let mut valid = Vec::new();
        direct.save(&mut valid).unwrap();

        let mut superseded_v2 = valid.clone();
        superseded_v2[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert_storage_load_fails_without_panic(
            &superseded_v2,
            "unsupported direct application storage version 2",
        );

        let mut unknown_scale = valid.clone();
        unknown_scale[metadata_offset + 2] = u8::MAX;
        assert_storage_load_fails_without_panic(&unknown_scale, "unknown direct output scale");

        let mut truncated = valid.clone();
        truncated.truncate(metadata_offset + 2);
        assert_storage_load_fails_without_panic(&truncated, "failed to fill whole buffer");

        let mut oversized_state_count = valid.clone();
        oversized_state_count[metadata_offset + 11..metadata_offset + 15]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert_storage_load_fails_without_panic(
            &oversized_state_count,
            "state binding count 4294967295 does not match source count 0",
        );

        direct.source.prog.builder.config.set_direct_arena(true);
        let mut invalid_config = Vec::new();
        direct.save(&mut invalid_config).unwrap();
        assert_storage_load_fails_without_panic(&invalid_config, "portable native Symbolica");

        let source = compiled_source_application();
        let metadata = compiled_metadata(&source);
        let mut source_storage = Vec::new();
        source.save(&mut source_storage).unwrap();
        source_storage.truncate(source_storage.len() / 2);
        let mut stored_source = source_storage.as_slice();
        let source_attempt = catch_unwind(AssertUnwindSafe(|| {
            DirectApplication::from_source_storage(&mut stored_source, &Config::default(), metadata)
        }));
        let source_result =
            source_attempt.expect("malformed shared source storage must not unwind");
        assert!(
            source_result.is_err(),
            "malformed shared source storage must fail closed"
        );
    }

    #[test]
    fn warmed_compiled_plane_calls_perform_zero_allocations() {
        let source = compiled_source_application();
        let metadata = compiled_metadata(&source);
        let callable = DirectApplication::new(source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let mut storage = vec![vec![0.0; 129]; 8];
        let planes = storage
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        let handle = callable.handle();

        assert_eq!(
            unsafe { handle.invoke(&planes, &[], 0, 129) },
            DIRECT_STATUS_OK
        );
        assert_eq!(
            unsafe { callable.invoke_unchecked(&planes, &[], 0, 129) },
            DIRECT_STATUS_OK
        );

        let (checked_status, checked_allocations) =
            crate::allocation_probe::count_allocations(|| {
                let mut status = DIRECT_STATUS_OK;
                for _ in 0..128 {
                    status |= unsafe { handle.invoke(&planes, &[], 0, 129) };
                }
                status
            });
        let (unchecked_status, unchecked_allocations) =
            crate::allocation_probe::count_allocations(|| {
                let mut status = DIRECT_STATUS_OK;
                for _ in 0..128 {
                    status |= unsafe { callable.invoke_unchecked(&planes, &[], 0, 129) };
                }
                status
            });
        assert_eq!(checked_status, DIRECT_STATUS_OK);
        assert_eq!(unchecked_status, DIRECT_STATUS_OK);
        assert_eq!(checked_allocations, 0);
        assert_eq!(unchecked_allocations, 0);
    }

    #[test]
    fn direct_callable_rejects_wrong_destination_alias() {
        let callable = DirectApplication::new(
            source_application(2),
            metadata(DirectDestinationOperation::Accumulate),
        )
        .unwrap()
        .seal()
        .unwrap()
        .into_callable();
        let mut storage = vec![0.0; 8 * 2];
        let mut planes = (0..8)
            .map(|index| unsafe {
                DirectPlane::from_raw_parts(storage.as_mut_ptr().add(index * 2), 2)
            })
            .collect::<Vec<_>>();
        let scalar = 1.0;
        let scalars = [
            unsafe { DirectScalar::from_raw(ptr::from_ref(&scalar)) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&scalar)) },
        ];
        let status = unsafe { callable.handle().invoke(&planes, &scalars, 0, 2) };
        assert_eq!(status, DIRECT_STATUS_EXECUTION_FAILED);
        planes.clear();
    }

    #[test]
    fn direct_descriptor_layout_matches_symjit_indirect_planes() {
        assert_eq!(
            std::mem::size_of::<DirectPlane>(),
            std::mem::size_of::<&mut [f64]>()
        );
        assert_eq!(
            std::mem::align_of::<DirectPlane>(),
            std::mem::align_of::<&mut [f64]>()
        );
        assert_eq!(
            std::mem::size_of::<DirectScalar>(),
            std::mem::size_of::<*const f64>()
        );
    }
}
