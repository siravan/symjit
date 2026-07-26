//! Portable descriptors and native execution for table-aware Direct-Arena.
//!
//! This is an additive ABI: the existing per-row [`crate::DirectCallable`]
//! remains unchanged. A table callable receives opaque fixed-width invocation
//! and attachment rows. Its generated machine function owns the row, point,
//! and attachment loops, while the source MIR body is emitted exactly once
//! inside the point loop. Descriptor construction, serialization, and checked
//! call-view validation are architecture-neutral. Sealing and invoking the
//! generated callable currently require AArch64 or x86-64.

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::mem;
use std::ptr;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};

#[cfg(target_arch = "x86_64")]
use crate::amd::{AmdScalarGenerator, AmdVectorF64x4Generator};
#[cfg(target_arch = "aarch64")]
use crate::arm::{ArmGenerator, ArmSimdGenerator};
use crate::config::SPILL_AREA;
use crate::direct::{
    DirectPlane, DirectScalar, DIRECT_STATUS_EXECUTION_FAILED, DIRECT_STATUS_INVALID_ARGUMENT,
    DIRECT_STATUS_INVALID_CONTEXT, DIRECT_STATUS_OK,
};
use crate::generator::Generator;
use crate::memory::{BranchProtection, Memory};
use crate::mir::{Instruction, Mir};
use crate::model::Program;
use crate::runnable::{Application, CompilerType};
use crate::serializer::MirWriter;
use crate::symbol::Loc;

pub const DIRECT_TABLE_BINDING_ABI: &str = "symjit-direct-table-binding-v1";
pub const DIRECT_TABLE_DESCRIPTOR_ABI: &str = "symjit-direct-table-descriptor-v1";
pub const DIRECT_TABLE_DESCRIPTOR_MAX_BYTES: usize = 16 * 1024;

#[cfg(target_arch = "aarch64")]
const DIRECT_TABLE_SIMD_LANES: u32 = 2;
#[cfg(target_arch = "x86_64")]
const DIRECT_TABLE_SIMD_LANES: u32 = 4;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const DIRECT_TABLE_SIMD_LANES: u32 = 1;
const MAX_ROW_STRIDE: u32 = 16_384;
const MAX_INPUT_PLANES: usize = 256;
const MAX_PARAMETER_BINDINGS: usize = 256;
const MAX_OUTPUT_COMPONENTS: usize = 1024;
const DIRECT_TABLE_DESCRIPTOR_MAGIC: [u8; 8] = *b"SJTD0001";
const DIRECT_TABLE_DESCRIPTOR_VERSION: u32 = 1;

/// Per-attachment destination semantics stored as a native `u32` row field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum DirectTableDestinationOperation {
    Overwrite = 0,
    Accumulate = 1,
}

/// Maps one source parameter into the table invocation ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectTableParameterBinding {
    /// Read a point-dependent plane selected by this invocation-row binding.
    Plane(u32),
    /// Read a point-independent value through this scalar catalog descriptor.
    Scalar(u32),
}

/// Authenticated locations of fields in one opaque invocation row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectTableInvocationLayout {
    pub row_stride: u32,
    pub input_plane_offsets: Vec<u32>,
    pub attachment_start_offset: u32,
    pub attachment_count_offset: u32,
}

impl DirectTableInvocationLayout {
    pub fn new(
        row_stride: u32,
        input_plane_offsets: Vec<u32>,
        attachment_start_offset: u32,
        attachment_count_offset: u32,
    ) -> Result<Self> {
        let layout = Self {
            row_stride,
            input_plane_offsets,
            attachment_start_offset,
            attachment_count_offset,
        };
        layout.validate()?;
        Ok(layout)
    }

    fn validate(&self) -> Result<()> {
        validate_stride("invocation", self.row_stride)?;
        if self.input_plane_offsets.is_empty() {
            bail!("direct-table invocation layout must bind at least one input plane");
        }
        if self.input_plane_offsets.len() > MAX_INPUT_PLANES {
            bail!(
                "direct-table invocation layout has {} input planes, maximum is {MAX_INPUT_PLANES}",
                self.input_plane_offsets.len()
            );
        }
        for (index, &offset) in self.input_plane_offsets.iter().enumerate() {
            validate_u32_field("invocation input-plane", index, offset, self.row_stride)?;
        }
        validate_u32_field(
            "invocation attachment-start",
            0,
            self.attachment_start_offset,
            self.row_stride,
        )?;
        validate_u32_field(
            "invocation attachment-count",
            0,
            self.attachment_count_offset,
            self.row_stride,
        )?;
        validate_distinct_u32_fields(
            "invocation",
            self.input_plane_offsets
                .iter()
                .copied()
                .chain([self.attachment_start_offset, self.attachment_count_offset]),
        )
    }
}

/// Authenticated locations of fields in one opaque attachment row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectTableAttachmentLayout {
    pub row_stride: u32,
    /// Split destination-plane fields in source output order:
    /// `[output0.re, output0.im, output1.re, output1.im, ...]`.
    pub destination_plane_offsets: Vec<u32>,
    pub scale_offset: u32,
    pub operation_offset: u32,
}

impl DirectTableAttachmentLayout {
    pub fn new(
        row_stride: u32,
        destination_real_offset: u32,
        destination_imag_offset: u32,
        scale_offset: u32,
        operation_offset: u32,
    ) -> Result<Self> {
        Self::new_with_destination_plane_offsets(
            row_stride,
            vec![destination_real_offset, destination_imag_offset],
            scale_offset,
            operation_offset,
        )
    }

    pub fn new_with_destination_plane_offsets(
        row_stride: u32,
        destination_plane_offsets: Vec<u32>,
        scale_offset: u32,
        operation_offset: u32,
    ) -> Result<Self> {
        let layout = Self {
            row_stride,
            destination_plane_offsets,
            scale_offset,
            operation_offset,
        };
        layout.validate()?;
        Ok(layout)
    }

    fn validate(&self) -> Result<()> {
        validate_stride("attachment", self.row_stride)?;
        if self.destination_plane_offsets.is_empty()
            || !self.destination_plane_offsets.len().is_multiple_of(2)
        {
            bail!(
                "direct-table attachment layout must contain complete split-complex destinations"
            );
        }
        if self.destination_plane_offsets.len() > MAX_OUTPUT_COMPONENTS {
            bail!(
                "direct-table attachment layout has {} output components, maximum is {MAX_OUTPUT_COMPONENTS}",
                self.destination_plane_offsets.len()
            );
        }
        for (component, &offset) in self.destination_plane_offsets.iter().enumerate() {
            validate_u32_field("attachment destination", component, offset, self.row_stride)?;
        }
        validate_u32_field("attachment scale", 0, self.scale_offset, self.row_stride)?;
        validate_u32_field(
            "attachment operation",
            0,
            self.operation_offset,
            self.row_stride,
        )?;
        validate_distinct_u32_fields(
            "attachment",
            self.destination_plane_offsets
                .iter()
                .copied()
                .chain([self.scale_offset, self.operation_offset]),
        )?;
        Ok(())
    }
}

/// Portable, model-fixed binding metadata for the two opaque row tables.
///
/// Source states are implicit leading invocation-plane bindings; each source
/// parameter is explicitly bound to an invocation-selected point plane or a
/// fixed scalar descriptor. Every attachment maps the complete split-complex
/// source output vector to one ordered destination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectTableApplicationMetadata {
    pub invocation: DirectTableInvocationLayout,
    pub attachment: DirectTableAttachmentLayout,
    pub parameter_bindings: Vec<DirectTableParameterBinding>,
    pub scalar_input_count: u32,
}

impl DirectTableApplicationMetadata {
    pub fn new(
        invocation: DirectTableInvocationLayout,
        attachment: DirectTableAttachmentLayout,
        scalar_input_count: u32,
    ) -> Result<Self> {
        let parameter_count = usize::try_from(scalar_input_count)
            .context("direct-table scalar binding count exceeds usize")?;
        if parameter_count > MAX_PARAMETER_BINDINGS {
            bail!(
                "direct-table metadata has {parameter_count} scalar bindings, maximum is {MAX_PARAMETER_BINDINGS}"
            );
        }
        let mut parameter_bindings = Vec::new();
        parameter_bindings
            .try_reserve_exact(parameter_count)
            .context("cannot reserve direct-table scalar bindings")?;
        parameter_bindings.extend((0..scalar_input_count).map(DirectTableParameterBinding::Scalar));
        Self::new_with_parameter_bindings(
            invocation,
            attachment,
            parameter_bindings,
            scalar_input_count,
        )
    }

    pub fn new_with_parameter_bindings(
        invocation: DirectTableInvocationLayout,
        attachment: DirectTableAttachmentLayout,
        parameter_bindings: Vec<DirectTableParameterBinding>,
        scalar_input_count: u32,
    ) -> Result<Self> {
        let metadata = Self {
            invocation,
            attachment,
            parameter_bindings,
            scalar_input_count,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<()> {
        self.invocation.validate()?;
        self.attachment.validate()?;
        if self.parameter_bindings.len() > MAX_PARAMETER_BINDINGS {
            bail!(
                "direct-table metadata has {} parameter bindings, maximum is {MAX_PARAMETER_BINDINGS}",
                self.parameter_bindings.len()
            );
        }
        let mut scalars = BTreeSet::new();
        for (parameter, binding) in self.parameter_bindings.iter().enumerate() {
            match *binding {
                DirectTableParameterBinding::Plane(plane) => {
                    if plane as usize >= self.invocation.input_plane_offsets.len() {
                        bail!(
                            "direct-table parameter {parameter} plane binding {plane} is out of bounds"
                        );
                    }
                }
                DirectTableParameterBinding::Scalar(scalar) => {
                    if scalar >= self.scalar_input_count {
                        bail!(
                            "direct-table parameter {parameter} scalar binding {scalar} is out of bounds"
                        );
                    }
                    scalars.insert(scalar);
                }
            }
        }
        if scalars.len() != self.scalar_input_count as usize {
            bail!("direct-table scalar parameter bindings must densely cover the scalar catalog");
        }
        Ok(())
    }

    /// Encode only portable, fixed-width descriptor metadata.
    ///
    /// The separately loaded immutable `source` is authoritative for state,
    /// parameter, and output counts; executable bytes and pointers are never
    /// included in this descriptor.
    pub fn encode_descriptor(&self, source: &Application) -> Result<Vec<u8>> {
        self.validate_source(source)?;
        let source_state_count = descriptor_source_count(source.count_states, "state")?;
        let source_parameter_count = descriptor_source_count(source.count_params, "parameter")?;
        let source_output_count = descriptor_source_count(source.count_obs, "output component")?;
        let encoded_bytes = descriptor_encoded_len(
            self.invocation.input_plane_offsets.len(),
            self.attachment.destination_plane_offsets.len(),
            self.parameter_bindings.len(),
        )?;
        let encoded_bytes_u32 = u32::try_from(encoded_bytes)
            .context("direct-table descriptor byte length exceeds u32")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(encoded_bytes)
            .context("cannot reserve direct-table descriptor bytes")?;
        bytes.extend_from_slice(&DIRECT_TABLE_DESCRIPTOR_MAGIC);
        push_descriptor_u32(&mut bytes, DIRECT_TABLE_DESCRIPTOR_VERSION);
        push_descriptor_u32(&mut bytes, encoded_bytes_u32);
        push_descriptor_u32(&mut bytes, source_state_count);
        push_descriptor_u32(&mut bytes, source_parameter_count);
        push_descriptor_u32(&mut bytes, source_output_count);
        push_descriptor_u32(&mut bytes, self.invocation.row_stride);
        push_descriptor_u32(
            &mut bytes,
            u32::try_from(self.invocation.input_plane_offsets.len())
                .context("direct-table invocation input count exceeds u32")?,
        );
        for &offset in &self.invocation.input_plane_offsets {
            push_descriptor_u32(&mut bytes, offset);
        }
        push_descriptor_u32(&mut bytes, self.invocation.attachment_start_offset);
        push_descriptor_u32(&mut bytes, self.invocation.attachment_count_offset);
        push_descriptor_u32(&mut bytes, self.attachment.row_stride);
        push_descriptor_u32(
            &mut bytes,
            u32::try_from(self.attachment.destination_plane_offsets.len())
                .context("direct-table output component count exceeds u32")?,
        );
        for &offset in &self.attachment.destination_plane_offsets {
            push_descriptor_u32(&mut bytes, offset);
        }
        push_descriptor_u32(&mut bytes, self.attachment.scale_offset);
        push_descriptor_u32(&mut bytes, self.attachment.operation_offset);
        push_descriptor_u32(&mut bytes, self.scalar_input_count);
        push_descriptor_u32(
            &mut bytes,
            u32::try_from(self.parameter_bindings.len())
                .context("direct-table parameter binding count exceeds u32")?,
        );
        for binding in &self.parameter_bindings {
            let (kind, value) = match *binding {
                DirectTableParameterBinding::Plane(value) => (0, value),
                DirectTableParameterBinding::Scalar(value) => (1, value),
            };
            push_descriptor_u32(&mut bytes, kind);
            push_descriptor_u32(&mut bytes, value);
        }
        debug_assert_eq!(bytes.len(), encoded_bytes);
        Ok(bytes)
    }

    /// Decode bounded descriptor bytes against an already loaded source.
    pub fn decode_descriptor(source: &Application, bytes: &[u8]) -> Result<Self> {
        if bytes.len() > DIRECT_TABLE_DESCRIPTOR_MAX_BYTES {
            bail!(
                "direct-table descriptor has {} bytes, maximum is {DIRECT_TABLE_DESCRIPTOR_MAX_BYTES}",
                bytes.len()
            );
        }
        let mut reader = DirectTableDescriptorReader::new(bytes);
        if reader.read_array::<8>()? != DIRECT_TABLE_DESCRIPTOR_MAGIC {
            bail!("invalid direct-table descriptor magic");
        }
        let version = reader.read_u32()?;
        if version != DIRECT_TABLE_DESCRIPTOR_VERSION {
            bail!("unsupported direct-table descriptor version {version}");
        }
        let declared_bytes = reader.read_u32()? as usize;
        if declared_bytes != bytes.len() {
            bail!(
                "direct-table descriptor declares {declared_bytes} bytes, received {}",
                bytes.len()
            );
        }

        read_expected_source_count(&mut reader, source.count_states, "state")?;
        read_expected_source_count(&mut reader, source.count_params, "parameter")?;
        read_expected_source_count(&mut reader, source.count_obs, "output component")?;

        let invocation_stride = reader.read_u32()?;
        let input_count = reader.read_bounded_count(MAX_INPUT_PLANES, "invocation input")?;
        let mut input_plane_offsets = Vec::new();
        input_plane_offsets
            .try_reserve_exact(input_count)
            .context("cannot reserve direct-table invocation input offsets")?;
        for _ in 0..input_count {
            input_plane_offsets.push(reader.read_u32()?);
        }
        let attachment_start_offset = reader.read_u32()?;
        let attachment_count_offset = reader.read_u32()?;

        let attachment_stride = reader.read_u32()?;
        let output_component_count =
            reader.read_bounded_count(MAX_OUTPUT_COMPONENTS, "output component")?;
        if output_component_count != source.count_obs {
            bail!(
                "direct-table output component count {output_component_count} does not match source count {}",
                source.count_obs
            );
        }
        let mut destination_plane_offsets = Vec::new();
        destination_plane_offsets
            .try_reserve_exact(output_component_count)
            .context("cannot reserve direct-table destination offsets")?;
        for _ in 0..output_component_count {
            destination_plane_offsets.push(reader.read_u32()?);
        }
        let scale_offset = reader.read_u32()?;
        let operation_offset = reader.read_u32()?;
        let scalar_input_count = reader.read_u32()?;

        let parameter_count =
            reader.read_bounded_count(MAX_PARAMETER_BINDINGS, "parameter binding")?;
        if parameter_count != source.count_params {
            bail!(
                "direct-table parameter binding count {parameter_count} does not match source count {}",
                source.count_params
            );
        }
        let mut parameter_bindings = Vec::new();
        parameter_bindings
            .try_reserve_exact(parameter_count)
            .context("cannot reserve direct-table parameter bindings")?;
        for parameter in 0..parameter_count {
            let kind = reader.read_u32()?;
            let value = reader.read_u32()?;
            parameter_bindings.push(match kind {
                0 => DirectTableParameterBinding::Plane(value),
                1 => DirectTableParameterBinding::Scalar(value),
                _ => bail!("direct-table parameter {parameter} has unknown binding kind {kind}"),
            });
        }
        reader.finish()?;

        let invocation = DirectTableInvocationLayout::new(
            invocation_stride,
            input_plane_offsets,
            attachment_start_offset,
            attachment_count_offset,
        )?;
        let attachment = DirectTableAttachmentLayout::new_with_destination_plane_offsets(
            attachment_stride,
            destination_plane_offsets,
            scale_offset,
            operation_offset,
        )?;
        let metadata = Self::new_with_parameter_bindings(
            invocation,
            attachment,
            parameter_bindings,
            scalar_input_count,
        )?;
        metadata.validate_source(source)?;
        Ok(metadata)
    }

    /// Validate one raw call view without constructing executable code.
    ///
    /// This validation path is architecture-neutral.
    ///
    /// # Safety
    ///
    /// Every raw pointer in `view` must denote live storage matching its count
    /// and stride for the duration of this call.
    pub unsafe fn validate_call_view(&self, view: &DirectTableCallViewV1) -> Result<()> {
        // Unlike sealed applets, this metadata remains caller-owned and its
        // public layout fields may have changed since construction. Recheck
        // every offset and bound before interpreting any row pointer.
        self.validate()?;
        validate_call_view_authenticated(self, view)
    }

    fn validate_source(&self, source: &Application) -> Result<()> {
        self.validate()?;
        if (!matches!(source.config.ty, CompilerType::Native)
            && !(cfg!(target_arch = "x86_64") && matches!(source.config.ty, CompilerType::AmdAVX)))
            || !matches!(source.config.opt_level(), 2 | 3)
            || !source.config.symbolica()
            || !source.config.is_complex()
        {
            bail!(
                "direct-table applications require a portable optimized native Symbolica complex source"
            );
        }
        if !source.prog.builder.ft.is_empty() {
            bail!("direct-table MRE does not support external function calls");
        }
        if source.count_states > self.invocation.input_plane_offsets.len() {
            bail!(
                "direct-table input binding count {} is smaller than source state count {}",
                self.invocation.input_plane_offsets.len(),
                source.count_states
            );
        }
        if source.count_params != self.parameter_bindings.len() {
            bail!(
                "direct-table parameter binding count {} does not match source parameter count {}",
                self.parameter_bindings.len(),
                source.count_params
            );
        }
        let mut planes = (0..source.count_states as u32).collect::<BTreeSet<_>>();
        for binding in &self.parameter_bindings {
            if let DirectTableParameterBinding::Plane(plane) = *binding {
                planes.insert(plane);
            }
        }
        if planes.len() != self.invocation.input_plane_offsets.len() {
            bail!(
                "direct-table state and parameter bindings must densely cover every invocation plane"
            );
        }
        if source.count_obs == 0
            || !source.count_obs.is_multiple_of(2)
            || source.count_obs != self.attachment.destination_plane_offsets.len()
            || source.count_diffs != 0
        {
            bail!(
                "direct-table applications require one or more complete complex outputs, matching attachment destinations, and no differentials"
            );
        }
        let stack_slots = source
            .prog
            .builder
            .count_stack
            .unwrap_or_else(|| source.prog.builder.block_shared().sym_table.num_stack);
        let mut output_stores = vec![0_u32; source.count_obs / 2];
        for instruction in source.bytecode.mir.code.iter() {
            match instruction {
                Instruction::Call { .. }
                | Instruction::Branch { .. }
                | Instruction::BranchIf { .. } => {
                    bail!("direct-table MRE requires call-free, branch-free source MIR");
                }
                Instruction::Load { loc, .. }
                | Instruction::LoadMath { loc, .. }
                | Instruction::IfElse { cond: loc, .. } => validate_source_read_location(
                    loc,
                    1,
                    source.count_states,
                    source.count_params,
                    stack_slots,
                )?,
                Instruction::LoadComplex { loc, .. } => validate_source_read_location(
                    loc,
                    2,
                    source.count_states,
                    source.count_params,
                    stack_slots,
                )?,
                Instruction::Save { loc, .. } => {
                    validate_source_write_location(
                        loc,
                        1,
                        source.count_states,
                        source.count_obs,
                        stack_slots,
                    )?;
                }
                Instruction::SaveComplex { loc, .. } => {
                    if let Some(output) = validate_source_write_location(
                        loc,
                        2,
                        source.count_states,
                        source.count_obs,
                        stack_slots,
                    )? {
                        output_stores[output] += 1;
                    }
                }
                Instruction::LoadConst { idx, .. } | Instruction::LoadConstMath { idx, .. }
                    if idx as usize >= source.bytecode.mir.consts.len() =>
                {
                    bail!("direct-table source constant index {idx} is out of bounds");
                }
                _ => {}
            }
        }
        for (output, stores) in output_stores.into_iter().enumerate() {
            if stores != 1 {
                bail!(
                    "direct-table complex output {output} must be stored exactly once, found {stores} stores"
                );
            }
        }
        Ok(())
    }
}

fn validate_source_read_location(
    location: Loc,
    width: usize,
    state_count: usize,
    parameter_count: usize,
    stack_slots: usize,
) -> Result<()> {
    let (name, index, bound) = match location {
        Loc::Mem(index) => ("input plane", index, state_count),
        Loc::Param(index) => ("scalar parameter", index, parameter_count),
        Loc::Stack(index) => ("stack slot", index, stack_slots),
    };
    let end = (index as usize)
        .checked_add(width)
        .ok_or_else(|| anyhow!("direct-table source {name} range overflows"))?;
    if end > bound {
        bail!(
            "direct-table source {name} range {}..{end} exceeds {bound}",
            index
        );
    }
    Ok(())
}

fn validate_source_write_location(
    location: Loc,
    width: usize,
    state_count: usize,
    output_component_count: usize,
    stack_slots: usize,
) -> Result<Option<usize>> {
    match location {
        Loc::Stack(index) => {
            let end = (index as usize)
                .checked_add(width)
                .ok_or_else(|| anyhow!("direct-table source stack store range overflows"))?;
            if end > stack_slots {
                bail!(
                    "direct-table source stack store range {}..{end} exceeds {stack_slots}",
                    index
                );
            }
            Ok(None)
        }
        Loc::Mem(index)
            if width == 2
                && index as usize >= state_count
                && (index as usize - state_count).is_multiple_of(2)
                && (index as usize - state_count)
                    .checked_add(width)
                    .is_some_and(|end| end <= output_component_count) =>
        {
            Ok(Some((index as usize - state_count) / 2))
        }
        Loc::Mem(index) => {
            bail!("direct-table source has unsupported memory store at plane {index}");
        }
        Loc::Param(index) => {
            bail!("direct-table source cannot write scalar parameter {index}");
        }
    }
}

fn descriptor_source_count(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("direct-table source {label} count exceeds u32"))
}

fn descriptor_encoded_len(
    input_count: usize,
    output_component_count: usize,
    parameter_count: usize,
) -> Result<usize> {
    let bytes = 68_usize
        .checked_add(
            input_count
                .checked_mul(4)
                .ok_or_else(|| anyhow!("direct-table descriptor input byte count overflows"))?,
        )
        .and_then(|bytes| {
            output_component_count
                .checked_mul(4)
                .and_then(|extra| bytes.checked_add(extra))
        })
        .and_then(|bytes| {
            parameter_count
                .checked_mul(8)
                .and_then(|extra| bytes.checked_add(extra))
        })
        .ok_or_else(|| anyhow!("direct-table descriptor byte count overflows"))?;
    if bytes > DIRECT_TABLE_DESCRIPTOR_MAX_BYTES {
        bail!(
            "direct-table descriptor requires {bytes} bytes, maximum is {DIRECT_TABLE_DESCRIPTOR_MAX_BYTES}"
        );
    }
    Ok(bytes)
}

fn push_descriptor_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct DirectTableDescriptorReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> DirectTableDescriptorReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| anyhow!("direct-table descriptor cursor overflows"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| anyhow!("truncated direct-table descriptor"))?;
        self.position = end;
        Ok(bytes.try_into().expect("slice length was checked"))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_bounded_count(&mut self, maximum: usize, label: &str) -> Result<usize> {
        let count = self.read_u32()? as usize;
        if count > maximum {
            bail!("direct-table descriptor {label} count {count} exceeds maximum {maximum}");
        }
        Ok(count)
    }

    fn finish(self) -> Result<()> {
        if self.position != self.bytes.len() {
            bail!(
                "direct-table descriptor has {} trailing bytes",
                self.bytes.len() - self.position
            );
        }
        Ok(())
    }
}

fn read_expected_source_count(
    reader: &mut DirectTableDescriptorReader<'_>,
    expected: usize,
    label: &str,
) -> Result<()> {
    let encoded = reader.read_u32()?;
    let expected = descriptor_source_count(expected, label)?;
    if encoded != expected {
        bail!(
            "direct-table descriptor source {label} count {encoded} does not match loaded source count {expected}"
        );
    }
    Ok(())
}

/// Host-only checked view passed to the generated table function.
///
/// Invocation and attachment row contents are opaque to SymJIT except for the
/// authenticated `u32` offsets in [`DirectTableApplicationMetadata`].
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DirectTableCallViewV1 {
    pub invocations: *const u8,
    pub invocation_count: u32,
    pub invocation_stride: u32,

    pub attachments: *const u8,
    pub attachment_count: u32,
    pub attachment_stride: u32,

    pub planes: *const DirectPlane,
    pub plane_count: u32,
    pub scalar_count: u32,

    pub scalars: *const DirectScalar,
    pub scale_re: *const f64,
    pub scale_im: *const f64,
    pub scale_count: u32,

    pub point_start: u32,
    pub point_count: u32,
}

/// Structural facts recorded while emitting one generated table function.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectTableCodeShape {
    pub simd_lanes: u32,
    pub machine_code_bytes: usize,
    pub executable_instruction_bytes: usize,
    pub inline_kernel_body_offset: usize,
    pub inline_kernel_body_bytes: usize,
    pub inline_kernel_copies: u32,
    pub branch_and_link_instructions: u32,
    /// Point packets materialized outside the direct plane descriptors.
    pub packet_materializations: u32,
    /// Input-plane gathers into temporary point packets.
    pub gather_materializations: u32,
    /// Result-packet scatters out to destination planes.
    pub scatter_materializations: u32,
}

/// Portable table application before host code is sealed.
pub struct DirectTableApplication {
    metadata: DirectTableApplicationMetadata,
    program: Program,
    mir: Mir,
    stack_slots: u32,
    result_stack_offset: u32,
}

impl DirectTableApplication {
    pub fn new(source: Application, metadata: DirectTableApplicationMetadata) -> Result<Self> {
        metadata.validate_source(&source)?;
        let (program, mir, stack_slots, result_stack_offset) =
            lower_table_application(&source, &metadata)?;
        Ok(Self {
            metadata,
            program,
            mir,
            stack_slots,
            result_stack_offset,
        })
    }

    /// Build from a separately loaded immutable source and its portable
    /// descriptor bytes. The descriptor never embeds source or executable
    /// payloads.
    pub fn from_descriptor(source: Application, descriptor: &[u8]) -> Result<Self> {
        let metadata = DirectTableApplicationMetadata::decode_descriptor(&source, descriptor)?;
        Self::new(source, metadata)
    }

    pub fn metadata(&self) -> &DirectTableApplicationMetadata {
        &self.metadata
    }

    pub fn seal(self) -> Result<DirectTableApplet> {
        ensure_table_callable_host()?;
        let (scalar_mir_stack_bytes, scalar_dynamic_stack_bytes) = codegen_stack_sizes(
            self.stack_slots,
            self.metadata.invocation.input_plane_offsets.len(),
            8,
        )?;
        let (simd_mir_stack_bytes, simd_dynamic_stack_bytes) = codegen_stack_sizes(
            self.stack_slots,
            self.metadata.invocation.input_plane_offsets.len(),
            DIRECT_TABLE_SIMD_LANES * 8,
        )?;
        let stack_limit = self.program.config().stack_limit();
        for (kind, dynamic_bytes) in [
            ("scalar", scalar_dynamic_stack_bytes),
            ("SIMD", simd_dynamic_stack_bytes),
        ] {
            if dynamic_bytes as usize > stack_limit {
                bail!(
                    "direct-table {kind} dynamic stack {dynamic_bytes} exceeds configured stack limit {stack_limit}"
                );
            }
        }
        let layout = DirectTableCodegenLayout {
            invocation: self.metadata.invocation.clone(),
            attachment: self.metadata.attachment.clone(),
            input_plane_row_offsets: self.metadata.invocation.input_plane_offsets.clone(),
            result_stack_offset: self.result_stack_offset,
            scalar_mir_stack_bytes,
            scalar_dynamic_stack_bytes,
            simd_mir_stack_bytes,
            simd_dynamic_stack_bytes,
        };

        let (scalar, simd) = compile_host_table_machines(&self.program, &self.mir, &layout)?;

        Ok(DirectTableApplet {
            metadata: self.metadata,
            scalar,
            simd,
        })
    }
}

/// Loaded owner of the scalar and native SIMD table machine functions.
pub struct DirectTableApplet {
    metadata: DirectTableApplicationMetadata,
    scalar: DirectTableMachineCode,
    simd: DirectTableMachineCode,
}

impl DirectTableApplet {
    pub fn metadata(&self) -> &DirectTableApplicationMetadata {
        &self.metadata
    }

    pub fn scalar_code_shape(&self) -> DirectTableCodeShape {
        self.scalar.shape
    }

    pub fn simd_code_shape(&self) -> DirectTableCodeShape {
        self.simd.shape
    }

    /// Validate all table fields and execute at most scalar-head, SIMD-middle,
    /// and scalar-tail generated calls.
    ///
    /// # Safety
    ///
    /// Every raw pointer in `view` must denote live storage matching its count
    /// and stride for the duration of this call.
    pub unsafe fn evaluate_table(&self, view: &DirectTableCallViewV1) -> Result<()> {
        validate_call_view_authenticated(&self.metadata, view)?;
        unsafe { self.dispatch_table_unchecked(view) }.map(|_| ())
    }

    /// Execute a call view already authenticated by an owning runtime.
    ///
    /// # Safety
    ///
    /// `view` must satisfy the complete contract checked by
    /// [`Self::evaluate_table`].
    pub unsafe fn evaluate_table_unchecked(&self, view: &DirectTableCallViewV1) -> Result<()> {
        unsafe { self.dispatch_table_unchecked(view) }.map(|_| ())
    }

    pub fn into_callable(self) -> DirectTableCallable {
        DirectTableCallable {
            context: Box::new(self),
        }
    }

    unsafe fn dispatch_table_unchecked(&self, view: &DirectTableCallViewV1) -> Result<u32> {
        let mut calls = 0_u32;
        // Keep invocation rows outermost even when the point range needs
        // scalar/SIMD segmentation. This is observable for ordered cross-row
        // aliases with shifted descriptor bases.
        for row_index in 0..view.invocation_count {
            let mut row_view = *view;
            row_view.invocations = unsafe {
                view.invocations
                    .add(row_index as usize * view.invocation_stride as usize)
            };
            row_view.invocation_count = 1;

            let mut start = view.point_start;
            let mut remaining = view.point_count;
            let head = ((DIRECT_TABLE_SIMD_LANES - start % DIRECT_TABLE_SIMD_LANES)
                % DIRECT_TABLE_SIMD_LANES)
                .min(remaining);
            if head != 0 {
                call_generated(&self.scalar, &row_view, start, head)?;
                calls = calls
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("direct-table generated call count overflows"))?;
                start += head;
                remaining -= head;
            }

            let middle = remaining / DIRECT_TABLE_SIMD_LANES * DIRECT_TABLE_SIMD_LANES;
            if middle != 0 {
                call_generated(&self.simd, &row_view, start, middle)?;
                calls = calls
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("direct-table generated call count overflows"))?;
                start += middle;
                remaining -= middle;
            }

            if remaining != 0 {
                call_generated(&self.scalar, &row_view, start, remaining)?;
                calls = calls
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("direct-table generated call count overflows"))?;
            }
        }
        Ok(calls)
    }
}

type DirectTableCompiledFunction = unsafe extern "C" fn(*const DirectTableCallViewV1) -> i32;

struct DirectTableMachineCode {
    _code: Arc<Memory>,
    function: DirectTableCompiledFunction,
    shape: DirectTableCodeShape,
}

unsafe impl Send for DirectTableMachineCode {}
unsafe impl Sync for DirectTableMachineCode {}

impl DirectTableMachineCode {
    fn new(
        machine_code: Vec<u8>,
        executable_instruction_bytes: usize,
        inline_kernel_body_offset: usize,
        inline_kernel_body_bytes: usize,
        simd_lanes: u32,
        huge: bool,
    ) -> Result<Self> {
        let machine_code_bytes = machine_code.len();
        let branch_and_link_instructions =
            count_out_of_line_calls(&machine_code[..executable_instruction_bytes]);
        let mut code = Memory::new(BranchProtection::None, huge);
        let pages = code
            .allocate(machine_code.len(), 4096)
            .context("cannot allocate direct-table executable memory")?;
        let destination = unsafe { std::slice::from_raw_parts_mut(pages, machine_code.len()) };
        destination.copy_from_slice(&machine_code);
        code.set_readable_and_executable()
            .map_err(|error| anyhow!("cannot seal direct-table executable memory: {error:?}"))?;
        let function = unsafe { mem::transmute::<*mut u8, DirectTableCompiledFunction>(pages) };
        Ok(Self {
            _code: Arc::new(code),
            function,
            shape: DirectTableCodeShape {
                simd_lanes,
                machine_code_bytes,
                executable_instruction_bytes,
                inline_kernel_body_offset,
                inline_kernel_body_bytes,
                inline_kernel_copies: 1,
                branch_and_link_instructions,
                packet_materializations: 0,
                gather_materializations: 0,
                scatter_materializations: 0,
            },
        })
    }
}

pub type DirectTableCallFunction =
    unsafe extern "C" fn(*const c_void, *const DirectTableCallViewV1) -> i32;

/// Borrowed context/function pair for one checked table call.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DirectTableCallableHandle {
    pub call: DirectTableCallFunction,
    pub context: *const c_void,
}

impl DirectTableCallableHandle {
    /// # Safety
    ///
    /// The callable owner and every raw pointer in `view` must remain live.
    pub unsafe fn invoke(self, view: &DirectTableCallViewV1) -> i32 {
        unsafe { (self.call)(self.context, ptr::from_ref(view)) }
    }
}

/// Owns the generated native scalar and SIMD table functions.
pub struct DirectTableCallable {
    context: Box<DirectTableApplet>,
}

impl DirectTableCallable {
    pub fn handle(&self) -> DirectTableCallableHandle {
        DirectTableCallableHandle {
            call: direct_table_call_trampoline,
            context: ptr::from_ref(self.context.as_ref()).cast(),
        }
    }

    pub fn metadata(&self) -> &DirectTableApplicationMetadata {
        self.context.metadata()
    }

    pub fn scalar_code_shape(&self) -> DirectTableCodeShape {
        self.context.scalar_code_shape()
    }

    pub fn simd_code_shape(&self) -> DirectTableCodeShape {
        self.context.simd_code_shape()
    }

    /// # Safety
    ///
    /// `view` must satisfy the complete checked call-view contract.
    pub unsafe fn invoke_unchecked(&self, view: &DirectTableCallViewV1) -> i32 {
        match unsafe { self.context.evaluate_table_unchecked(view) } {
            Ok(()) => DIRECT_STATUS_OK,
            Err(_) => DIRECT_STATUS_EXECUTION_FAILED,
        }
    }
}

unsafe extern "C" fn direct_table_call_trampoline(
    context: *const c_void,
    view: *const DirectTableCallViewV1,
) -> i32 {
    if context.is_null() {
        return DIRECT_STATUS_INVALID_CONTEXT;
    }
    if view.is_null() {
        return DIRECT_STATUS_INVALID_ARGUMENT;
    }
    let applet = unsafe { &*context.cast::<DirectTableApplet>() };
    let view = unsafe { &*view };
    if let Err(error) = validate_call_view_authenticated(&applet.metadata, view) {
        if std::env::var_os("SYMJIT_DIRECT_DEBUG").is_some() {
            eprintln!("SymJIT Direct-Table call rejected: {error:#}");
        }
        return DIRECT_STATUS_INVALID_ARGUMENT;
    }
    match unsafe { applet.dispatch_table_unchecked(view) } {
        Ok(_) => DIRECT_STATUS_OK,
        Err(error) => {
            if std::env::var_os("SYMJIT_DIRECT_DEBUG").is_some() {
                eprintln!("SymJIT Direct-Table execution failed: {error:#}");
            }
            DIRECT_STATUS_EXECUTION_FAILED
        }
    }
}

#[derive(Clone)]
pub(crate) struct DirectTableCodegenLayout {
    pub invocation: DirectTableInvocationLayout,
    pub attachment: DirectTableAttachmentLayout,
    pub input_plane_row_offsets: Vec<u32>,
    pub result_stack_offset: u32,
    scalar_mir_stack_bytes: u32,
    scalar_dynamic_stack_bytes: u32,
    simd_mir_stack_bytes: u32,
    simd_dynamic_stack_bytes: u32,
}

pub(crate) trait DirectTableCodegen {
    fn direct_table_prologue(&mut self, layout: &DirectTableCodegenLayout);
    fn direct_table_begin_loops(&mut self, layout: &DirectTableCodegenLayout);
    fn direct_table_end_loops(&mut self, layout: &DirectTableCodegenLayout);
    fn direct_table_epilogue(&mut self, layout: &DirectTableCodegenLayout);
    fn direct_table_ip(&self) -> usize;
}

impl DirectTableCodegenLayout {
    pub fn mir_stack_bytes(&self, simd: bool) -> u32 {
        if simd {
            self.simd_mir_stack_bytes
        } else {
            self.scalar_mir_stack_bytes
        }
    }

    pub fn dynamic_stack_bytes(&self, simd: bool) -> u32 {
        if simd {
            self.simd_dynamic_stack_bytes
        } else {
            self.scalar_dynamic_stack_bytes
        }
    }
}

fn compile_table_machine<G>(
    program: &Program,
    mir: &Mir,
    mut generator: G,
    layout: &DirectTableCodegenLayout,
    simd_lanes: u32,
) -> Result<DirectTableMachineCode>
where
    G: Generator + DirectTableCodegen,
{
    generator.direct_table_prologue(layout);
    let used = mir.used_registers();
    generator.save_used_registers(&used);
    generator.direct_table_begin_loops(layout);
    let inline_kernel_body_offset = generator.direct_table_ip();
    mir.rerun(&mut generator)?;
    let inline_kernel_body_bytes = generator
        .direct_table_ip()
        .checked_sub(inline_kernel_body_offset)
        .ok_or_else(|| anyhow!("direct-table kernel body offset underflows"))?;
    generator.direct_table_end_loops(layout);
    generator.load_used_registers(&used);
    generator.direct_table_epilogue(layout);
    generator.align();
    let executable_instruction_bytes = generator.direct_table_ip();
    generator.add_consts(&mir.consts);
    debug_assert!(program.builder.ft.is_empty());
    generator.seal();
    let bytes = generator.bytes();
    DirectTableMachineCode::new(
        bytes,
        executable_instruction_bytes,
        inline_kernel_body_offset,
        inline_kernel_body_bytes,
        simd_lanes,
        program.config().huge(),
    )
}

#[cfg(target_arch = "aarch64")]
fn compile_host_table_machines(
    program: &Program,
    mir: &Mir,
    layout: &DirectTableCodegenLayout,
) -> Result<(DirectTableMachineCode, DirectTableMachineCode)> {
    let mut scalar_config = mir.config.clone();
    scalar_config.set_simd(false);
    scalar_config.set_direct_arena(true);
    let scalar = compile_table_machine(program, mir, ArmGenerator::new(scalar_config), layout, 1)?;

    let mut simd_config = mir.config.clone();
    simd_config.set_simd(true);
    simd_config.set_direct_arena(true);
    let simd = compile_table_machine(
        program,
        mir,
        ArmSimdGenerator::new(simd_config),
        layout,
        DIRECT_TABLE_SIMD_LANES,
    )?;
    Ok((scalar, simd))
}

#[cfg(target_arch = "x86_64")]
fn compile_host_table_machines(
    program: &Program,
    mir: &Mir,
    layout: &DirectTableCodegenLayout,
) -> Result<(DirectTableMachineCode, DirectTableMachineCode)> {
    let mut scalar_config = mir.config.clone();
    scalar_config.set_simd(false);
    scalar_config.set_direct_arena(true);
    let scalar = compile_table_machine(
        program,
        mir,
        AmdScalarGenerator::new(scalar_config),
        layout,
        1,
    )?;

    let mut simd_config = mir.config.clone();
    simd_config.set_simd(true);
    simd_config.set_direct_arena(true);
    let simd = compile_table_machine(
        program,
        mir,
        AmdVectorF64x4Generator::new(simd_config),
        layout,
        DIRECT_TABLE_SIMD_LANES,
    )?;
    Ok((scalar, simd))
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn compile_host_table_machines(
    _program: &Program,
    _mir: &Mir,
    _layout: &DirectTableCodegenLayout,
) -> Result<(DirectTableMachineCode, DirectTableMachineCode)> {
    bail!("direct-table callable code generation is unsupported on this architecture")
}

fn lower_table_application(
    source: &Application,
    metadata: &DirectTableApplicationMetadata,
) -> Result<(Program, Mir, u32, u32)> {
    let mut config = source.config.clone();
    #[cfg(target_arch = "x86_64")]
    {
        // The x86 table envelope has a fixed AVX4 middle loop. An explicit
        // target keeps that ABI stable for cross-builds as well as native JITs.
        config.ty = crate::CompilerType::AmdAVX;
    }
    config.set_symbolica(false);
    config.set_complex(false);
    config.set_fast_complex(false);
    config.set_dicect(false);
    config.set_direct_arena(false);
    config.set_threads(false);
    config.set_opt_level(2);

    let mut program = source.prog.clone();
    program.builder.config = config.clone();
    program.builder.primary_block.config = config.clone();
    program.count_states = metadata.invocation.input_plane_offsets.len();
    program.count_params = metadata.scalar_input_count as usize;
    program.count_obs = 0;
    program.count_diffs = 0;
    program.count_loops = 0;

    let result_stack_offset = u32::try_from(program.builder.stack_size().max(SPILL_AREA))
        .context("direct-table stack size exceeds u32")?;
    let output_component_count = u32::try_from(source.count_obs)
        .context("direct-table output component count exceeds u32")?;
    let stack_slots = result_stack_offset
        .checked_add(output_component_count)
        .ok_or_else(|| anyhow!("direct-table result stack range overflows"))?;
    program.builder.count_stack = Some(stack_slots as usize);

    let mut code = MirWriter::new();
    for instruction in source.bytecode.mir.code.iter() {
        append_table_remapped(
            &mut code,
            instruction,
            source.count_states,
            source.count_obs,
            &metadata.parameter_bindings,
            result_stack_offset,
        )?;
    }
    let mut mir = Mir {
        code,
        consts: source.bytecode.mir.consts.clone(),
        labels: source.bytecode.mir.labels.clone(),
        config,
    };
    mir.populate_labels();
    Ok((program, mir, stack_slots, result_stack_offset))
}

fn append_table_remapped(
    code: &mut MirWriter,
    instruction: Instruction,
    source_state_count: usize,
    source_output_count: usize,
    parameter_bindings: &[DirectTableParameterBinding],
    result_stack_offset: u32,
) -> Result<()> {
    match instruction {
        Instruction::LoadComplex { xd, yd, loc } => {
            code.push(&Instruction::Load {
                dst: xd,
                loc: table_input_location(
                    loc,
                    source_state_count,
                    source_output_count,
                    parameter_bindings,
                )?,
            });
            code.push(&Instruction::Load {
                dst: yd,
                loc: table_input_location(
                    loc.imag(),
                    source_state_count,
                    source_output_count,
                    parameter_bindings,
                )?,
            });
        }
        Instruction::SaveComplex { xs, ys, loc } => {
            let Loc::Mem(index) = loc else {
                code.push(&Instruction::SaveComplex { xs, ys, loc });
                return Ok(());
            };
            let output_component = (index as usize)
                .checked_sub(source_state_count)
                .ok_or_else(|| {
                    anyhow!(
                        "direct-table source has unsupported complex memory store at plane {index}"
                    )
                })?;
            if output_component >= source_output_count
                || !output_component.is_multiple_of(2)
                || output_component + 2 > source_output_count
            {
                bail!(
                    "direct-table complex output store at plane {index} is outside the source output vector"
                );
            }
            code.push(&Instruction::Save {
                src: xs,
                loc: Loc::Stack(result_stack_offset + output_component as u32),
            });
            code.push(&Instruction::Save {
                src: ys,
                loc: Loc::Stack(result_stack_offset + output_component as u32 + 1),
            });
        }
        Instruction::Load { dst, loc } => code.push(&Instruction::Load {
            dst,
            loc: table_input_location(
                loc,
                source_state_count,
                source_output_count,
                parameter_bindings,
            )?,
        }),
        Instruction::Save { src, loc } => {
            if matches!(loc, Loc::Mem(index) if index as usize >= source_state_count) {
                bail!("direct-table complex outputs must use paired stores");
            }
            code.push(&Instruction::Save { src, loc });
        }
        Instruction::LoadMath { op, dst, s1, loc } => {
            code.push(&Instruction::LoadMath {
                op,
                dst,
                s1,
                loc: table_input_location(
                    loc,
                    source_state_count,
                    source_output_count,
                    parameter_bindings,
                )?,
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
            cond: table_input_location(
                cond,
                source_state_count,
                source_output_count,
                parameter_bindings,
            )?,
        }),
        other => code.push(&other),
    }
    Ok(())
}

fn table_input_location(
    location: Loc,
    source_state_count: usize,
    source_output_count: usize,
    parameter_bindings: &[DirectTableParameterBinding],
) -> Result<Loc> {
    match location {
        Loc::Param(index) => {
            let binding = parameter_bindings.get(index as usize).with_context(|| {
                format!("direct-table source parameter {index} is out of bounds")
            })?;
            Ok(match *binding {
                DirectTableParameterBinding::Plane(plane) => Loc::Mem(plane),
                DirectTableParameterBinding::Scalar(scalar) => Loc::Param(scalar),
            })
        }
        Loc::Mem(index) if index as usize >= source_state_count => {
            let output = index as usize - source_state_count;
            if output < source_output_count {
                bail!("direct-table kernel cannot read its output plane");
            }
            bail!("direct-table source memory location {index} is out of bounds");
        }
        other => Ok(other),
    }
}

/// Validate a call view against metadata already authenticated and privately
/// owned by a constructed application/applet.
fn validate_call_view_authenticated(
    metadata: &DirectTableApplicationMetadata,
    view: &DirectTableCallViewV1,
) -> Result<()> {
    // `DirectTableApplet` owns metadata that was authenticated before sealing.
    // Revalidating it here both repeats immutable work and used to allocate via
    // attachment-field diagnostic formatting on every successful checked call.
    if view.invocation_stride != metadata.invocation.row_stride {
        bail!("direct-table invocation stride does not match authenticated metadata");
    }
    if view.attachment_stride != metadata.attachment.row_stride {
        bail!("direct-table attachment stride does not match authenticated metadata");
    }
    if view.invocation_count == 0 {
        bail!("direct-table call must contain at least one invocation");
    }
    if view.point_count == 0 {
        bail!("direct-table call point count must be positive");
    }
    let point_end =
        view.point_start
            .checked_add(view.point_count)
            .ok_or_else(|| anyhow!("direct-table point range overflows u32"))? as usize;

    require_table_pointer(
        "invocation",
        view.invocations,
        view.invocation_count,
        view.invocation_stride,
    )?;
    require_table_pointer(
        "attachment",
        view.attachments,
        view.attachment_count,
        view.attachment_stride,
    )?;
    if view.plane_count == 0 || view.planes.is_null() {
        bail!("direct-table plane catalog is empty or null");
    }
    require_aligned(
        "plane catalog",
        view.planes.cast(),
        mem::align_of::<DirectPlane>(),
    )?;
    if view.scalar_count != metadata.scalar_input_count {
        bail!(
            "direct-table call has {} scalars, expected {}",
            view.scalar_count,
            metadata.scalar_input_count
        );
    }
    if view.scalar_count != 0 && view.scalars.is_null() {
        bail!("direct-table scalar catalog is null");
    }
    if view.scalar_count != 0 {
        require_aligned(
            "scalar catalog",
            view.scalars.cast(),
            mem::align_of::<DirectScalar>(),
        )?;
    }
    if view.scale_count == 0 || view.scale_re.is_null() || view.scale_im.is_null() {
        bail!("direct-table scale catalog is empty or null");
    }
    require_aligned(
        "real scale catalog",
        view.scale_re.cast(),
        mem::align_of::<f64>(),
    )?;
    require_aligned(
        "imaginary scale catalog",
        view.scale_im.cast(),
        mem::align_of::<f64>(),
    )?;

    let planes = unsafe { std::slice::from_raw_parts(view.planes, view.plane_count as usize) };
    for (index, plane) in planes.iter().enumerate() {
        if plane.values.is_null() || plane.len < point_end {
            bail!("direct-table plane {index} does not cover the point range");
        }
    }
    let scalars = if view.scalar_count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(view.scalars, view.scalar_count as usize) }
    };
    for (index, scalar) in scalars.iter().enumerate() {
        if scalar.value.is_null() {
            bail!("direct-table scalar descriptor {index} is null");
        }
    }

    for row_index in 0..view.invocation_count {
        let row = unsafe {
            view.invocations
                .add(row_index as usize * view.invocation_stride as usize)
        };
        for (binding, &offset) in metadata.invocation.input_plane_offsets.iter().enumerate() {
            let plane = unsafe { read_row_u32(row, offset) };
            if plane >= view.plane_count {
                bail!(
                    "direct-table invocation {row_index} input binding {binding} references plane {plane}, catalog has {}",
                    view.plane_count
                );
            }
        }
        let start = unsafe { read_row_u32(row, metadata.invocation.attachment_start_offset) };
        let count = unsafe { read_row_u32(row, metadata.invocation.attachment_count_offset) };
        let stop = start
            .checked_add(count)
            .ok_or_else(|| anyhow!("direct-table invocation attachment range overflows"))?;
        if stop > view.attachment_count {
            bail!(
                "direct-table invocation {row_index} attachment range {start}..{stop} exceeds {}",
                view.attachment_count
            );
        }
    }

    for attachment_index in 0..view.attachment_count {
        let row = unsafe {
            view.attachments
                .add(attachment_index as usize * view.attachment_stride as usize)
        };
        for (component, &offset) in metadata
            .attachment
            .destination_plane_offsets
            .iter()
            .enumerate()
        {
            let plane = unsafe { read_row_u32(row, offset) };
            if plane >= view.plane_count {
                bail!(
                    "direct-table attachment {attachment_index} destination component {component} references plane {plane}, catalog has {}",
                    view.plane_count
                );
            }
        }
        let scale = unsafe { read_row_u32(row, metadata.attachment.scale_offset) };
        if scale >= view.scale_count {
            bail!(
                "direct-table attachment {attachment_index} scale {scale} exceeds {}",
                view.scale_count
            );
        }
        let operation = unsafe { read_row_u32(row, metadata.attachment.operation_offset) };
        if operation > 1 {
            bail!(
                "direct-table attachment {attachment_index} operation {operation} is not Overwrite(0) or Accumulate(1)"
            );
        }
    }
    validate_no_table_aliases(metadata, view, planes, scalars)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct TableAddressRange {
    start: usize,
    end: usize,
}

impl TableAddressRange {
    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

fn checked_address_range(
    name: &str,
    pointer: *const u8,
    count: usize,
    element_bytes: usize,
) -> Result<TableAddressRange> {
    let bytes = count
        .checked_mul(element_bytes)
        .ok_or_else(|| anyhow!("direct-table {name} byte range overflows"))?;
    if bytes > isize::MAX as usize {
        bail!("direct-table {name} byte range exceeds isize");
    }
    let start = pointer as usize;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("direct-table {name} address range overflows"))?;
    Ok(TableAddressRange { start, end })
}

fn plane_address_range(index: u32, planes: &[DirectPlane]) -> Result<TableAddressRange> {
    let plane = planes[index as usize];
    checked_address_range(
        "plane descriptor",
        plane.values.cast(),
        plane.len,
        mem::size_of::<f64>(),
    )
}

unsafe fn attachment_plane_index(
    metadata: &DirectTableApplicationMetadata,
    view: &DirectTableCallViewV1,
    attachment_index: u32,
    destination_component: usize,
) -> u32 {
    let row = unsafe {
        view.attachments
            .add(attachment_index as usize * view.attachment_stride as usize)
    };
    let offset = metadata.attachment.destination_plane_offsets[destination_component];
    unsafe { read_row_u32(row, offset) }
}

fn validate_no_table_aliases(
    metadata: &DirectTableApplicationMetadata,
    view: &DirectTableCallViewV1,
    planes: &[DirectPlane],
    scalars: &[DirectScalar],
) -> Result<()> {
    let destination_components = metadata.attachment.destination_plane_offsets.len();

    // Rows execute in strict table order. Inputs are live only for their own
    // row, while destinations are written immediately after the inline kernel.
    // Therefore a row may consume an earlier row's output, overwrite an
    // expired earlier input, or accumulate into an earlier destination. Within
    // one row, input/output and destination/destination overlap remain invalid.
    // Complete descriptor ranges are compared so shifted partial aliases fail
    // closed even when the active point windows happen not to intersect.
    for row_index in 0..view.invocation_count {
        let row = unsafe {
            view.invocations
                .add(row_index as usize * view.invocation_stride as usize)
        };
        let attachment_start =
            unsafe { read_row_u32(row, metadata.invocation.attachment_start_offset) };
        let attachment_count =
            unsafe { read_row_u32(row, metadata.invocation.attachment_count_offset) };
        for (binding, &offset) in metadata.invocation.input_plane_offsets.iter().enumerate() {
            let input_index = unsafe { read_row_u32(row, offset) };
            let input = plane_address_range(input_index, planes)?;
            for attachment_index in attachment_start..attachment_start + attachment_count {
                for component in 0..destination_components {
                    let destination_index = unsafe {
                        attachment_plane_index(metadata, view, attachment_index, component)
                    };
                    if input.overlaps(plane_address_range(destination_index, planes)?) {
                        bail!(
                            "direct-table invocation {row_index} input binding {binding} overlaps attachment {attachment_index} destination component {component}"
                        );
                    }
                }
            }
        }

        let row_destination_count = (attachment_count as usize)
            .checked_mul(destination_components)
            .ok_or_else(|| anyhow!("direct-table row destination count overflows"))?;
        for left in 0..row_destination_count {
            let left_attachment = attachment_start + (left / destination_components) as u32;
            let left_component = left % destination_components;
            let left_index =
                unsafe { attachment_plane_index(metadata, view, left_attachment, left_component) };
            let left_range = plane_address_range(left_index, planes)?;
            for right in left + 1..row_destination_count {
                let right_attachment = attachment_start + (right / destination_components) as u32;
                let right_component = right % destination_components;
                let right_index = unsafe {
                    attachment_plane_index(metadata, view, right_attachment, right_component)
                };
                if left_range.overlaps(plane_address_range(right_index, planes)?) {
                    bail!(
                        "direct-table invocation {row_index} attachment {left_attachment} destination component {left_component} overlaps attachment {right_attachment} destination component {right_component}"
                    );
                }
            }
        }
    }

    // Scalars, scales, and table/catalog storage remain live for every row.
    // No destination in any row may overwrite any part of those global inputs.
    let global_buffers = [
        (
            "invocation table",
            checked_address_range(
                "invocation table",
                view.invocations,
                view.invocation_count as usize,
                view.invocation_stride as usize,
            )?,
        ),
        (
            "attachment table",
            checked_address_range(
                "attachment table",
                view.attachments,
                view.attachment_count as usize,
                view.attachment_stride as usize,
            )?,
        ),
        (
            "plane catalog",
            checked_address_range(
                "plane catalog",
                view.planes.cast(),
                view.plane_count as usize,
                mem::size_of::<DirectPlane>(),
            )?,
        ),
        (
            "scalar catalog",
            checked_address_range(
                "scalar catalog",
                view.scalars.cast(),
                view.scalar_count as usize,
                mem::size_of::<DirectScalar>(),
            )?,
        ),
        (
            "real scale catalog",
            checked_address_range(
                "real scale catalog",
                view.scale_re.cast(),
                view.scale_count as usize,
                mem::size_of::<f64>(),
            )?,
        ),
        (
            "imaginary scale catalog",
            checked_address_range(
                "imaginary scale catalog",
                view.scale_im.cast(),
                view.scale_count as usize,
                mem::size_of::<f64>(),
            )?,
        ),
    ];
    for global_buffer in global_buffers {
        for attachment_index in 0..view.attachment_count {
            for component in 0..destination_components {
                let destination_index =
                    unsafe { attachment_plane_index(metadata, view, attachment_index, component) };
                if global_buffer
                    .1
                    .overlaps(plane_address_range(destination_index, planes)?)
                {
                    bail!(
                        "direct-table {} overlaps attachment {attachment_index} destination component {component}",
                        global_buffer.0
                    );
                }
            }
        }
    }

    for (scalar_index, scalar) in scalars.iter().enumerate() {
        let input = checked_address_range(
            "scalar value",
            scalar.value.cast(),
            1,
            mem::size_of::<f64>(),
        )?;
        for attachment_index in 0..view.attachment_count {
            for component in 0..destination_components {
                let destination_index =
                    unsafe { attachment_plane_index(metadata, view, attachment_index, component) };
                if input.overlaps(plane_address_range(destination_index, planes)?) {
                    bail!(
                        "direct-table scalar descriptor {scalar_index} overlaps attachment {attachment_index} destination component {component}"
                    );
                }
            }
        }
    }
    Ok(())
}

fn call_generated(
    machine: &DirectTableMachineCode,
    view: &DirectTableCallViewV1,
    point_start: u32,
    point_count: u32,
) -> Result<()> {
    let mut segment = *view;
    segment.point_start = point_start;
    segment.point_count = point_count;
    let status = unsafe { (machine.function)(ptr::from_ref(&segment)) };
    if status != DIRECT_STATUS_OK {
        bail!("direct-table generated machine function returned status {status}");
    }
    Ok(())
}

fn require_table_pointer(name: &str, pointer: *const u8, count: u32, stride: u32) -> Result<()> {
    if count != 0 && pointer.is_null() {
        bail!("direct-table {name} pointer is null");
    }
    let bytes = (count as usize)
        .checked_mul(stride as usize)
        .ok_or_else(|| anyhow!("direct-table {name} byte range overflows"))?;
    if bytes > isize::MAX as usize {
        bail!("direct-table {name} byte range exceeds isize");
    }
    Ok(())
}

unsafe fn read_row_u32(row: *const u8, offset: u32) -> u32 {
    unsafe { ptr::read_unaligned(row.add(offset as usize).cast::<u32>()) }
}

fn validate_stride(name: &str, stride: u32) -> Result<()> {
    if stride == 0 || stride & 3 != 0 {
        bail!("direct-table {name} stride must be a nonzero multiple of four");
    }
    if stride > MAX_ROW_STRIDE {
        bail!("direct-table {name} stride exceeds ABI limit {MAX_ROW_STRIDE}");
    }
    Ok(())
}

fn validate_u32_field(name: &str, index: usize, offset: u32, stride: u32) -> Result<()> {
    if offset & 3 != 0 {
        bail!("direct-table {name} field {index} is not u32-aligned");
    }
    if offset.checked_add(4).is_none_or(|stop| stop > stride) {
        bail!("direct-table {name} field {index} lies outside row stride {stride}");
    }
    Ok(())
}

fn validate_distinct_u32_fields(name: &str, offsets: impl IntoIterator<Item = u32>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for offset in offsets {
        if !seen.insert(offset) {
            bail!("direct-table {name} contains overlapping u32 fields at offset {offset}");
        }
    }
    Ok(())
}

fn ensure_table_callable_host() -> Result<()> {
    if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        Ok(())
    } else {
        bail!(
            "direct-table descriptors and validation are portable, but generated callable execution is only implemented on AArch64 and x86-64 hosts"
        )
    }
}

fn codegen_stack_sizes(
    stack_slots: u32,
    input_plane_count: usize,
    lane_bytes: u32,
) -> Result<(u32, u32)> {
    let mir_bytes = stack_slots
        .checked_mul(lane_bytes)
        .and_then(checked_align_16)
        .ok_or_else(|| anyhow!("direct-table MIR stack size overflows u32"))?;
    let descriptor_bytes = u32::try_from(input_plane_count)
        .ok()
        .and_then(|count| count.checked_mul(16))
        .ok_or_else(|| anyhow!("direct-table descriptor stack size overflows u32"))?;
    #[cfg(target_arch = "x86_64")]
    let envelope_scratch_bytes = 16;
    #[cfg(not(target_arch = "x86_64"))]
    let envelope_scratch_bytes = 0;
    let dynamic_bytes = mir_bytes
        .checked_add(descriptor_bytes)
        .and_then(|bytes| bytes.checked_add(envelope_scratch_bytes))
        .and_then(checked_align_16)
        .ok_or_else(|| anyhow!("direct-table dynamic stack size overflows u32"))?;
    if dynamic_bytes >= 1 << 24 {
        bail!("direct-table dynamic stack exceeds generated-code immediate range");
    }
    Ok((mir_bytes, dynamic_bytes))
}

fn checked_align_16(value: u32) -> Option<u32> {
    value.checked_add(15).map(|aligned| aligned & !15)
}

fn require_aligned(name: &str, pointer: *const u8, alignment: usize) -> Result<()> {
    if !(pointer as usize).is_multiple_of(alignment) {
        bail!("direct-table {name} pointer is not aligned to {alignment} bytes");
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn count_out_of_line_calls(bytes: &[u8]) -> u32 {
    bytes
        .chunks_exact(4)
        .filter(|chunk| {
            let instruction =
                u32::from_le_bytes((*chunk).try_into().expect("four-byte instruction"));
            instruction & 0xfc00_0000 == 0x9400_0000 || instruction & 0xffff_fc1f == 0xd63f_0000
        })
        .count() as u32
}

#[cfg(not(target_arch = "aarch64"))]
fn count_out_of_line_calls(_bytes: &[u8]) -> u32 {
    // The x86 table envelope emits no calls. Any source-level external calls
    // are validated separately by the MIR contract and are not dense-kernel
    // fallbacks. Keep this legacy AArch64-shaped statistic at zero on x86.
    0
}

#[cfg(test)]
mod portable_descriptor_tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::expr::Expr;
    use crate::{Compiler, Config};

    fn portable_source() -> Application {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let x = Expr::var("x");
        let y = Expr::var("y");
        let product = &x * &y;
        let sum = &x + &y;
        Compiler::with_config(config)
            .compile(&[x, y], &[product, sum])
            .unwrap()
    }

    fn portable_metadata() -> DirectTableApplicationMetadata {
        DirectTableApplicationMetadata::new(
            DirectTableInvocationLayout::new(24, vec![0, 4, 8, 12], 16, 20).unwrap(),
            DirectTableAttachmentLayout::new_with_destination_plane_offsets(
                24,
                vec![0, 4, 8, 12],
                16,
                20,
            )
            .unwrap(),
            0,
        )
        .unwrap()
    }

    fn portable_parameter_source() -> Application {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        let x = Expr::var("x");
        let z = Expr::var("z");
        let output = &x + &z;
        Compiler::with_config(config)
            .compile_params(
                std::slice::from_ref(&x),
                &[output],
                std::slice::from_ref(&z),
            )
            .unwrap()
    }

    fn portable_parameter_metadata() -> DirectTableApplicationMetadata {
        DirectTableApplicationMetadata::new_with_parameter_bindings(
            DirectTableInvocationLayout::new(16, vec![0, 4], 8, 12).unwrap(),
            DirectTableAttachmentLayout::new(16, 0, 4, 8, 12).unwrap(),
            vec![
                DirectTableParameterBinding::Scalar(0),
                DirectTableParameterBinding::Scalar(1),
            ],
            2,
        )
        .unwrap()
    }

    fn assert_decode_fails_closed(bytes: &[u8]) {
        let source = portable_source();
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            DirectTableApplicationMetadata::decode_descriptor(&source, bytes)
        }));
        assert!(attempt.is_ok(), "malformed descriptor must not unwind");
        assert!(
            attempt.unwrap().is_err(),
            "malformed descriptor must fail closed"
        );
    }

    #[test]
    fn descriptor_codec_is_little_endian_fixed_width_bounded_and_source_first() {
        let source = portable_source();
        let metadata = portable_metadata();
        let encoded = metadata.encode_descriptor(&source).unwrap();
        assert_eq!(&encoded[..8], &DIRECT_TABLE_DESCRIPTOR_MAGIC);
        assert_eq!(
            u32::from_le_bytes(encoded[8..12].try_into().unwrap()),
            DIRECT_TABLE_DESCRIPTOR_VERSION
        );
        assert_eq!(
            u32::from_le_bytes(encoded[12..16].try_into().unwrap()) as usize,
            encoded.len()
        );
        assert_eq!(encoded.len(), descriptor_encoded_len(4, 4, 0).unwrap());
        assert_eq!(
            DirectTableApplicationMetadata::decode_descriptor(&source, &encoded).unwrap(),
            metadata
        );
        assert_eq!(metadata.encode_descriptor(&source).unwrap(), encoded);
        DirectTableApplication::from_descriptor(portable_source(), &encoded).unwrap();
        let invocation = [0_u32, 1, 2, 3, 0, 1];
        let attachment = [4_u32, 5, 6, 7, 0, 0];
        let mut storage = [[0.0_f64; 1]; 8];
        let planes = storage
            .iter_mut()
            .map(|plane| unsafe { DirectPlane::from_raw_parts(plane.as_mut_ptr(), plane.len()) })
            .collect::<Vec<_>>();
        let scale_re = [1.0];
        let scale_im = [0.0];
        let view = DirectTableCallViewV1 {
            invocations: invocation.as_ptr().cast(),
            invocation_count: 1,
            invocation_stride: 24,
            attachments: attachment.as_ptr().cast(),
            attachment_count: 1,
            attachment_stride: 24,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: 1,
            point_start: 0,
            point_count: 1,
        };
        unsafe { metadata.validate_call_view(&view) }.unwrap();

        let mut malformed = encoded.clone();
        malformed.pop();
        let malformed_len = malformed.len() as u32;
        malformed[12..16].copy_from_slice(&malformed_len.to_le_bytes());
        assert_decode_fails_closed(&malformed);

        let mut malformed = encoded.clone();
        malformed.extend_from_slice(&[0, 0, 0, 0]);
        let malformed_len = malformed.len() as u32;
        malformed[12..16].copy_from_slice(&malformed_len.to_le_bytes());
        assert_decode_fails_closed(&malformed);

        let mut malformed = encoded.clone();
        malformed[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_decode_fails_closed(&malformed);

        let mut malformed = encoded.clone();
        malformed[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_decode_fails_closed(&malformed);

        let mut malformed = encoded.clone();
        // Invocation input count follows the fixed header and row stride.
        malformed[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_decode_fails_closed(&malformed);

        let mut malformed = encoded.clone();
        // With four invocation offsets, output-component count is at byte 64.
        malformed[64..68].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_decode_fails_closed(&malformed);

        let mut malformed = encoded.clone();
        malformed[72..76].copy_from_slice(&encoded[68..72]);
        assert_decode_fails_closed(&malformed);

        assert_decode_fails_closed(&vec![0; DIRECT_TABLE_DESCRIPTOR_MAX_BYTES + 1]);

        let single_output = {
            let mut config = Config::default();
            config.set_opt_level(2);
            config.set_complex(true);
            config.set_symbolica(true);
            let x = Expr::var("x");
            Compiler::with_config(config)
                .compile(std::slice::from_ref(&x), std::slice::from_ref(&x))
                .unwrap()
        };
        assert!(
            DirectTableApplicationMetadata::decode_descriptor(&single_output, &encoded).is_err()
        );

        let parameter_source = portable_parameter_source();
        let parameter_encoded = portable_parameter_metadata()
            .encode_descriptor(&parameter_source)
            .unwrap();
        assert_eq!(
            DirectTableApplicationMetadata::decode_descriptor(
                &parameter_source,
                &parameter_encoded
            )
            .unwrap(),
            portable_parameter_metadata()
        );
        let mut malformed = parameter_encoded;
        // Two-input/two-output descriptor parameter records begin at byte 84.
        malformed[84..88].copy_from_slice(&2_u32.to_le_bytes());
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            DirectTableApplicationMetadata::decode_descriptor(&parameter_source, &malformed)
        }));
        assert!(attempt.is_ok());
        assert!(attempt.unwrap().is_err());
    }

    #[test]
    fn public_checked_paths_reauthenticate_mutated_metadata_before_row_reads() {
        let source = portable_source();
        let metadata = portable_metadata();
        let unreadable_view = DirectTableCallViewV1 {
            invocations: ptr::null(),
            invocation_count: 1,
            invocation_stride: metadata.invocation.row_stride,
            attachments: ptr::null(),
            attachment_count: 1,
            attachment_stride: metadata.attachment.row_stride,
            planes: ptr::null(),
            plane_count: 0,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: ptr::null(),
            scale_im: ptr::null(),
            scale_count: 0,
            point_start: 0,
            point_count: 1,
        };

        let mut mutated_invocation = metadata.clone();
        mutated_invocation.invocation.input_plane_offsets[0] =
            mutated_invocation.invocation.row_stride;
        let error = unsafe { mutated_invocation.validate_call_view(&unreadable_view) }
            .expect_err("mutated invocation metadata must fail before reading its row");
        assert!(error
            .to_string()
            .contains("invocation input-plane field 0 lies outside row stride"));
        assert!(
            mutated_invocation.encode_descriptor(&source).is_err(),
            "descriptor encoding must revalidate caller-mutated metadata"
        );
        assert!(
            DirectTableApplication::new(portable_source(), mutated_invocation).is_err(),
            "application construction must revalidate caller-mutated metadata"
        );

        let mut mutated_attachment = metadata.clone();
        mutated_attachment.attachment.destination_plane_offsets[0] =
            mutated_attachment.attachment.row_stride;
        let error = unsafe { mutated_attachment.validate_call_view(&unreadable_view) }
            .expect_err("mutated attachment metadata must fail before reading its row");
        assert!(error
            .to_string()
            .contains("attachment destination field 0 lies outside row stride"));

        assert!(
            DirectTableApplicationMetadata::new(
                metadata.invocation,
                metadata.attachment,
                u32::MAX,
            )
            .is_err(),
            "scalar-binding construction must reject unbounded counts before allocation"
        );
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    #[test]
    fn unsupported_host_can_decode_and_validate_but_cannot_seal_callable() {
        let source = portable_source();
        let descriptor = portable_metadata().encode_descriptor(&source).unwrap();
        let application = DirectTableApplication::from_descriptor(source, &descriptor).unwrap();
        let error = match application.seal() {
            Ok(_) => panic!("non-AArch64 direct-table sealing must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("only implemented on AArch64 and x86-64"));
    }
}

#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64")))]
mod tests {
    use super::*;
    use std::rc::Rc;

    use crate::direct::{
        DirectApplication, DirectApplicationMetadata, DirectDestinationOperation,
        DirectInputBinding, DirectInputSnapshot, DirectOutputScale, DIRECT_STATUS_OK,
    };
    use crate::expr::Expr;
    use crate::{Compiler, Config};

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct InvocationRow {
        input_re_0: u32,
        input_im_0: u32,
        input_re_1: u32,
        input_im_1: u32,
        input_re_2: u32,
        input_im_2: u32,
        input_re_3: u32,
        input_im_3: u32,
        input_re_4: u32,
        input_im_4: u32,
        input_re_5: u32,
        input_im_5: u32,
        attachment_start: u32,
        attachment_count: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct AttachmentRow {
        destination_re: u32,
        destination_im: u32,
        scale: u32,
        operation: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct MultiOutputAttachmentRow {
        output_0_re: u32,
        output_0_im: u32,
        output_1_re: u32,
        output_1_im: u32,
        scale: u32,
        operation: u32,
    }

    fn source_application() -> Application {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let x = Expr::var("x");
        let y = Expr::var("y");
        let product = &x * &y;
        Compiler::with_config(config)
            .compile(&[x, y], &[product])
            .unwrap()
    }

    fn multi_output_source_application() -> Application {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let x = Expr::var("x");
        let y = Expr::var("y");
        let product = &x * &y;
        let sum = &x + &y;
        Compiler::with_config(config)
            .compile(&[x, y], &[product, sum])
            .unwrap()
    }

    fn prepared_checkpoint_application() -> Application {
        let mut config = Config::default();
        if cfg!(target_arch = "x86_64") {
            config.ty = crate::CompilerType::AmdAVX;
        }
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        config.set_cse(true);
        config.set_compact(true);
        config.set_parallel_mul(true);

        let parameters = (0..6)
            .map(|index| Expr::var(&format!("p{index}")))
            .collect::<Vec<_>>();
        let scale = Expr::from(0.707106781186547_f64);
        let i_scale = Expr::binary("complex", &Expr::from(0.0), &scale);
        let minus_i_scale = Expr::binary("complex", &Expr::from(0.0), &-&scale);
        let a = &parameters[3] * &minus_i_scale;
        let b = &(&parameters[2] * &i_scale) + &(&parameters[5] * &minus_i_scale);
        let output_0 =
            &(&parameters[0] * &b) + &(&parameters[1] * &(&(&parameters[4] * &scale) + &a));
        let output_1 = &(&parameters[1] * &(&(&parameters[2] + &parameters[5]) * &i_scale))
            + &(&parameters[0] * &(&(&parameters[4] * &-&scale) + &a));

        let mut application = Compiler::with_config(config)
            .compile_params(&[], &[output_0, output_1], &parameters)
            .unwrap();
        application.prepare_simd();
        application
    }

    fn evaluate_source_with_table_partition(
        application: &Application,
        input_planes: &[Vec<f64>],
        plane_len: usize,
        point_start: usize,
        point_count: usize,
    ) -> Vec<f64> {
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (point_start, point_count);
            let mut inputs = Vec::with_capacity(plane_len * input_planes.len());
            for point in 0..plane_len {
                inputs.extend(input_planes.iter().map(|plane| plane[point]));
            }
            let mut outputs = vec![f64::NAN; plane_len * application.count_obs];
            application.evaluate_matrix(&inputs, &mut outputs, plane_len);
            return outputs;
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Mirror the table callable's scalar-head/AVX4-middle/scalar-tail
            // partition so strict bitwise comparisons exercise like-for-like
            // arithmetic schedules rather than a whole-matrix oracle.
            let mut outputs = vec![f64::NAN; plane_len * application.count_obs];
            let lanes = DIRECT_TABLE_SIMD_LANES as usize;
            let head = ((lanes - point_start % lanes) % lanes).min(point_count);
            let remaining = point_count - head;
            let middle = remaining / lanes * lanes;

            let evaluate_scalar = |point: usize, outputs: &mut [f64]| {
                let inputs = input_planes
                    .iter()
                    .map(|plane| plane[point])
                    .collect::<Vec<_>>();
                let mut point_outputs = vec![f64::NAN; application.count_obs];
                application.evaluate(&inputs, &mut point_outputs);
                outputs[point * application.count_obs..(point + 1) * application.count_obs]
                    .copy_from_slice(&point_outputs);
            };

            for point in point_start..point_start + head {
                evaluate_scalar(point, &mut outputs);
            }
            if middle != 0 {
                let middle_start = point_start + head;
                let mut inputs = Vec::with_capacity(middle * input_planes.len());
                for point in middle_start..middle_start + middle {
                    inputs.extend(input_planes.iter().map(|plane| plane[point]));
                }
                let mut middle_outputs = vec![f64::NAN; middle * application.count_obs];
                application.evaluate_matrix(&inputs, &mut middle_outputs, middle);
                outputs[middle_start * application.count_obs
                    ..(middle_start + middle) * application.count_obs]
                    .copy_from_slice(&middle_outputs);
            }
            for point in point_start + head + middle..point_start + point_count {
                evaluate_scalar(point, &mut outputs);
            }
            outputs
        }
    }

    fn prepared_multi_output_application() -> Application {
        let mut application = prepared_checkpoint_application();
        let mut code = MirWriter::new();
        for instruction in application.bytecode.mir.code.iter() {
            match instruction {
                Instruction::SaveComplex {
                    loc: Loc::Mem(index),
                    ..
                } if index >= 2 => {}
                Instruction::Save {
                    loc: Loc::Mem(index),
                    ..
                } if index >= 2 => {}
                _ => code.push(&instruction),
            }
        }
        let mir = Rc::make_mut(&mut application.bytecode.mir);
        mir.code = code;
        mir.populate_labels();
        application.count_obs = 2;
        application.count_diffs = 0;
        application.prog.count_obs = 1;
        application.prog.count_diffs = 0;
        application
    }

    #[test]
    fn prepared_application_has_expected_shape() {
        let application = prepared_checkpoint_application();
        assert_eq!(application.count_states, 0);
        assert_eq!(application.count_params, 12);
        assert_eq!(application.count_obs, 4);
        assert_eq!(application.count_diffs, 0);
    }

    fn source_application_with_scalars() -> Application {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let x = Expr::var("x");
        let y = Expr::var("y");
        let z = Expr::var("z");
        let product = &x * &y;
        let output = &product + &z;
        Compiler::with_config(config)
            .compile_params(&[x, y], &[output], &[z])
            .unwrap()
    }

    fn invocation_layout() -> DirectTableInvocationLayout {
        DirectTableInvocationLayout::new(
            mem::size_of::<InvocationRow>() as u32,
            vec![
                mem::offset_of!(InvocationRow, input_re_0) as u32,
                mem::offset_of!(InvocationRow, input_im_0) as u32,
                mem::offset_of!(InvocationRow, input_re_1) as u32,
                mem::offset_of!(InvocationRow, input_im_1) as u32,
            ],
            mem::offset_of!(InvocationRow, attachment_start) as u32,
            mem::offset_of!(InvocationRow, attachment_count) as u32,
        )
        .unwrap()
    }

    fn prepared_invocation_layout() -> DirectTableInvocationLayout {
        DirectTableInvocationLayout::new(
            mem::size_of::<InvocationRow>() as u32,
            vec![
                mem::offset_of!(InvocationRow, input_re_0) as u32,
                mem::offset_of!(InvocationRow, input_im_0) as u32,
                mem::offset_of!(InvocationRow, input_re_1) as u32,
                mem::offset_of!(InvocationRow, input_im_1) as u32,
                mem::offset_of!(InvocationRow, input_re_2) as u32,
                mem::offset_of!(InvocationRow, input_im_2) as u32,
                mem::offset_of!(InvocationRow, input_re_3) as u32,
                mem::offset_of!(InvocationRow, input_im_3) as u32,
                mem::offset_of!(InvocationRow, input_re_4) as u32,
                mem::offset_of!(InvocationRow, input_im_4) as u32,
                mem::offset_of!(InvocationRow, input_re_5) as u32,
                mem::offset_of!(InvocationRow, input_im_5) as u32,
            ],
            mem::offset_of!(InvocationRow, attachment_start) as u32,
            mem::offset_of!(InvocationRow, attachment_count) as u32,
        )
        .unwrap()
    }

    fn attachment_layout() -> DirectTableAttachmentLayout {
        DirectTableAttachmentLayout::new(
            mem::size_of::<AttachmentRow>() as u32,
            mem::offset_of!(AttachmentRow, destination_re) as u32,
            mem::offset_of!(AttachmentRow, destination_im) as u32,
            mem::offset_of!(AttachmentRow, scale) as u32,
            mem::offset_of!(AttachmentRow, operation) as u32,
        )
        .unwrap()
    }

    fn multi_output_attachment_layout() -> DirectTableAttachmentLayout {
        DirectTableAttachmentLayout::new_with_destination_plane_offsets(
            mem::size_of::<MultiOutputAttachmentRow>() as u32,
            vec![
                mem::offset_of!(MultiOutputAttachmentRow, output_0_re) as u32,
                mem::offset_of!(MultiOutputAttachmentRow, output_0_im) as u32,
                mem::offset_of!(MultiOutputAttachmentRow, output_1_re) as u32,
                mem::offset_of!(MultiOutputAttachmentRow, output_1_im) as u32,
            ],
            mem::offset_of!(MultiOutputAttachmentRow, scale) as u32,
            mem::offset_of!(MultiOutputAttachmentRow, operation) as u32,
        )
        .unwrap()
    }

    fn table_metadata_with_scalars(scalar_input_count: u32) -> DirectTableApplicationMetadata {
        DirectTableApplicationMetadata::new(
            invocation_layout(),
            attachment_layout(),
            scalar_input_count,
        )
        .unwrap()
    }

    fn table_metadata() -> DirectTableApplicationMetadata {
        table_metadata_with_scalars(0)
    }

    fn multi_output_table_metadata() -> DirectTableApplicationMetadata {
        DirectTableApplicationMetadata::new(
            invocation_layout(),
            multi_output_attachment_layout(),
            0,
        )
        .unwrap()
    }

    fn prepared_table_metadata() -> DirectTableApplicationMetadata {
        DirectTableApplicationMetadata::new_with_parameter_bindings(
            prepared_invocation_layout(),
            attachment_layout(),
            vec![
                DirectTableParameterBinding::Plane(0),
                DirectTableParameterBinding::Plane(1),
                DirectTableParameterBinding::Plane(2),
                DirectTableParameterBinding::Plane(3),
                DirectTableParameterBinding::Plane(4),
                DirectTableParameterBinding::Plane(5),
                DirectTableParameterBinding::Plane(6),
                DirectTableParameterBinding::Plane(7),
                DirectTableParameterBinding::Plane(8),
                DirectTableParameterBinding::Plane(9),
                DirectTableParameterBinding::Plane(10),
                DirectTableParameterBinding::Plane(11),
            ],
            0,
        )
        .unwrap()
    }

    fn prepared_multi_output_table_metadata() -> DirectTableApplicationMetadata {
        DirectTableApplicationMetadata::new_with_parameter_bindings(
            prepared_invocation_layout(),
            DirectTableAttachmentLayout::new_with_destination_plane_offsets(
                mem::size_of::<MultiOutputAttachmentRow>() as u32,
                vec![
                    mem::offset_of!(MultiOutputAttachmentRow, output_0_re) as u32,
                    mem::offset_of!(MultiOutputAttachmentRow, output_0_im) as u32,
                    mem::offset_of!(MultiOutputAttachmentRow, output_1_re) as u32,
                    mem::offset_of!(MultiOutputAttachmentRow, output_1_im) as u32,
                ],
                mem::offset_of!(MultiOutputAttachmentRow, scale) as u32,
                mem::offset_of!(MultiOutputAttachmentRow, operation) as u32,
            )
            .unwrap(),
            (0..12).map(DirectTableParameterBinding::Plane).collect(),
            0,
        )
        .unwrap()
    }

    fn mixed_table_metadata() -> DirectTableApplicationMetadata {
        DirectTableApplicationMetadata::new_with_parameter_bindings(
            invocation_layout(),
            attachment_layout(),
            vec![
                DirectTableParameterBinding::Plane(0),
                DirectTableParameterBinding::Scalar(0),
            ],
            1,
        )
        .unwrap()
    }

    fn oracle_callable(operation: DirectDestinationOperation) -> crate::DirectCallable {
        let metadata = DirectApplicationMetadata::new(
            operation,
            DirectInputSnapshot::Live,
            DirectOutputScale::ComplexScalar,
            vec![0, 1, 2, 3],
            Vec::new(),
            6,
            2,
            vec![4, 5],
        )
        .unwrap();
        DirectApplication::new(source_application(), metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable()
    }

    fn scalar_oracle_callable(operation: DirectDestinationOperation) -> crate::DirectCallable {
        let metadata = DirectApplicationMetadata::new(
            operation,
            DirectInputSnapshot::Live,
            DirectOutputScale::ComplexScalar,
            vec![0, 1, 2, 3],
            vec![DirectInputBinding::Scalar(2), DirectInputBinding::Scalar(3)],
            6,
            4,
            vec![4, 5],
        )
        .unwrap();
        DirectApplication::new(source_application_with_scalars(), metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable()
    }

    fn prepared_oracle_callable(operation: DirectDestinationOperation) -> crate::DirectCallable {
        let metadata = DirectApplicationMetadata::new(
            operation,
            DirectInputSnapshot::Live,
            DirectOutputScale::ComplexScalar,
            Vec::new(),
            vec![
                DirectInputBinding::Plane(0),
                DirectInputBinding::Plane(1),
                DirectInputBinding::Plane(2),
                DirectInputBinding::Plane(3),
                DirectInputBinding::Plane(4),
                DirectInputBinding::Plane(5),
                DirectInputBinding::Plane(6),
                DirectInputBinding::Plane(7),
                DirectInputBinding::Plane(8),
                DirectInputBinding::Plane(9),
                DirectInputBinding::Plane(10),
                DirectInputBinding::Plane(11),
            ],
            14,
            2,
            vec![12, 13],
        )
        .unwrap();
        // The table path keeps the prepared source at O3. Its per-row oracle
        // lowers the same retained MIR through the O2 complex-scale path.
        let mut oracle_source = prepared_multi_output_application();
        oracle_source.config.set_opt_level(2);
        DirectApplication::new(oracle_source, metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable()
    }

    fn mixed_oracle_callable(operation: DirectDestinationOperation) -> crate::DirectCallable {
        let metadata = DirectApplicationMetadata::new(
            operation,
            DirectInputSnapshot::Live,
            DirectOutputScale::ComplexScalar,
            vec![0, 1, 2, 3],
            vec![DirectInputBinding::Plane(0), DirectInputBinding::Scalar(2)],
            6,
            3,
            vec![4, 5],
        )
        .unwrap();
        DirectApplication::new(source_application_with_scalars(), metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable()
    }

    fn add_plane(table: &mut Vec<Vec<f64>>, oracle: &mut Vec<Vec<f64>>, values: Vec<f64>) -> u32 {
        let index = table.len() as u32;
        table.push(values.clone());
        oracle.push(values);
        index
    }

    fn assert_storage_bits(
        actual: &[Vec<f64>],
        expected: &[Vec<f64>],
        context: impl std::fmt::Display,
    ) {
        assert_eq!(actual.len(), expected.len(), "{context}: plane count");
        for (plane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.len(),
                expected.len(),
                "{context}: plane {plane} point count"
            );
            for (point, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{context}: plane {plane}, point {point}, actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_oracle(
        oracle_factory: fn(DirectDestinationOperation) -> crate::DirectCallable,
        invocations: &[InvocationRow],
        attachments: &[AttachmentRow],
        values: &mut [Vec<f64>],
        scale_re: &[f64],
        scale_im: &[f64],
        point_start: usize,
        point_count: usize,
        input_plane_count: usize,
    ) {
        let overwrite = oracle_factory(DirectDestinationOperation::Overwrite);
        let accumulate = oracle_factory(DirectDestinationOperation::Accumulate);

        for invocation in invocations {
            for attachment_index in invocation.attachment_start
                ..invocation.attachment_start + invocation.attachment_count
            {
                let attachment = attachments[attachment_index as usize];
                let input_indices = [
                    invocation.input_re_0,
                    invocation.input_im_0,
                    invocation.input_re_1,
                    invocation.input_im_1,
                    invocation.input_re_2,
                    invocation.input_im_2,
                    invocation.input_re_3,
                    invocation.input_im_3,
                    invocation.input_re_4,
                    invocation.input_im_4,
                    invocation.input_re_5,
                    invocation.input_im_5,
                ];
                let mut planes = Vec::with_capacity(input_plane_count + 4);
                for &index in &input_indices[..input_plane_count] {
                    let values = &mut values[index as usize];
                    planes.push(unsafe {
                        DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
                    });
                }
                for index in [attachment.destination_re, attachment.destination_im] {
                    let values = &mut values[index as usize];
                    planes.push(unsafe {
                        DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
                    });
                }
                planes.push(planes[input_plane_count]);
                planes.push(planes[input_plane_count + 1]);

                let scale_real = scale_re[attachment.scale as usize];
                let scale_imag = scale_im[attachment.scale as usize];
                let scalars = [
                    unsafe { DirectScalar::from_raw(ptr::from_ref(&scale_real)) },
                    unsafe { DirectScalar::from_raw(ptr::from_ref(&scale_imag)) },
                ];
                let callable = if attachment.operation == 0 {
                    &overwrite
                } else {
                    &accumulate
                };
                assert_eq!(
                    unsafe {
                        callable.handle().invoke(
                            &planes,
                            &scalars,
                            point_start as u32,
                            point_count as u32,
                        )
                    },
                    DIRECT_STATUS_OK
                );
            }
        }
    }

    fn run_case(
        row_count: usize,
        fanout: usize,
        point_start: usize,
        point_count: usize,
        expected_generated_calls: u32,
    ) {
        run_case_with_stack(
            row_count,
            fanout,
            point_start,
            point_count,
            expected_generated_calls,
            None,
        );
    }

    fn expected_generated_calls(row_count: usize, point_start: usize, point_count: usize) -> u32 {
        let lanes = DIRECT_TABLE_SIMD_LANES as usize;
        let head = ((lanes - point_start % lanes) % lanes).min(point_count);
        let remaining = point_count - head;
        let middle = remaining / lanes * lanes;
        let tail = remaining - middle;
        let segments = usize::from(head != 0) + usize::from(middle != 0) + usize::from(tail != 0);
        u32::try_from(row_count * segments).unwrap()
    }

    fn run_case_with_stack(
        row_count: usize,
        fanout: usize,
        point_start: usize,
        point_count: usize,
        expected_generated_calls: u32,
        table_stack_slots: Option<usize>,
    ) {
        run_case_with_source(
            row_count,
            fanout,
            point_start,
            point_count,
            expected_generated_calls,
            table_stack_slots,
            source_application,
            table_metadata,
            oracle_callable,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case_with_source(
        row_count: usize,
        fanout: usize,
        point_start: usize,
        point_count: usize,
        expected_generated_calls: u32,
        table_stack_slots: Option<usize>,
        source_factory: fn() -> Application,
        metadata_factory: fn() -> DirectTableApplicationMetadata,
        oracle_factory: fn(DirectDestinationOperation) -> crate::DirectCallable,
    ) {
        const PLANE_LEN: usize = 9;

        let scale_re = [0.75, -1.25, 0.5];
        let scale_im = [-0.5, 0.25, 1.5];
        let table_metadata = metadata_factory();
        let input_plane_count = table_metadata.invocation.input_plane_offsets.len();
        let mut table_values = Vec::<Vec<f64>>::new();
        let mut oracle_values = Vec::<Vec<f64>>::new();
        let mut invocations = Vec::with_capacity(row_count);
        let mut attachments = Vec::with_capacity(row_count * fanout);

        for row in 0..row_count {
            let mut inputs = [0_u32; 12];
            for (component, slot) in inputs[..input_plane_count].iter_mut().enumerate() {
                let values = (0..PLANE_LEN)
                    .map(|point| {
                        0.125 * (1 + row) as f64
                            + 0.03125 * (1 + component) as f64
                            + 0.0078125 * point as f64
                    })
                    .collect();
                *slot = add_plane(&mut table_values, &mut oracle_values, values);
            }

            let attachment_start = attachments.len() as u32;
            for output in 0..fanout {
                let initial_re = vec![-7.0 - row as f64 - output as f64; PLANE_LEN];
                let initial_im = vec![11.0 + row as f64 + output as f64; PLANE_LEN];
                let destination_re = add_plane(&mut table_values, &mut oracle_values, initial_re);
                let destination_im = add_plane(&mut table_values, &mut oracle_values, initial_im);
                attachments.push(AttachmentRow {
                    destination_re,
                    destination_im,
                    scale: ((row + output) % scale_re.len()) as u32,
                    operation: ((row + output) & 1) as u32,
                });
            }
            invocations.push(InvocationRow {
                input_re_0: inputs[0],
                input_im_0: inputs[1],
                input_re_1: inputs[2],
                input_im_1: inputs[3],
                input_re_2: inputs[4],
                input_im_2: inputs[5],
                input_re_3: inputs[6],
                input_im_3: inputs[7],
                input_re_4: inputs[8],
                input_im_4: inputs[9],
                input_re_5: inputs[10],
                input_im_5: inputs[11],
                attachment_start,
                attachment_count: fanout as u32,
            });
        }

        run_oracle(
            oracle_factory,
            &invocations,
            &attachments,
            &mut oracle_values,
            &scale_re,
            &scale_im,
            point_start,
            point_count,
            input_plane_count,
        );

        let mut plane_descriptors = table_values
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        let view = DirectTableCallViewV1 {
            invocations: invocations.as_ptr().cast(),
            invocation_count: invocations.len() as u32,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: plane_descriptors.as_mut_ptr(),
            plane_count: plane_descriptors.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: scale_re.len() as u32,
            point_start: point_start as u32,
            point_count: point_count as u32,
        };

        let mut table_source = source_factory();
        if let Some(stack_slots) = table_stack_slots {
            table_source.prog.builder.count_stack = Some(stack_slots);
        }
        let callable = DirectTableApplication::new(table_source, table_metadata)
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);

        assert_storage_bits(
            &table_values,
            &oracle_values,
            format_args!("row_count={row_count}, fanout={fanout}"),
        );

        let generated_calls = unsafe { callable.context.dispatch_table_unchecked(&view).unwrap() };
        assert_eq!(generated_calls, expected_generated_calls);
    }

    #[test]
    fn table_callable_matches_per_row_oracle_for_requested_matrix() {
        for row_count in [1, 2, 64] {
            for fanout in [1, 2] {
                run_case(row_count, fanout, 0, 7, (2 * row_count) as u32);
            }
        }
    }

    #[test]
    fn packaged_prepared_o3_first_output_matches_single_attachment_table_path() {
        const PLANE_LEN: usize = 9;
        const POINT_START: usize = 1;
        const POINT_COUNT: usize = 5;
        const DESTINATION_RE_SENTINEL: f64 = -77.0;
        const DESTINATION_IM_SENTINEL: f64 = 91.0;

        let input_planes = (0..12)
            .map(|component| {
                (0..PLANE_LEN)
                    .map(|point| {
                        0.125 + 0.03125 * (component + 1) as f64 + 0.0078125 * point as f64
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Evaluate the untouched 360-byte O3 payload, retaining both of its
        // complex outputs. The table source below independently derives only
        // the first complex store from that same pinned payload.
        let mut untouched_o3 = prepared_checkpoint_application();
        untouched_o3.prepare_simd();
        assert_eq!(untouched_o3.count_params, 12);
        assert_eq!(untouched_o3.count_obs, 4);
        let untouched_outputs = evaluate_source_with_table_partition(
            &untouched_o3,
            &input_planes,
            PLANE_LEN,
            POINT_START,
            POINT_COUNT,
        );

        let mut table_values = input_planes;
        let destination_re = table_values.len() as u32;
        table_values.push(vec![DESTINATION_RE_SENTINEL; PLANE_LEN]);
        let destination_im = table_values.len() as u32;
        table_values.push(vec![DESTINATION_IM_SENTINEL; PLANE_LEN]);
        let invocations = [InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 4,
            input_im_2: 5,
            input_re_3: 6,
            input_im_3: 7,
            input_re_4: 8,
            input_im_4: 9,
            input_re_5: 10,
            input_im_5: 11,
            attachment_start: 0,
            attachment_count: 1,
        }];
        let attachments = [AttachmentRow {
            destination_re,
            destination_im,
            scale: 0,
            operation: DirectTableDestinationOperation::Overwrite as u32,
        }];
        // A single unit-scale attachment exposes the retained kernel output
        // before any multi-destination fanout can obscure source fidelity.
        let scale_re = [1.0];
        let scale_im = [0.0];
        let mut plane_descriptors = table_values
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        let view = DirectTableCallViewV1 {
            invocations: invocations.as_ptr().cast(),
            invocation_count: invocations.len() as u32,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: plane_descriptors.as_mut_ptr(),
            plane_count: plane_descriptors.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: scale_re.len() as u32,
            point_start: POINT_START as u32,
            point_count: POINT_COUNT as u32,
        };
        let table_source = prepared_multi_output_application();
        let callable = DirectTableApplication::new(table_source, prepared_table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);

        for point in 0..PLANE_LEN {
            let actual_re = table_values[destination_re as usize][point];
            let actual_im = table_values[destination_im as usize][point];
            if (POINT_START..POINT_START + POINT_COUNT).contains(&point) {
                assert_eq!(
                    actual_re.to_bits(),
                    untouched_outputs[point * untouched_o3.count_obs].to_bits(),
                    "untouched O3 real output mismatch at point {point}"
                );
                assert_eq!(
                    actual_im.to_bits(),
                    untouched_outputs[point * untouched_o3.count_obs + 1].to_bits(),
                    "untouched O3 imaginary output mismatch at point {point}"
                );
                assert_ne!(
                    actual_re.to_bits(),
                    DESTINATION_RE_SENTINEL.to_bits(),
                    "active real destination remained at its sentinel"
                );
                assert_ne!(
                    actual_im.to_bits(),
                    DESTINATION_IM_SENTINEL.to_bits(),
                    "active imaginary destination remained at its sentinel"
                );
            } else {
                assert_eq!(actual_re.to_bits(), DESTINATION_RE_SENTINEL.to_bits());
                assert_eq!(actual_im.to_bits(), DESTINATION_IM_SENTINEL.to_bits());
            }
        }
    }

    #[test]
    fn packaged_prepared_o3_all_outputs_fan_out_as_one_ordered_attachment_vector() {
        const PLANE_LEN: usize = 9;
        const POINT_COUNT: usize = 7;
        let input_planes = (0..12)
            .map(|component| {
                (0..PLANE_LEN)
                    .map(|point| {
                        0.125 + 0.03125 * (component + 1) as f64 + 0.0078125 * point as f64
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut untouched = prepared_checkpoint_application();
        untouched.prepare_simd();
        let source_outputs = evaluate_source_with_table_partition(
            &untouched,
            &input_planes,
            PLANE_LEN,
            0,
            POINT_COUNT,
        );

        let mut table_values = input_planes;
        let mut destination_sets = [[0_u32; 4]; 2];
        for (fanout, destinations) in destination_sets.iter_mut().enumerate() {
            for (component, destination) in destinations.iter_mut().enumerate() {
                *destination = table_values.len() as u32;
                table_values.push(vec![-100.0 - (fanout * 4 + component) as f64; PLANE_LEN]);
            }
        }
        let invocation = InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 4,
            input_im_2: 5,
            input_re_3: 6,
            input_im_3: 7,
            input_re_4: 8,
            input_im_4: 9,
            input_re_5: 10,
            input_im_5: 11,
            attachment_start: 0,
            attachment_count: 2,
        };
        let attachments = destination_sets.map(|destination| MultiOutputAttachmentRow {
            output_0_re: destination[0],
            output_0_im: destination[1],
            output_1_re: destination[2],
            output_1_im: destination[3],
            scale: 0,
            operation: DirectTableDestinationOperation::Overwrite as u32,
        });
        let scale_re = [1.0];
        let scale_im = [0.0];
        let planes = table_values
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        let view = DirectTableCallViewV1 {
            invocations: ptr::from_ref(&invocation).cast(),
            invocation_count: 1,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<MultiOutputAttachmentRow>() as u32,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: 1,
            point_start: 0,
            point_count: POINT_COUNT as u32,
        };
        let callable = DirectTableApplication::new(
            prepared_checkpoint_application(),
            prepared_multi_output_table_metadata(),
        )
        .unwrap()
        .seal()
        .unwrap()
        .into_callable();
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);

        for destinations in destination_sets {
            for (component, destination) in destinations.into_iter().enumerate() {
                for point in 0..PLANE_LEN {
                    let actual = table_values[destination as usize][point];
                    if point < POINT_COUNT {
                        assert_eq!(
                            actual.to_bits(),
                            source_outputs[point * 4 + component].to_bits(),
                            "output component {component}, point {point}"
                        );
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            (-100.0
                                - destination_sets
                                    .iter()
                                    .flatten()
                                    .position(|&candidate| candidate == destination)
                                    .unwrap() as f64)
                                .to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn multiple_complex_outputs_preserve_row_attachment_and_output_order() {
        const POINT_COUNT: usize = 7;
        let scale_re = [1.0, 0.5];
        let scale_im = [0.0, 0.0];
        let mut values = Vec::<Vec<f64>>::new();
        for row in 0..2 {
            for component in 0..4 {
                values.push(
                    (0..POINT_COUNT)
                        .map(|point| (row * 16 + component * 2 + point + 1) as f64 / 16.0)
                        .collect(),
                );
            }
        }
        for destination in 0..8 {
            values.push(vec![-4.0 + destination as f64 / 8.0; POINT_COUNT]);
        }
        let mut expected = values.clone();
        let invocations = [
            InvocationRow {
                input_re_0: 0,
                input_im_0: 1,
                input_re_1: 2,
                input_im_1: 3,
                input_re_2: 0,
                input_im_2: 0,
                input_re_3: 0,
                input_im_3: 0,
                input_re_4: 0,
                input_im_4: 0,
                input_re_5: 0,
                input_im_5: 0,
                attachment_start: 0,
                attachment_count: 2,
            },
            InvocationRow {
                input_re_0: 4,
                input_im_0: 5,
                input_re_1: 6,
                input_im_1: 7,
                input_re_2: 0,
                input_im_2: 0,
                input_re_3: 0,
                input_im_3: 0,
                input_re_4: 0,
                input_im_4: 0,
                input_re_5: 0,
                input_im_5: 0,
                attachment_start: 2,
                attachment_count: 2,
            },
        ];
        let attachments = [
            MultiOutputAttachmentRow {
                output_0_re: 8,
                output_0_im: 9,
                output_1_re: 10,
                output_1_im: 11,
                scale: 0,
                operation: 0,
            },
            MultiOutputAttachmentRow {
                output_0_re: 12,
                output_0_im: 13,
                output_1_re: 14,
                output_1_im: 15,
                scale: 1,
                operation: 1,
            },
            MultiOutputAttachmentRow {
                output_0_re: 8,
                output_0_im: 9,
                output_1_re: 10,
                output_1_im: 11,
                scale: 1,
                operation: 1,
            },
            MultiOutputAttachmentRow {
                output_0_re: 12,
                output_0_im: 13,
                output_1_re: 14,
                output_1_im: 15,
                scale: 0,
                operation: 0,
            },
        ];
        for (row_index, invocation) in invocations.iter().enumerate() {
            for attachment_index in invocation.attachment_start
                ..invocation.attachment_start + invocation.attachment_count
            {
                let attachment = attachments[attachment_index as usize];
                let x_re = &values[row_index * 4];
                let x_im = &values[row_index * 4 + 1];
                let y_re = &values[row_index * 4 + 2];
                let y_im = &values[row_index * 4 + 3];
                let destinations = [
                    attachment.output_0_re,
                    attachment.output_0_im,
                    attachment.output_1_re,
                    attachment.output_1_im,
                ];
                for point in 0..POINT_COUNT {
                    let outputs = [
                        x_re[point] * y_re[point] - x_im[point] * y_im[point],
                        x_re[point] * y_im[point] + x_im[point] * y_re[point],
                        x_re[point] + y_re[point],
                        x_im[point] + y_im[point],
                    ];
                    for output in 0..2 {
                        let real = outputs[2 * output];
                        let imag = outputs[2 * output + 1];
                        let scale = attachment.scale as usize;
                        let scaled = [
                            real * scale_re[scale] - imag * scale_im[scale],
                            real * scale_im[scale] + imag * scale_re[scale],
                        ];
                        for component in 0..2 {
                            let destination = destinations[2 * output + component] as usize;
                            if attachment.operation == 0 {
                                expected[destination][point] = scaled[component];
                            } else {
                                expected[destination][point] += scaled[component];
                            }
                        }
                    }
                }
            }
        }

        let planes = values
            .iter_mut()
            .map(|plane| unsafe { DirectPlane::from_raw_parts(plane.as_mut_ptr(), plane.len()) })
            .collect::<Vec<_>>();
        let view = DirectTableCallViewV1 {
            invocations: invocations.as_ptr().cast(),
            invocation_count: invocations.len() as u32,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<MultiOutputAttachmentRow>() as u32,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: scale_re.len() as u32,
            point_start: 0,
            point_count: POINT_COUNT as u32,
        };
        let callable = DirectTableApplication::new(
            multi_output_source_application(),
            multi_output_table_metadata(),
        )
        .unwrap()
        .seal()
        .unwrap()
        .into_callable();
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);
        assert_storage_bits(&values, &expected, "multi-output ordered fanout");
        assert_eq!(
            unsafe { callable.context.dispatch_table_unchecked(&view).unwrap() },
            4
        );
    }

    #[test]
    fn generated_multi_output_application_matches_per_row_oracle() {
        let prepared = prepared_multi_output_application();
        assert_eq!(prepared.count_states, 0);
        assert_eq!(prepared.count_params, 12);
        assert_eq!(prepared.count_obs, 2);
        assert_eq!(prepared.count_diffs, 0);

        let callable = DirectTableApplication::new(prepared, prepared_table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        for shape in [callable.scalar_code_shape(), callable.simd_code_shape()] {
            assert_eq!(shape.packet_materializations, 0);
            assert_eq!(shape.gather_materializations, 0);
            assert_eq!(shape.scatter_materializations, 0);
        }

        run_case_with_source(
            3,
            2,
            0,
            7,
            6,
            None,
            prepared_multi_output_application,
            prepared_table_metadata,
            prepared_oracle_callable,
        );
        run_case_with_source(
            3,
            2,
            1,
            5,
            6,
            None,
            prepared_multi_output_application,
            prepared_table_metadata,
            prepared_oracle_callable,
        );
    }

    #[test]
    fn bitwise_oracle_covers_signed_zero_finite_edges_and_nan_payloads() {
        const QNAN_A: u64 = 0x7ff8_0000_0000_1234;
        const QNAN_B: u64 = 0xfff8_0000_0000_5678;
        let input_re_0 = vec![
            0.0,
            -0.0,
            f64::from_bits(1),
            f64::from_bits(0x8000_0000_0000_0001),
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            1.0,
            -1.0,
            f64::MAX,
            -f64::MAX,
            f64::from_bits(QNAN_A),
            f64::from_bits(QNAN_B),
        ];
        let point_count = input_re_0.len();
        let mut table_values = vec![
            input_re_0,
            vec![0.0; point_count],
            vec![1.0; point_count],
            vec![0.0; point_count],
            vec![-0.0; point_count],
            vec![0.0; point_count],
            vec![f64::from_bits(1); point_count],
            vec![f64::from_bits(0x8000_0000_0000_0001); point_count],
        ];
        let mut oracle_values = table_values.clone();
        let invocations = [InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 0,
            input_im_2: 0,
            input_re_3: 0,
            input_im_3: 0,
            input_re_4: 0,
            input_im_4: 0,
            input_re_5: 0,
            input_im_5: 0,
            attachment_start: 0,
            attachment_count: 2,
        }];
        let attachments = [
            AttachmentRow {
                destination_re: 4,
                destination_im: 5,
                scale: 0,
                operation: DirectTableDestinationOperation::Overwrite as u32,
            },
            AttachmentRow {
                destination_re: 6,
                destination_im: 7,
                scale: 1,
                operation: DirectTableDestinationOperation::Accumulate as u32,
            },
        ];
        let scale_re = [1.0, -0.5];
        let scale_im = [0.0, 0.25];

        run_oracle(
            oracle_callable,
            &invocations,
            &attachments,
            &mut oracle_values,
            &scale_re,
            &scale_im,
            0,
            point_count,
            4,
        );

        let planes = table_values
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        let view = DirectTableCallViewV1 {
            invocations: invocations.as_ptr().cast(),
            invocation_count: invocations.len() as u32,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: scale_re.len() as u32,
            point_start: 0,
            point_count: point_count as u32,
        };
        let callable = DirectTableApplication::new(source_application(), table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);

        assert_storage_bits(&table_values, &oracle_values, "floating-point edge matrix");
        assert_eq!(table_values[4][0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(table_values[4][1].to_bits(), (-0.0_f64).to_bits());
        // These points introduce one quiet NaN payload and otherwise only
        // finite/zero operands, so the AArch64 non-default-NaN path preserves
        // that payload through the kernel and unit-scale attachment.
        assert_eq!(table_values[4][10].to_bits(), QNAN_A);
        assert_eq!(table_values[4][11].to_bits(), QNAN_B);
    }

    #[test]
    fn scalar_descriptors_broadcast_and_match_per_row_oracle() {
        const POINT_COUNT: usize = 7;

        let scale_re = [0.75, -1.25];
        let scale_im = [-0.5, 0.25];
        let parameter_re = 0.375;
        let parameter_im = -0.625;
        let invocations = [InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 0,
            input_im_2: 0,
            input_re_3: 0,
            input_im_3: 0,
            input_re_4: 0,
            input_im_4: 0,
            input_re_5: 0,
            input_im_5: 0,
            attachment_start: 0,
            attachment_count: 2,
        }];
        let attachments = [
            AttachmentRow {
                destination_re: 4,
                destination_im: 5,
                scale: 0,
                operation: 0,
            },
            AttachmentRow {
                destination_re: 6,
                destination_im: 7,
                scale: 1,
                operation: 1,
            },
        ];
        let mut table_values = vec![
            (0..POINT_COUNT).map(|i| 0.25 + i as f64 / 64.0).collect(),
            (0..POINT_COUNT)
                .map(|i| -0.125 + i as f64 / 128.0)
                .collect(),
            (0..POINT_COUNT).map(|i| 0.5 - i as f64 / 96.0).collect(),
            (0..POINT_COUNT)
                .map(|i| 0.1875 + i as f64 / 160.0)
                .collect(),
            vec![-3.0; POINT_COUNT],
            vec![5.0; POINT_COUNT],
            vec![7.0; POINT_COUNT],
            vec![-11.0; POINT_COUNT],
        ];
        let mut oracle_values = table_values.clone();

        let overwrite = scalar_oracle_callable(DirectDestinationOperation::Overwrite);
        let accumulate = scalar_oracle_callable(DirectDestinationOperation::Accumulate);
        for attachment in attachments {
            let mut planes = Vec::with_capacity(8);
            for index in [
                0,
                1,
                2,
                3,
                attachment.destination_re,
                attachment.destination_im,
            ] {
                let values = &mut oracle_values[index as usize];
                planes.push(unsafe {
                    DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
                });
            }
            planes.push(planes[4]);
            planes.push(planes[5]);

            let exact_re = scale_re[attachment.scale as usize];
            let exact_im = scale_im[attachment.scale as usize];
            let scalars = [
                unsafe { DirectScalar::from_raw(ptr::from_ref(&exact_re)) },
                unsafe { DirectScalar::from_raw(ptr::from_ref(&exact_im)) },
                unsafe { DirectScalar::from_raw(ptr::from_ref(&parameter_re)) },
                unsafe { DirectScalar::from_raw(ptr::from_ref(&parameter_im)) },
            ];
            let callable = if attachment.operation == 0 {
                &overwrite
            } else {
                &accumulate
            };
            assert_eq!(
                unsafe {
                    callable
                        .handle()
                        .invoke(&planes, &scalars, 0, POINT_COUNT as u32)
                },
                DIRECT_STATUS_OK
            );
        }

        let mut plane_descriptors = table_values
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        let scalar_descriptors = [
            unsafe { DirectScalar::from_raw(ptr::from_ref(&parameter_re)) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&parameter_im)) },
        ];
        let view = DirectTableCallViewV1 {
            invocations: invocations.as_ptr().cast(),
            invocation_count: invocations.len() as u32,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: plane_descriptors.as_mut_ptr(),
            plane_count: plane_descriptors.len() as u32,
            scalar_count: scalar_descriptors.len() as u32,
            scalars: scalar_descriptors.as_ptr(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: scale_re.len() as u32,
            point_start: 0,
            point_count: POINT_COUNT as u32,
        };
        let callable = DirectTableApplication::new(
            source_application_with_scalars(),
            table_metadata_with_scalars(2),
        )
        .unwrap()
        .seal()
        .unwrap()
        .into_callable();
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);
        assert_storage_bits(&table_values, &oracle_values, "scalar descriptors");
    }

    #[test]
    fn plane_and_scalar_parameter_bindings_can_be_mixed_in_one_source() {
        const POINT_COUNT: usize = 7;

        let scale_re = [0.75, -1.25];
        let scale_im = [-0.5, 0.25];
        let parameter_im = -0.625;
        let invocations = [InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 0,
            input_im_2: 0,
            input_re_3: 0,
            input_im_3: 0,
            input_re_4: 0,
            input_im_4: 0,
            input_re_5: 0,
            input_im_5: 0,
            attachment_start: 0,
            attachment_count: 2,
        }];
        let attachments = [
            AttachmentRow {
                destination_re: 4,
                destination_im: 5,
                scale: 0,
                operation: 0,
            },
            AttachmentRow {
                destination_re: 6,
                destination_im: 7,
                scale: 1,
                operation: 1,
            },
        ];
        let mut table_values = vec![
            (0..POINT_COUNT).map(|i| 0.25 + i as f64 / 64.0).collect(),
            (0..POINT_COUNT)
                .map(|i| -0.125 + i as f64 / 128.0)
                .collect(),
            (0..POINT_COUNT).map(|i| 0.5 - i as f64 / 96.0).collect(),
            (0..POINT_COUNT)
                .map(|i| 0.1875 + i as f64 / 160.0)
                .collect(),
            vec![-3.0; POINT_COUNT],
            vec![5.0; POINT_COUNT],
            vec![7.0; POINT_COUNT],
            vec![-11.0; POINT_COUNT],
        ];
        let mut oracle_values = table_values.clone();

        let overwrite = mixed_oracle_callable(DirectDestinationOperation::Overwrite);
        let accumulate = mixed_oracle_callable(DirectDestinationOperation::Accumulate);
        for attachment in attachments {
            let mut planes = Vec::with_capacity(8);
            for index in [
                0,
                1,
                2,
                3,
                attachment.destination_re,
                attachment.destination_im,
            ] {
                let values = &mut oracle_values[index as usize];
                planes.push(unsafe {
                    DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len())
                });
            }
            planes.push(planes[4]);
            planes.push(planes[5]);

            let exact_re = scale_re[attachment.scale as usize];
            let exact_im = scale_im[attachment.scale as usize];
            let scalars = [
                unsafe { DirectScalar::from_raw(ptr::from_ref(&exact_re)) },
                unsafe { DirectScalar::from_raw(ptr::from_ref(&exact_im)) },
                unsafe { DirectScalar::from_raw(ptr::from_ref(&parameter_im)) },
            ];
            let callable = if attachment.operation == 0 {
                &overwrite
            } else {
                &accumulate
            };
            assert_eq!(
                unsafe {
                    callable
                        .handle()
                        .invoke(&planes, &scalars, 0, POINT_COUNT as u32)
                },
                DIRECT_STATUS_OK
            );
        }

        let plane_descriptors = table_values
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        let scalar_descriptors = [unsafe { DirectScalar::from_raw(ptr::from_ref(&parameter_im)) }];
        let view = DirectTableCallViewV1 {
            invocations: invocations.as_ptr().cast(),
            invocation_count: invocations.len() as u32,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: plane_descriptors.as_ptr(),
            plane_count: plane_descriptors.len() as u32,
            scalar_count: scalar_descriptors.len() as u32,
            scalars: scalar_descriptors.as_ptr(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: scale_re.len() as u32,
            point_start: 0,
            point_count: POINT_COUNT as u32,
        };
        let callable =
            DirectTableApplication::new(source_application_with_scalars(), mixed_table_metadata())
                .unwrap()
                .seal()
                .unwrap()
                .into_callable();
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);
        assert_storage_bits(
            &table_values,
            &oracle_values,
            "mixed plane/scalar parameter descriptors",
        );
    }

    #[test]
    fn unaligned_range_dispatches_scalar_head_simd_middle_and_scalar_tail() {
        run_case(2, 2, 1, 4, expected_generated_calls(2, 1, 4));
    }

    #[test]
    fn warmed_checked_and_unchecked_calls_perform_zero_allocations() {
        let mut input_values = [[0.0; 7]; 12];
        for (component, values) in input_values.iter_mut().enumerate() {
            for (point, value) in values.iter_mut().enumerate() {
                *value = 0.125 * (component + 1) as f64 + 0.015625 * point as f64;
            }
        }
        let mut output_0_re = [0.0; 7];
        let mut output_0_im = [-0.0; 7];
        let mut output_1_re = [1.0; 7];
        let mut output_1_im = [-1.0; 7];
        let mut planes = input_values
            .iter_mut()
            .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
            .collect::<Vec<_>>();
        for output in [
            output_0_re.as_mut_slice(),
            output_0_im.as_mut_slice(),
            output_1_re.as_mut_slice(),
            output_1_im.as_mut_slice(),
        ] {
            planes.push(unsafe { DirectPlane::from_raw_parts(output.as_mut_ptr(), output.len()) });
        }
        let invocation = InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 4,
            input_im_2: 5,
            input_re_3: 6,
            input_im_3: 7,
            input_re_4: 8,
            input_im_4: 9,
            input_re_5: 10,
            input_im_5: 11,
            attachment_start: 0,
            attachment_count: 1,
        };
        let attachment = MultiOutputAttachmentRow {
            output_0_re: 12,
            output_0_im: 13,
            output_1_re: 14,
            output_1_im: 15,
            scale: 0,
            operation: DirectTableDestinationOperation::Overwrite as u32,
        };
        let scale_re = [0.75];
        let scale_im = [-0.5];
        let view = DirectTableCallViewV1 {
            invocations: ptr::from_ref(&invocation).cast(),
            invocation_count: 1,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: ptr::from_ref(&attachment).cast(),
            attachment_count: 1,
            attachment_stride: mem::size_of::<MultiOutputAttachmentRow>() as u32,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: scale_re.len() as u32,
            point_start: 0,
            point_count: input_values[0].len() as u32,
        };
        let callable = DirectTableApplication::new(
            prepared_checkpoint_application(),
            prepared_multi_output_table_metadata(),
        )
        .unwrap()
        .seal()
        .unwrap()
        .into_callable();
        let handle = callable.handle();

        assert_eq!(unsafe { handle.invoke(&view) }, DIRECT_STATUS_OK);
        assert_eq!(
            unsafe { callable.invoke_unchecked(&view) },
            DIRECT_STATUS_OK
        );

        let (checked_status, checked_allocations) =
            crate::allocation_probe::count_allocations(|| {
                let mut status = DIRECT_STATUS_OK;
                for _ in 0..128 {
                    status |= unsafe { handle.invoke(&view) };
                }
                status
            });
        let (unchecked_status, unchecked_allocations) =
            crate::allocation_probe::count_allocations(|| {
                let mut status = DIRECT_STATUS_OK;
                for _ in 0..128 {
                    status |= unsafe { callable.invoke_unchecked(&view) };
                }
                status
            });

        assert_eq!(checked_status, DIRECT_STATUS_OK);
        assert_eq!(unchecked_status, DIRECT_STATUS_OK);
        assert_eq!(checked_allocations, 0);
        assert_eq!(unchecked_allocations, 0);
    }

    #[test]
    fn ordered_cross_row_aliases_preserve_full_row_lifetimes_and_odd_tails() {
        const PLANE_LEN: usize = 9;
        const POINT_START: usize = 1;
        const POINT_COUNT: usize = 6;
        let make_plane = |seed: usize| {
            (0..PLANE_LEN)
                .map(|point| (seed * 16 + point + 1) as f64 / 32.0)
                .collect::<Vec<_>>()
        };
        let mut row0_x_re = make_plane(0);
        let mut row0_x_im = make_plane(1);
        let mut row0_y_re = make_plane(2);
        let mut row0_y_im = make_plane(3);
        let mut shared_re = vec![0.0; PLANE_LEN + 1];
        let mut shared_im = vec![0.0; PLANE_LEN + 1];
        let mut unit_re = vec![1.0; PLANE_LEN];
        let mut unit_im = vec![0.0; PLANE_LEN];
        let mut final_re = vec![-17.0; PLANE_LEN];
        let mut final_im = vec![19.0; PLANE_LEN];
        let mut row2_x_re = make_plane(4);
        let mut row2_x_im = make_plane(5);
        let mut row2_y_re = make_plane(6);
        let mut row2_y_im = make_plane(7);
        let mut row3_x_re = make_plane(8);
        let mut row3_x_im = make_plane(9);
        let mut row3_y_re = make_plane(10);
        let mut row3_y_im = make_plane(11);

        let mut expected_row0_x_re = row0_x_re.clone();
        let mut expected_row0_x_im = row0_x_im.clone();
        let mut expected_shared_re = shared_re.clone();
        let mut expected_shared_im = shared_im.clone();
        let mut expected_final_re = final_re.clone();
        let mut expected_final_im = final_im.clone();
        for point in POINT_START..POINT_START + POINT_COUNT {
            let product_re =
                row0_x_re[point] * row0_y_re[point] - row0_x_im[point] * row0_y_im[point];
            let product_im =
                row0_x_re[point] * row0_y_im[point] + row0_x_im[point] * row0_y_re[point];
            expected_shared_re[point] += product_re;
            expected_shared_im[point] += product_im;
        }
        // Row 1 consumes row 0's completed output through descriptors shifted
        // by one element. Segment-first dispatch would read stale future
        // elements here.
        expected_final_re[POINT_START..POINT_START + POINT_COUNT]
            .copy_from_slice(&expected_shared_re[POINT_START + 1..POINT_START + POINT_COUNT + 1]);
        expected_final_im[POINT_START..POINT_START + POINT_COUNT]
            .copy_from_slice(&expected_shared_im[POINT_START + 1..POINT_START + POINT_COUNT + 1]);
        for point in POINT_START..POINT_START + POINT_COUNT {
            expected_row0_x_re[point] =
                row2_x_re[point] * row2_y_re[point] - row2_x_im[point] * row2_y_im[point];
            expected_row0_x_im[point] =
                row2_x_re[point] * row2_y_im[point] + row2_x_im[point] * row2_y_re[point];
        }
        for point in POINT_START..POINT_START + POINT_COUNT {
            expected_shared_re[point] +=
                row3_x_re[point] * row3_y_re[point] - row3_x_im[point] * row3_y_im[point];
            expected_shared_im[point] +=
                row3_x_re[point] * row3_y_im[point] + row3_x_im[point] * row3_y_re[point];
        }

        let planes = [
            unsafe { DirectPlane::from_raw_parts(row0_x_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row0_x_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row0_y_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row0_y_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(shared_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(shared_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(shared_re.as_mut_ptr().add(1), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(shared_im.as_mut_ptr().add(1), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(unit_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(unit_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(final_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(final_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row2_x_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row2_x_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row2_y_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row2_y_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row3_x_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row3_x_im.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row3_y_re.as_mut_ptr(), PLANE_LEN) },
            unsafe { DirectPlane::from_raw_parts(row3_y_im.as_mut_ptr(), PLANE_LEN) },
        ];
        let invocations = [
            InvocationRow {
                input_re_0: 0,
                input_im_0: 1,
                input_re_1: 2,
                input_im_1: 3,
                input_re_2: 0,
                input_im_2: 0,
                input_re_3: 0,
                input_im_3: 0,
                input_re_4: 0,
                input_im_4: 0,
                input_re_5: 0,
                input_im_5: 0,
                attachment_start: 0,
                attachment_count: 1,
            },
            InvocationRow {
                input_re_0: 6,
                input_im_0: 7,
                input_re_1: 8,
                input_im_1: 9,
                input_re_2: 0,
                input_im_2: 0,
                input_re_3: 0,
                input_im_3: 0,
                input_re_4: 0,
                input_im_4: 0,
                input_re_5: 0,
                input_im_5: 0,
                attachment_start: 1,
                attachment_count: 1,
            },
            InvocationRow {
                input_re_0: 12,
                input_im_0: 13,
                input_re_1: 14,
                input_im_1: 15,
                input_re_2: 0,
                input_im_2: 0,
                input_re_3: 0,
                input_im_3: 0,
                input_re_4: 0,
                input_im_4: 0,
                input_re_5: 0,
                input_im_5: 0,
                attachment_start: 2,
                attachment_count: 1,
            },
            InvocationRow {
                input_re_0: 16,
                input_im_0: 17,
                input_re_1: 18,
                input_im_1: 19,
                input_re_2: 0,
                input_im_2: 0,
                input_re_3: 0,
                input_im_3: 0,
                input_re_4: 0,
                input_im_4: 0,
                input_re_5: 0,
                input_im_5: 0,
                // Reuse row 0's Add attachment after rows 1 and 2.
                attachment_start: 0,
                attachment_count: 1,
            },
        ];
        let attachments = [
            AttachmentRow {
                destination_re: 4,
                destination_im: 5,
                scale: 0,
                operation: DirectTableDestinationOperation::Accumulate as u32,
            },
            AttachmentRow {
                destination_re: 10,
                destination_im: 11,
                scale: 0,
                operation: DirectTableDestinationOperation::Overwrite as u32,
            },
            AttachmentRow {
                destination_re: 0,
                destination_im: 1,
                scale: 0,
                operation: DirectTableDestinationOperation::Overwrite as u32,
            },
        ];
        let scale_re = [1.0];
        let scale_im = [0.0];
        let view = DirectTableCallViewV1 {
            invocations: invocations.as_ptr().cast(),
            invocation_count: invocations.len() as u32,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: attachments.as_ptr().cast(),
            attachment_count: attachments.len() as u32,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: 1,
            point_start: POINT_START as u32,
            point_count: POINT_COUNT as u32,
        };
        let callable = DirectTableApplication::new(source_application(), table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        unsafe { callable.metadata().validate_call_view(&view) }.unwrap();
        assert_eq!(
            unsafe { callable.context.dispatch_table_unchecked(&view).unwrap() },
            expected_generated_calls(invocations.len(), POINT_START, POINT_COUNT,)
        );
        for (actual, expected) in [
            (&row0_x_re, &expected_row0_x_re),
            (&row0_x_im, &expected_row0_x_im),
            (&shared_re, &expected_shared_re),
            (&shared_im, &expected_shared_im),
            (&final_re, &expected_final_re),
            (&final_im, &expected_final_im),
        ] {
            for (&actual, &expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn checked_calls_reject_same_row_aliases_but_allow_ordered_cross_row_reuse() {
        let callable = DirectTableApplication::new(source_application(), table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let scale_re = [1.0];
        let scale_im = [0.0];
        let invocation = InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 0,
            input_im_2: 0,
            input_re_3: 0,
            input_im_3: 0,
            input_re_4: 0,
            input_im_4: 0,
            input_re_5: 0,
            input_im_5: 0,
            attachment_start: 0,
            attachment_count: 1,
        };

        let mut exact_storage = [[0.0; 2]; 6];
        let exact_planes = exact_storage
            .iter_mut()
            .map(|plane| unsafe { DirectPlane::from_raw_parts(plane.as_mut_ptr(), plane.len()) })
            .collect::<Vec<_>>();
        let exact_attachment = AttachmentRow {
            destination_re: 0,
            destination_im: 5,
            scale: 0,
            operation: 0,
        };
        let mut view = DirectTableCallViewV1 {
            invocations: ptr::from_ref(&invocation).cast(),
            invocation_count: 1,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: ptr::from_ref(&exact_attachment).cast(),
            attachment_count: 1,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: exact_planes.as_ptr(),
            plane_count: exact_planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: 1,
            point_start: 0,
            point_count: 2,
        };
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );

        // The active two-point input and destination windows do not touch, but
        // their complete six-element descriptors overlap at elements 4 and 5.
        let mut partial_storage = [0.0; 10];
        let mut partial_other = [[0.0; 2]; 4];
        let partial_planes = [
            unsafe { DirectPlane::from_raw_parts(partial_storage.as_mut_ptr(), 6) },
            unsafe {
                DirectPlane::from_raw_parts(partial_other[0].as_mut_ptr(), partial_other[0].len())
            },
            unsafe {
                DirectPlane::from_raw_parts(partial_other[1].as_mut_ptr(), partial_other[1].len())
            },
            unsafe {
                DirectPlane::from_raw_parts(partial_other[2].as_mut_ptr(), partial_other[2].len())
            },
            unsafe { DirectPlane::from_raw_parts(partial_storage.as_mut_ptr().add(4), 6) },
            unsafe {
                DirectPlane::from_raw_parts(partial_other[3].as_mut_ptr(), partial_other[3].len())
            },
        ];
        let normal_attachment = AttachmentRow {
            destination_re: 4,
            destination_im: 5,
            scale: 0,
            operation: 0,
        };
        view.attachments = ptr::from_ref(&normal_attachment).cast();
        view.planes = partial_planes.as_ptr();
        view.plane_count = partial_planes.len() as u32;
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );

        let mut distinct_inputs = [[0.0; 2]; 4];
        let mut overlapping_destinations = [0.0; 10];
        let destination_planes = [
            unsafe {
                DirectPlane::from_raw_parts(
                    distinct_inputs[0].as_mut_ptr(),
                    distinct_inputs[0].len(),
                )
            },
            unsafe {
                DirectPlane::from_raw_parts(
                    distinct_inputs[1].as_mut_ptr(),
                    distinct_inputs[1].len(),
                )
            },
            unsafe {
                DirectPlane::from_raw_parts(
                    distinct_inputs[2].as_mut_ptr(),
                    distinct_inputs[2].len(),
                )
            },
            unsafe {
                DirectPlane::from_raw_parts(
                    distinct_inputs[3].as_mut_ptr(),
                    distinct_inputs[3].len(),
                )
            },
            unsafe { DirectPlane::from_raw_parts(overlapping_destinations.as_mut_ptr(), 6) },
            unsafe { DirectPlane::from_raw_parts(overlapping_destinations.as_mut_ptr().add(4), 6) },
        ];
        view.planes = destination_planes.as_ptr();
        view.plane_count = destination_planes.len() as u32;
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );

        let mut distinct_storage = [[0.0; 2]; 6];
        let distinct_planes = distinct_storage
            .iter_mut()
            .map(|plane| unsafe { DirectPlane::from_raw_parts(plane.as_mut_ptr(), plane.len()) })
            .collect::<Vec<_>>();
        let duplicate_attachments = [normal_attachment, normal_attachment];
        let duplicate_invocation = InvocationRow {
            attachment_count: 2,
            ..invocation
        };
        view.invocations = ptr::from_ref(&duplicate_invocation).cast();
        view.attachments = duplicate_attachments.as_ptr().cast();
        view.attachment_count = duplicate_attachments.len() as u32;
        view.planes = distinct_planes.as_ptr();
        view.plane_count = distinct_planes.len() as u32;
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );

        let repeated_invocations = [invocation, invocation];
        view.invocations = repeated_invocations.as_ptr().cast();
        view.invocation_count = repeated_invocations.len() as u32;
        view.attachments = ptr::from_ref(&normal_attachment).cast();
        view.attachment_count = 1;
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);
    }

    #[test]
    fn checked_calls_reject_destination_overlap_with_global_scalar_scale_and_table_buffers() {
        let callable = DirectTableApplication::new(source_application(), table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let invocation = InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 0,
            input_im_2: 0,
            input_re_3: 0,
            input_im_3: 0,
            input_re_4: 0,
            input_im_4: 0,
            input_re_5: 0,
            input_im_5: 0,
            attachment_start: 0,
            attachment_count: 1,
        };
        let attachment = AttachmentRow {
            destination_re: 4,
            destination_im: 5,
            scale: 0,
            operation: 0,
        };
        let mut inputs = [[1.0; 2]; 4];
        let mut destination_im = [0.0; 2];
        let mut scale_re = [1.0, 0.0];
        let scale_im = [0.0, 0.0];
        let mut planes = [
            unsafe { DirectPlane::from_raw_parts(inputs[0].as_mut_ptr(), 2) },
            unsafe { DirectPlane::from_raw_parts(inputs[1].as_mut_ptr(), 2) },
            unsafe { DirectPlane::from_raw_parts(inputs[2].as_mut_ptr(), 2) },
            unsafe { DirectPlane::from_raw_parts(inputs[3].as_mut_ptr(), 2) },
            unsafe { DirectPlane::from_raw_parts(scale_re.as_mut_ptr(), 2) },
            unsafe { DirectPlane::from_raw_parts(destination_im.as_mut_ptr(), 2) },
        ];
        let mut view = DirectTableCallViewV1 {
            invocations: ptr::from_ref(&invocation).cast(),
            invocation_count: 1,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: ptr::from_ref(&attachment).cast(),
            attachment_count: 1,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale_re.as_ptr(),
            scale_im: scale_im.as_ptr(),
            scale_count: 1,
            point_start: 0,
            point_count: 2,
        };
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );

        #[repr(C, align(8))]
        struct AlignedAttachment(AttachmentRow);
        let mut table_storage = AlignedAttachment(attachment);
        planes[4] = unsafe {
            DirectPlane::from_raw_parts(
                ptr::from_mut(&mut table_storage).cast::<f64>(),
                mem::size_of::<AttachmentRow>() / mem::size_of::<f64>(),
            )
        };
        view.attachments = ptr::from_ref(&table_storage.0).cast();
        view.planes = planes.as_ptr();
        let independent_scale_re = [1.0];
        let independent_scale_im = [0.0];
        view.scale_re = independent_scale_re.as_ptr();
        view.scale_im = independent_scale_im.as_ptr();
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );

        let scalar_callable = DirectTableApplication::new(
            source_application_with_scalars(),
            table_metadata_with_scalars(2),
        )
        .unwrap()
        .seal()
        .unwrap()
        .into_callable();
        let mut scalar_destination = [0.25; 2];
        planes[4] = unsafe {
            DirectPlane::from_raw_parts(scalar_destination.as_mut_ptr(), scalar_destination.len())
        };
        view.attachments = ptr::from_ref(&attachment).cast();
        view.planes = planes.as_ptr();
        let scalar_im = -0.5;
        let scalars = [
            unsafe { DirectScalar::from_raw(scalar_destination.as_ptr()) },
            unsafe { DirectScalar::from_raw(ptr::from_ref(&scalar_im)) },
        ];
        view.scalars = scalars.as_ptr();
        view.scalar_count = scalars.len() as u32;
        assert_eq!(
            unsafe { scalar_callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn large_spill_frame_does_not_clobber_attachment_cursor() {
        run_case_with_stack(2, 2, 0, 7, 4, Some(4097));
    }

    #[test]
    fn branchy_source_is_rejected_until_simd_fallback_exists() {
        let mut source = source_application();
        Rc::make_mut(&mut source.bytecode.mir)
            .code
            .push(&Instruction::Branch {
                label: "unreachable".into(),
            });
        assert!(DirectTableApplication::new(source, table_metadata()).is_err());
    }

    #[test]
    fn crafted_call_instruction_is_rejected_before_codegen() {
        let mut source = source_application();
        Rc::make_mut(&mut source.bytecode.mir)
            .code
            .push(&Instruction::Call {
                label: "missing".into(),
                num_args: 0,
            });
        assert!(DirectTableApplication::new(source, table_metadata()).is_err());
    }

    #[test]
    fn generated_frame_respects_configured_stack_limit() {
        let mut source = source_application();
        source.prog.builder.count_stack = Some(source.config.stack_limit() / 16 + 1);
        let application = DirectTableApplication::new(source, table_metadata()).unwrap();
        assert!(application.seal().is_err());
    }

    #[test]
    fn generated_functions_inline_one_mir_body_and_never_call_old_kernel() {
        let callable = DirectTableApplication::new(source_application(), table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        for shape in [callable.scalar_code_shape(), callable.simd_code_shape()] {
            println!("direct-table-code-shape: {shape:?}");
            assert_eq!(shape.inline_kernel_copies, 1);
            assert!(shape.inline_kernel_body_bytes > 0);
            assert_eq!(shape.branch_and_link_instructions, 0);
            assert_eq!(shape.packet_materializations, 0);
            assert_eq!(shape.gather_materializations, 0);
            assert_eq!(shape.scatter_materializations, 0);
            assert!(shape.inline_kernel_body_offset < shape.executable_instruction_bytes);
            assert!(
                shape.inline_kernel_body_offset + shape.inline_kernel_body_bytes
                    <= shape.executable_instruction_bytes
            );
        }
    }

    #[test]
    fn call_view_layout_is_stable_on_supported_64_bit_hosts() {
        assert_eq!(mem::align_of::<DirectTableCallViewV1>(), 8);
        assert_eq!(mem::size_of::<DirectTableCallViewV1>(), 88);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, invocations), 0);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, attachments), 16);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, planes), 32);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, scalars), 48);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, scale_re), 56);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, scale_im), 64);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, point_start), 76);
        assert_eq!(mem::offset_of!(DirectTableCallViewV1, point_count), 80);
    }

    #[test]
    fn malformed_parameter_bindings_fail_closed() {
        assert!(DirectTableApplicationMetadata::new_with_parameter_bindings(
            invocation_layout(),
            attachment_layout(),
            vec![DirectTableParameterBinding::Plane(4)],
            0,
        )
        .is_err());
        assert!(DirectTableApplicationMetadata::new_with_parameter_bindings(
            invocation_layout(),
            attachment_layout(),
            vec![DirectTableParameterBinding::Scalar(0)],
            0,
        )
        .is_err());
        assert!(DirectTableApplicationMetadata::new_with_parameter_bindings(
            invocation_layout(),
            attachment_layout(),
            vec![DirectTableParameterBinding::Scalar(1)],
            2,
        )
        .is_err());

        let too_few_bindings = DirectTableApplicationMetadata::new_with_parameter_bindings(
            invocation_layout(),
            attachment_layout(),
            vec![
                DirectTableParameterBinding::Plane(0),
                DirectTableParameterBinding::Plane(1),
                DirectTableParameterBinding::Plane(2),
            ],
            0,
        )
        .unwrap();
        assert!(
            DirectTableApplication::new(prepared_multi_output_application(), too_few_bindings)
                .is_err()
        );

        let uncovered_invocation_plane =
            DirectTableApplicationMetadata::new_with_parameter_bindings(
                invocation_layout(),
                attachment_layout(),
                vec![
                    DirectTableParameterBinding::Plane(0),
                    DirectTableParameterBinding::Plane(1),
                    DirectTableParameterBinding::Plane(2),
                    DirectTableParameterBinding::Plane(2),
                ],
                0,
            )
            .unwrap();
        assert!(DirectTableApplication::new(
            prepared_multi_output_application(),
            uncovered_invocation_plane,
        )
        .is_err());
    }

    #[test]
    fn metadata_and_call_view_fail_closed() {
        assert!(DirectTableInvocationLayout::new(6, vec![0, 4], 0, 4).is_err());
        assert!(DirectTableAttachmentLayout::new(16, 0, 4, 8, 16).is_err());

        let callable = DirectTableApplication::new(source_application(), table_metadata())
            .unwrap()
            .seal()
            .unwrap()
            .into_callable();
        let scale = [1.0];
        let mut plane = vec![0.0; 4];
        let descriptor = unsafe { DirectPlane::from_raw_parts(plane.as_mut_ptr(), plane.len()) };
        let mut view = DirectTableCallViewV1 {
            invocations: ptr::null(),
            invocation_count: 1,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: ptr::null(),
            attachment_count: 1,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: ptr::from_ref(&descriptor),
            plane_count: 1,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale.as_ptr(),
            scale_im: scale.as_ptr(),
            scale_count: 1,
            point_start: 0,
            point_count: 1,
        };
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.point_count = 0;
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );

        let invocation = InvocationRow {
            input_re_0: 0,
            input_im_0: 1,
            input_re_1: 2,
            input_im_1: 3,
            input_re_2: 0,
            input_im_2: 0,
            input_re_3: 0,
            input_im_3: 0,
            input_re_4: 0,
            input_im_4: 0,
            input_re_5: 0,
            input_im_5: 0,
            attachment_start: 0,
            attachment_count: 1,
        };
        let attachment = AttachmentRow {
            destination_re: 4,
            destination_im: 5,
            scale: 0,
            operation: 0,
        };
        let mut values = [[1.0; 4]; 6];
        let planes = values
            .iter_mut()
            .map(|plane| unsafe { DirectPlane::from_raw_parts(plane.as_mut_ptr(), plane.len()) })
            .collect::<Vec<_>>();
        view = DirectTableCallViewV1 {
            invocations: ptr::from_ref(&invocation).cast(),
            invocation_count: 1,
            invocation_stride: mem::size_of::<InvocationRow>() as u32,
            attachments: ptr::from_ref(&attachment).cast(),
            attachment_count: 1,
            attachment_stride: mem::size_of::<AttachmentRow>() as u32,
            planes: planes.as_ptr(),
            plane_count: planes.len() as u32,
            scalar_count: 0,
            scalars: ptr::null(),
            scale_re: scale.as_ptr(),
            scale_im: scale.as_ptr(),
            scale_count: 1,
            point_start: 0,
            point_count: 1,
        };
        assert_eq!(unsafe { callable.handle().invoke(&view) }, DIRECT_STATUS_OK);

        view.invocation_stride += 4;
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.invocation_stride = mem::size_of::<InvocationRow>() as u32;

        let bad_attachment_range = InvocationRow {
            attachment_count: 2,
            ..invocation
        };
        view.invocations = ptr::from_ref(&bad_attachment_range).cast();
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.invocations = ptr::from_ref(&invocation).cast();

        let bad_input_plane = InvocationRow {
            input_re_0: planes.len() as u32,
            ..invocation
        };
        view.invocations = ptr::from_ref(&bad_input_plane).cast();
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.invocations = ptr::from_ref(&invocation).cast();

        let bad_destination = AttachmentRow {
            destination_re: planes.len() as u32,
            ..attachment
        };
        view.attachments = ptr::from_ref(&bad_destination).cast();
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.attachments = ptr::from_ref(&attachment).cast();

        let bad_scale = AttachmentRow {
            scale: 1,
            ..attachment
        };
        view.attachments = ptr::from_ref(&bad_scale).cast();
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.attachments = ptr::from_ref(&attachment).cast();

        let bad_operation = AttachmentRow {
            operation: 2,
            ..attachment
        };
        view.attachments = ptr::from_ref(&bad_operation).cast();
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.attachments = ptr::from_ref(&attachment).cast();

        view.point_start = u32::MAX;
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.point_start = 0;
        view.point_count = values[0].len() as u32 + 1;
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
        view.point_count = 1;

        view.planes = unsafe { planes.as_ptr().cast::<u8>().add(1).cast() };
        assert_eq!(
            unsafe { callable.handle().invoke(&view) },
            DIRECT_STATUS_INVALID_ARGUMENT
        );
    }
}
