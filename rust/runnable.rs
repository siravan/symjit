use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::io::{Read, Write};
use wide::{f64x2, f64x4, f64x8};

use super::amd::{
    AmdComplexGenerator, AmdSSEGenerator, AmdScalarGenerator, AmdVectorF64x4Generator,
    AmdVectorF64x8Generator,
};
use super::applet::Applet;
use super::arm::{ArmComplexGenerator, ArmGenerator, ArmSimdGenerator};
use super::code::Func;
use super::complexify::Complexifier;
use super::config::Config;
#[cfg(feature = "obj")]
use super::generator::{kernel_symbol, user_symbol};
use super::generator::{Generator, GeneratorType};
#[cfg(feature = "obj")]
use super::linker::{
    header, lower_for_object, Arch, Format, HeaderInfo, ObjectBuilder, Relocation, Target,
};
use super::machine::MachineCode;
use super::matrix::{combine_matrixes, Matrix};
use super::mir::{CompiledMir, Mir};
use super::model::Program;
use super::riscv64::RiscV;
use super::symbol::Loc;
use super::utils::*;
#[cfg(feature = "wasm")]
use super::wasm::WasmGenerator;

use rayon::prelude::*;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum CompilerType {
    /// generates bytecode (interpreter).
    ByteCode,
    /// generates code for the detected CPU (default)
    Native,
    /// generates x86-64 (AMD64) code.
    Amd,
    /// generates AVX code for x86-64 architecture.
    AmdAVX,
    /// generates SSE2 code for x86-64 architecture.
    AmdSSE,
    /// generates aarch64 (ARM64) code.
    Arm,
    /// generates riscv64 (RISC V) code.
    RiscV,
    /// generates a WebAssembly module (not executed in-process;
    /// calls run on the bytecode interpreter). Needs the `wasm` feature.
    Wasm,
    /// debug mode, generates both bytecode and native codes
    /// and compares the outputs.
    Debug,
}

#[repr(C)] // to ensure binary compatibility with Applet
pub struct Application {
    // Applet compatibility
    // Important! The order of these fields is critical and should be
    // the same as the order of Applet fields.
    pub compiled: Option<MachineCode<f64>>,
    pub compiled_simd: Option<MachineCode<f64>>,
    pub use_simd: bool,
    pub use_threads: bool,
    pub config: Config,
    pub params: Vec<f64>,
    pub first_state: usize,
    pub first_obs: usize,
    pub count_states: usize,
    pub count_obs: usize,
    pub count_params: usize,
    // Non-Applet fields
    pub prog: Program,
    pub compiled_fast: Option<MachineCode<f64>>,
    pub bytecode: CompiledMir,
    pub can_fast: bool,
    pub reals: HashSet<Loc>,
    pub original: Option<Mir>,
    pub first_param: usize,
    pub first_diff: usize,
    pub count_diffs: usize,
    pub wasm: Option<Vec<u8>>,
}

impl Application {
    pub fn new(mut prog: Program, reals: HashSet<Loc>) -> Result<Application> {
        /*
         * Stop-gap measure. A better solution would be to add `times_real`,
         * `divide_real`, and `load_param_real` to generators.
         */
        if !reals.is_empty() {
            prog.builder.config.set_fast_complex(false);
        }

        let mut mir = Mir::new(prog.config().clone());
        prog.builder.compile_mir(&mut mir)?;
        prog.builder.optimize_mir(&mut mir)?;
        Self::with_mir(prog, reals, mir)
    }

    pub fn with_mir(mut prog: Program, reals: HashSet<Loc>, mut mir: Mir) -> Result<Application> {
        #[cfg(not(feature = "wasm"))]
        if prog.config().is_wasm() {
            return Err(anyhow!(
                "symjit was built without the `wasm` feature (`ty = \"wasm\"`); rebuild with `--features wasm`"
            ));
        }

        let first_state = 0;
        let first_param = 0;
        let first_obs = first_state + prog.count_states;
        let first_diff = first_obs + prog.count_obs;

        let count_states = prog.count_states;
        let count_params = prog.count_params;
        let count_obs = prog.count_obs;
        let count_diffs = prog.count_diffs;

        let params = vec![0.0; count_params + 1];

        let config = prog.config().clone();
        let mut original: Option<Mir> = None;
        let compiled: Option<MachineCode<f64>>;

        if config.is_complex() {
            original = Some(mir.clone());
            let complexified = Complexifier::new(&reals, config.clone()).complexify(&mir)?;
            // the constants of the math library's funclets (`libm::append_funclets`)
            if complexified.consts.len() > prog.builder.consts.len() {
                prog.builder.consts = complexified.consts.clone();
            }

            if config.fast_complex() {
                /*
                let mut c = config.clone();
                c.set_simd(false);
                let n = c.count_scratch() as usize;
                super::allocator::GreedyAllocator::new(c, n).optimize(&mut mir)?;
                */
                compiled = Self::compile_ty(&config, &mir, &mut prog)?;
            } else {
                compiled = Self::compile_ty(&config, &complexified, &mut prog)?;
            }

            mir = complexified;
        } else {
            compiled = Self::compile_ty(&config, &mir, &mut prog)?;
        }

        #[cfg(feature = "wasm")]
        let wasm = if config.is_wasm() {
            Some(Self::compile_wasm(&mir, &mut prog)?)
        } else {
            None
        };
        #[cfg(not(feature = "wasm"))]
        let wasm = None;

        let use_simd = config.use_simd() && prog.count_loops == 0;
        let use_threads = config.use_threads();

        let can_fast = config.may_fast()
            && (count_states <= 4 || (count_states <= 8 && !config.is_window()))
            && count_params == 0
            && count_obs == 1
            && count_diffs == 0;

        // bytecode takes the ownership of mir
        let bytecode = Self::compile_bytecode(mir, &mut prog)?;

        Ok(Application {
            prog,
            compiled,
            compiled_simd: None,
            compiled_fast: None,
            bytecode,
            params,
            use_simd,
            use_threads,
            can_fast,
            first_state,
            first_param,
            first_obs,
            first_diff,
            count_states,
            count_params,
            count_obs,
            count_diffs,
            config,
            reals,
            original,
            wasm,
        })
    }

    fn compile_ty(
        config: &Config,
        mir: &Mir,
        prog: &mut Program,
    ) -> Result<Option<MachineCode<f64>>> {
        let compiled = match config.compiler_type() {
            CompilerType::AmdAVX => Some(Self::compile_avx(mir, prog)?),
            CompilerType::AmdSSE => Some(Self::compile_sse(mir, prog)?),
            CompilerType::Arm => Some(Self::compile_arm(mir, prog)?),
            CompilerType::RiscV => Some(Self::compile_riscv(mir, prog)?),
            CompilerType::ByteCode | CompilerType::Wasm => None,
            CompilerType::Debug => {
                println!("`ty = debug` is deprecated");
                None
            }
            _ => return Err(anyhow!("unrecognized `ty`")),
        };

        Ok(compiled)
    }

    pub fn seal(self) -> Result<Applet> {
        Applet::new(self)
    }

    pub fn as_applet(&self) -> &Applet {
        unsafe { std::mem::transmute(self) }
    }

    /********************* compile_* functions *************************/

    fn compile<G: Generator>(
        mir: &Mir,
        prog: &mut Program,
        mut generator: G,
        size: usize,
        arch: &str,
        lanes: usize,
    ) -> Result<MachineCode<f64>> {
        let stack_limit = prog.config().stack_limit();
        if prog.builder.stack_size() >= stack_limit * std::mem::size_of::<f64>() {
            return Err(anyhow!(
                "Cannot compile due to reaching the stack limit (= {}). Consider increasing the increasing the limit using `Config.set_stack_limit`.",
                stack_limit
            ));
        }

        let mem: Vec<f64> = vec![0.0; size];
        prog.builder.compile_from_mir(
            mir,
            &mut generator,
            prog.count_states,
            prog.count_obs,
            prog.count_params,
        )?;

        Ok(MachineCode::new(
            arch,
            generator.bytes(),
            mem,
            false,
            lanes,
            prog.config().huge(),
        ))
    }

    fn compile_fast<G: Generator>(
        mir: &Mir,
        prog: &mut Program,
        mut generator: G,
        idx_ret: u32,
        arch: &str,
    ) -> Result<MachineCode<f64>> {
        let mem: Vec<f64> = Vec::new();
        prog.builder.compile_fast_from_mir(
            mir,
            &mut generator,
            prog.count_states,
            prog.count_obs,
            idx_ret as i32,
        )?;

        Ok(MachineCode::new(
            arch,
            generator.bytes(),
            mem,
            true,
            1,
            prog.config().huge(),
        ))
    }

    #[cfg(feature = "wasm")]
    fn compile_wasm(mir: &Mir, prog: &mut Program) -> Result<Vec<u8>> {
        let mut generator = WasmGenerator::new(prog.config().clone());
        generator.set_mem_size(prog.mem_size());
        if prog.count_params == 0 && prog.count_obs == 1 && prog.count_diffs == 0 {
            generator.export_fast(prog.mem_size());
        }
        prog.builder.compile_from_mir(
            mir,
            &mut generator,
            prog.count_states,
            prog.count_obs,
            prog.count_params,
        )?;
        generator.check()?;
        Ok(generator.bytes())
    }

    fn compile_bytecode(mir: Mir, prog: &mut Program) -> Result<CompiledMir> {
        let mem: Vec<f64> = vec![0.0; prog.mem_size()];
        let stack: Vec<f64> = vec![0.0; prog.builder.stack_size()];

        Ok(CompiledMir::new(mir, mem, stack))
    }

    fn compile_sse(mir: &Mir, prog: &mut Program) -> Result<MachineCode<f64>> {
        Self::compile::<AmdSSEGenerator>(
            mir,
            prog,
            AmdSSEGenerator::new(prog.config().clone()),
            prog.mem_size(),
            "x86_64",
            1,
        )
    }

    fn compile_avx(mir: &Mir, prog: &mut Program) -> Result<MachineCode<f64>> {
        if prog.config().is_complex() && prog.config().fast_complex() {
            Self::compile::<AmdComplexGenerator>(
                mir,
                prog,
                AmdComplexGenerator::new(prog.config().clone()),
                prog.mem_size(),
                "x86_64",
                1,
            )
        } else {
            Self::compile::<AmdScalarGenerator>(
                mir,
                prog,
                AmdScalarGenerator::new(prog.config().clone()),
                prog.mem_size(),
                "x86_64",
                1,
            )
        }
    }

    fn compile_avx_simd(mir: &Mir, prog: &mut Program) -> Result<MachineCode<f64>> {
        let stack_size = prog.builder.stack_size();
        let stack_limit = prog.config().stack_limit();
        let is_complex = prog.config().is_complex();

        let p = if stack_size * std::mem::size_of::<f64x8>() <= stack_limit
            && prog.config().use_simd512()
        {
            GeneratorType::AmdVectorF64x8(is_complex)
        } else if stack_size * std::mem::size_of::<f64x4>() <= stack_limit
            && prog.config().use_simd()
        {
            GeneratorType::AmdVectorF64x4(is_complex)
        } else {
            GeneratorType::AmdScalar(is_complex)
        };

        match p {
            GeneratorType::AmdVectorF64x8(_) => Self::compile::<AmdVectorF64x8Generator>(
                mir,
                prog,
                AmdVectorF64x8Generator::new(prog.config().clone()),
                prog.mem_size() * 8,
                "x86_64",
                8,
            ),
            GeneratorType::AmdVectorF64x4(_) => Self::compile::<AmdVectorF64x4Generator>(
                mir,
                prog,
                AmdVectorF64x4Generator::new(prog.config().clone()),
                prog.mem_size() * 4,
                "x86_64",
                4,
            ),
            _ => Err(anyhow!(
                "Cannot use SIMD (f64x4) due to the stack limit (= {}).",
                stack_limit
            )),
        }
    }

    fn compile_arm(mir: &Mir, prog: &mut Program) -> Result<MachineCode<f64>> {
        if prog.config().is_complex() && prog.config().fast_complex() {
            Self::compile::<ArmComplexGenerator>(
                mir,
                prog,
                ArmComplexGenerator::new(prog.config().clone()),
                prog.mem_size(),
                "aarch64",
                1,
            )
        } else {
            Self::compile::<ArmGenerator>(
                mir,
                prog,
                ArmGenerator::new(prog.config().clone()),
                prog.mem_size(),
                "aarch64",
                1,
            )
        }
    }

    fn compile_arm_simd(mir: &Mir, prog: &mut Program) -> Result<MachineCode<f64>> {
        let stack_size = prog.builder.stack_size();
        let stack_limit = prog.config().stack_limit();

        if prog.config().use_simd() {
            if stack_size * std::mem::size_of::<f64x2>() <= stack_limit {
                return Self::compile::<ArmSimdGenerator>(
                    mir,
                    prog,
                    ArmSimdGenerator::new(prog.config().clone()),
                    prog.mem_size() * 2,
                    "aarch64",
                    2,
                );
            } else {
                eprintln!(
                    "SIMD (f64x2) request downgraded to scalar f64 due to the stack limit (= {}).",
                    stack_limit
                )
            }
        }

        Err(anyhow!(
            "Cannot use SIMD (f64x2) due to the stack limit (= {}).",
            stack_limit
        ))
    }

    fn compile_riscv(mir: &Mir, prog: &mut Program) -> Result<MachineCode<f64>> {
        Self::compile::<RiscV>(
            mir,
            prog,
            RiscV::new(prog.config().clone()),
            prog.mem_size(),
            "riscv64",
            1,
        )
    }

    fn compile_amd_fast(mir: &Mir, prog: &mut Program, idx_ret: u32) -> Result<MachineCode<f64>> {
        if prog.config().has_avx() {
            Self::compile_fast(
                mir,
                prog,
                AmdScalarGenerator::new(prog.config().clone()),
                idx_ret,
                "x86_64",
            )
        } else {
            Self::compile_fast(
                mir,
                prog,
                AmdSSEGenerator::new(prog.config().clone()),
                idx_ret,
                "x86_64",
            )
        }
    }

    fn compile_arm_fast(mir: &Mir, prog: &mut Program, idx_ret: u32) -> Result<MachineCode<f64>> {
        Self::compile_fast(
            mir,
            prog,
            ArmGenerator::new(prog.config().clone()),
            idx_ret,
            "aarch64",
        )
    }

    fn compile_riscv_fast(mir: &Mir, prog: &mut Program, idx_ret: u32) -> Result<MachineCode<f64>> {
        Self::compile_fast(
            mir,
            prog,
            RiscV::new(prog.config().clone()),
            idx_ret,
            "riscv64",
        )
    }

    /**********************************************************/

    #[inline]
    pub fn exec(&mut self) {
        if let Some(compiled) = &mut self.compiled {
            compiled.exec(&self.params[..])
        } else {
            self.bytecode.exec(&self.params[..]);
        }
    }

    pub fn exec_callable(&mut self, xx: &[f64]) -> f64 {
        if let Some(compiled) = &mut self.compiled {
            let mem = compiled.mem_mut();
            mem[self.first_state..self.first_state + self.count_states].copy_from_slice(xx);
            compiled.exec(&self.params[..]);
            compiled.mem()[self.first_obs]
        } else {
            let mem = self.bytecode.mem_mut();
            mem[self.first_state..self.first_state + self.count_states].copy_from_slice(xx);
            self.bytecode.exec(&self.params[..]);
            self.bytecode.mem()[self.first_obs]
        }
    }

    pub fn prepare_simd(&mut self) {
        // SIMD compilation is lazy!
        // A SIMD kernel calls the SIMD kernels of the applets it calls (`_simd_<name>_`):
        // without one (e.g. an applet sealed from `Compiler::compile`, which does not
        // prepare SIMD), it stays scalar, as when its compilation fails.
        let applets_have_simd = self.config.df.as_ref().is_none_or(|df| {
            df.funcs
                .values()
                .all(|f| !matches!(f, Func::App(app) if app.simd_kernel().is_none()))
        });
        if self.compiled_simd.is_none() && self.use_simd && applets_have_simd {
            if self.config.has_avx() {
                self.compiled_simd =
                    Self::compile_avx_simd(&self.bytecode.mir, &mut self.prog).ok();
            } else if self.config.is_arm64() {
                self.compiled_simd =
                    Self::compile_arm_simd(&self.bytecode.mir, &mut self.prog).ok();
            }
        };
    }

    fn prepare_fast(&mut self) {
        // fast func compilation is lazy!
        if self.compiled_fast.is_none() && self.can_fast {
            if self.config.is_amd64() {
                self.compiled_fast = Self::compile_amd_fast(
                    &self.bytecode.mir,
                    &mut self.prog,
                    self.first_obs as u32,
                )
                .ok();
            } else if self.config.is_arm64() {
                self.compiled_fast = Self::compile_arm_fast(
                    &self.bytecode.mir,
                    &mut self.prog,
                    self.first_obs as u32,
                )
                .ok();
            } else if self.config.is_riscv64() {
                self.compiled_fast = Self::compile_riscv_fast(
                    &self.bytecode.mir,
                    &mut self.prog,
                    self.first_obs as u32,
                )
                .ok();
            }
        };
    }

    pub fn get_fast(&mut self) -> Option<CompiledFunc<f64>> {
        self.prepare_fast();
        self.compiled_fast.as_ref().map(|c| c.func())
    }

    pub fn exec_vectorized(&mut self, states: &mut Matrix, obs: &mut Matrix) {
        if let Some(compiled) = &self.compiled {
            if !compiled.support_indirect() {
                self.exec_vectorized_simple(states, obs);
                return;
            }

            self.prepare_simd();

            if let Some(simd) = &self.compiled_simd {
                self.exec_vectorized_simd(states, obs, self.use_threads, simd.count_lanes());
            } else {
                self.exec_vectorized_scalar(states, obs, self.use_threads);
            }
        } else {
            // no machine code (`ty = "bytecode"` or `"wasm"`): one column at a time on
            // the bytecode interpreter
            self.exec_vectorized_simple(states, obs);
        }
    }

    pub fn exec_vectorized_simple(&mut self, states: &Matrix, obs: &mut Matrix) {
        assert!(states.vecsize == obs.vecsize);
        let n = states.vecsize;
        let params = &self.params[..];

        if let Some(compiled) = &mut self.compiled {
            for t in 0..n {
                {
                    let mem = compiled.mem_mut();
                    for i in 0..self.count_states {
                        mem[self.first_state + i] = states.get(i, t);
                    }
                }

                compiled.exec(params);

                {
                    let mem = compiled.mem_mut();
                    for i in 0..self.count_obs {
                        obs.set(i, t, mem[self.first_obs + i]);
                    }
                }
            }
        } else {
            for t in 0..n {
                {
                    let mem = self.bytecode.mem_mut();
                    for i in 0..self.count_states {
                        mem[self.first_state + i] = states.get(i, t);
                    }
                }

                self.bytecode.exec(params);

                {
                    let mem = self.bytecode.mem_mut();
                    for i in 0..self.count_obs {
                        obs.set(i, t, mem[self.first_obs + i]);
                    }
                }
            }
        }
    }

    fn exec_single(t: usize, v: &Matrix, params: &[f64], f: CompiledFunc<f64>) -> i32 {
        let p = v.p.as_ptr();
        f(std::ptr::null(), p, t, params.as_ptr())
    }

    /// Runs the SIMD kernel `f_simd` on the `l` points from `t`; when it does not finish (its
    /// lanes take different sides of a branch, `Generator::branch_if`, and it returns
    /// non-zero), the scalar kernel `f` computes the points one at a time.
    fn exec_simd_row(
        t: usize,
        l: usize,
        v: &Matrix,
        params: &[f64],
        f_simd: CompiledFunc<f64>,
        f: CompiledFunc<f64>,
    ) {
        if Self::exec_single(t, v, params, f_simd) != 0 {
            for i in t..t + l {
                Self::exec_single(i, v, params, f);
            }
        }
    }

    pub fn exec_vectorized_scalar(&mut self, states: &mut Matrix, obs: &mut Matrix, threads: bool) {
        if let Some(compiled) = &mut self.compiled {
            assert!(states.vecsize == obs.vecsize);
            let n = states.vecsize;
            let f = compiled.func();
            let params = &self.params[..];
            let v = combine_matrixes(states, obs);

            if threads {
                (0..n).into_par_iter().for_each(|t| {
                    Self::exec_single(t, &v, params, f);
                });
            } else {
                (0..n).for_each(|t| {
                    Self::exec_single(t, &v, params, f);
                });
            }
        }
    }

    pub fn exec_vectorized_simd(
        &mut self,
        states: &mut Matrix,
        obs: &mut Matrix,
        threads: bool,
        l: usize,
    ) {
        if let Some(compiled) = &mut self.compiled {
            assert!(states.vecsize == obs.vecsize);
            let n = states.vecsize;
            let params = &self.params[..];
            let n0 = l * (n / l);
            let v = combine_matrixes(states, obs);

            let f = compiled.func();

            if let Some(g) = &mut self.compiled_simd {
                let f_simd = g.func();
                if threads {
                    (0..n / l)
                        .into_par_iter()
                        .for_each(|t| Self::exec_simd_row(t * l, l, &v, params, f_simd, f));
                } else {
                    (0..n / l).for_each(|t| Self::exec_simd_row(t * l, l, &v, params, f_simd, f));
                }
            }

            if threads {
                (n0..n).into_par_iter().for_each(|t| {
                    Self::exec_single(t, &v, params, f);
                });
            } else {
                (n0..n).for_each(|t| {
                    Self::exec_single(t, &v, params, f);
                });
            }
        }
    }

    pub fn dump(&mut self, name: &str, what: &str) -> bool {
        match what {
            "scalar" | "wasm" if self.wasm.is_some() => {
                std::fs::write(name, self.wasm.as_ref().unwrap()).is_ok()
            }
            "scalar" => {
                if let Some(f) = &self.compiled {
                    f.dump(name);
                    true
                } else {
                    false
                }
            }
            "simd" => {
                self.prepare_simd();

                if let Some(f) = &self.compiled_simd {
                    f.dump(name);
                    true
                } else {
                    false
                }
            }
            "fast" => {
                self.prepare_fast();

                if let Some(f) = &self.compiled_fast {
                    f.dump(name);
                    true
                } else {
                    false
                }
            }
            "bytecode" => {
                self.bytecode.dump(name);
                true
            }
            "stats" => {
                let size = if let Some(f) = &self.compiled {
                    f.as_machine().unwrap().size
                } else {
                    0
                };
                self.bytecode.mir.print_stats(name, size);
                true
            }
            _ => false,
        }
    }

    pub fn dumps(&self) -> Vec<u8> {
        if let Some(f) = &self.compiled {
            f.dumps()
        } else {
            Vec::new()
        }
    }

    pub fn measure(&self, what: &str) -> usize {
        match what {
            "ker-scalar-size" => {
                if let Some(f) = &self.compiled {
                    f.size
                } else {
                    0
                }
            }
            "ker-simd-size" => {
                if let Some(f) = &self.compiled_simd {
                    f.size
                } else {
                    0
                }
            }
            "ker-fast-size" => {
                if let Some(f) = &self.compiled_fast {
                    f.size
                } else {
                    0
                }
            }
            "mir-size" => self.bytecode.mir.code.ip,
            "mir-bytes" => self.bytecode.mir.code.buf.len(),
            // the general registers of the saved (target-dependent) MIR, see `load_for`
            "registers" => self
                .original
                .as_ref()
                .unwrap_or(&self.bytecode.mir)
                .max_register()
                .map_or(0, |r| r as usize + 1),
            "stack-size" => self.prog.builder.stack_size(),
            #[cfg(feature = "async")]
            "async" => 1,
            #[cfg(not(feature = "async"))]
            "async" => 0,
            "version" => {
                let major: usize = env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(99);
                let minor: usize = env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(99);
                let patch: usize = env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(99);
                major * 10000 + minor * 100 + patch
            }
            _ => 0,
        }
    }

    /************************** save/load ******************************/

    const MAGIC: usize = 0x40568795410d08e9;

    pub fn load(stream: &mut impl Read, config: &Config) -> Result<Self> {
        Self::load_with_version(stream, config, 0)
    }

    /// Loads a saved application to export it (`write_obj`, `object_files`, the wasm module)
    /// for `target`, or for the type it was saved with if `target` is None (`Native` means
    /// this machine). The SIMD and fast kernels are not prepared.
    ///
    /// The saved MIR is register-allocated for the machine it was compiled for. Models
    /// compiled with opt level 2 or lower use the same registers everywhere, but opt level 3
    /// gives ARM64 and RISC-V more registers than x86-64 has; such a model is rejected here
    /// (it would compute wrong results) instead of being compiled.
    pub fn load_for(
        stream: &mut impl Read,
        config: &Config,
        target: Option<CompilerType>,
    ) -> Result<Self> {
        Self::load_as(stream, config, target, false)
    }

    /// The MIR uses no more registers than the generators of `config` have (the wasm
    /// module keeps registers in locals, so any number fits).
    fn check_registers(mir: &Mir, config: &Config) -> Result<()> {
        if config.is_wasm() {
            return Ok(());
        }
        let available = config.count_scratch();
        match mir.max_register() {
            Some(r) if r >= available => Err(anyhow!(
                "the model uses {} registers (it was compiled with opt level {} for a machine with \
                 more registers), but {} has {}; compile it for this target or with opt level 2 \
                 or lower",
                r + 1,
                config.opt_level(),
                Self::arch_name(config),
                available
            )),
            _ => Ok(()),
        }
    }

    fn arch_name(config: &Config) -> &'static str {
        if config.is_amd64() {
            "x86-64"
        } else if config.is_arm64() {
            "ARM64"
        } else if config.is_riscv64() {
            "RISC-V"
        } else if config.is_wasm() {
            "WebAssembly"
        } else {
            "the bytecode interpreter"
        }
    }
}

fn save_reals(stream: &mut impl Write, reals: &HashSet<Loc>) -> Result<()> {
    let num_elems = reals.len();
    stream.write_all(&num_elems.to_le_bytes())?;

    for r in reals.iter() {
        let b = match r {
            Loc::Mem(idx) => 0x100000000 | (*idx as usize),
            Loc::Stack(idx) => 0x200000000 | (*idx as usize),
            Loc::Param(idx) => 0x300000000 | (*idx as usize),
        };
        stream.write_all(&b.to_le_bytes())?;
    }

    Ok(())
}

fn load_reals(stream: &mut impl Read) -> Result<HashSet<Loc>> {
    let mut bytes: [u8; 8] = [0; 8];

    stream.read_exact(&mut bytes)?;
    let num_elems = usize::from_le_bytes(bytes);

    let mut reals: HashSet<Loc> = HashSet::new();

    for _ in 0..num_elems {
        stream.read_exact(&mut bytes)?;
        let b = usize::from_le_bytes(bytes);

        let r = match b >> 32 {
            1 => Loc::Mem((b & 0xffffffff) as u32),
            2 => Loc::Stack((b & 0xffffffff) as u32),
            3 => Loc::Param((b & 0xffffffff) as u32),
            _ => return Err(anyhow!("invalid loc")),
        };
        reals.insert(r);
    }

    Ok(reals)
}

#[cfg(feature = "obj")]
impl Application {
    /// Writes the relocatable object file `<name>.o` in the host's format (ELF on Linux,
    /// Mach-O on macOS) and its C header `<name>.h`. The kernels are named after the last
    /// component of `name`, which must be a C identifier (see `object`).
    pub fn write_obj(&mut self, name: &str) -> Result<()> {
        let target = self.object_target("")?;
        self.write_obj_for(name, target)
    }

    /// `write_obj` for another object format.
    pub fn write_obj_for(&mut self, name: &str, target: Target) -> Result<()> {
        let symbol = std::path::Path::new(name)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("`{}` does not end in a file name", name))?;

        let object_file = format!("{}.o", symbol);
        let header_file = format!("{}.h", symbol);
        let (bytes, text) =
            self.object_bytes(symbol, target, &object_file, &header_file, &mut [])?;

        std::fs::write(format!("{}.o", name), bytes)?;
        std::fs::write(format!("{}.h", name), text)?;
        Ok(())
    }

    /// The object file (in `format`, see `object_target`) and its C header, in memory, for
    /// writing them under other names than `write_obj` does: the kernels are named `symbol`,
    /// and the header names its files `object_file` and `header_file` (e.g. "model.o",
    /// "model.h").
    pub fn object_files(
        &mut self,
        symbol: &str,
        format: &str,
        object_file: &str,
        header_file: &str,
    ) -> Result<(Vec<u8>, String)> {
        self.object_files_with(symbol, format, object_file, header_file, &mut [])
    }

    /// `object_files`, with the kernels this one calls (the applets of its configuration's
    /// `Defuns`) compiled into the object too: `callees` holds their applications, under their
    /// names in the `Defuns`, compiled for the same architecture. Their functions are named
    /// `kernel_symbol(name)`, which the calls go to; a kernel called but not given here is left
    /// to another object.
    pub fn object_files_with(
        &mut self,
        symbol: &str,
        format: &str,
        object_file: &str,
        header_file: &str,
        callees: &mut [(String, Application)],
    ) -> Result<(Vec<u8>, String)> {
        let target = self.object_target(format)?;
        self.object_bytes(symbol, target, object_file, header_file, callees)
    }

    fn object_bytes(
        &mut self,
        symbol: &str,
        target: Target,
        object_file: &str,
        header_file: &str,
        callees: &mut [(String, Application)],
    ) -> Result<(Vec<u8>, String)> {
        let mut obj = self.object(symbol, target)?;
        let fast = obj.functions.len() > 1;

        // the called kernels' scalar kernels
        let mut included = Vec::new();
        for (name, app) in callees.iter_mut() {
            let callee = kernel_symbol(name);
            if app.object_target("")?.arch != target.arch {
                return Err(anyhow!(
                    "the kernel `{}` is compiled for another architecture than its caller",
                    name
                ));
            }
            let mut cobj = app.object(&callee, target)?;
            cobj.functions.truncate(1);
            obj.append(&cobj)?;
            included.push(callee);
        }

        // the undefined symbols: user-defined functions, kernels of other objects and the C
        // math library
        let mut user: Vec<(String, usize)> = Vec::new();
        let mut kernels: HashSet<String> = HashSet::new();
        let configs = std::iter::once(&self.config).chain(callees.iter().map(|(_, a)| &a.config));
        for config in configs {
            if let Some(df) = &config.df {
                for (name, f) in df.funcs.iter() {
                    match f {
                        Func::Unary(_) => user.push((user_symbol(name).to_string(), 1)),
                        Func::Binary(_) => user.push((user_symbol(name).to_string(), 2)),
                        Func::App(_) => {
                            kernels.insert(kernel_symbol(name));
                        }
                        _ => {}
                    }
                }
            }
        }
        let (mut functions, mut external, mut imports) = (Vec::new(), Vec::new(), Vec::new());
        for s in obj.undefined_symbols() {
            if let Some((_, arity)) = user.iter().find(|(u, _)| *u == s) {
                functions.push((s, *arity));
            } else if kernels.contains(&s) {
                external.push(s);
            } else {
                imports.push(s);
            }
        }
        let bytes = obj.build()?;

        let cpu = match target.arch {
            Arch::X86_64 if self.config.has_avx() => "AVX and FMA (x86-64-v3)",
            Arch::X86_64 => "SSE2",
            // packed complex products use fcmla where the compiling CPU has FCMA
            // (not for models compiled through `complexify`, see `object`)
            Arch::Aarch64
                if self.config.is_complex()
                    && self.config.fast_complex()
                    && !Self::calls_complex_functions(
                        self.original.as_ref().unwrap_or(&self.bytecode.mir),
                        &self.config,
                    )
                    && host_has_fcma() =>
            {
                "ARMv8.3-A (with FCMA)"
            }
            Arch::Aarch64 => "ARMv8-A",
        };
        let text = header(&HeaderInfo {
            name: symbol,
            target,
            count_states: self.count_states,
            count_obs: self.count_obs,
            count_params: self.count_params,
            count_diffs: self.count_diffs,
            mem_size: self.prog.mem_size(),
            complex: self.config.is_complex(),
            fast,
            symbolica: self.config.symbolica(),
            cpu,
            imports: &imports,
            functions: &functions,
            external_kernels: &external,
            included_kernels: &included,
            version: env!("CARGO_PKG_VERSION"),
            object_file,
            header_file,
        });

        Ok((bytes, text))
    }

    /// The target of the model's object file: the format named `format` ("" for the
    /// host's, "elf" or "macho") and the architecture the model is compiled for
    /// (x86-64, or AArch64, e.g. with `ty = "arm"`).
    pub fn object_target(&self, format: &str) -> Result<Target> {
        let format = match format {
            "" if cfg!(target_os = "macos") => Format::MachO,
            "" if cfg!(target_os = "linux") => Format::Elf,
            "" => {
                return Err(anyhow!(
                    "name the object format (\"elf\" or \"macho\") on this system"
                ))
            }
            "elf" => Format::Elf,
            "macho" => Format::MachO,
            _ => {
                return Err(anyhow!(
                    "unknown object format `{}` (use \"elf\" or \"macho\")",
                    format
                ))
            }
        };
        let arch = if self.config.is_amd64() {
            Arch::X86_64
        } else if self.config.is_arm64() {
            Arch::Aarch64
        } else {
            return Err(anyhow!(
                "object files hold x86-64 or AArch64 kernels (the model is compiled for {:?})",
                self.config.ty
            ));
        };
        Ok(Target::new(format, arch))
    }

    /// The kernels of the model as functions of a relocatable object file: `name`, the
    /// scalar kernel (with the native kernels' calling convention,
    /// `int name(double *mem, const void *states, size_t idx, const double *params)`),
    /// and `<name>_fast` (`double name_fast(double x0, ...)`) for models that have a
    /// fast kernel. Calls to math functions become relocations against the C math
    /// library; models calling other functions are rejected. The target architecture
    /// must be the one the model is compiled for (x86-64 or AArch64).
    pub fn object(&mut self, name: &str, target: Target) -> Result<ObjectBuilder> {
        let mut config = self.config.clone();
        config.set_relocatable(true);
        let idx_ret = self.can_fast.then_some(self.first_obs as u32);
        let mut obj = ObjectBuilder::new(target);

        // the code of the object, which calls only the C library: a complex model that calls
        // complex functions is compiled through `complexify` (as with `fast_complex=false`), whose
        // calls go to C99's complex functions, and the real functions C does not have are
        // composed of C functions (`lower_for_object`); the program gets their constants and
        // stack slots
        let mut prog = self.prog.clone();
        let original = self.original.as_ref().unwrap_or(&self.bytecode.mir);
        let complex_calls = config.is_complex() && Self::calls_complex_functions(original, &config);
        if complex_calls {
            config.set_fast_complex(false);
        }
        let complex_fast = config.is_complex() && config.fast_complex();
        let mir = if complex_calls {
            let scratch = prog.builder.stack_size();
            prog.builder.count_stack = Some(scratch + 2);
            let mir = Complexifier::for_object(&self.reals, config.clone(), scratch as u32)
                .complexify(original)?;
            if mir.consts.len() > prog.builder.consts.len() {
                prog.builder.consts = mir.consts.clone();
            }
            mir
        } else if config.is_complex() {
            (*self.bytecode.mir).clone()
        } else {
            lower_for_object(&self.bytecode.mir, &mut prog)?
        };
        let prog = &mut prog;

        // the kernels, compiled as `compile_ty` and `compile_*_fast` do
        let (scalar, fast) = match target.arch {
            Arch::X86_64 if self.config.is_amd64() => {
                let scalar = if complex_fast {
                    let mir = original;
                    Self::object_code(mir, prog, AmdComplexGenerator::new(config.clone()), None)?
                } else if config.has_avx() {
                    Self::object_code(&mir, prog, AmdScalarGenerator::new(config.clone()), None)?
                } else {
                    Self::object_code(&mir, prog, AmdSSEGenerator::new(config.clone()), None)?
                };
                let fast = match idx_ret {
                    Some(_) if config.has_avx() => {
                        Some(Self::object_code(&mir, prog, AmdScalarGenerator::new(config), idx_ret)?)
                    }
                    Some(_) => Some(Self::object_code(&mir, prog, AmdSSEGenerator::new(config), idx_ret)?),
                    None => None,
                };
                (scalar, fast)
            }
            Arch::Aarch64 if self.config.is_arm64() => {
                // constants are loaded with PC-relative literal loads, which reach 1 MB (the
                // JIT kernel is a little larger than the object's)
                let size = self.compiled.as_ref().map_or(0, |c| c.size);
                if size >= ARM_OBJECT_LIMIT {
                    return Err(anyhow!(
                        "AArch64 object files hold kernels up to {} bytes (this one is about {})",
                        ARM_OBJECT_LIMIT,
                        size
                    ));
                }
                let scalar = if complex_fast {
                    let mir = original;
                    Self::object_code(mir, prog, ArmComplexGenerator::new(config.clone()), None)?
                } else {
                    Self::object_code(&mir, prog, ArmGenerator::new(config.clone()), None)?
                };
                let fast = match idx_ret {
                    Some(_) => Some(Self::object_code(&mir, prog, ArmGenerator::new(config), idx_ret)?),
                    None => None,
                };
                (scalar, fast)
            }
            arch => {
                return Err(anyhow!(
                    "cannot write {:?} object files for a model compiled for {:?} (use `ty` to choose the architecture)",
                    arch,
                    self.config.ty
                ))
            }
        };

        obj.add_function(name, &scalar.0, &scalar.1)?;
        if let Some((code, relocs)) = fast {
            obj.add_function(&format!("{}_fast", name), &code, &relocs)?;
        }
        Ok(obj)
    }

    /// Whether the complex model `mir` calls complex functions (not kernels, callbacks or
    /// other user functions).
    fn calls_complex_functions(mir: &Mir, config: &Config) -> bool {
        mir.code.iter().any(|ins| {
            matches!(&ins, crate::mir::Instruction::Call { label, num_args } if *num_args > 0
                && !label.starts_with('@')
                && !config.is_user_func(label)
                && !config.is_external_func(label))
        })
    }

    // the machine code and relocations of a kernel (a fast kernel if `idx_ret` is given)
    fn object_code<G: Generator>(
        mir: &Mir,
        prog: &mut Program,
        mut generator: G,
        idx_ret: Option<u32>,
    ) -> Result<(Vec<u8>, Vec<Relocation>)> {
        match idx_ret {
            None => prog.builder.compile_from_mir(
                mir,
                &mut generator,
                prog.count_states,
                prog.count_obs,
                prog.count_params,
            )?,
            Some(idx) => prog.builder.compile_fast_from_mir(
                mir,
                &mut generator,
                prog.count_states,
                prog.count_obs,
                idx as i32,
            )?,
        }

        let relocs = generator
            .relocations()
            .iter()
            .map(|(at, symbol)| Relocation::call(*at, symbol))
            .collect();
        Ok((generator.bytes(), relocs))
    }
}

// the reach of AArch64 PC-relative literal loads (`ldr d, label`, +-1 MB)
#[cfg(feature = "obj")]
const ARM_OBJECT_LIMIT: usize = 1 << 20;

// whether the compiling CPU has FCMA (used by the packed-complex ARM generator)
#[cfg(feature = "obj")]
fn host_has_fcma() -> bool {
    #[cfg(target_arch = "aarch64")]
    return std::arch::is_aarch64_feature_detected!("fcma");
    #[cfg(not(target_arch = "aarch64"))]
    return false;
}

impl Storage for Application {
    fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&Self::MAGIC.to_le_bytes())?;

        let version: usize = 4;
        stream.write_all(&version.to_le_bytes())?;

        self.prog.save(stream)?;

        let mut mask: usize = 0;

        if self.compiled.is_some() && self.compiled.as_ref().unwrap().as_machine().is_some() {
            mask |= 1;
        };

        if self.compiled_fast.is_some()
            && self.compiled_fast.as_ref().unwrap().as_machine().is_some()
        {
            mask |= 2;
        }

        if self.compiled_simd.is_some()
            && self.compiled_simd.as_ref().unwrap().as_machine().is_some()
        {
            mask |= 4;
        }

        stream.write_all(&mask.to_le_bytes())?;

        match &self.original {
            Some(mir) => mir.save(stream)?,
            None => self.bytecode.mir.save(stream)?,
        }

        save_reals(stream, &self.reals)?;

        Ok(())
    }

    fn load_with_version(stream: &mut impl Read, config: &Config, _version: usize) -> Result<Self> {
        Self::load_as(stream, config, None, true)
    }
}

impl Application {
    /// `load_with_version` (`target` None, `prepare` true) and `load_for` (`prepare` false,
    /// with the register check).
    fn load_as(
        stream: &mut impl Read,
        config: &Config,
        target: Option<CompilerType>,
        prepare: bool,
    ) -> Result<Self> {
        let mut bytes: [u8; 8] = [0; 8];

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != Self::MAGIC {
            return Err(anyhow!("invalid magic number (Application)"));
        }

        stream.read_exact(&mut bytes)?;

        let version = usize::from_le_bytes(bytes);

        if version < 3 {
            return Err(anyhow!("invalid sjb version"));
        }

        let prog = Program::load_retargeted(stream, config, version, target)?;

        stream.read_exact(&mut bytes)?;
        let mask = usize::from_le_bytes(bytes);

        let mir = Mir::load_with_version(stream, prog.config(), version)?;

        let reals = load_reals(stream)?;

        if !prepare {
            Self::check_registers(&mir, prog.config())?;
        }

        let mut app = Application::with_mir(prog, reals, mir)?;

        if prepare && mask & 2 != 0 {
            app.prepare_fast();
        }

        if prepare && mask & 4 != 0 {
            app.prepare_simd();
        }

        Ok(app)
    }
}
