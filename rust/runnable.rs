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
use super::complexify::Complexifier;
use super::config::Config;
use super::generator::{Generator, GeneratorType};
use super::machine::MachineCode;
use super::matrix::{combine_matrixes, Matrix};
use super::mir::{CompiledMir, Mir};
use super::model::Program;
use super::riscv64::RiscV;
use super::symbol::Loc;
use super::utils::*;
#[cfg(feature = "wasm")]
use super::wasm::WasmGenerator;
#[cfg(feature = "obj")]
use super::linker::{header, Arch, Format, HeaderInfo, ObjectBuilder, Relocation, Target};

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
        if self.compiled_simd.is_none() && self.use_simd {
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

    fn exec_single(t: usize, v: &Matrix, params: &[f64], f: CompiledFunc<f64>) {
        let p = v.p.as_ptr();
        f(std::ptr::null(), p, t, params.as_ptr());
    }

    pub fn exec_vectorized_scalar(&mut self, states: &mut Matrix, obs: &mut Matrix, threads: bool) {
        if let Some(compiled) = &mut self.compiled {
            assert!(states.vecsize == obs.vecsize);
            let n = states.vecsize;
            let f = compiled.func();
            let params = &self.params[..];
            let v = combine_matrixes(states, obs);

            if threads {
                (0..n)
                    .into_par_iter()
                    .for_each(|t| Self::exec_single(t, &v, params, f));
            } else {
                (0..n)
                    //.into_iter()
                    .for_each(|t| Self::exec_single(t, &v, params, f));
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

            if let Some(g) = &mut self.compiled_simd {
                let f = g.func();
                if threads {
                    (0..n / l)
                        .into_par_iter()
                        .for_each(|t| Self::exec_single(t * l, &v, params, f));
                } else {
                    (0..n / l).for_each(|t| Self::exec_single(t * l, &v, params, f));
                }
            }

            let f = compiled.func();

            if threads {
                (n0..n)
                    .into_par_iter()
                    .for_each(|t| Self::exec_single(t, &v, params, f));
            } else {
                (n0..n).for_each(|t| Self::exec_single(t, &v, params, f));
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
            "stack-size" => self.prog.builder.stack_size(),
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
    /// Writes the relocatable object file `<name>.o` for the host (ELF on Linux, Mach-O
    /// on macOS) and its C header `<name>.h`. The kernels are named after the last
    /// component of `name`, which must be a C identifier (see `object`).
    pub fn write_obj(&mut self, name: &str) -> Result<()> {
        self.write_obj_for(name, Target::host()?)
    }

    /// `write_obj` for another object format.
    pub fn write_obj_for(&mut self, name: &str, target: Target) -> Result<()> {
        let symbol = std::path::Path::new(name)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("`{}` does not end in a file name", name))?;

        let obj = self.object(symbol, target)?;
        let fast = obj.functions.len() > 1;
        let imports = obj.undefined_symbols();
        let bytes = obj.build()?;

        let cpu = if self.config.has_avx() {
            "AVX and FMA (x86-64-v3)"
        } else {
            "SSE2"
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
            version: env!("CARGO_PKG_VERSION"),
        });

        std::fs::write(format!("{}.o", name), bytes)?;
        std::fs::write(format!("{}.h", name), text)?;
        Ok(())
    }

    /// The object format named `format`: "" (the host), "elf" or "macho".
    pub fn object_target(format: &str) -> Result<Target> {
        match format {
            "" => Target::host(),
            "elf" => Ok(Target::new(Format::Elf, Arch::X86_64)),
            "macho" => Ok(Target::new(Format::MachO, Arch::X86_64)),
            _ => Err(anyhow!("unknown object format `{}` (use \"elf\" or \"macho\")", format)),
        }
    }

    /// The kernels of the model as functions of a relocatable object file: `name`, the
    /// scalar kernel (with the native kernels' calling convention,
    /// `int name(double *mem, const void *states, size_t idx, const double *params)`),
    /// and `<name>_fast` (`double name_fast(double x0, ...)`) for models that have a
    /// fast kernel. Calls to math functions become relocations against the C math
    /// library; models calling other functions are rejected. x86-64 only (so far).
    pub fn object(&mut self, name: &str, target: Target) -> Result<ObjectBuilder> {
        if !matches!(target.arch, Arch::X86_64) || !self.config.is_amd64() {
            return Err(anyhow!(
                "object files are supported for x86-64 kernels only (so far)"
            ));
        }

        let mut config = self.config.clone();
        config.set_relocatable(true);
        let mut obj = ObjectBuilder::new(target);

        // the scalar kernel, compiled as `compile_ty` does
        let (code, relocs) = if config.is_complex() && config.fast_complex() {
            let mir = self.original.as_ref().unwrap_or(&self.bytecode.mir);
            Self::object_code(mir, &mut self.prog, AmdComplexGenerator::new(config.clone()), None)?
        } else if config.has_avx() {
            Self::object_code(&self.bytecode.mir, &mut self.prog, AmdScalarGenerator::new(config.clone()), None)?
        } else {
            Self::object_code(&self.bytecode.mir, &mut self.prog, AmdSSEGenerator::new(config.clone()), None)?
        };
        obj.add_function(name, &code, &relocs)?;

        // the fast kernel, as `compile_amd_fast`
        if self.can_fast {
            let idx_ret = Some(self.first_obs as u32);
            let (code, relocs) = if config.has_avx() {
                Self::object_code(&self.bytecode.mir, &mut self.prog, AmdScalarGenerator::new(config), idx_ret)?
            } else {
                Self::object_code(&self.bytecode.mir, &mut self.prog, AmdSSEGenerator::new(config), idx_ret)?
            };
            obj.add_function(&format!("{}_fast", name), &code, &relocs)?;
        }

        Ok(obj)
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

    fn load(stream: &mut impl Read, config: &Config) -> Result<Self> {
        let mut bytes: [u8; 8] = [0; 8];

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != Self::MAGIC {
            return Err(anyhow!("invalid magic number (Application)"));
        }

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != 4 {
            return Err(anyhow!("invalid sjb version"));
        }

        let prog = Program::load(stream, config)?;

        stream.read_exact(&mut bytes)?;
        let mask = usize::from_le_bytes(bytes);

        let mir = Mir::load(stream, prog.config())?;

        let reals = load_reals(stream)?;

        let mut app = Application::with_mir(prog, reals, mir)?;

        if mask & 2 != 0 {
            app.prepare_fast();
        }

        if mask & 4 != 0 {
            app.prepare_simd();
        }

        Ok(app)
    }
}
