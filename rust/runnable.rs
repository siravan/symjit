use anyhow::Result;

use crate::amd::{AmdFamily, AmdGenerator};
use crate::arm::ArmGenerator;
use crate::builder::Builder;
use crate::generator::Generator;
use crate::machine::MachineCode;
use crate::matrix::{combine_matrixes, Matrix};
use crate::mir::{CompiledMir, Mir};
use crate::model::Program;
use crate::symbol::Loc;
use crate::utils::*;
use crate::{FASTMATH, USE_SIMD, USE_THREADS};

use rayon::prelude::*;

#[derive(PartialEq)]
pub enum CompilerType {
    ByteCode,
    Native,
    Amd,
    AmdAVX,
    AmdSSE,
    Arm,
    Debug,
}

pub struct Platform;

impl Platform {
    pub fn is_amd64() -> bool {
        #[cfg(target_arch = "x86_64")]
        return true;
        #[cfg(not(target_arch = "x86_64"))]
        return false;
    }

    pub fn is_arm64() -> bool {
        #[cfg(target_arch = "aarch64")]
        return true;
        #[cfg(not(target_arch = "aarch64"))]
        return false;
    }

    pub fn has_avx() -> bool {
        #[cfg(target_arch = "x86_64")]
        return is_x86_feature_detected!("avx");
        #[cfg(not(target_arch = "x86_64"))]
        return false;
    }
}

pub struct Runnable {
    pub prog: Program,
    pub mir: Mir,
    pub compiled: Box<dyn Compiled<f64>>,
    pub compiled_simd: Option<Box<dyn Compiled<f64x4>>>,
    pub compiled_fast: Option<Box<dyn Compiled<f64>>>,
    pub params: Vec<f64>,
    pub use_simd: bool,
    pub use_threads: bool,
    pub can_fast: bool,
    pub first_state: usize,
    pub first_param: usize,
    pub first_obs: usize,
    pub first_diff: usize,
    pub idx_iv: usize, // independent variable index
    pub count_states: usize,
    pub count_params: usize,
    pub count_obs: usize,
    pub count_diffs: usize,
    pub size: usize,
}

impl Runnable {
    pub fn new(mut prog: Program, ty: CompilerType, opt: u32) -> Result<Runnable> {
        let first_state = 0;
        let first_param = 0;
        let idx_iv = prog.count_states;
        let first_obs = first_state + prog.count_states + 1; // +1 for the independent variable
        let first_diff = first_obs + prog.count_obs;
        let size = first_diff + prog.count_diffs + 1; // +1 is here for padding, so that we can return
                                                      // diff vector even if count_diff is 0

        let count_states = prog.count_states;
        let count_params = prog.count_params;
        let count_obs = prog.count_obs;
        let count_diffs = prog.count_diffs;

        let params = vec![0.0; count_params + 1];

        let fastmath = opt & FASTMATH != 0;
        let mir = prog.builder.create_mir(fastmath)?;

        let compiled = match ty {
            CompilerType::ByteCode => Self::compile_debugger(&mir, &mut prog, size, false)?,
            CompilerType::Native => Self::compile_native(&mir, &mut prog, size)?,
            CompilerType::Amd if Platform::has_avx() => Self::compile_avx(&mir, &mut prog, size)?,
            CompilerType::Amd if !Platform::has_avx() => Self::compile_avx(&mir, &mut prog, size)?,
            CompilerType::AmdAVX => Self::compile_avx(&mir, &mut prog, size)?,
            CompilerType::AmdSSE => Self::compile_sse(&mir, &mut prog, size)?,
            CompilerType::Arm => Self::compile_arm(&mir, &mut prog, size)?,
            CompilerType::Debug => Self::compile_debugger(&mir, &mut prog, size, true)?,
            _ => {
                unreachable!()
            }
        };

        let use_simd = (opt & USE_SIMD != 0)
            && Platform::has_avx()
            && (matches!(ty, CompilerType::Amd)
                | matches!(ty, CompilerType::AmdAVX)
                | matches!(ty, CompilerType::Native));

        let use_threads = opt & USE_THREADS != 0 && size < 128;

        let can_fast = count_states < 8
            && count_params == 0
            && count_obs == 1
            && count_diffs == 0
            && !matches!(ty, CompilerType::ByteCode)
            && !matches!(ty, CompilerType::Debug);

        Ok(Runnable {
            prog,
            mir,
            compiled,
            compiled_simd: None,
            compiled_fast: None,
            params,
            use_simd,
            use_threads,
            can_fast,
            first_state,
            first_param,
            first_obs,
            first_diff,
            idx_iv,
            count_states,
            count_params,
            count_obs,
            count_diffs,
            size,
        })
    }

    /********************* compile_* functions *************************/

    fn compile_native(
        mir: &Mir,
        prog: &mut Program,
        size: usize,
    ) -> Result<Box<dyn Compiled<f64>>> {
        if Platform::is_amd64() && Platform::has_avx() {
            Self::compile_avx(mir, prog, size)
        } else if Platform::is_amd64() && !Platform::has_avx() {
            Self::compile_sse(mir, prog, size)
        } else if Platform::is_arm64() {
            Self::compile_arm(mir, prog, size)
        } else {
            println!("cpu not supported, falling back to bytecode.");
            Self::compile_bytecode(mir, prog, size)
        }
    }

    fn compile_sse(mir: &Mir, prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = AmdGenerator::new(AmdFamily::SSEScalar);
        let mem: Vec<f64> = vec![0.0; size];
        prog.builder
            .compile_from_mir(mir, &mut generator, prog.count_states, prog.count_obs)?;
        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_avx(mir: &Mir, prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = AmdGenerator::new(AmdFamily::AvxScalar);
        let mem: Vec<f64> = vec![0.0; size];
        prog.builder
            .compile_from_mir(mir, &mut generator, prog.count_states, prog.count_obs)?;
        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_simd(
        mir: &Mir,
        prog: &mut Program,
        size: usize,
    ) -> Result<Box<dyn Compiled<f64x4>>> {
        let mut generator = AmdGenerator::new(AmdFamily::AvxVector);
        let mem: Vec<f64x4> = vec![f64x4::splat(0.0); size];
        prog.builder
            .compile_from_mir(mir, &mut generator, prog.count_states, prog.count_obs)?;

        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64x4>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_arm(mir: &Mir, prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = ArmGenerator::new();
        let mem: Vec<f64> = vec![0.0; size];
        prog.builder
            .compile_from_mir(mir, &mut generator, prog.count_states, prog.count_obs)?;
        let code = MachineCode::new("aarch64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_bytecode(
        mir: &Mir,
        prog: &mut Program,
        size: usize,
    ) -> Result<Box<dyn Compiled<f64>>> {
        // println!("{:#?}", &mir);
        let mem: Vec<f64> = vec![0.0; size];
        let stack: Vec<f64> = vec![0.0; prog.builder.block.sym_table.num_stack];
        let code = CompiledMir::new(mir.clone(), mem, stack);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);
        Ok(compiled)
    }

    fn compile_avx_fast(
        mir: &Mir,
        prog: &mut Program,
        _size: usize,
        idx_ret: u32,
    ) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = AmdGenerator::new(AmdFamily::AvxScalar);
        let mem: Vec<f64> = Vec::new();
        prog.builder.compile_fast_from_mir(
            mir,
            &mut generator,
            prog.count_states as u32,
            idx_ret as i32,
        )?;
        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_arm_fast(
        mir: &Mir,
        prog: &mut Program,
        _size: usize,
        idx_ret: u32,
    ) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = ArmGenerator::new();
        let mem: Vec<f64> = Vec::new();
        prog.builder.compile_fast_from_mir(
            mir,
            &mut generator,
            prog.count_states as u32,
            idx_ret as i32,
        )?;
        let code = MachineCode::new("aarch64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_debugger(
        mir: &Mir,
        prog: &mut Program,
        size: usize,
        debug: bool,
    ) -> Result<Box<dyn Compiled<f64>>> {
        let compiled = Self::compile_native(mir, prog, size)?;
        let bytecode = Self::compile_bytecode(mir, prog, size)?;
        let debugger: Box<dyn Compiled<f64>> = Box::new(Debugger::new(
            prog.builder.clone(),
            compiled,
            bytecode,
            debug,
        ));
        Ok(debugger)
    }

    /**********************************************************/

    pub fn exec(&mut self, t: f64) {
        let mem = self.compiled.mem_mut();
        mem[self.idx_iv] = t;
        self.compiled.exec(&self.params[..]);
    }

    pub fn exec_callable(&mut self, xx: &[f64]) -> f64 {
        let mem = self.compiled.mem_mut();
        mem[self.first_state..self.first_state + self.count_states].copy_from_slice(xx);
        mem[self.idx_iv] = 0.0;
        self.compiled.exec(&self.params[..]);
        self.compiled.mem()[self.first_obs]
    }

    fn prepare_simd(&mut self) {
        // SIMD compilation is lazy!
        if self.compiled_simd.is_none() && self.use_simd {
            self.compiled_simd = Self::compile_simd(&self.mir, &mut self.prog, self.size).ok();
        };
    }

    fn prepare_fast(&mut self) {
        // fast func compilation is lazy!
        if self.compiled_simd.is_none() && self.can_fast {
            if Platform::is_amd64() && Platform::has_avx() {
                self.compiled_fast = Self::compile_avx_fast(
                    &self.mir,
                    &mut self.prog,
                    self.size,
                    self.first_obs as u32,
                )
                .ok();
            } else if Platform::is_arm64() {
                self.compiled_fast = Self::compile_arm_fast(
                    &self.mir,
                    &mut self.prog,
                    self.size,
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

    pub fn exec_vectorized(&mut self, states: &Matrix, obs: &mut Matrix) {
        if !self.compiled.support_indirect() {
            self.exec_vectorized_simple(states, obs);
            return;
        }

        self.prepare_simd();

        if self.compiled_simd.is_none() {
            self.exec_vectorized_scalar(states, obs, self.use_threads);
        } else {
            self.exec_vectorized_simd(states, obs, self.use_threads);
        }
    }

    pub fn exec_vectorized_simple(&mut self, states: &Matrix, obs: &mut Matrix) {
        assert!(states.ncols == obs.ncols);
        let n = states.ncols;
        let params = &self.params[..];

        for t in 0..n {
            {
                let mem = self.compiled.mem_mut();
                mem[self.idx_iv] = t as f64;
                for i in 0..self.count_states {
                    mem[self.first_state + i] = states.get(i, t);
                }
            }

            self.compiled.exec(params);

            {
                let mem = self.compiled.mem_mut();
                for i in 0..self.count_obs {
                    obs.set(i, t, mem[self.first_obs + i]);
                }
            }
        }
    }

    fn exec_single<T>(t: usize, v: &Matrix, params: &[f64], f: CompiledFunc<T>) {
        f(std::ptr::null(), v.p.as_ptr(), t, params.as_ptr());
    }

    pub fn exec_vectorized_scalar(&mut self, states: &Matrix, obs: &mut Matrix, threads: bool) {
        assert!(states.ncols == obs.ncols);
        let n = states.ncols;
        let f = self.compiled.func();
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

    pub fn exec_vectorized_simd(&mut self, states: &Matrix, obs: &mut Matrix, threads: bool) {
        assert!(states.ncols == obs.ncols);
        let n = states.ncols;
        let params = &self.params[..];
        let n0 = 4 * (n / 4);
        let v = combine_matrixes(states, obs);

        if let Some(g) = &mut self.compiled_simd {
            let f = g.func();
            if threads {
                (0..n / 4)
                    .into_par_iter()
                    .for_each(|t| Self::exec_single(4 * t, &v, params, f));
            } else {
                (0..n / 4)
                    //.into_iter()
                    .for_each(|t| Self::exec_single(4 * t, &v, params, f));
            }
        }

        let f = self.compiled.func();

        if threads {
            (n0..n)
                .into_par_iter()
                .for_each(|t| Self::exec_single(t, &v, params, f));
        } else {
            (n0..n)
                //.into_iter()
                .for_each(|t| Self::exec_single(t, &v, params, f));
        }
    }

    // call interface to Julia ODESolver
    pub fn call(&mut self, du: &mut [f64], u: &[f64], p: &[f64], t: f64) {
        {
            let mem = self.compiled.mem_mut();
            mem[self.idx_iv] = t;
            let _ =
                &mut mem[self.first_state..self.first_state + self.count_states].copy_from_slice(u);
            let _ =
                &mut mem[self.first_param..self.first_param + self.count_params].copy_from_slice(p);
        }

        self.compiled.exec(&self.params[..]);

        {
            let mem = self.compiled.mem();
            du.copy_from_slice(&mem[self.first_diff..self.first_diff + self.count_diffs]);
        }
    }

    pub fn dump(&mut self, name: &str, what: &str) -> bool {
        match what {
            "scalar" => {
                self.compiled.dump(name);
                true
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
            _ => false,
        }
    }
}

/***************************************************/

pub struct Debugger {
    builder: Builder,
    compiled: Box<dyn Compiled<f64>>,
    bytecode: Box<dyn Compiled<f64>>,
    debug: bool,
}

impl Debugger {
    pub fn new(
        builder: Builder,
        compiled: Box<dyn Compiled<f64>>,
        bytecode: Box<dyn Compiled<f64>>,
        debug: bool,
    ) -> Debugger {
        Debugger {
            builder,
            compiled,
            bytecode,
            debug,
        }
    }

    fn assert_equal(&self) {
        let p = self.compiled.mem();
        let q = self.bytecode.mem();

        // accept if the difference is less that 1e-15 to count for rounding error
        // because of different operation order
        if p.iter().zip(q).any(|(x, y)| !(f64::abs(*x - *y) < 1e-6)) {
            for (key, sym) in self.builder.block.sym_table.syms.iter() {
                match sym.borrow().loc {
                    Loc::Mem(idx) => {
                        let a = p[idx as usize];
                        let b = q[idx as usize];
                        let eq = if a == b { "pass" } else { "fail" };
                        println!("{:14.8} {:14.8} {} -> \t{}", a, b, eq, key);
                    }
                    Loc::Param(..) => {}
                    Loc::Stack(..) => {}
                }
            }
            panic!("discrepencies detected!");
        }
    }
}

impl Compiled<f64> for Debugger {
    fn exec(&mut self, params: &[f64]) {
        if !self.debug {
            self.bytecode.exec(params);
            return;
        }

        let p = self.compiled.mem_mut();
        let q = self.bytecode.mem();
        p.copy_from_slice(q);

        self.bytecode.exec(params);
        self.compiled.exec(params);
        self.assert_equal();
    }

    fn mem(&self) -> &[f64] {
        self.bytecode.mem()
    }

    fn mem_mut(&mut self) -> &mut [f64] {
        self.bytecode.mem_mut()
    }

    fn dump(&self, name: &str) {
        self.bytecode.dump(name);
    }

    fn func(&self) -> CompiledFunc<f64> {
        unreachable!()
    }

    fn support_indirect(&self) -> bool {
        false
    }
}
