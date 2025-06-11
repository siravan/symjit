use anyhow::Result;

use crate::amd::{AmdFamily, AmdGenerator};
use crate::arm::ArmGenerator;
use crate::builder::ByteCode;
use crate::generator::Generator;
use crate::machine::MachineCode;
use crate::model::Program;
use crate::utils::*;

#[derive(PartialEq)]
pub enum CompilerType {
    ByteCode,
    Native,
    Amd,
    AmdAVX,
    AmdSSE,
    Arm,
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
    pub compiled: Box<dyn Compiled<f64>>,
    pub compiled_simd: Option<Box<dyn Compiled<f64x4>>>,
    pub compiled_fast: Option<Box<dyn Compiled<f64>>>,
    pub use_simd: bool,
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
    pub fn new(mut prog: Program, ty: CompilerType, use_simd: bool) -> Result<Runnable> {
        let first_state = 0;
        let idx_iv = prog.count_states;
        let first_param = first_state + prog.count_states + 1; // +1 for the independent variable
        let first_obs = first_param + prog.count_params;
        let first_diff = first_obs + prog.count_obs;
        let size = first_diff + prog.count_diffs + 1; // +1 is here for padding, so that we can return
                                                      // diff vector even if count_diff is 0

        let count_states = prog.count_states;
        let count_params = prog.count_params;
        let count_obs = prog.count_obs;
        let count_diffs = prog.count_diffs;

        let ty = if matches!(ty, CompilerType::Native) {
            if Platform::is_amd64() {
                CompilerType::Amd
            } else if Platform::is_arm64() {
                CompilerType::Arm
            } else {
                println!("cpu not supported, falling back to bytecode.");
                CompilerType::ByteCode
            }
        } else {
            ty
        };

        let compiled = match ty {
            CompilerType::ByteCode => Self::compile_bytecode(&mut prog, size)?,
            CompilerType::Amd | CompilerType::Native => {
                if Platform::has_avx() {
                    Self::compile_avx(&mut prog, size)?
                } else {
                    Self::compile_sse(&mut prog, size)?
                }
            }
            CompilerType::AmdAVX => Self::compile_avx(&mut prog, size)?,
            CompilerType::AmdSSE => Self::compile_sse(&mut prog, size)?,
            CompilerType::Arm => Self::compile_arm(&mut prog, size)?,
        };

        let use_simd = use_simd
            && Platform::has_avx()
            && (matches!(ty, CompilerType::Amd) | matches!(ty, CompilerType::AmdAVX));

        let can_fast = count_states < 8 && count_params == 0 && count_obs == 1 && count_diffs == 0;

        Ok(Runnable {
            prog,
            compiled,
            compiled_simd: None,
            compiled_fast: None,
            use_simd,
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

    fn compile_sse(prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = AmdGenerator::new(AmdFamily::SSEScalar);
        let mem: Vec<f64> = vec![0.0; size];
        prog.builder.compile(&mut generator)?;
        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_avx(prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = AmdGenerator::new(AmdFamily::AvxScalar);
        let mem: Vec<f64> = vec![0.0; size];
        prog.builder.compile(&mut generator)?;
        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_simd(prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64x4>>> {
        let mut generator = AmdGenerator::new(AmdFamily::AvxVector);
        let mem: Vec<f64x4> = vec![f64x4::splat(0.0); size];
        prog.builder.compile(&mut generator)?;
        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64x4>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_arm(prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = ArmGenerator::new();
        let mem: Vec<f64> = vec![0.0; size];
        prog.builder.compile(&mut generator)?;
        let code = MachineCode::new("aarch64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_bytecode(prog: &mut Program, size: usize) -> Result<Box<dyn Compiled<f64>>> {
        let mem: Vec<f64> = vec![0.0; size];
        let code = ByteCode::new(prog.builder.clone(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_avx_fast(
        prog: &mut Program,
        _size: usize,
        idx_ret: u32,
    ) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = AmdGenerator::new(AmdFamily::AvxScalar);
        let mem: Vec<f64> = Vec::new();
        prog.builder
            .compile_fast(&mut generator, prog.count_states as u32, idx_ret as i32)?;
        let code = MachineCode::new("x86_64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    fn compile_arm_fast(
        prog: &mut Program,
        _size: usize,
        idx_ret: u32,
    ) -> Result<Box<dyn Compiled<f64>>> {
        let mut generator = ArmGenerator::new();
        let mem: Vec<f64> = Vec::new();
        prog.builder
            .compile_fast(&mut generator, prog.count_states as u32, idx_ret as i32)?;
        let code = MachineCode::new("aarch64", generator.bytes(), mem);
        let compiled: Box<dyn Compiled<f64>> = Box::new(code);

        Ok(compiled)
    }

    pub fn exec(&mut self, t: f64) {
        {
            let mem = self.compiled.mem_mut();
            mem[self.idx_iv] = t;
        }
        self.compiled.exec();
    }

    fn prepare_simd(&mut self) {
        // SIMD compilation is lazy!
        if self.compiled_simd.is_none() && self.use_simd {
            self.compiled_simd = Self::compile_simd(&mut self.prog, self.size).ok();
        };
    }

    fn prepare_fast(&mut self) {
        // fast func compilation is lazy!
        if self.compiled_simd.is_none() && self.can_fast {
            if Platform::is_amd64() && Platform::has_avx() {
                self.compiled_fast =
                    Self::compile_avx_fast(&mut self.prog, self.size, self.first_obs as u32).ok();
            } else if Platform::is_arm64() {
                self.compiled_fast =
                    Self::compile_arm_fast(&mut self.prog, self.size, self.first_obs as u32).ok();
            }
        };
    }

    pub fn get_fast(&mut self) -> Option<fn(&[f64])> {
        self.prepare_fast();
        self.compiled_fast.as_ref().map(|c| c.func())

        /*
        match &self.compiled_fast {
            Some(c) => Some(c.func()),
            None => None,
        }
        */
    }

    pub fn exec_vectorized(&mut self, buf: &mut [f64], n: usize) {
        self.prepare_simd();

        if self.compiled_simd.is_none() {
            self.exec_vectorized_scalar(buf, n);
        } else {
            self.exec_vectorized_simd(buf, n);
        }
    }

    pub fn exec_vectorized_scalar(&mut self, buf: &mut [f64], n: usize) {
        let h = usize::max(self.count_states, self.count_obs);
        assert!(buf.len() == n * h);

        for t in 0..n {
            {
                let mem = self.compiled.mem_mut();
                mem[self.idx_iv] = t as f64;
                for i in 0..self.count_states {
                    let x = buf[i * n + t];
                    mem[self.first_state + i] = x;
                }
            }

            self.compiled.exec();

            {
                let mem = self.compiled.mem_mut();
                for i in 0..self.count_obs {
                    buf[i * n + t] = mem[self.first_obs + i];
                }
            }
        }
    }

    pub fn exec_vectorized_simd(&mut self, buf: &mut [f64], n: usize) {
        let h = usize::max(self.count_states, self.count_obs);
        assert!(buf.len() == n * h);

        self.set_simd_params();

        if let Some(f) = &mut self.compiled_simd {
            let n0 = 4 * (n / 4);

            for t in (0..n0).step_by(4) {
                {
                    let mem = f.mem_mut();
                    mem[self.idx_iv] = f64x4::splat(t as f64);
                    for i in 0..self.count_states {
                        let x = f64x4::from_slice(&buf[i * n + t..i * n + t + 4]);
                        mem[self.first_state + i] = x;
                    }
                }

                f.exec();

                {
                    let mem = f.mem_mut();
                    for i in 0..self.count_obs {
                        mem[self.first_obs + i].copy_to_slice(&mut buf[i * n + t..i * n + t + 4]);
                    }
                }
            }

            for t in n0..n {
                {
                    let mem = self.compiled.mem_mut();
                    mem[self.idx_iv] = t as f64;
                    for i in 0..self.count_states {
                        mem[self.first_state + i] = buf[i * n + t];
                    }
                }

                self.compiled.exec();

                {
                    let mem = self.compiled.mem_mut();
                    for i in 0..self.count_obs {
                        buf[i * n + t] = mem[self.first_obs + i];
                    }
                }
            }
        }
    }

    fn set_simd_params(&mut self) {
        if let Some(f) = &mut self.compiled_simd {
            let mem = self.compiled.mem();
            let simd_mem = f.mem_mut();

            for i in 0..self.count_params {
                simd_mem[self.first_param + i] = f64x4::splat(mem[self.first_param + i]);
            }
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

        self.compiled.exec();

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
