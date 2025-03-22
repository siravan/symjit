use crate::model::Program;
use crate::utils::*;
//use std::simd::f64x4;

use crate::amd::AmdCompiler;
use crate::arm::ArmCompiler;
use crate::avx::AmdCompilerSimd;
use crate::interpreter::Interpreter;
#[cfg(feature = "wasm")]
use crate::wasm::WasmCompiler;

#[derive(PartialEq)]
pub enum CompilerType {
    ByteCode,
    Native,
    Amd,
    Arm,
    #[cfg(feature = "wasm")]
    Wasm,
}

pub struct Runnable {
    // pub prog: Program,
    pub compiled: Box<dyn Compiled<f64>>,
    pub compiled_simd: Option<Box<dyn Compiled<f64x4>>>,
    pub first_state: usize,
    pub first_param: usize,
    pub first_obs: usize,
    pub first_diff: usize,
    pub count_states: usize,
    pub count_params: usize,
    pub count_obs: usize,
    pub count_diffs: usize,
}

impl Runnable {
    pub fn new(prog: Program, ty: CompilerType, use_simd: bool) -> Runnable {
        let compiled: Box<dyn Compiled<f64>> = match ty {
            CompilerType::ByteCode => Box::new(Interpreter::new().compile(&prog)),
            CompilerType::Amd => Box::new(AmdCompiler::new().compile(&prog)),
            CompilerType::Arm => Box::new(ArmCompiler::new().compile(&prog)),

            #[cfg(feature = "wasm")]
            CompilerType::Wasm => Box::new(WasmCompiler::new().compile(&prog)),
            #[cfg(target_arch = "x86_64")]
            CompilerType::Native => Box::new(AmdCompiler::new().compile(&prog)),
            #[cfg(target_arch = "aarch64")]
            CompilerType::Native => Box::new(ArmCompiler::new().compile(&prog)),
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            CompilerType::Native => Box::new(Interpreter::new().compile(&prog)),
        };

        #[cfg(target_arch = "x86_64")]
        let compiled_simd: Option<Box<dyn Compiled<f64x4>>> =
            if use_simd && is_x86_feature_detected!("avx") {
                Some(Box::new(AmdCompilerSimd::new().compile(&prog)))
            } else {
                None
            };
        #[cfg(not(target_arch = "x86_64"))]
        let compiled_simd = None; // Box::new(Compiled::<f64x4>::dummy());;

        let first_state = prog.frame.first_state().unwrap();
        let first_param = prog.frame.first_param().unwrap_or(first_state);
        let first_obs = prog.frame.first_obs().unwrap_or(first_param);
        let first_diff = prog.frame.first_diff().unwrap_or(first_obs);

        let count_states = prog.frame.count_states();
        let count_params = prog.frame.count_params();
        let count_obs = prog.frame.count_obs();
        let count_diffs = prog.frame.count_diffs();

        Runnable {
            // prog,
            compiled,
            compiled_simd,
            first_state,
            first_param,
            first_obs,
            first_diff,
            count_states,
            count_params,
            count_obs,
            count_diffs,
        }
    }

    #[inline]
    pub fn exec(&mut self, t: f64) {
        {
            let mem = self.compiled.mem_mut();
            mem[self.first_state - 1] = t;
        }
        self.compiled.exec();
    }

    pub fn exec_vectorized(&mut self, buf: &mut [f64], n: usize) {
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
                mem[self.first_state - 1] = t as f64;
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

    pub fn exec_vectorized_simd(&mut self, buf: &mut [f64], n: usize) {
        let h = usize::max(self.count_states, self.count_obs);
        assert!(buf.len() == n * h);

        self.set_simd_params();

        if let Some(f) = &mut self.compiled_simd {
            let n0 = 4 * (n / 4);

            for t in (0..n0).step_by(4) {
                {
                    let mem = f.mem_mut();
                    mem[self.first_state - 1] = f64x4::splat(t as f64);
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
                    mem[self.first_state - 1] = t as f64;
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
            mem[self.first_state - 1] = t;
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

    pub fn dump(&self, name: &str, what: &str) -> bool {
        match what {
            "scalar" => {
                self.compiled.dump(name);
                true
            }
            "simd" => {
                if let Some(f) = &self.compiled_simd {
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
