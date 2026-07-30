use anyhow::{anyhow, Result};
use rayon::prelude::*;

use super::config::Config;
use super::machine::MachineCode;
use super::matrix::{combine_matrixes, Matrix};
use super::runnable::Application;
use super::types::{ElemType, Element};
use super::utils::*;

#[derive(Clone)]
#[repr(C)]
pub struct Applet {
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
}

impl Applet {
    pub fn new(app: Application) -> Result<Applet> {
        if app.config.is_bytecode() {
            return Err(anyhow!("Bytecode Application cannot be sealed."));
        }

        Ok(Applet {
            compiled: app.compiled,
            compiled_simd: app.compiled_simd,
            use_simd: app.use_simd,
            use_threads: app.use_threads,
            config: app.config.clone(),
            params: app.params.clone(),
            first_state: app.first_state,
            first_obs: app.first_obs,
            count_states: app.count_states,
            count_obs: app.count_obs,
            count_params: app.count_params,
        })
    }

    /// Generic evaluate function for compiled Symbolica expressions
    pub fn evaluate<T>(&self, args: &[T], outs: &mut [T])
    where
        T: Element,
    {
        let args = recast_as_f64(args);
        let outs = recast_as_f64_mut(outs);

        let simd = matches!(
            T::get_type(T::default()),
            ElemType::RealF64x2(_)
                | ElemType::RealF64x4(_)
                | ElemType::ComplexF64x2(_)
                | ElemType::ComplexF64x4(_)
        );

        if let Some(f) = &self.compiled {
            if !simd {
                f.func()(outs.as_mut_ptr(), std::ptr::null(), 0, args.as_ptr());
            } else if let Some(g) = &self.compiled_simd {
                g.func()(outs.as_mut_ptr(), std::ptr::null(), 0, args.as_ptr());
            }
        }
    }

    /// Generic evaluate_single function for compiled Symbolica expressions
    #[inline(always)]
    pub fn evaluate_single<T>(&self, args: &[T]) -> T
    where
        T: Element + Copy,
    {
        let mut outs = [T::default(); 1];
        self.evaluate(args, &mut outs);
        outs[0]
    }

    /// Evaluates a single logical row. It could be a combinatino of multiple
    /// physical rows because of implicit SIMD.
    fn evaluate_row(
        args: &[f64],
        args_idx: usize,
        outs: &[f64],
        outs_idx: usize,
        f: CompiledFunc<f64>,
        transpose: bool,
    ) -> i32 {
        unsafe {
            f(
                outs.as_ptr().add(outs_idx),
                std::ptr::null(),
                if transpose { 1 } else { 0 },
                args.as_ptr().add(args_idx),
            )
        }
    }

    fn evaluate_matrix_with_threads(&self, args: &[f64], outs: &mut [f64], n: usize) {
        if let Some(f) = &self.compiled {
            let count_params = self.count_params;
            let count_obs = self.count_obs;
            let f_scalar = f.func();

            (0..n).into_par_iter().for_each(|t| {
                Self::evaluate_row(args, t * count_params, outs, t * count_obs, f_scalar, false);
            });
        }
    }

    fn evaluate_matrix_without_threads(&self, args: &[f64], outs: &mut [f64], n: usize) {
        if let Some(f) = &self.compiled {
            let count_params = self.count_params;
            let count_obs = self.count_obs;
            let f_scalar = f.func();

            for t in 0..n {
                Self::evaluate_row(args, t * count_params, outs, t * count_obs, f_scalar, false);
            }
        }
    }

    fn evaluate_matrix_with_threads_simd(
        &self,
        args: &[f64],
        outs: &mut [f64],
        n: usize,
        transpose: bool,
    ) {
        if let Some(f) = &self.compiled {
            let count_params = self.count_params;
            let count_obs = self.count_obs;

            if let Some(compiled) = &self.compiled_simd {
                let f_simd = compiled.func();
                let f_scalar = f.func();
                let lanes = compiled.count_lanes();
                let step = if transpose { lanes } else { 1 };

                (0..n / step).into_par_iter().for_each(|k| {
                    let top = k * lanes;
                    if Self::evaluate_row(
                        args,
                        top * count_params,
                        outs,
                        top * count_obs,
                        f_simd,
                        transpose,
                    ) != 0
                    {
                        for i in 0..lanes {
                            Self::evaluate_row(
                                args,
                                (top + i) * count_params,
                                outs,
                                (top + i) * count_obs,
                                f_scalar,
                                false,
                            );
                        }
                    }
                });

                for t in step * (n / step)..n {
                    Self::evaluate_row(
                        args,
                        t * count_params,
                        outs,
                        t * count_obs,
                        f_scalar,
                        false,
                    );
                }
            }
        }
    }

    fn evaluate_matrix_without_threads_simd(
        &self,
        args: &[f64],
        outs: &mut [f64],
        n: usize,
        transpose: bool,
    ) {
        if let Some(f) = &self.compiled {
            let count_params = self.count_params;
            let count_obs = self.count_obs;

            if let Some(compiled) = &self.compiled_simd {
                let f_simd = compiled.func();
                let f_scalar = f.func();
                let lanes = compiled.count_lanes();
                let step = if transpose { lanes } else { 1 };

                for k in 0..n / step {
                    let top = k * lanes;
                    if Self::evaluate_row(
                        args,
                        top * count_params,
                        outs,
                        top * count_obs,
                        f_simd,
                        transpose,
                    ) != 0
                    {
                        for i in 0..lanes {
                            Self::evaluate_row(
                                args,
                                (top + i) * count_params,
                                outs,
                                (top + i) * count_obs,
                                f_scalar,
                                false,
                            );
                        }
                    }
                }

                for t in step * (n / step)..n {
                    Self::evaluate_row(
                        args,
                        t * count_params,
                        outs,
                        t * count_obs,
                        f_scalar,
                        false,
                    );
                }
            }
        }
    }

    /// Generic evaluate function for compiled Symbolica expressions
    /// The main entry point to compute matrices.
    /// The actual dispatched method depends on the configuration and the
    /// type of the arguments.
    pub fn evaluate_matrix<T>(&self, args: &[T], outs: &mut [T], n: usize)
    where
        T: Element,
    {
        let args = recast_as_f64(args);
        let outs = recast_as_f64_mut(outs);

        let transpose = !matches!(
            T::get_type(T::default()),
            ElemType::RealF64x2(_)
                | ElemType::RealF64x4(_)
                | ElemType::ComplexF64x2(_)
                | ElemType::ComplexF64x4(_)
        );

        if self.use_threads && n > 1 {
            if self.compiled_simd.is_some() {
                self.evaluate_matrix_with_threads_simd(args, outs, n, transpose);
            } else {
                self.evaluate_matrix_with_threads(args, outs, n);
            }
        } else {
            if self.compiled_simd.is_some() {
                self.evaluate_matrix_without_threads_simd(args, outs, n, transpose);
            } else {
                self.evaluate_matrix_without_threads(args, outs, n);
            }
        }
    }

    /* P-kernel execution */

    pub fn exec_vectorized(&mut self, states: &mut Matrix, obs: &mut Matrix) {
        if let Some(compiled) = &self.compiled {
            if !compiled.support_indirect() {
                self.exec_vectorized_simple(states, obs);
                return;
            }

            // self.prepare_simd();

            if let Some(simd) = &self.compiled_simd {
                self.exec_vectorized_simd(states, obs, self.use_threads, simd.count_lanes());
            } else {
                self.exec_vectorized_scalar(states, obs, self.use_threads);
            }
        }
    }

    fn exec_vectorized_simple(&mut self, states: &Matrix, obs: &mut Matrix) {
        assert!(states.ncols == obs.ncols);
        let n = states.ncols;
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
        }
    }

    fn exec_single(t: usize, v: &Matrix, params: &[f64], f: CompiledFunc<f64>) {
        let p = v.p.as_ptr();
        f(std::ptr::null(), p, t, params.as_ptr());
    }

    fn exec_vectorized_scalar(&mut self, states: &mut Matrix, obs: &mut Matrix, threads: bool) {
        if let Some(compiled) = &mut self.compiled {
            assert!(states.ncols == obs.ncols);
            let n = states.ncols;
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

    fn exec_vectorized_simd(
        &mut self,
        states: &mut Matrix,
        obs: &mut Matrix,
        threads: bool,
        l: usize,
    ) {
        if let Some(compiled) = &mut self.compiled {
            assert!(states.ncols == obs.ncols);
            let n = states.ncols;
            let params = &self.params[..];
            let n0 = l * (n / l);
            let v = combine_matrixes(states, obs);

            if let Some(g) = &mut self.compiled_simd {
                let f = g.func();
                if threads {
                    (0..n / l)
                        .into_par_iter()
                        .for_each(|t| Self::exec_single(t, &v, params, f));
                } else {
                    (0..n / l).for_each(|t| Self::exec_single(t, &v, params, f));
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
}

pub fn recast_as_f64<T>(v: &[T]) -> &[f64]
where
    T: Sized,
{
    let s = std::mem::size_of::<T>() / std::mem::size_of::<f64>();
    let p: *const f64 = v.as_ptr() as _;
    let q: &[f64] = unsafe { std::slice::from_raw_parts(p, s * v.len()) };
    q
}

pub fn recast_as_f64_mut<T>(v: &mut [T]) -> &mut [f64]
where
    T: Sized,
{
    let s = std::mem::size_of::<T>() / std::mem::size_of::<f64>();
    let p: *mut f64 = v.as_ptr() as _;
    let q: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(p, s * v.len()) };
    q
}
