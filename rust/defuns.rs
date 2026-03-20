use std::fmt;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use num_complex::Complex;
use std::collections::HashMap;
use std::ffi::c_void;
use wide::{f64x2, f64x4};

type ExternalFunction<T> = Box<dyn Fn(&[T]) -> T + Send + Sync>;

use crate::code::{BinaryFunc, BinaryFuncCplx, Func, UnaryFunc, UnaryFuncCplx, VirtualTable};
use crate::config::SLICE_CAP;
use crate::types::{ElemType, Element};

#[derive(Debug, Clone)]
pub struct RawBox {
    func_ptr: *mut c_void,
    elem_type: ElemType,
}

pub extern "C" fn closure_trampoline<T>(
    env: *const c_void,
    slice_ptr: *const T,
    slice_len: usize,
) -> T
where
    T: Sized + Copy + Default,
{
    // Reconstruct the closure and the slice from the raw C arguments
    let closure: Box<ExternalFunction<T>> = unsafe { std::mem::transmute(env) };
    let slice = unsafe { std::slice::from_raw_parts(slice_ptr, slice_len) };

    // Execute the actual Rust closure
    let val = closure(&slice[..slice_len]);

    // Important! We need to unbox closure to prevent it from dropping
    let _ = Box::into_raw(closure);
    val
}

extern "C" fn closure_trampoline_simd<T>(
    env: *mut c_void,
    slice_ptr: *const T,
    slice_len: usize,
    step: usize,
) -> T
where
    T: Sized + Copy + Default,
{
    // Reconstruct the closure and the slice from the raw C arguments
    let closure: Box<ExternalFunction<T>> = unsafe { std::mem::transmute(env) };
    let mut slice = [T::default(); SLICE_CAP];
    assert!(slice_len <= SLICE_CAP);
    let mut p = slice_ptr;

    for i in 0..slice_len {
        unsafe {
            slice[i] = *p;
            p = p.add(step);
        }
    }

    // Execute the actual Rust closure
    let val = closure(&slice[..slice_len]);

    // Important! We need to unbox closure to prevent it from dropping
    let _ = Box::into_raw(closure);
    val
}

#[derive(Clone, Default)]
pub struct Defuns {
    pub funcs: HashMap<String, Func>,
    pub boxes: Vec<Arc<RawBox>>,
}

impl fmt::Debug for Defuns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?}", &self.funcs)?;
        Ok(())
    }
}

impl Defuns {
    pub fn new() -> Defuns {
        Defuns {
            funcs: HashMap::new(),
            boxes: Vec::new(),
        }
    }

    pub fn add_func(&mut self, name: &str, p: *const usize, num_args: usize) {
        match num_args {
            1 => {
                let f: UnaryFunc = unsafe { std::mem::transmute(p) };
                self.funcs.insert(name.to_string(), Func::Unary(f));
            }
            2 => {
                let f: BinaryFunc = unsafe { std::mem::transmute(p) };
                self.funcs.insert(name.to_string(), Func::Binary(f));
            }
            _ => {
                panic!("only unary and binary functions are supported")
            }
        }
    }

    pub fn add_unary(&mut self, name: &str, f: UnaryFunc) {
        self.funcs.insert(name.to_string(), Func::Unary(f));
    }

    pub fn add_binary(&mut self, name: &str, f: BinaryFunc) {
        self.funcs.insert(name.to_string(), Func::Binary(f));
    }

    pub fn add_unary_complex(&mut self, name: &str, f: UnaryFuncCplx) {
        self.funcs
            .insert(format!("cplx_{}", name), Func::UnaryCplx(f));
    }

    pub fn add_binary_complex(&mut self, name: &str, f: BinaryFuncCplx) {
        self.funcs
            .insert(format!("cplx_{}", name), Func::BinaryCplx(f));
    }

    // pub fn add_sliced_func<F, T>(&mut self, name: &str, mut closure: F) -> Result<()>
    // where
    //     F: Fn(&[T]) -> T,
    //     T: Copy + Sized + Default,
    // {
    //     if VirtualTable::from_str(name).is_ok() {
    //         return Err(anyhow!("cannot redefine function {}.", &name));
    //     }

    //     let env_ptr = &mut closure as *mut _ as *mut c_void;
    //     let trampoline = closure_trampoline::<F, T> as *const c_void;
    //     let trampoline_simd = closure_trampoline_simd::<F, T> as *const c_void;
    //     let op = format!("${}", name);

    //     self.funcs.insert(
    //         op,
    //         Func::Slice {
    //             f_scalar: trampoline,
    //             f_simd: trampoline_simd,
    //             env: env_ptr,
    //         },
    //     );

    //     Ok(())
    // }

    pub fn add_sliced_func<T>(&mut self, name: &str, closure: ExternalFunction<T>) -> Result<()>
    where
        T: Copy + Sized + Default + Element,
    {
        if VirtualTable::from_str(name).is_ok() {
            return Err(anyhow!("cannot redefine function {}.", &name));
        }

        let ext = Box::new(closure);
        let env = ext.as_ref() as *const _ as *const c_void;
        let trampoline = closure_trampoline::<T> as *const c_void;
        let trampoline_simd = closure_trampoline_simd::<T> as *const c_void;
        let op = format!("${}", name);

        self.funcs.insert(
            op,
            Func::Slice {
                f_scalar: trampoline,
                f_simd: trampoline_simd,
                env,
            },
        );

        let func_ptr = Box::into_raw(ext);

        self.boxes.push(Arc::new(RawBox {
            func_ptr: func_ptr as *mut _,
            elem_type: T::get_type(),
        }));

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.funcs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for RawBox {
    fn drop(&mut self) {
        unsafe {
            match self.elem_type {
                ElemType::RealF64 => {
                    let p: *mut ExternalFunction<f64> = self.func_ptr as *mut _;
                    let _: Box<ExternalFunction<f64>> = Box::from_raw(p);
                }
                ElemType::ComplexF64 => {
                    let p: *mut ExternalFunction<Complex<f64>> = self.func_ptr as *mut _;
                    let _: Box<ExternalFunction<Complex<f64>>> = Box::from_raw(p);
                }
                ElemType::RealF64x2 => {
                    let p: *mut ExternalFunction<f64x2> = self.func_ptr as *mut _;
                    let _: Box<ExternalFunction<f64x2>> = Box::from_raw(p);
                }
                ElemType::ComplexF64x2 => {
                    let p: *mut ExternalFunction<Complex<f64x2>> = self.func_ptr as *mut _;
                    let _: Box<ExternalFunction<Complex<f64x2>>> = Box::from_raw(p);
                }
                ElemType::RealF64x4 => {
                    let p: *mut ExternalFunction<f64x4> = self.func_ptr as *mut _;
                    let _: Box<ExternalFunction<f64x4>> = Box::from_raw(p);
                }
                ElemType::ComplexF64x4 => {
                    let p: *mut ExternalFunction<Complex<f64x4>> = self.func_ptr as *mut _;
                    let _: Box<ExternalFunction<Complex<f64x4>>> = Box::from_raw(p);
                }
                _ => {}
            }
        }
    }
}
