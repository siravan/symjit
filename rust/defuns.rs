use anyhow::{anyhow, Result};
use num_complex::Complex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::mem::size_of;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::Arc;
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

pub unsafe extern "C" fn trampoline_homogenous<T>(
    env: *const c_void,
    slice_ptr: *const T,
    slice_len: usize,
    res: *mut T,
) where
    T: Sized + Copy + Default,
{
    let closure = &*(env as *const ExternalFunction<T>);
    let slice = from_raw_parts(slice_ptr as *const T, slice_len);
    *res = closure(slice);
}

pub unsafe extern "C" fn trampoline_call_scalar<T, F>(
    env: *const c_void,
    slice_ptr: *const T,
    slice_len: usize,
    res: *mut T,
) where
    T: Sized + Copy + Default,
    F: Sized + Copy + Default,
{
    assert!(slice_len <= SLICE_CAP && size_of::<T>() > size_of::<F>());

    let closure = &*(env as *const ExternalFunction<F>);
    let mut buf = [F::default(); SLICE_CAP];
    let step = size_of::<T>() / size_of::<F>();
    let slice = from_raw_parts(slice_ptr as *mut F, step * slice_len);
    let res = from_raw_parts_mut(res as *mut F, step);

    for i in 0..step {
        for j in 0..slice_len {
            buf[j] = slice[j * step + i];
        }
        res[i] = closure(&buf[..slice_len]);
    }
}

pub unsafe extern "C" fn trampoline_call_simd<T, F>(
    env: *const c_void,
    slice_ptr: *const T,
    slice_len: usize,
    res: *mut T,
) where
    T: Sized + Copy + Default + Element,
    F: Sized + Copy + Default + Element,
{
    assert!(slice_len <= SLICE_CAP && size_of::<T>() < size_of::<F>());

    let closure = &*(env as *const ExternalFunction<F>);
    let buf = [F::default(); SLICE_CAP];
    let step = size_of::<F>() / size_of::<T>();
    let slice = from_raw_parts(slice_ptr, slice_len);
    let p = from_raw_parts_mut(buf.as_ptr() as *mut T, step * slice_len);

    for j in 0..slice_len {
        for i in 0..step {
            p[j * step + i] = slice[j];
        }
    }

    let val = closure(&buf[..slice_len]);
    *res = *(&val as *const F as *const T);
    // let q = from_raw_parts(&val as *const F as *const f64, 8);
    // let mut res: *mut f64 = res as _;
    // *res = q[0];
    // res = res.add(1);

    // match F::get_type() {
    //     ElemType::ComplexF64 => {
    //         *res = q[1];
    //     }
    //     ElemType::ComplexF64x2 => {
    //         *res = q[2];
    //     }
    //     ElemType::ComplexF64x4 => {
    //         *res = q[4];
    //     }
    //     _ => {}
    // }
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

    pub fn add_sliced_func<T>(&mut self, name: &str, closure: ExternalFunction<T>) -> Result<()>
    where
        T: Copy + Sized + Default + Element,
    {
        if VirtualTable::from_str(name).is_ok() {
            return Err(anyhow!("cannot redefine function {}.", &name));
        }

        let ext = Box::new(closure);
        let env = ext.as_ref() as *const _ as *const c_void;

        let trampoline: *const c_void = match T::get_type() {
            ElemType::RealF64 | ElemType::ComplexF64 => trampoline_homogenous::<T> as *const c_void,
            _ => trampoline_call_simd::<f64, T> as *const c_void,
        };

        let trampoline_simd: *const c_void = match T::get_type() {
            ElemType::RealF64 => trampoline_call_scalar::<f64x4, T> as *const c_void,
            ElemType::ComplexF64 => trampoline_call_scalar::<Complex<f64x4>, T> as *const c_void,
            _ => trampoline_homogenous::<T> as *const c_void,
        };

        let (complex, shuffle) = match T::get_type() {
            ElemType::ComplexF64 => (true, true),
            ElemType::ComplexF64x2 | ElemType::ComplexF64x4 => (true, false),
            _ => (false, false),
        };

        let op = format!("${}", name);

        self.funcs.insert(
            op,
            Func::Slice {
                f_scalar: trampoline,
                f_simd: trampoline_simd,
                env,
                complex,
                shuffle,
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
            }
        }
    }
}
