use std::collections::HashMap;
use std::ffi::c_void;

use crate::code::{BinaryFunc, BinaryFuncCplx, Func, UnaryFunc, UnaryFuncCplx};
use crate::config::SLICE_CAP;

#[derive(Debug, Clone, Default)]
pub struct Defuns {
    pub funcs: HashMap<String, Func>,
}

pub extern "C" fn closure_trampoline<F, T>(
    env: *mut c_void,
    slice_ptr: *const T,
    slice_len: usize,
) -> T
where
    F: Fn(&[T]) -> T,
    T: Sized + Copy + Default,
{
    // Reconstruct the closure and the slice from the raw C arguments
    let closure = unsafe { &mut *(env as *mut F) };
    let slice = unsafe { std::slice::from_raw_parts(slice_ptr, slice_len) };

    // Execute the actual Rust closure
    closure(slice)
}

extern "C" fn closure_trampoline_simd<F, T>(
    env: *mut c_void,
    slice_ptr: *const T,
    slice_len: usize,
    step: usize,
) -> T
where
    F: Fn(&[T]) -> T,
    T: Sized + Copy + Default,
{
    // Reconstruct the closure and the slice from the raw C arguments
    let closure = unsafe { &mut *(env as *mut F) };
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
    closure(&slice[..slice_len])
}

impl Defuns {
    pub fn new() -> Defuns {
        Defuns {
            funcs: HashMap::new(),
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

    pub fn add_sliced_func<F, T>(&mut self, name: &str, mut closure: F)
    where
        F: Fn(&[T]) -> T,
        T: Copy + Sized + Default,
    {
        let env_ptr = &mut closure as *mut _ as *mut c_void;
        let trampoline = closure_trampoline::<F, T> as *const c_void;
        let trampoline_simd = closure_trampoline_simd::<F, T> as *const c_void;
        let op = format!("${}", name);

        self.funcs.insert(
            op,
            Func::Slice {
                f_scalar: trampoline,
                f_simd: trampoline_simd,
                env: env_ptr,
            },
        );
    }

    pub fn len(&self) -> usize {
        self.funcs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
