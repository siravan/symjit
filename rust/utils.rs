use std::fmt;

use crate::COUNT_SCRATCH;

pub type CompiledFunc<T> = fn(*const T, *const *mut T, usize, *const T);

pub trait Compiled<T> {
    fn exec(&mut self, params: &[T]);
    fn mem(&self) -> &[T];
    fn mem_mut(&mut self) -> &mut [T];
    fn dump(&self, name: &str);
    fn dumps(&self) -> Vec<u8>;
    fn func(&self) -> CompiledFunc<T>;
    fn support_indirect(&self) -> bool;
    fn count_lanes(&self) -> usize;
}

pub fn bool_to_f64(b: bool) -> f64 {
    const T: f64 = f64::from_bits(!0);
    const F: f64 = f64::from_bits(0);
    if b {
        T
    } else {
        F
    }
}

/// aligns at a multiple of 32 (to cover different ABIs)
pub fn align_stack(n: u32) -> u32 {
    n + 16 - (n & 15)
}

/*****************************************/

#[derive(Copy, Clone, PartialEq)]
pub enum DataType {
    F32,
    F64,
}

#[derive(Copy, Clone, PartialEq)]
pub enum Reg {
    Ret,
    Temp,
    Gen(u8),
    Left,
    Right,
    Static(u32),
}

impl fmt::Debug for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::Ret => write!(f, "%$"),
            Reg::Left => write!(f, "%l"),
            Reg::Right => write!(f, "%r"),
            Reg::Temp => write!(f, "%t"),
            Reg::Gen(r) => write!(f, "%{}", r),
            Reg::Static(r) => write!(f, "σ{}", r),
        }
    }
}

pub fn reg(r: u8) -> Reg {
    assert!(r < COUNT_SCRATCH);
    Reg::Gen(r)
}
