use anyhow::Result;
use std::fmt;
use std::io::{Read, Write};

use super::config::Config;
use super::machine::MachineCode;

pub type CompiledFunc<T> = fn(*const T, *const &mut [T], usize, *const T) -> i32;

/// Stable raw descriptor for one column-first P-kernel state plane.
///
/// Unlike `&mut [T]`, this representation can describe duplicate inputs and
/// intentional input/output aliases without creating overlapping Rust
/// references. Callers remain responsible for keeping the allocation alive
/// and for synchronizing every access while a kernel is executing.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PlaneDescriptor<T> {
    pub data: *mut T,
    pub len: usize,
}

impl<T> PlaneDescriptor<T> {
    /// Constructs a descriptor from raw allocation metadata.
    ///
    /// # Safety
    ///
    /// `data` must remain valid and suitably aligned for `len` consecutive
    /// values throughout every kernel call which receives this descriptor.
    pub const unsafe fn from_raw_parts(data: *mut T, len: usize) -> Self {
        Self { data, len }
    }
}

/// Raw-descriptor variant of a compiled kernel entry point.
///
/// The machine-code ABI is identical to [`CompiledFunc`], whose second
/// argument is already consumed as an array of `(data, len)` descriptors.
///
/// Calling the function is unsafe because the generated code dereferences
/// descriptors and parameters without validating their lifetimes, lengths,
/// alignment, alias synchronization, or mutability.
pub type CompiledPlaneFunc<T> =
    unsafe fn(*const T, *const PlaneDescriptor<T>, usize, *const T) -> i32;

pub trait Compiled<T: Sized + Copy + Default> {
    fn exec(&mut self, params: &[T]);
    fn evaluate(&mut self, args: &[T], outs: &mut [T]);
    fn evaluate_single(&mut self, args: &[T]) -> T;
    fn mem(&self) -> &[T];
    fn mem_mut(&mut self) -> &mut [T];
    fn dump(&self, name: &str);
    fn dumps(&self) -> Vec<u8>;
    fn func(&self) -> CompiledFunc<T>;
    fn support_indirect(&self) -> bool;
    fn count_lanes(&self) -> usize;
    fn as_machine(&self) -> Option<&MachineCode<T>>;
}

pub trait Storage: Sized {
    fn save(&self, stream: &mut impl Write) -> Result<()>;
    fn load(stream: &mut impl Read, config: &Config) -> Result<Self>;
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

#[derive(PartialEq)]
#[allow(unused)]
pub enum DataType {
    F32,
    F64,
}

#[derive(Copy, Clone, PartialEq, Hash, Eq)]
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
    Reg::Gen(r)
}

pub fn is_external_func(op: &str) -> bool {
    op.starts_with("$")
}
