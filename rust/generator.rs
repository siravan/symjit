use anyhow::{anyhow, Result};

use super::code::{Func, VirtualTable};
use super::config::Config;
#[cfg(feature = "experimental")]
use super::mir::ArithOp;
use super::symbol::Loc;
use super::utils::Reg;

// Object mode (`Config::is_relocatable`): math functions are called directly in the C
// library through relocations, and there is no table of function addresses.

/// The C library symbol a call to `op` goes to in object mode, or the error to report.
pub fn object_symbol(config: &Config, op: &str) -> Result<&'static str> {
    if config.is_user_func(op) {
        return Err(object_external_error(op));
    }
    VirtualTable::c_name(op).ok_or_else(|| {
        anyhow!(
            "`{}` is not a C math library function, so it cannot be called from an object file",
            op
        )
    })
}

/// User-defined (Python or Rust) functions live in the compiling process.
pub fn object_external_error(op: &str) -> anyhow::Error {
    anyhow!("the user-defined function `{}` cannot be called from an object file", op)
}

pub fn object_complex_error(op: &str) -> anyhow::Error {
    anyhow!("the complex function `{}` cannot be called from an object file", op)
}

#[derive(Clone, Debug)]
pub struct StackRegions {
    pub cap: u32,
    pub count_states: u32,
    pub count_obs: u32,
    pub count_params: u32,
}

impl StackRegions {
    pub fn new(
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) -> StackRegions {
        StackRegions {
            cap: cap as u32,
            count_states: count_states as u32,
            count_obs: count_obs as u32,
            count_params: count_params as u32,
        }
    }
}

pub enum GeneratorType {
    Unknown,
    Complexifier,
    AmdScalar(bool),
    AmdVectorF64x4(bool),
    AmdVectorF64x8(bool),
    AmdComplex,
    AmdSSE(bool),
    ArmScalar(bool),
    ArmVector(bool),
    ArmComplex,
    RiscvScalar(bool),
    #[cfg(feature = "wasm")]
    Wasm,
}

#[allow(dead_code)]
pub trait Generator {
    fn count_shadows(&self) -> u8;
    fn three_address(&self) -> bool;
    fn bytes(&mut self) -> Vec<u8>;
    /// (offset, symbol) of the calls to external functions in `bytes` (object mode)
    fn relocations(&self) -> Vec<(usize, String)> {
        Vec::new()
    }
    fn what(&self) -> GeneratorType;

    fn seal(&mut self);
    fn align(&mut self);
    fn set_label(&mut self, label: &str);
    fn branch(&mut self, label: &str);
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool);

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg);
    fn fxchg(&mut self, dst: Reg, s1: Reg);
    fn load_const(&mut self, dst: Reg, idx: u32);
    fn load_mem(&mut self, dst: Reg, idx: u32);
    fn save_mem(&mut self, dst: Reg, idx: u32);
    fn load_param(&mut self, dst: Reg, idx: u32);
    fn load_stack(&mut self, dst: Reg, idx: u32);
    fn save_stack(&mut self, dst: Reg, idx: u32);

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32);
    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32);
    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32);
    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32);
    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32);

    fn load_args(&mut self, locs: Vec<Loc>, ultra: bool);
    fn save_args(&mut self, num_args: u8, ultra: bool);
    fn load_args_complex(&mut self, locs: Vec<Loc>, ultra: bool);
    fn save_args_complex(&mut self, num_args: u8, ultra: bool);

    fn save_mem_result(&mut self, idx: u32);
    fn save_stack_result(&mut self, idx: u32);

    fn neg(&mut self, dst: Reg, s1: Reg);
    fn sign(&mut self, dst: Reg, s1: Reg);
    fn abs(&mut self, dst: Reg, s1: Reg);
    fn abs2(&mut self, dst: Reg, s1: Reg);
    fn times_i(&mut self, dst: Reg, s1: Reg);
    fn times_neg_i(&mut self, dst: Reg, s1: Reg);
    fn root(&mut self, dst: Reg, s1: Reg);
    fn real_root(&mut self, dst: Reg, s1: Reg);
    fn recip(&mut self, dst: Reg, s1: Reg);
    fn half(&mut self, dst: Reg, s1: Reg);

    fn round(&mut self, dst: Reg, s1: Reg);
    fn floor(&mut self, dst: Reg, s1: Reg);
    fn ceiling(&mut self, dst: Reg, s1: Reg);
    fn trunc(&mut self, dst: Reg, s1: Reg);
    fn frac(&mut self, dst: Reg, s1: Reg);

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg);

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool;
    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool;

    fn fuse_load_math(&mut self);

    /// `dst := s1 op [loc]` with the operand read from memory (a `LoadMath` MIR
    /// instruction). Returns false if the generator has no such form; the caller then
    /// loads the operand into a scratch register and emits the register operation.
    #[cfg(feature = "experimental")]
    fn op_loc(&mut self, _op: ArithOp, _dst: Reg, _s1: Reg, _loc: Loc) -> bool {
        false
    }

    /// Whether `Mir::rerun` should fold a load followed by an arithmetic instruction into
    /// `op_loc`/`complex_op_loc` when the loaded register is dead afterwards (the loads
    /// the complexifier and the tree compiler emit into scratch registers).
    #[cfg(feature = "experimental")]
    fn fold_loads(&self) -> bool {
        false
    }

    /// `(xd, yd) := (ox, oy) op [loc]`, or `[loc] op (ox, oy)` if `mem_first`, with the
    /// complex operand (real part at `loc`, imaginary part at the next slot) read from
    /// memory; false if not supported.
    #[cfg(feature = "experimental")]
    #[allow(clippy::too_many_arguments)]
    fn complex_op_loc(
        &mut self,
        _op: ArithOp,
        _xd: Reg,
        _yd: Reg,
        _ox: Reg,
        _oy: Reg,
        _loc: Loc,
        _mem_first: bool,
    ) -> bool {
        false
    }
    fn support_times2(&self) -> bool;
    fn times2_loc(&mut self, d1: Reg, s1: Reg, l1: Loc, d2: Reg, s2: Reg, l2: Loc);

    fn real(&mut self, dst: Reg, s1: Reg);
    fn imaginary(&mut self, dst: Reg, s1: Reg);
    fn conjugate(&mut self, dst: Reg, s1: Reg);
    fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg);

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg);

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn not(&mut self, dst: Reg, s1: Reg);

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);

    fn add_consts(&mut self, consts: &[f64]);
    fn add_func(&mut self, f: &str, p: Func);
    fn call(&mut self, op: &str, num_args: usize) -> Result<()>;
    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()>;

    fn call_funclet(&mut self, label: &str);
    fn ret(&mut self);

    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize);
    fn epilogue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize, idx_ret: i32);

    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    );
    fn epilogue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    );

    fn save_used_registers(&mut self, used: &[Reg]);
    fn load_used_registers(&mut self, used: &[Reg]);

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32);
}
