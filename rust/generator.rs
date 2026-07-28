use anyhow::Result;

use crate::code::Func;
use crate::symbol::Loc;
use crate::utils::Reg;

pub enum FuncletType {
    None,
    Real,
    Complex,
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

#[allow(dead_code)]
pub trait Generator {
    fn count_shadows(&self) -> u8;
    fn three_address(&self) -> bool;
    fn bytes(&mut self) -> Vec<u8>;
    fn support_funclet(&self) -> FuncletType;

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

    fn save_mem_result(&mut self, idx: u32);
    fn save_stack_result(&mut self, idx: u32);

    fn neg(&mut self, dst: Reg, s1: Reg);
    fn abs(&mut self, dst: Reg, s1: Reg);
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
