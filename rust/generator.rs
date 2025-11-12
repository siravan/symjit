use crate::code::Func;
use crate::utils::Reg;

#[allow(dead_code)]
pub trait Generator {
    fn count_shadows(&self) -> u8;
    fn three_address(&self) -> bool;
    fn bytes(&mut self) -> Vec<u8>;
    fn seal(&mut self);
    fn align(&mut self);
    fn set_label(&mut self, label: &str);
    fn branch_if(&mut self, cond: Reg, label: &str);

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg);
    fn fxchg(&mut self, dst: Reg, s1: Reg);
    fn load_const(&mut self, dst: Reg, idx: u32);
    fn load_mem(&mut self, dst: Reg, idx: u32);
    fn save_mem(&mut self, dst: Reg, idx: u32);
    fn load_param(&mut self, dst: Reg, idx: u32);
    fn load_stack(&mut self, dst: Reg, idx: u32);
    fn save_stack(&mut self, dst: Reg, idx: u32);

    fn save_mem_result(&mut self, idx: u32);
    fn save_stack_result(&mut self, idx: u32);

    fn neg(&mut self, dst: Reg, s1: Reg);
    fn abs(&mut self, dst: Reg, s1: Reg);
    fn root(&mut self, dst: Reg, s1: Reg);
    fn recip(&mut self, dst: Reg, s1: Reg);

    fn round(&mut self, dst: Reg, s1: Reg);
    fn floor(&mut self, dst: Reg, s1: Reg);
    fn ceiling(&mut self, dst: Reg, s1: Reg);
    fn trunc(&mut self, dst: Reg, s1: Reg);
    fn frac(&mut self, dst: Reg, s1: Reg);

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg);

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
    fn call(&mut self, op: &str, num_args: usize);

    fn prologue_fast(&mut self, cap: u32, num_args: u32);
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32);

    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize);
    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize);

    fn save_used_registers(&mut self, used: &[u8]);
    fn load_used_registers(&mut self, used: &[u8]);

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32);
}
