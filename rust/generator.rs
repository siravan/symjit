use crate::assembler::Assembler;
use crate::utils::Reg;

#[allow(dead_code)]
pub trait Generator {
    fn count_shadows(&self) -> u8;
    fn reg_size(&self) -> u32;
    fn a(&mut self) -> &mut Assembler;
    fn three_address(&self) -> bool;

    // assembler's methods
    fn bytes(&mut self) -> Vec<u8> {
        self.a().bytes()
    }

    fn append_byte(&mut self, b: u8) {
        self.a().append_byte(b);
    }

    fn append_bytes(&mut self, bs: &[u8]) {
        self.a().append_bytes(bs);
    }

    fn append_word(&mut self, u: u32) {
        self.a().append_word(u);
    }

    fn append_quad(&mut self, u: u64) {
        self.a().append_quad(u);
    }

    fn ip(&mut self) -> usize {
        self.a().ip()
    }

    fn set_label(&mut self, label: &str) {
        self.a().set_label(label);
    }

    fn jump(&mut self, label: &str, code: u32) {
        self.a().jump(label, code)
    }

    fn apply_jumps(&mut self) {
        self.a().apply_jumps();
    }

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg);
    fn fxchg(&mut self, dst: Reg, s1: Reg);
    fn load_const(&mut self, dst: Reg, label: &str);
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
    fn square(&mut self, dst: Reg, s1: Reg);
    fn cube(&mut self, dst: Reg, s1: Reg);
    fn recip(&mut self, dst: Reg, s1: Reg);
    fn powi(&mut self, dst: Reg, s1: Reg, power: i32);
    fn powi_mod(&mut self, dst: Reg, s1: Reg, power: i32, modulus: Reg);

    fn round(&mut self, dst: Reg, s1: Reg);
    fn floor(&mut self, dst: Reg, s1: Reg);
    fn ceiling(&mut self, dst: Reg, s1: Reg);
    fn trunc(&mut self, dst: Reg, s1: Reg);
    fn frac(&mut self, dst: Reg, s1: Reg);
    fn fmod(&mut self, dst: Reg, s1: Reg, s2: Reg);

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

    fn setup_call_unary(&mut self, s1: Reg);
    fn setup_call_binary(&mut self, s1: Reg, s2: Reg);
    fn call(&mut self, label: &str, num_args: usize);
    fn select_if(&mut self, dst: Reg, cond: Reg, s1: Reg);
    fn select_else(&mut self, dst: Reg, cond: Reg, s1: Reg);

    fn prologue(&mut self, n: u32);
    fn epilogue(&mut self, n: u32);

    fn prologue_fast(&mut self, cap: u32, num_args: u32);
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32);

    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize);
    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize);
}

pub fn powi<T: Generator>(ir: &mut T, dst: Reg, s1: Reg, power: i32) {
    if power == 0 {
        ir.divide(dst, dst, dst); // this is a generic way to make 1, but should be
                                  // overrided by the calling Generator for efficiency
    } else if power > 0 {
        let t = power.trailing_zeros();
        let mut n = power >> (t + 1);
        let mut s = s1;

        ir.fmov(dst, s1);

        while n > 0 {
            ir.times(Reg::Temp, s, s);
            s = Reg::Temp;

            if n & 1 != 0 {
                ir.times(dst, dst, Reg::Temp);
            };
            n >>= 1;
        }

        for _ in 0..t {
            ir.times(dst, dst, dst);
        }
    } else {
        powi(ir, dst, s1, -power);
        ir.recip(dst, dst);
    }
}

pub fn powi_mod<T: Generator>(ir: &mut T, dst: Reg, s1: Reg, power: i32, modulus: Reg) {
    assert!(dst != Reg::Ret && s1 != Reg::Ret);

    if power == 0 {
        ir.divide(dst, dst, dst); // this is a generic way to make 1, but should be
                                  // overrided by the calling Generator for efficiency
    } else if power > 0 {
        let t = power.trailing_zeros();
        let mut n = power >> (t + 1);
        let mut s = s1;

        ir.fmov(dst, s);

        while n > 0 {
            ir.times(Reg::Temp, s, s);
            ir.fmod(Reg::Temp, Reg::Temp, modulus);
            s = Reg::Temp;

            if n & 1 != 0 {
                ir.times(dst, dst, Reg::Temp);
                ir.fmod(dst, dst, modulus);
            };
            n >>= 1;
        }

        for _ in 0..t {
            ir.times(dst, dst, dst);
            ir.fmod(dst, dst, modulus);
        }
    } else {
        powi(ir, dst, s1, -power);
        ir.recip(dst, dst);
    }
}

pub fn fmod<T: Generator>(ir: &mut T, dst: Reg, s1: Reg, s2: Reg) {
    assert!(dst != Reg::Ret && s1 != Reg::Ret && s2 != Reg::Ret);
    ir.divide(Reg::Ret, s1, s2);
    ir.floor(Reg::Ret, Reg::Ret);
    ir.times(Reg::Ret, Reg::Ret, s2);
    ir.minus(dst, s1, Reg::Ret);
}

pub fn setup_call_unary<T: Generator>(ir: &mut T, s1: Reg) {
    if s1 != Reg::Left {
        ir.fmov(Reg::Left, s1);
    };
}

pub fn setup_call_binary<T: Generator>(ir: &mut T, s1: Reg, s2: Reg) {
    if s1 == Reg::Right && s2 == Reg::Left {
        ir.fxchg(Reg::Right, Reg::Left);
    } else if s2 == Reg::Left {
        ir.fmov(Reg::Right, Reg::Left);
        if s1 != Reg::Left {
            ir.fmov(Reg::Left, s1);
        }
    } else {
        if s1 != Reg::Left {
            ir.fmov(Reg::Left, s1);
        }
        if s2 != Reg::Right {
            ir.fmov(Reg::Right, s2);
        }
    };
}
