use crate::assembler::Assembler;

pub trait Generator {
    fn first_shadow(&self) -> u8;
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

    //***********************************
    fn fmov(&mut self, dst: u8, r: u8);
    fn fxchg(&mut self, a: u8, b: u8);
    fn load_const(&mut self, dst: u8, label: &str);
    fn load_mem(&mut self, dst: u8, idx: u32);
    fn save_mem(&mut self, src: u8, idx: u32);
    fn load_stack(&mut self, dst: u8, idx: u32);
    fn save_stack(&mut self, src: u8, idx: u32);

    fn neg(&mut self, dst: u8, r: u8);
    fn abs(&mut self, dst: u8, r: u8);
    fn root(&mut self, dst: u8, r: u8);
    fn square(&mut self, dst: u8, r: u8);
    fn cube(&mut self, dst: u8, r: u8);
    fn recip(&mut self, dst: u8, r: u8);
    fn powi(&mut self, dst: u8, r: u8, n: i32);

    fn round(&mut self, dst: u8, r: u8);
    fn floor(&mut self, dst: u8, r: u8);
    fn ceiling(&mut self, dst: u8, r: u8);
    fn trunc(&mut self, dst: u8, r: u8);
    fn fmod(&mut self, dst: u8, a: u8, b: u8);

    fn plus(&mut self, dst: u8, a: u8, b: u8);
    fn minus(&mut self, dst: u8, a: u8, b: u8);
    fn times(&mut self, dst: u8, a: u8, b: u8);
    fn divide(&mut self, dst: u8, a: u8, b: u8);

    fn gt(&mut self, dst: u8, a: u8, b: u8);
    fn geq(&mut self, dst: u8, a: u8, b: u8);
    fn lt(&mut self, dst: u8, a: u8, b: u8);
    fn leq(&mut self, dst: u8, a: u8, b: u8);
    fn eq(&mut self, dst: u8, a: u8, b: u8);
    fn neq(&mut self, dst: u8, a: u8, b: u8);

    fn and(&mut self, dst: u8, a: u8, b: u8);
    fn andnot(&mut self, dst: u8, a: u8, b: u8);
    fn or(&mut self, dst: u8, a: u8, b: u8);
    fn xor(&mut self, dst: u8, a: u8, b: u8);
    fn not(&mut self, dst: u8, r: u8);

    fn call(&mut self, label: &str, num_args: usize);
    fn select_if(&mut self, dst: u8, cond: u8, a: u8);
    fn select_else(&mut self, dst: u8, cond: u8, a: u8);

    fn prologue(&mut self, n: u32);
    fn epilogue(&mut self, n: u32);
}

pub fn powi<T: Generator>(g: &mut T, dst: u8, r: u8, n: i32) {
    if n == 0 {
        g.divide(dst, dst, dst); // this is a generic way to make 1, but should be
                                 // overrided by the calling Generator for efficiency
    } else if n > 0 {
        let t = n.trailing_zeros();
        let mut s = r;
        let mut n = n >> (t + 1);

        g.fmov(dst, s);

        while n > 0 {
            g.times(1, s, s);
            s = 1;

            if n & 1 != 0 {
                g.times(dst, dst, 1);
            };
            n = n >> 1;
        }

        for _ in 0..t {
            g.times(dst, dst, dst);
        }
    } else {
        powi(g, dst, r, -n);
        g.recip(dst, dst);
    }
}

pub fn fmod<T: Generator>(g: &mut T, dst: u8, a: u8, b: u8) {
    g.divide(1, a, b);
    g.floor(1, 1);
    g.times(1, 1, b);
    g.minus(dst, a, 1);
}
