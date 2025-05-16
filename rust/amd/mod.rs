mod asm;

use asm::Amd;

use crate::code::BinaryFunc;

pub enum AmdFamily {
    AvxScalar,
    AvxVector,
    SSEScalar,
}

pub struct AmdCompiler {
    amd: Amd,
    family: AmdFamily,
}

impl AmdCompiler {
    pub fn new(family: AmdFamily) -> AmdCompiler {
        AmdCompiler {
            amd: Amd::new(),
            family,
        }
    }

    pub fn first_shadow(&self) -> u8 {
        return 2;
    }

    pub fn count_shadows(&self) -> u8 {
        #[cfg(target_family = "windows")]
        return 4;
        #[cfg(not(target_family = "windows"))]
        return 14;
    }
    
    pub fn reg_size(&self) -> u32 {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => 8,
            AmdFamily::AvxVector => 32,
        }   
    }

    // assembler's methods
    pub fn bytes(&self) -> Vec<u8> {
        self.amd.a.bytes()
    }

    pub fn append_byte(&mut self, b: u8) {
        self.amd.a.append_byte(b);
    }

    pub fn append_bytes(&mut self, bs: &[u8]) {
        self.amd.a.append_bytes(bs);
    }

    pub fn append_word(&mut self, u: u32) {
        self.amd.a.append_word(u);
    }

    pub fn append_quad(&mut self, u: u64) {
        self.amd.a.append_quad(u);
    }

    pub fn ip(&self) -> usize {
        self.amd.a.ip()
    }

    pub fn set_label(&mut self, label: &str) {
        self.amd.a.set_label(label);
    }

    pub fn jump(&mut self, label: &str, code: u32) {
        self.amd.a.jump(label, code)
    }

    pub fn apply_jumps(&mut self) {
        self.amd.a.apply_jumps();
    }

    //***********************************

    /*
        shrink is a helper function used to generate
        SSE codes from 3-address inputs.

        IMPORTANT! this function can overwrite the values of
        a and/or b. Therefore, cannot assume a and b are intact
        after calling this function.
    */
    fn shrink(&mut self, dst: u8, a: u8, b: u8, commutative: bool) -> (u8, u8) {
        if dst == a {
            (a, b)
        } else if dst == b {
            // difficult case: dst == b && dst != a
            if !commutative {
                self.amd.xchg_xmm(a, b);
            };
            (dst, a)
        } else {
            self.fmov(dst, a);
            (dst, b)
        }
    }

    //***********************************
    pub fn fmov(&mut self, dst: u8, r: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vmovapd(dst, r),
            AmdFamily::SSEScalar => self.amd.movapd(dst, r),
        }
    }

    pub fn load_const(&mut self, dst: u8, label: &str) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_label(dst, label),
            AmdFamily::AvxVector => self.amd.vbroadcastsd_label(dst, label),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_label(dst, label),
        }
    }

    pub fn load_mem(&mut self, dst: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_mem(dst, Amd::RBP, (idx * 8) as i32),
            AmdFamily::AvxVector => self.amd.vmovpd_ymm_mem(dst, Amd::RBP, (idx * 32) as i32),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, Amd::RBP, (idx * 8) as i32),
        }
    }

    pub fn save_mem(&mut self, src: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(Amd::RBP, (idx * 8) as i32, src),
            AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(Amd::RBP, (idx * 32) as i32, src),
            AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RBP, (idx * 8) as i32, src),
        }
    }

    pub fn load_stack(&mut self, dst: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_mem(dst, Amd::RSP, (idx * 8) as i32),
            AmdFamily::AvxVector => self.amd.vmovpd_ymm_mem(dst, Amd::RSP, (idx * 32) as i32),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, Amd::RSP, (idx * 8) as i32),
        }
    }

    pub fn save_stack(&mut self, src: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(Amd::RSP, (idx * 8) as i32, src),
            AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(Amd::RSP, (idx * 32) as i32, src),
            AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RSP, (idx * 8) as i32, src),
        }
    }

    pub fn neg(&mut self, dst: u8) {
        self.load_const(1, "_minus_zero_");
        self.xor(dst, dst, 1);
    }

    pub fn abs(&mut self, dst: u8) {
        self.load_const(1, "_minus_zero_");
        self.andnot(dst, 1, dst);
    }

    pub fn root(&mut self, dst: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vsqrtsd(dst, dst),
            AmdFamily::AvxVector => self.amd.vsqrtpd(dst, dst),
            AmdFamily::SSEScalar => self.amd.sqrtsd(dst, dst),
        }
    }

    pub fn square(&mut self, dst: u8) {
        self.times(dst, dst, dst);
    }

    pub fn cube(&mut self, dst: u8) {
        self.times(1, dst, dst);
        self.times(dst, dst, 1);
    }

    pub fn recip(&mut self, dst: u8) {
        self.load_const(1, "_one_");
        self.divide(dst, 1, dst);
    }

    pub fn plus(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vaddsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vaddpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.addsd(x, y);
            }
        }
    }

    pub fn minus(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vsubsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vsubpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.subsd(x, y);
            }
        }
    }

    pub fn times(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmulsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vmulpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.mulsd(x, y);
            }
        }
    }

    pub fn divide(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vdivsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vdivpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.divsd(x, y);
            }
        }
    }

    pub fn gt(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpnlesd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpnlepd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpnlesd(x, y);
            }
        }
    }

    pub fn geq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpnltsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpnltpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpnltsd(x, y);
            }
        }
    }

    pub fn lt(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpltsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpltpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpltsd(x, y);
            }
        }
    }

    pub fn leq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmplesd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmplepd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmplesd(x, y);
            }
        }
    }

    pub fn eq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpeqsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpeqpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpeqsd(x, y);
            }
        }
    }

    pub fn neq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpneqsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpneqpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpneqsd(x, y);
            }
        }
    }

    pub fn and(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vandpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.andnpd(x, y);
            }
        }
    }

    pub fn andnot(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vandnpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.andnpd(x, y);
            }
        }
    }

    pub fn or(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vorpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.orpd(x, y);
            }
        }
    }

    pub fn xor(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vxorpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.xorpd(x, y);
            }
        }
    }

    pub fn not(&mut self, dst: u8) {
        self.xor(1, 1, 1);
        self.eq(1, 1, 1);
        self.xor(dst, dst, 1);
    }

    pub fn call(&mut self, label: &str) {
        if matches!(self.family, AmdFamily::AvxScalar)
            || matches!(self.family, AmdFamily::AvxVector)
        {
            self.amd.vzeroupper();
        }

        // Windows 32-byte home area
        #[cfg(target_family = "windows")]
        self.amd.sub_rsp(32);

        self.amd.mov_reg_label(Amd::RAX, label);
        self.amd.call(Amd::RAX);

        #[cfg(target_family = "windows")]
        self.amd.add_rsp(32);
    }

    pub fn branch(&mut self, label: &str) {
        self.amd.jmp(label);
    }

    pub fn branch_if(&mut self, cond: u8, true_label: &str) {
        self.amd.vucomisd(cond, cond);
        self.amd.jpe(true_label);
    }

    pub fn branch_if_else(&mut self, cond: u8, true_label: &str, false_label: &str) {
        self.amd.vucomisd(cond, cond);
        self.amd.jpe(true_label);
        self.amd.jmp(false_label);
    }

    pub fn select_if(&mut self, dst: u8, cond: u8, a: u8) {
        self.amd.vandpd(dst, cond, a);
    }

    pub fn select_else(&mut self, dst: u8, cond: u8, a: u8) {
        self.amd.vandnpd(dst, cond, a);
    }

    #[cfg(target_family = "unix")]
    pub fn prologue(&mut self, n: u32) {
        self.amd.push(Amd::RBP);
        self.amd.mov(Amd::RBP, Amd::RDI);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.sub_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.sub_rsp(32 * n),
        }
    }

    #[cfg(target_family = "windows")]
    pub fn prologue(&mut self, n: u32) {
        self.amd.mov_mem_reg(Amd::RSP, 0x08, Amd::RBP);
        self.amd.mov(Amd::RBP, Amd::RCX);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.sub_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.sub_rsp(32 * n),
        }
    }

    #[cfg(target_family = "unix")]
    pub fn epilogue(&mut self, n: u32) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.add_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.add_rsp(32 * n),
        }

        self.amd.pop(Amd::RBP);
        self.amd.ret();
        self.predefined_consts();
    }

    #[cfg(target_family = "windows")]
    pub fn epilogue(&mut self, n: u32) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.add_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.add_rsp(32 * n),
        }

        self.amd.mov(Amd::RBP, Amd::RSP, 0x08);
        self.amd.ret();
        self.predefined_consts();
    }

    fn predefined_consts(&mut self) {
        self.align();

        self.set_label("_minus_zero_");
        let u: u64 = unsafe { std::mem::transmute(-0.0f64) };
        self.append_quad(u);

        self.set_label("_one_");
        let u: u64 = unsafe { std::mem::transmute(1.0f64) };
        self.append_quad(u);
    }

    fn align(&mut self) {
        let mut n = self.amd.a.ip();

        while (n & 7) != 0 {
            self.amd.nop();
            n += 1
        }
    }
}
