use crate::assembler::Assembler;
use crate::code::BinaryFunc;
use crate::generator::Generator;

mod asm;
use asm::Amd;

pub enum AmdFamily {
    AvxScalar,
    AvxVector,
    SSEScalar,
}

pub struct AmdGenerator {
    amd: Amd,
    family: AmdFamily,
}

impl AmdGenerator {
    pub fn new(family: AmdFamily) -> AmdGenerator {
        AmdGenerator {
            amd: Amd::new(),
            family,
        }
    }

    /*
        shrink is a helper function used to generate
        SSE codes from 3-address inputs.

        IMPORTANT! this function can overwrite the values of
        a and/or b. Therefore, cannot assume a and b are intact
        after calling this function.
    */
    fn shrink(&mut self, dst: u8, a: u8, b: u8, commutative: bool) -> (u8, u8) {
        if dst == a {
            (dst, b)
        } else if dst == b {
            // difficult case: dst == b, dst != a
            if !commutative {
                self.fxchg(a, b);
            };
            (dst, a)
        } else {
            // dst != a, dst != b, a ?= b
            self.fmov(dst, a);
            (dst, b)
        }
    }

    fn vzeroupper(&mut self) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vzeroupper(),
            AmdFamily::SSEScalar => {}
        }
    }

    fn call_vector_unary(&mut self) {
        // reserves 64 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        self.amd.sub_rsp(32 * 2);
        self.amd.vmovpd_mem_ymm(Amd::RSP, 32, 0);

        self.vzeroupper();

        for i in 0..4 {
            self.amd.vmovsd_xmm_mem(0, Amd::RSP, 32 + i * 8);
            self.amd.call(Amd::RBX);
            self.amd.vmovsd_mem_xmm(Amd::RSP, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, Amd::RSP, 32);
        self.amd.add_rsp(32 * 2);
    }

    fn call_vector_binary(&mut self) {
        // reserves 96 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        // 32 bytes to save ymm1
        self.amd.sub_rsp(32 * 3);
        self.amd.vmovpd_mem_ymm(Amd::RSP, 32, 0);
        self.amd.vmovpd_mem_ymm(Amd::RSP, 64, 1);

        self.vzeroupper();

        for i in 0..4 {
            self.amd.vmovsd_xmm_mem(0, Amd::RSP, 32 + i * 8);
            self.amd.vmovsd_xmm_mem(1, Amd::RSP, 64 + i * 8);
            self.amd.call(Amd::RBX);
            self.amd.vmovsd_mem_xmm(Amd::RSP, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, Amd::RSP, 32);
        self.amd.add_rsp(32 * 3);
    }

    fn predefined_consts(&mut self) {
        self.align();

        self.set_label("_minus_zero_");
        let u: u64 = unsafe { std::mem::transmute(-0.0f64) };
        self.append_quad(u);

        self.set_label("_one_");
        let u: u64 = unsafe { std::mem::transmute(1.0f64) };
        self.append_quad(u);

        self.set_label("_all_ones_");
        self.append_quad(0xffffffffffffffff);
    }

    fn align(&mut self) {
        let mut n = self.amd.a.ip();

        while (n & 7) != 0 {
            self.amd.nop();
            n += 1
        }
    }
}

impl Generator for AmdGenerator {
    fn first_shadow(&self) -> u8 {
        return 2;
    }

    fn count_shadows(&self) -> u8 {
        #[cfg(target_family = "windows")]
        return 4;
        #[cfg(not(target_family = "windows"))]
        return 14;
    }

    fn reg_size(&self) -> u32 {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => 8,
            AmdFamily::AvxVector => 32,
        }
    }

    fn a(&mut self) -> &mut Assembler {
        &mut self.amd.a
    }

    //***********************************
    fn fmov(&mut self, dst: u8, r: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vmovapd(dst, r),
            AmdFamily::SSEScalar => self.amd.movapd(dst, r),
        }
    }

    fn fxchg(&mut self, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => {
                self.amd.vxorpd(a, a, b);
                self.amd.vxorpd(b, a, b);
                self.amd.vxorpd(a, a, b);
            }
            AmdFamily::SSEScalar => {
                self.amd.xorpd(a, b);
                self.amd.xorpd(b, a);
                self.amd.xorpd(a, b);
            }
        }
    }

    fn load_const(&mut self, dst: u8, label: &str) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_label(dst, label),
            AmdFamily::AvxVector => self.amd.vbroadcastsd_label(dst, label),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_label(dst, label),
        }
    }

    fn load_mem(&mut self, dst: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_mem(dst, Amd::RBP, (idx * 8) as i32),
            AmdFamily::AvxVector => self.amd.vmovpd_ymm_mem(dst, Amd::RBP, (idx * 32) as i32),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, Amd::RBP, (idx * 8) as i32),
        }
    }

    fn save_mem(&mut self, src: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(Amd::RBP, (idx * 8) as i32, src),
            AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(Amd::RBP, (idx * 32) as i32, src),
            AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RBP, (idx * 8) as i32, src),
        }
    }

    fn load_stack(&mut self, dst: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_mem(dst, Amd::RSP, (idx * 8) as i32),
            AmdFamily::AvxVector => self.amd.vmovpd_ymm_mem(dst, Amd::RSP, (idx * 32) as i32),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, Amd::RSP, (idx * 8) as i32),
        }
    }

    fn save_stack(&mut self, src: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(Amd::RSP, (idx * 8) as i32, src),
            AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(Amd::RSP, (idx * 32) as i32, src),
            AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RSP, (idx * 8) as i32, src),
        }
    }

    fn neg(&mut self, dst: u8) {
        self.load_const(1, "_minus_zero_");
        self.xor(dst, dst, 1);
    }

    fn abs(&mut self, dst: u8) {
        self.load_const(1, "_minus_zero_");
        self.andnot(dst, 1, dst);
    }

    fn root(&mut self, dst: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vsqrtsd(dst, dst),
            AmdFamily::AvxVector => self.amd.vsqrtpd(dst, dst),
            AmdFamily::SSEScalar => self.amd.sqrtsd(dst, dst),
        }
    }

    fn square(&mut self, dst: u8) {
        self.times(dst, dst, dst);
    }

    fn cube(&mut self, dst: u8) {
        self.times(1, dst, dst);
        self.times(dst, dst, 1);
    }

    fn recip(&mut self, dst: u8) {
        self.load_const(1, "_one_");
        self.divide(dst, 1, dst);
    }

    fn plus(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vaddsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vaddpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.addsd(x, y);
            }
        }
    }

    fn minus(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vsubsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vsubpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.subsd(x, y);
            }
        }
    }

    fn times(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmulsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vmulpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.mulsd(x, y);
            }
        }
    }

    fn divide(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vdivsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vdivpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.divsd(x, y);
            }
        }
    }

    fn gt(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpnlesd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpnlepd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpnlesd(x, y);
            }
        }
    }

    fn geq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpnltsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpnltpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpnltsd(x, y);
            }
        }
    }

    fn lt(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpltsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpltpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpltsd(x, y);
            }
        }
    }

    fn leq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmplesd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmplepd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmplesd(x, y);
            }
        }
    }

    fn eq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpeqsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpeqpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpeqsd(x, y);
            }
        }
    }

    fn neq(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vcmpneqsd(dst, a, b),
            AmdFamily::AvxVector => self.amd.vcmpneqpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.cmpneqsd(x, y);
            }
        }
    }

    fn and(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vandpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.andpd(x, y);
            }
        }
    }

    fn andnot(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vandnpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.andnpd(x, y);
            }
        }
    }

    fn or(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vorpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.orpd(x, y);
            }
        }
    }

    fn xor(&mut self, dst: u8, a: u8, b: u8) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vxorpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.xorpd(x, y);
            }
        }
    }

    fn not(&mut self, dst: u8) {
        self.load_const(1, "_all_ones_");
        self.xor(dst, dst, 1);
    }

    fn call(&mut self, label: &str, num_args: usize) {
        self.amd.mov_reg_label(Amd::RBX, label);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => {
                self.vzeroupper();
                #[cfg(target_family = "windows")]
                self.amd.sub_rsp(32);

                self.amd.call(Amd::RBX);

                #[cfg(target_family = "windows")]
                self.amd.add_rsp(32);
            }
            AmdFamily::AvxVector => match num_args {
                1 => self.call_vector_unary(),
                2 => self.call_vector_binary(),
                _ => {
                    panic!("invalid number of arguments")
                }
            },
        }
    }
    /*
        fn branch(&mut self, label: &str) {
            self.amd.jmp(label);
        }

        fn branch_if(&mut self, cond: u8, true_label: &str) {
            self.amd.vucomisd(cond, cond);
            self.amd.jpe(true_label);
        }

        fn branch_if_else(&mut self, cond: u8, true_label: &str, false_label: &str) {
            self.amd.vucomisd(cond, cond);
            self.amd.jpe(true_label);
            self.amd.jmp(false_label);
        }
    */
    fn select_if(&mut self, dst: u8, cond: u8, a: u8) {
        self.amd.vandpd(dst, cond, a);
    }

    fn select_else(&mut self, dst: u8, cond: u8, a: u8) {
        self.amd.vandnpd(dst, cond, a);
    }

    #[cfg(target_family = "unix")]
    fn prologue(&mut self, n: u32) {
        self.amd.push(Amd::RBP);
        self.amd.push(Amd::RBX);
        self.amd.mov(Amd::RBP, Amd::RDI);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.sub_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.sub_rsp(32 * n),
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue(&mut self, n: u32) {
        self.amd.mov_mem_reg(Amd::RSP, 0x08, Amd::RBP);
        self.amd.mov_mem_reg(Amd::RSP, 0x10, Amd::RBX);
        self.amd.mov(Amd::RBP, Amd::RCX);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.sub_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.sub_rsp(32 * n),
        }
    }

    #[cfg(target_family = "unix")]
    fn epilogue(&mut self, n: u32) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.add_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.add_rsp(32 * n),
        }

        self.amd.pop(Amd::RBX);
        self.amd.pop(Amd::RBP);
        self.amd.ret();
        self.predefined_consts();
    }

    #[cfg(target_family = "windows")]
    fn epilogue(&mut self, n: u32) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => self.amd.add_rsp(8 * n),
            AmdFamily::AvxVector => self.amd.add_rsp(32 * n),
        }

        self.amd.mov(Amd::RBX, Amd::RSP, 0x10);
        self.amd.mov(Amd::RBP, Amd::RSP, 0x08);
        self.amd.ret();
        self.predefined_consts();
    }
}
