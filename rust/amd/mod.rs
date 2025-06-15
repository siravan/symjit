use crate::assembler::Assembler;
use crate::generator::{fmod, powi, powi_mod, Generator};
use crate::utils::align_stack;

mod asm;
use asm::{Amd, RoundingMode};

pub enum AmdFamily {
    AvxScalar,
    AvxVector,
    SSEScalar,
}

pub struct AmdGenerator {
    amd: Amd,
    family: AmdFamily,
    r0: Option<u32>,
    mask: u32,
}

impl AmdGenerator {
    pub fn new(family: AmdFamily) -> AmdGenerator {
        AmdGenerator {
            amd: Amd::new(),
            family,
            r0: None,
            mask: if cfg!(target_family = "windows") {
                0x003f
            } else {
                0xffff
            },
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
            self.amd.movsd_xmm_mem(0, Amd::RSP, 32 + i * 8);
            self.amd.call(Amd::RBX);
            self.amd.movsd_mem_xmm(Amd::RSP, 32 + i * 8, 0);
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
            self.amd.movsd_xmm_mem(0, Amd::RSP, 32 + i * 8);
            self.amd.movsd_xmm_mem(1, Amd::RSP, 64 + i * 8);
            self.amd.call(Amd::RBX);
            self.amd.movsd_mem_xmm(Amd::RSP, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, Amd::RSP, 32);
        self.amd.add_rsp(32 * 3);
    }

    fn predefined_consts(&mut self) {
        self.align();

        self.set_label("_minus_zero_");
        self.append_quad((-0.0f64).to_bits());

        self.set_label("_one_");
        self.append_quad(1.0f64.to_bits());

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

    fn flush(&mut self, dst: u8) {
        if dst == 0 {
            if let Some(idx) = self.r0 {
                match self.family {
                    AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(Amd::RSP, (idx * 8) as i32, 0),
                    AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(Amd::RSP, (idx * 32) as i32, 0),
                    AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RSP, (idx * 8) as i32, 0),
                };
            };

            self.r0 = None;
        } else {
            let m = 1 << dst;

            if self.mask & m == 0 {
                // println!("saving reg {}", dst);
                match self.family {
                    AmdFamily::AvxScalar => {
                        self.amd.vmovsd_mem_xmm(Amd::RSP, 8 * (dst as i32), dst)
                    }
                    AmdFamily::AvxVector => {
                        self.amd.vmovpd_mem_ymm(Amd::RSP, 32 * (dst as i32), dst)
                    }
                    AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RSP, 8 * (dst as i32), dst),
                };
            }

            self.mask |= m;
        }
    }

    fn restore_regs(&mut self) {
        let last = self.first_shadow() + self.count_shadows();

        for dst in last..16 {
            let m = 1 << dst;

            if self.mask & m != 0 {
                match self.family {
                    AmdFamily::AvxScalar => {
                        self.amd.vmovsd_xmm_mem(dst, Amd::RSP, 8 * (dst as i32))
                    }
                    AmdFamily::AvxVector => {
                        self.amd.vmovpd_ymm_mem(dst, Amd::RSP, 32 * (dst as i32))
                    }
                    AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, Amd::RSP, 8 * (dst as i32)),
                }
            }
        }
    }

    fn frame_size(&self, cap: u32) -> u32 {
        align_stack(self.reg_size() * cap + 8) - 8
    }
}

impl Generator for AmdGenerator {
    fn first_shadow(&self) -> u8 {
        2
    }

    fn count_shadows(&self) -> u8 {
        if cfg!(target_family = "windows") {
            4
        } else {
            14
        }
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

    fn three_address(&self) -> bool {
        match self.family {
            AmdFamily::AvxScalar => true,
            AmdFamily::AvxVector => true,
            AmdFamily::SSEScalar => false,
        }
    }

    //***********************************
    fn fmov(&mut self, dst: u8, r: u8) {
        if dst == r {
            return;
        }

        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vmovapd(dst, r),
            AmdFamily::SSEScalar => self.amd.movapd(dst, r),
        }
    }

    fn fxchg(&mut self, a: u8, b: u8) {
        self.flush(a);
        self.flush(b);

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
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_label(dst, label),
            AmdFamily::AvxVector => self.amd.vbroadcastsd_label(dst, label),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_label(dst, label),
        }
    }

    fn load_mem(&mut self, dst: u8, idx: u32) {
        self.flush(dst);

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
        if let Some(k) = self.r0 {
            if k == idx {
                match self.family {
                    AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vmovapd(dst, 0),
                    AmdFamily::SSEScalar => {}
                };
                self.r0 = None;
                return;
            }
        };

        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_mem(dst, Amd::RSP, (idx * 8) as i32),
            AmdFamily::AvxVector => self.amd.vmovpd_ymm_mem(dst, Amd::RSP, (idx * 32) as i32),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, Amd::RSP, (idx * 8) as i32),
        }
    }

    fn save_stack(&mut self, src: u8, idx: u32) {
        if src == 0 && !matches!(self.family, AmdFamily::SSEScalar) {
            self.r0 = Some(idx);
            return;
        }

        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(Amd::RSP, (idx * 8) as i32, src),
            AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(Amd::RSP, (idx * 32) as i32, src),
            AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RSP, (idx * 8) as i32, src),
        }
    }

    fn neg(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.load_const(1, "_minus_zero_");
        self.xor(dst, r, 1);
    }

    fn abs(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.load_const(1, "_minus_zero_");
        self.andnot(dst, 1, r);
    }

    fn root(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        match self.family {
            AmdFamily::AvxScalar => self.amd.vsqrtsd(dst, r),
            AmdFamily::AvxVector => self.amd.vsqrtpd(dst, r),
            AmdFamily::SSEScalar => self.amd.sqrtsd(dst, r),
        }
    }

    fn square(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.times(dst, r, r);
    }

    fn cube(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.times(1, r, r);
        self.times(dst, r, 1);
    }

    fn powi(&mut self, dst: u8, r: u8, power: i32) {
        self.flush(dst);
        if power == 0 {
            self.load_const(dst, "_one_");
        } else {
            powi(self, dst, r, power);
        }
    }

    fn powi_mod(&mut self, dst: u8, r: u8, power: i32, modulus: u8) {
        self.flush(dst);
        if power == 0 {
            self.load_const(dst, "_one_");
        } else {
            powi_mod(self, dst, r, power, modulus);
        }
    }

    fn recip(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.load_const(1, "_one_");
        self.divide(dst, 1, r);
    }

    fn round(&mut self, dst: u8, r: u8) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar => self.amd.vroundsd(dst, r, RoundingMode::Round),
            AmdFamily::AvxVector => self.amd.vroundpd(dst, r, RoundingMode::Round),
            AmdFamily::SSEScalar => self.amd.roundsd(dst, r, RoundingMode::Round),
        }
    }

    fn floor(&mut self, dst: u8, r: u8) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar => self.amd.vroundsd(dst, r, RoundingMode::Floor),
            AmdFamily::AvxVector => self.amd.vroundpd(dst, r, RoundingMode::Floor),
            AmdFamily::SSEScalar => self.amd.roundsd(dst, r, RoundingMode::Floor),
        }
    }

    fn ceiling(&mut self, dst: u8, r: u8) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar => self.amd.vroundsd(dst, r, RoundingMode::Ceiling),
            AmdFamily::AvxVector => self.amd.vroundpd(dst, r, RoundingMode::Ceiling),
            AmdFamily::SSEScalar => self.amd.roundsd(dst, r, RoundingMode::Ceiling),
        }
    }

    fn trunc(&mut self, dst: u8, r: u8) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar => self.amd.vroundsd(dst, r, RoundingMode::Trunc),
            AmdFamily::AvxVector => self.amd.vroundpd(dst, r, RoundingMode::Trunc),
            AmdFamily::SSEScalar => self.amd.roundsd(dst, r, RoundingMode::Trunc),
        }
    }

    fn fmod(&mut self, dst: u8, a: u8, b: u8) {
        fmod(self, dst, a, b);
    }

    fn plus(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

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
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vandpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.andpd(x, y);
            }
        }
    }

    fn andnot(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vandnpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, false);
                self.amd.andnpd(x, y);
            }
        }
    }

    fn or(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vorpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.orpd(x, y);
            }
        }
    }

    fn xor(&mut self, dst: u8, a: u8, b: u8) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vxorpd(dst, a, b),
            AmdFamily::SSEScalar => {
                let (x, y) = self.shrink(dst, a, b, true);
                self.amd.xorpd(x, y);
            }
        }
    }

    fn not(&mut self, dst: u8, r: u8) {
        self.flush(dst);
        self.load_const(1, "_all_ones_");
        self.xor(dst, r, 1);
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
        self.flush(dst);
        self.amd.vandpd(dst, cond, a);
    }

    fn select_else(&mut self, dst: u8, cond: u8, a: u8) {
        self.flush(dst);
        self.amd.vandnpd(dst, cond, a);
    }

    #[cfg(target_family = "unix")]
    fn prologue(&mut self, cap: u32) {
        self.amd.push(Amd::RBP);
        self.amd.push(Amd::RBX);
        self.amd.mov(Amd::RBP, Amd::RDI);
        self.amd.sub_rsp(self.frame_size(cap));
    }

    #[cfg(target_family = "unix")]
    fn epilogue(&mut self, cap: u32) {
        self.restore_regs();
        self.vzeroupper();

        self.amd.add_rsp(self.frame_size(cap));

        self.amd.pop(Amd::RBX);
        self.amd.pop(Amd::RBP);
        self.amd.ret();
        self.predefined_consts();
    }

    #[cfg(target_family = "unix")]
    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.amd.push(Amd::RBP);
        self.amd.push(Amd::RBX);

        self.amd.sub_rsp(self.frame_size(cap));

        self.amd.mov(Amd::RBP, Amd::RSP);

        for i in 0..num_args {
            self.amd.movsd_mem_xmm(Amd::RSP, (i * 8) as i32, i as u8);
        }
    }

    #[cfg(target_family = "unix")]
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.restore_regs();
        self.vzeroupper();
        self.amd.movsd_xmm_mem(0, Amd::RSP, 8 * idx_ret);

        self.amd.add_rsp(self.frame_size(cap));

        self.amd.pop(Amd::RBX);
        self.amd.pop(Amd::RBP);
        self.amd.ret();
        self.predefined_consts();
    }

    #[cfg(target_family = "windows")]
    fn prologue(&mut self, cap: u32) {
        self.amd.mov_mem_reg(Amd::RSP, 0x08, Amd::RBP);
        self.amd.mov_mem_reg(Amd::RSP, 0x10, Amd::RBX);
        self.amd.mov(Amd::RBP, Amd::RCX);

        self.amd.sub_rsp(self.frame_size(cap));
    }

    #[cfg(target_family = "windows")]
    fn epilogue(&mut self, cap: u32) {
        self.restore_regs();
        self.vzeroupper();

        self.amd.add_rsp(self.frame_size(cap));

        self.amd.mov_reg_mem(Amd::RBX, Amd::RSP, 0x10);
        self.amd.mov_reg_mem(Amd::RBP, Amd::RSP, 0x08);
        self.amd.ret();
        self.predefined_consts();
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.amd.mov_mem_reg(Amd::RSP, 0x08, Amd::RBP);
        self.amd.mov_mem_reg(Amd::RSP, 0x10, Amd::RBX);

        let frame_size = self.frame_size(cap);
        self.amd.sub_rsp(frame_size);

        self.amd.mov(Amd::RBP, Amd::RSP);

        for i in 0..num_args.min(4) {
            self.amd.movsd_mem_xmm(Amd::RSP, (i * 8) as i32, i as u8);
        }

        for i in 4..num_args {
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            self.amd.movsd_xmm_mem(
                0,
                Amd::RSP,
                (frame_size + self.reg_size() * (4 + 1 + i - 4)) as i32,
            );
            self.amd.movsd_mem_xmm(Amd::RSP, (i * 8) as i32, 0);
        }
    }

    #[cfg(target_family = "windows")]
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.restore_regs();
        self.vzeroupper();
        self.amd.movsd_xmm_mem(0, Amd::RSP, 8 * idx_ret);

        self.amd.add_rsp(self.frame_size(cap));

        self.amd.mov_reg_mem(Amd::RBX, Amd::RSP, 0x10);
        self.amd.mov_reg_mem(Amd::RBP, Amd::RSP, 0x08);
        self.amd.ret();
        self.predefined_consts();
    }
}
