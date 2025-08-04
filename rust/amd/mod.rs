use crate::assembler::Assembler;
use crate::generator::{
    fmod, powi, powi_mod, setup_call_binary, setup_call_unary, Generator, RET, TEMP,
};
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

const MEM: u8 = Amd::RBP;
const STATES: u8 = Amd::R13;
const IDX: u8 = Amd::R12;
const PARAMS: u8 = Amd::RBX;

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
    fn shrink(&mut self, dst: u8, s1: u8, s2: u8, commutative: bool) -> (u8, u8) {
        if dst == s1 {
            (dst, s2)
        } else if dst == s2 {
            // difficult case: dst == b, dst != a
            if !commutative {
                self.fxchg(s1, s2);
            };
            (dst, s1)
        } else {
            // dst != a, dst != b, a ?= b
            self.fmov(dst, s1);
            (dst, s2)
        }
    }

    fn vzeroupper(&mut self) {
        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vzeroupper(),
            AmdFamily::SSEScalar => {}
        }
    }

    fn call_vector_unary(&mut self, label: &str) {
        // reserves 64 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save ymm0
        self.amd.sub_rsp(32 * 2);
        self.amd.vmovpd_mem_ymm(Amd::RSP, 32, 0);

        self.vzeroupper();

        for i in 0..4 {
            self.amd.movsd_xmm_mem(0, Amd::RSP, 32 + i * 8);
            //self.amd.call(Amd::R12);
            self.amd.call_indirect(label);
            self.amd.movsd_mem_xmm(Amd::RSP, 32 + i * 8, 0);
        }

        self.amd.vmovpd_ymm_mem(0, Amd::RSP, 32);
        self.amd.add_rsp(32 * 2);
    }

    fn call_vector_binary(&mut self, label: &str) {
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
            //self.amd.call(Amd::R12);
            self.amd.call_indirect(label);
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
        if dst == RET {
            if let Some(idx) = self.r0 {
                match self.family {
                    AmdFamily::AvxScalar => {
                        self.amd.vmovsd_mem_xmm(Amd::RSP, (idx * 8) as i32, RET)
                    }
                    AmdFamily::AvxVector => {
                        self.amd.vmovpd_mem_ymm(Amd::RSP, (idx * 32) as i32, RET)
                    }
                    AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RSP, (idx * 8) as i32, RET),
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

    fn save_nonvolatile_regs(&mut self) {
        if cfg!(target_family = "windows") {
            self.amd.mov_mem_reg(Amd::RSP, 0x08, MEM);
            self.amd.mov_mem_reg(Amd::RSP, 0x10, PARAMS);
            self.amd.mov_mem_reg(Amd::RSP, 0x18, IDX);
            self.amd.mov_mem_reg(Amd::RSP, 0x20, Amd::R13);
        } else {
            self.amd.sub_rsp(32);
            self.amd.mov_mem_reg(Amd::RSP, 0x00, MEM);
            self.amd.mov_mem_reg(Amd::RSP, 0x08, PARAMS);
            self.amd.mov_mem_reg(Amd::RSP, 0x10, IDX);
            self.amd.mov_mem_reg(Amd::RSP, 0x18, STATES);
        }
    }

    fn load_nonvolatile_regs(&mut self) {
        if cfg!(target_family = "windows") {
            self.amd.mov_reg_mem(STATES, Amd::RSP, 0x20);
            self.amd.mov_reg_mem(IDX, Amd::RSP, 0x18);
            self.amd.mov_reg_mem(PARAMS, Amd::RSP, 0x10);
            self.amd.mov_reg_mem(MEM, Amd::RSP, 0x08);
        } else {
            self.amd.mov_reg_mem(STATES, Amd::RSP, 0x18);
            self.amd.mov_reg_mem(IDX, Amd::RSP, 0x10);
            self.amd.mov_reg_mem(PARAMS, Amd::RSP, 0x08);
            self.amd.mov_reg_mem(MEM, Amd::RSP, 0x00);
            self.amd.add_rsp(32);
        }
    }
}

macro_rules! binop {
    ($self:ident, $sse:ident, $avx:ident, $simd:ident, $dst:expr, $s1: expr, $s2: expr, $com:ident) => {
        $self.flush($dst);

        match $self.family {
            AmdFamily::AvxScalar => $self.amd.$avx($dst, $s1, $s2),
            AmdFamily::AvxVector => $self.amd.$simd($dst, $s1, $s2),
            AmdFamily::SSEScalar => {
                let (x, y) = $self.shrink($dst, $s1, $s2, $com);
                $self.amd.$sse(x, y);
            }
        }
    };
}

macro_rules! roundop {
    ($self:ident, $dst:expr, $s1: expr, $mode: expr) => {
        $self.flush($dst);

        match $self.family {
            AmdFamily::AvxScalar => $self.amd.vroundsd($dst, $s1, $mode),
            AmdFamily::AvxVector => $self.amd.vroundpd($dst, $s1, $mode),
            AmdFamily::SSEScalar => $self.amd.roundsd($dst, $s1, $mode),
        }
    };
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
    fn fmov(&mut self, dst: u8, s1: u8) {
        if dst == s1 {
            return;
        }

        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vmovapd(dst, s1),
            AmdFamily::SSEScalar => self.amd.movapd(dst, s1),
        }
    }

    fn fxchg(&mut self, s1: u8, s2: u8) {
        self.flush(s1);
        self.flush(s2);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::AvxVector => {
                self.amd.vxorpd(s1, s1, s2);
                self.amd.vxorpd(s2, s1, s2);
                self.amd.vxorpd(s1, s1, s2);
            }
            AmdFamily::SSEScalar => {
                self.amd.xorpd(s1, s2);
                self.amd.xorpd(s2, s1);
                self.amd.xorpd(s1, s2);
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
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_mem(dst, MEM, (idx * 8) as i32),
            AmdFamily::AvxVector => self.amd.vmovpd_ymm_mem(dst, MEM, (idx * 32) as i32),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, MEM, (idx * 8) as i32),
        }
    }

    fn save_mem(&mut self, dst: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(MEM, (idx * 8) as i32, dst),
            AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(MEM, (idx * 32) as i32, dst),
            AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(MEM, (idx * 8) as i32, dst),
        }
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(RET, idx);
    }

    fn load_param(&mut self, dst: u8, idx: u32) {
        self.flush(dst);

        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_xmm_mem(dst, PARAMS, (idx * 8) as i32),
            AmdFamily::AvxVector => self.amd.vbroadcastsd(dst, PARAMS, (idx * 8) as i32),
            AmdFamily::SSEScalar => self.amd.movsd_xmm_mem(dst, PARAMS, (idx * 8) as i32),
        }
    }

    fn load_stack(&mut self, dst: u8, idx: u32) {
        if let Some(k) = self.r0 {
            if k == idx {
                match self.family {
                    AmdFamily::AvxScalar | AmdFamily::AvxVector => self.amd.vmovapd(dst, RET),
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

    fn save_stack(&mut self, dst: u8, idx: u32) {
        match self.family {
            AmdFamily::AvxScalar => self.amd.vmovsd_mem_xmm(Amd::RSP, (idx * 8) as i32, dst),
            AmdFamily::AvxVector => self.amd.vmovpd_mem_ymm(Amd::RSP, (idx * 32) as i32, dst),
            AmdFamily::SSEScalar => self.amd.movsd_mem_xmm(Amd::RSP, (idx * 8) as i32, dst),
        }
    }

    fn save_stack_result(&mut self, idx: u32) {
        if !matches!(self.family, AmdFamily::SSEScalar) {
            assert!(self.r0.is_none());
            self.r0 = Some(idx);
            return;
        }

        self.save_stack(RET, idx);
    }

    fn neg(&mut self, dst: u8, s1: u8) {
        self.flush(dst);
        self.load_const(TEMP, "_minus_zero_");
        self.xor(dst, s1, TEMP);
    }

    fn abs(&mut self, dst: u8, s1: u8) {
        self.flush(dst);
        self.load_const(TEMP, "_minus_zero_");
        self.andnot(dst, TEMP, s1);
    }

    fn root(&mut self, dst: u8, s1: u8) {
        self.flush(dst);
        match self.family {
            AmdFamily::AvxScalar => self.amd.vsqrtsd(dst, s1),
            AmdFamily::AvxVector => self.amd.vsqrtpd(dst, s1),
            AmdFamily::SSEScalar => self.amd.sqrtsd(dst, s1),
        }
    }

    fn square(&mut self, dst: u8, s1: u8) {
        self.flush(dst);
        self.times(dst, s1, s1);
    }

    fn cube(&mut self, dst: u8, s1: u8) {
        self.flush(dst);
        self.times(TEMP, s1, s1);
        self.times(dst, s1, TEMP);
    }

    fn powi(&mut self, dst: u8, s1: u8, power: i32) {
        self.flush(dst);
        if power == 0 {
            self.load_const(dst, "_one_");
        } else {
            powi(self, dst, s1, power);
        }
    }

    fn powi_mod(&mut self, dst: u8, s1: u8, power: i32, modulus: u8) {
        self.flush(dst);
        if power == 0 {
            self.load_const(dst, "_one_");
        } else {
            powi_mod(self, dst, s1, power, modulus);
        }
    }

    fn recip(&mut self, dst: u8, s1: u8) {
        self.flush(dst);
        self.load_const(TEMP, "_one_");
        self.divide(dst, TEMP, s1);
    }

    fn round(&mut self, dst: u8, s1: u8) {
        roundop!(self, dst, s1, RoundingMode::Round);
    }

    fn floor(&mut self, dst: u8, s1: u8) {
        roundop!(self, dst, s1, RoundingMode::Floor);
    }

    fn ceiling(&mut self, dst: u8, s1: u8) {
        roundop!(self, dst, s1, RoundingMode::Ceiling);
    }

    fn trunc(&mut self, dst: u8, s1: u8) {
        roundop!(self, dst, s1, RoundingMode::Trunc);
    }

    fn fmod(&mut self, dst: u8, s1: u8, s2: u8) {
        fmod(self, dst, s1, s2);
    }

    fn plus(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, addsd, vaddsd, vaddpd, dst, s1, s2, true);
    }

    fn minus(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, subsd, vsubsd, vsubpd, dst, s1, s2, false);
    }

    fn times(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, mulsd, vmulsd, vmulpd, dst, s1, s2, true);
    }

    fn divide(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, divsd, vdivsd, vdivpd, dst, s1, s2, false);
    }

    fn gt(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, cmpnlesd, vcmpnlesd, vcmpnlepd, dst, s1, s2, false);
    }

    fn geq(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, cmpnltsd, vcmpnltsd, vcmpnltpd, dst, s1, s2, false);
    }

    fn lt(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, cmpltsd, vcmpltsd, vcmpltpd, dst, s1, s2, false);
    }

    fn leq(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, cmplesd, vcmplesd, vcmplepd, dst, s1, s2, false);
    }

    fn eq(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, cmpeqsd, vcmpeqsd, vcmpeqpd, dst, s1, s2, true);
    }

    fn neq(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, cmpneqsd, vcmpneqsd, vcmpneqpd, dst, s1, s2, true);
    }

    fn and(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, andpd, vandpd, vandpd, dst, s1, s2, true);
    }

    fn andnot(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, andnpd, vandnpd, vandnpd, dst, s1, s2, false);
    }

    fn or(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, orpd, vorpd, vorpd, dst, s1, s2, true);
    }

    fn xor(&mut self, dst: u8, s1: u8, s2: u8) {
        binop!(self, xorpd, vxorpd, vxorpd, dst, s1, s2, true);
    }

    fn not(&mut self, dst: u8, s1: u8) {
        self.flush(dst);
        self.load_const(TEMP, "_all_ones_");
        self.xor(dst, s1, TEMP);
    }

    fn setup_call_unary(&mut self, s1: u8) {
        setup_call_unary(self, s1);
    }

    fn setup_call_binary(&mut self, s1: u8, s2: u8) {
        setup_call_binary(self, s1, s2);
    }

    fn call(&mut self, label: &str, num_args: usize) {
        //self.amd.mov_reg_label(Amd::R12, label);

        match self.family {
            AmdFamily::AvxScalar | AmdFamily::SSEScalar => {
                self.vzeroupper();
                #[cfg(target_family = "windows")]
                self.amd.sub_rsp(32);

                //self.amd.call(Amd::R12);
                self.amd.call_indirect(label);

                #[cfg(target_family = "windows")]
                self.amd.add_rsp(32);
            }
            AmdFamily::AvxVector => match num_args {
                1 => self.call_vector_unary(label),
                2 => self.call_vector_binary(label),
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
    fn select_if(&mut self, dst: u8, cond: u8, s1: u8) {
        self.flush(dst);
        self.amd.vandpd(dst, cond, s1);
    }

    fn select_else(&mut self, dst: u8, cond: u8, s1: u8) {
        self.flush(dst);
        self.amd.vandnpd(dst, cond, s1);
    }

    /****************** Prologues/Epilogues ********************/

    #[cfg(target_family = "unix")]
    fn prologue(&mut self, cap: u32) {
        self.amd.sub_rsp(32);
        self.save_nonvolatile_regs();
        self.amd.mov(MEM, Amd::RDI);
        self.amd.mov(PARAMS, Amd::RSI);
        self.amd.sub_rsp(self.frame_size(cap));
    }

    #[cfg(target_family = "unix")]
    fn epilogue(&mut self, cap: u32) {
        self.restore_regs();
        self.vzeroupper();

        self.amd.add_rsp(self.frame_size(cap));
        self.load_nonvolatile_regs();
        self.amd.add_rsp(32);
        self.amd.ret();

        self.predefined_consts();
    }

    #[cfg(target_family = "unix")]
    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.amd.push(MEM);
        self.amd.push(PARAMS);
        self.amd.sub_rsp(self.frame_size(cap));
        self.amd.mov(MEM, Amd::RSP);

        for i in 0..num_args {
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
        }
    }

    #[cfg(target_family = "unix")]
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.restore_regs();
        self.vzeroupper();
        self.amd.movsd_xmm_mem(0, Amd::RSP, 8 * idx_ret);

        self.amd.add_rsp(self.frame_size(cap));
        self.amd.pop(PARAMS);
        self.amd.pop(MEM);
        self.amd.ret();
        self.predefined_consts();
    }

    /*
     * prologue_indirect generates the stack frame. It works in two modes:
     *  Direct mode: MEM (state variables + obs) is passed directly as the first argument. The second argument is null.
     *  Indirect mode: the second argument is a pointer to an array of pointers to states and obs. The third argument
     *      is the index into these arrays. MEM is allocated on the stack and filled based on the second and thirds args.
     *
     * Noth that the second argument determines whether it is the direct (args[1] == null) or indirect mode.
     * In both modes, the fourth argument points to an array of params.
     */
    fn prologue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        let win = cfg!(target_family = "windows");
        self.save_nonvolatile_regs();

        self.amd.mov(MEM, if win { Amd::RCX } else { Amd::RDI }); // first arg = mem if direct mode, otherwise null
        self.amd.mov(STATES, if win { Amd::RDX } else { Amd::RSI }); // second arg = states+obs if indirect mode, otherwise null
        self.amd.mov(IDX, if win { Amd::R8 } else { Amd::RDX }); // third arg = index if indirect mode
        self.amd.mov(PARAMS, if win { Amd::R9 } else { Amd::RCX }); // fourth arg = params

        self.amd.or(STATES, STATES);
        self.amd.jz("@main");

        let size = (count_states + count_obs + 1) as u32 * self.reg_size();
        self.amd.sub_rsp(size);
        self.amd.mov(MEM, Amd::RSP); // in indirect mode, MEM is allocated on the stack

        for i in 0..count_states {
            self.amd.mov_reg_mem(Amd::RAX, STATES, 8 * i as i32);
            let k = i as u32 * self.reg_size();

            match self.family {
                AmdFamily::AvxScalar => {
                    self.amd.vmovsd_xmm_indexed(0, Amd::RAX, IDX, 8);
                    self.amd.vmovsd_mem_xmm(MEM, k as i32, 0);
                }
                AmdFamily::AvxVector => {
                    self.amd.vmovpd_ymm_indexed(0, Amd::RAX, IDX, 8);
                    self.amd.vmovpd_mem_ymm(MEM, k as i32, 0);
                }
                AmdFamily::SSEScalar => {
                    self.amd.movsd_xmm_indexed(0, Amd::RAX, IDX, 8);
                    self.amd.movsd_mem_xmm(MEM, k as i32, 0);
                }
            }
        }

        // may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");
        self.amd.sub_rsp(self.frame_size(cap));
    }

    fn epilogue_indirect(&mut self, cap: u32, count_states: usize, count_obs: usize) {
        self.restore_regs();

        self.amd.add_rsp(self.frame_size(cap));

        self.amd.or(STATES, STATES);
        self.amd.jz("@done");

        let size = (count_states + count_obs + 1) as u32 * self.reg_size();

        for i in 0..count_obs {
            self.amd
                .mov_reg_mem(Amd::RAX, STATES, 8 * (count_states + i) as i32);
            let k = (count_states + i + 1) as u32 * self.reg_size();

            match self.family {
                AmdFamily::AvxScalar => {
                    self.amd.vmovsd_xmm_mem(0, MEM, k as i32);
                    self.amd.vmovsd_indexed_xmm(Amd::RAX, IDX, 8, 0);
                }
                AmdFamily::AvxVector => {
                    self.amd.vmovpd_ymm_mem(0, MEM, k as i32);
                    self.amd.vmovpd_indexed_ymm(Amd::RAX, IDX, 8, 0);
                }
                AmdFamily::SSEScalar => {
                    self.amd.movsd_xmm_mem(0, MEM, k as i32);
                    self.amd.movsd_indexed_xmm(Amd::RAX, IDX, 8, 0);
                }
            }
        }

        self.amd.add_rsp(size);
        self.set_label("@done");

        self.vzeroupper();

        self.load_nonvolatile_regs();
        self.amd.ret();

        self.predefined_consts();
    }

    #[cfg(target_family = "windows")]
    fn prologue(&mut self, cap: u32) {
        self.save_nonvolatile_regs();
        self.amd.mov(MEM, Amd::RCX);
        self.amd.mov(PARAMS, Amd::RDX);
        self.amd.sub_rsp(self.frame_size(cap));
    }

    #[cfg(target_family = "windows")]
    fn epilogue(&mut self, cap: u32) {
        self.restore_regs();
        self.vzeroupper();

        self.amd.add_rsp(self.frame_size(cap));
        self.load_nonvolatile_regs();
        self.amd.ret();
        self.predefined_consts();
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: u32, num_args: u32) {
        self.amd.mov_mem_reg(Amd::RSP, 0x08, MEM);
        self.amd.mov_mem_reg(Amd::RSP, 0x10, PARAMS);

        let frame_size = self.frame_size(cap);
        self.amd.sub_rsp(frame_size);
        self.amd.mov(MEM, Amd::RSP);

        for i in 0..num_args.min(4) {
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
        }

        for i in 4..num_args {
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            self.amd.movsd_xmm_mem(
                0,
                MEM,
                (frame_size + self.reg_size() * (4 + 1 + i - 4)) as i32,
            );
            self.amd.movsd_mem_xmm(MEM, (i * 8) as i32, 0);
        }
    }

    #[cfg(target_family = "windows")]
    fn epilogue_fast(&mut self, cap: u32, idx_ret: i32) {
        self.restore_regs();
        self.vzeroupper();
        self.amd.movsd_xmm_mem(0, MEM, 8 * idx_ret);

        self.amd.add_rsp(self.frame_size(cap));
        self.amd.mov_reg_mem(PARAMS, Amd::RSP, 0x10);
        self.amd.mov_reg_mem(MEM, Amd::RSP, 0x08);
        self.amd.ret();
        self.predefined_consts();
    }
}
