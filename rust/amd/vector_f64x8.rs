use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::{FuncletType, Generator};
use crate::symbol::Loc;
use crate::utils::align_stack;
use crate::utils::{is_external_func, reg, DataType, Reg};
use anyhow::{anyhow, Result};

use super::asm::{Amd, RoundingMode};
use super::*;

const REG_SIZE: u32 = 32 * 2;
const NUM_LANES: u32 = 8;

pub struct AmdVectorF64x8Generator {
    amd: Amd,
    config: Config,
    last_load: usize,
}

impl AmdVectorF64x8Generator {
    pub fn new(config: Config) -> AmdVectorF64x8Generator {
        AmdVectorF64x8Generator {
            amd: Amd::new(DataType::F64),
            config,
            last_load: 0,
        }
    }

    fn append_quad(&mut self, u: u64) {
        self.amd.a.append_quad(u);
    }

    fn apply_jumps(&mut self) {
        self.amd.a.apply_jumps();
    }

    fn load_const_by_name(&mut self, dst: Reg, label: &str) {
        amd! {vbroadcastsd zmm(ϕ(dst)), label; self.amd};
    }

    fn call_vector_unary(&mut self, label: &str) {
        amd! {vmovupd [r(STACK) + REG_SIZE], zmm(0); self.amd};
        amd! {vzeroupper; self.amd};

        for i in 0..NUM_LANES {
            if i > 0 {
                amd! {vmovsd xmm(0), [r(STACK) + REG_SIZE + i * 8]; self.amd};
            }

            amd! {call label; self.amd};
            amd! {vmovsd [r(STACK) + REG_SIZE + i * 8], xmm(0); self.amd};
        }

        amd! {vmovupd zmm(0), [r(STACK) + REG_SIZE]; self.amd};
    }

    fn call_vector_binary(&mut self, label: &str) {
        // reserves 96 bytes in the stack
        // 32 bytes for shadow store (mandatory in Windows)
        // 32 bytes to save zmm0
        // 32 bytes to save zmm1
        amd! {vmovupd [r(STACK) + REG_SIZE], zmm(0); self.amd};
        amd! {vmovupd [r(STACK) + REG_SIZE * 2], zmm(1); self.amd};
        amd! {vzeroupper; self.amd};

        for i in 0..NUM_LANES {
            if i > 0 {
                amd! {vmovsd xmm(0), [r(STACK) + REG_SIZE + i * 8]; self.amd};
                amd! {vmovsd xmm(1), [r(STACK) + REG_SIZE * 2 + i * 8]; self.amd};
            }

            amd! {call label; self.amd};
            amd! {vmovsd [r(STACK) + REG_SIZE + i * 8], xmm(0); self.amd};
        }

        amd! {vmovupd zmm(0), [r(STACK) + REG_SIZE]; self.amd};
    }

    fn call_complex_vector_unary(&mut self, label: &str) {
        amd! {vmovupd [r(STACK) + REG_SIZE * 2], zmm(0); self.amd};
        amd! {vmovupd [r(STACK) + REG_SIZE * 3], zmm(1); self.amd};
        amd! {vzeroupper; self.amd};

        for i in 0..NUM_LANES {
            if i > 0 {
                amd! {vmovsd xmm(0), [r(STACK) + REG_SIZE * 2 + i * 8]; self.amd};
                amd! {vmovsd xmm(1), [r(STACK) + REG_SIZE * 3 + i * 8]; self.amd};
            }

            if cfg!(target_family = "windows") {
                amd! {lea r(Amd::R8), [r(STACK) + REG_SIZE]; self.amd};
            } else {
                amd! {lea r(Amd::RDI), [r(STACK) + REG_SIZE]; self.amd};
            }

            amd! {call label; self.amd};
            amd! {vmovsd xmm(0), [r(STACK) + REG_SIZE]; self.amd};
            amd! {vmovsd xmm(1), [r(STACK) + REG_SIZE + 8]; self.amd};
            amd! {vmovsd [r(STACK) + REG_SIZE * 2 + i * 8], xmm(0); self.amd};
            amd! {vmovsd [r(STACK) + REG_SIZE * 3 + i * 8], xmm(1); self.amd};
        }

        amd! {vmovupd zmm(0), [r(STACK) + REG_SIZE * 2]; self.amd};
        amd! {vmovupd zmm(1), [r(STACK) + REG_SIZE * 3]; self.amd};
    }

    fn call_complex_vector_binary(&mut self, label: &str) {
        amd! {vmovupd [r(STACK) + REG_SIZE * 2], zmm(0); self.amd};
        amd! {vmovupd [r(STACK) + REG_SIZE * 3], zmm(1); self.amd};
        amd! {vmovupd [r(STACK) + REG_SIZE * 4], zmm(2); self.amd};
        amd! {vmovupd [r(STACK) + REG_SIZE * 5], zmm(3); self.amd};
        amd! {vzeroupper; self.amd};

        for i in 0..NUM_LANES {
            if i > 0 {
                amd! {vmovsd xmm(0), [r(STACK) + REG_SIZE * 2 + i * 8]; self.amd};
                amd! {vmovsd xmm(1), [r(STACK) + REG_SIZE * 3 + i * 8]; self.amd};
                amd! {vmovsd xmm(2), [r(STACK) + REG_SIZE * 4 + i * 8]; self.amd};
                amd! {vmovsd xmm(3), [r(STACK) + REG_SIZE * 5 + i * 8]; self.amd};
            }

            amd! {vmovsd [r(STACK) + REG_SIZE], xmm(2); self.amd};
            amd! {vmovsd [r(STACK) + REG_SIZE + 8], xmm(3); self.amd};

            if cfg!(target_family = "windows") {
                amd! {lea r(Amd::R8), [r(STACK) + 32]; self.amd};
            } else {
                amd! {lea r(Amd::RDI), [r(STACK) + 32]; self.amd};
            }

            amd! {call label; self.amd};
            amd! {vmovsd xmm(0), [r(STACK) + REG_SIZE]; self.amd};
            amd! {vmovsd xmm(1), [r(STACK) + REG_SIZE + 8]; self.amd};
            amd! {vmovsd [r(STACK) + REG_SIZE * 2 + i * 8], xmm(0); self.amd};
            amd! {vmovsd [r(STACK) + REG_SIZE * 3 + i * 8], xmm(1); self.amd};
        }

        amd! {vmovupd zmm(0), [r(STACK) + REG_SIZE * 2]; self.amd};
        amd! {vmovupd zmm(1), [r(STACK) + REG_SIZE * 3]; self.amd};
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = SPILL_AREA as u32;

        amd! {mov r(ARGS[0]), [&format!("_env_{}_", op)]; self.amd};
        amd! {lea r(ARGS[1]), [r(STACK) + cap * REG_SIZE]; self.amd};
        amd! {mov r(ARGS[2]), num_args; self.amd};
        amd! {lea r(ARGS[3]), [r(STACK) + 4 * REG_SIZE]; self.amd};
        amd! {vzeroupper; self.amd};
        amd! {call &format!("_simd_{}_", op); self.amd};

        if self.config.is_complex() {
            let l1 = format!(".P{}", self.amd.a.ip());
            let l2 = format!(".Q{}", self.amd.a.ip());

            amd! {or r(Amd::RAX), r(Amd::RAX); self.amd};
            amd! {jz &l1; self.amd};

            amd! {vmovupd zmm(2), [r(STACK) + 4 * REG_SIZE]; self.amd};
            amd! {vmovupd zmm(3), [r(STACK) + 5 * REG_SIZE]; self.amd};
            amd! {vshufpd zmm(0), zmm(2), zmm(3), 0; self.amd};
            amd! {vshufpd zmm(1), zmm(2), zmm(3), 0; self.amd};

            amd! {jmp &l2; self.amd};
            self.set_label(&l1);

            amd! {vmovupd zmm(0), [r(STACK) + 4 * REG_SIZE]; self.amd};
            amd! {vmovupd zmm(1), [r(STACK) + 5 * REG_SIZE]; self.amd};

            self.set_label(&l2);
        } else {
            amd! {vmovupd zmm(0), [r(STACK) + 4 * REG_SIZE]; self.amd};
        }

        Ok(())
    }

    fn predefined_consts(&mut self) {
        self.align();
        predefined_consts(&mut self.amd);
    }
}

impl Generator for AmdVectorF64x8Generator {
    fn bytes(&mut self) -> Vec<u8> {
        self.amd.a.bytes()
    }

    fn count_shadows(&self) -> u8 {
        if cfg!(target_family = "windows") {
            4 // xmm2-xmm5
        } else {
            14 // xmm2-xmm15
        }
    }

    fn three_address(&self) -> bool {
        true
    }

    fn support_funclet(&self) -> FuncletType {
        FuncletType::Complex
    }

    fn seal(&mut self) {
        self.predefined_consts();
        self.apply_jumps();
    }

    fn align(&mut self) {
        let mut n = self.amd.a.ip();

        while (n & 7) != 0 {
            amd! {nop; self.amd};
            n += 1
        }
    }

    fn set_label(&mut self, label: &str) {
        self.amd.a.set_label(label);
    }

    fn branch(&mut self, label: &str) {
        amd! {jmp label; self.amd};
    }

    /// jump to label if all bits of cond == is_else
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        amd! {vpmovq2m k(1), zmm(ϕ(cond)); self.amd};
        amd! {kmovw r(Amd::RAX), k(1); self.amd};
        amd! {and r(Amd::RAX), (1 << NUM_LANES) - 1; self.amd};

        if is_else {
            amd! {cmp r(Amd::RAX), (1 << NUM_LANES) - 1; self.amd};
        }

        amd! {jz label; self.amd};

        if !self.config.simd_branch() {
            amd! {or r(Amd::RAX), r(Amd::RAX); self.amd};
            amd! {jnz "@epilogue"; self.amd};
        }
    }

    fn fuse_load_math(&mut self) {
        fuse_load_math(&mut self.amd, self.last_load);
    }

    //***********************************

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst != s1 {
            amd! {vmovapd zmm(ϕ(dst)), zmm(ϕ(s1)); self.amd};
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        amd! {vxorpd zmm(ϕ(s1)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vxorpd zmm(ϕ(s2)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vxorpd zmm(ϕ(s1)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        let label = format!("_const_{}_", idx);
        amd! {vbroadcastsd zmm(ϕ(dst)), label.as_str(); self.amd};
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        amd! {vmovupd zmm(ϕ(dst)), [r(MEM) + idx * REG_SIZE]; self.amd};
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        amd! {vmovupd [r(MEM) + idx * REG_SIZE], zmm(ϕ(dst)); self.amd};
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();

        if self.config.symbolica() {
            amd! {vmovupd zmm(ϕ(dst)), [r(PARAMS) + idx * REG_SIZE]; self.amd};
        } else {
            amd! {vbroadcastsd zmm(ϕ(dst)), [r(PARAMS) + 8 * idx]; self.amd};
        }
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        amd! {vmovupd zmm(ϕ(dst)), [r(STACK) + idx * REG_SIZE]; self.amd};
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        amd! {vmovupd [r(STACK) + idx * REG_SIZE], zmm(ϕ(dst)); self.amd};
    }

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_mem(xd, idx);
        self.load_mem(yd, idx + 1);
    }

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save_mem(xs, idx);
        self.save_mem(ys, idx + 1);
    }

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_param(xd, idx);
        self.load_param(yd, idx + 1);
    }

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_stack(xd, idx);
        self.load_stack(yd, idx + 1);
    } /* */

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save_stack(xs, idx);
        self.save_stack(ys, idx + 1);
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_minus_zero_");
        self.xor(dst, s1, Reg::Temp);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_minus_zero_");
        self.andnot(dst, Reg::Temp, s1);
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        amd! {vsqrtpd zmm(ϕ(dst)), zmm(ϕ(s1)); self.amd};
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.root(dst, s1);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_one_");
        self.divide(dst, Reg::Temp, s1);
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_half_");
        self.times(dst, s1, Reg::Temp);
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd zmm(ϕ(dst)), zmm(ϕ(s1)), RoundingMode::Round; self.amd};
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd zmm(ϕ(dst)), zmm(ϕ(s1)), RoundingMode::Floor; self.amd};
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd zmm(ϕ(dst)), zmm(ϕ(s1)), RoundingMode::Ceiling; self.amd};
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd zmm(ϕ(dst)), zmm(ϕ(s1)), RoundingMode::Trunc; self.amd};
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vaddpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vsubpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vmulpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vdivpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let xt = Reg::Gen(2);
        let yt = Reg::Gen(3);

        if xd != x1 && xd != x2 {
            self.times(xd, y1, y2);
            self.fused_mul_sub(xd, x1, x2, xd);
            self.times(yd, x1, y2);
            self.fused_mul_add(yd, x2, y1, yd);
        } else if xd == x1 && xd != x2 {
            self.times(xt, y1, y2);
            self.fused_mul_sub(xt, x1, x2, xt);
            self.times(yd, x2, y1);
            self.fused_mul_add(yd, x1, y2, yd);
            self.fmov(xd, xt);
        } else if xd != x1 && xd == x2 {
            self.times(xt, y1, y2);
            self.fused_mul_sub(xt, x1, x2, xt);
            self.times(yd, x1, y2);
            self.fused_mul_add(yd, x2, y1, yd);
            self.fmov(xd, xt);
        } else {
            self.times(xt, y1, y2);
            self.fused_mul_sub(xt, x1, x2, xt);
            self.times(yt, x2, y1);
            self.fused_mul_add(yt, x1, y2, yt);
            self.fmov(xd, xt);
            self.fmov(yd, yt);
        }

        true
    }

    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let xt = Reg::Gen(2);
        let yt = Reg::Gen(3);
        let t = Reg::Temp;

        self.times(xt, y1, y2);
        self.fused_mul_add(xt, x1, x2, xt);
        self.times(yt, x1, y2);
        self.fused_mul_sub(yt, x2, y1, yt);
        self.times(t, x2, x2);
        self.fused_mul_add(t, y2, y2, t);
        self.divide(xd, xt, t);
        self.divide(yd, yt, t);

        true
    }

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!()
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn imaginary(&mut self, dst: Reg, _s1: Reg) {
        self.xor(dst, dst, dst);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, _s2: Reg) {
        self.fmov(dst, s1);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpnlepd k(1), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vpmovm2q zmm(ϕ(dst)), k(1); self.amd};
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpnltpd k(1), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vpmovm2q zmm(ϕ(dst)), k(1); self.amd};
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpltpd k(1), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vpmovm2q zmm(ϕ(dst)), k(1); self.amd};
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmplepd k(1), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vpmovm2q zmm(ϕ(dst)), k(1); self.amd};
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpeqpd k(1), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vpmovm2q zmm(ϕ(dst)), k(1); self.amd};
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpneqpd k(1), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
        amd! {vpmovm2q zmm(ϕ(dst)), k(1); self.amd};
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vandpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vandnpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vorpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vxorpd zmm(ϕ(dst)), zmm(ϕ(s1)), zmm(ϕ(s2)); self.amd};
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_all_ones_");
        self.xor(dst, s1, Reg::Temp);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match fused_perm(dst, s1, s2, s3) {
            FusedAction::Use132(a, b, c) => {
                amd! {vfmadd132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use213(a, b, c) => {
                amd! {vfmadd213pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use231(a, b, c) => {
                amd! {vfmadd231pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Copy132(a, b, c) => {
                amd! {vmovapd zmm(ϕ(dst)), zmm(ϕ(s1)); self.amd};
                amd! {vfmadd132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd};
            }
        }
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match fused_perm(dst, s1, s2, s3) {
            FusedAction::Use132(a, b, c) => {
                amd! {vfmsub132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use213(a, b, c) => {
                amd! {vfmsub213pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use231(a, b, c) => {
                amd! {vfmsub231pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Copy132(a, b, c) => {
                amd! {vmovapd zmm(ϕ(dst)), zmm(ϕ(s1)); self.amd};
                amd! {vfmsub132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd};
            }
        }
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match fused_perm(dst, s1, s2, s3) {
            FusedAction::Use132(a, b, c) => {
                amd! {vfnmadd132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use213(a, b, c) => {
                amd! {vfnmadd213pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use231(a, b, c) => {
                amd! {vfnmadd231pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Copy132(a, b, c) => {
                amd! {vmovapd zmm(ϕ(dst)), zmm(ϕ(s1)); self.amd};
                amd! {vfnmadd132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd};
            }
        }
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        match fused_perm(dst, s1, s2, s3) {
            FusedAction::Use132(a, b, c) => {
                amd! {vfnmsub132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use213(a, b, c) => {
                amd! {vfnmsub213pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Use231(a, b, c) => {
                amd! {vfnmsub231pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd}
            }
            FusedAction::Copy132(a, b, c) => {
                amd! {vmovapd zmm(ϕ(dst)), zmm(ϕ(s1)); self.amd};
                amd! {vfnmsub132pd zmm(ϕ(a)), zmm(ϕ(b)), zmm(ϕ(c)); self.amd};
            }
        }
    }

    fn add_consts(&mut self, consts: &[f64]) {
        for (idx, val) in consts.iter().enumerate() {
            let label = format!("_const_{}_", idx);
            self.set_label(label.as_str());
            self.append_quad((*val).to_bits());
        }
    }

    fn add_func(&mut self, op: &str, f: Func) {
        add_func(&mut self.amd, op, f);
    }

    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        if is_external_func(op) {
            return self.call_external(op, num_args);
        }

        let label = format!("_func_{}_", op);

        match num_args {
            1 => self.call_vector_unary(&label),
            2 => self.call_vector_binary(&label),
            _ => return Err(anyhow!("invalid number of arguments")),
        }

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let label = format!("_func_{}_", op);

        match num_args {
            1 => self.call_complex_vector_unary(&label),
            2 => self.call_complex_vector_binary(&label),
            _ => return Err(anyhow!("invalid number of arguments")),
        }

        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        self.amd.call_relative(label);
    }

    fn ret(&mut self) {
        self.amd.ret();
    }

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        if true_val == false_val {
            self.fmov(dst, true_val);
        }

        self.load_stack(Reg::Temp, idx);
        amd! {vpmovq2m k(1), zmm(ϕ(Reg::Temp)); self.amd};

        if dst == true_val {
            amd! {knotw k(1), k(1); self.amd};
            amd! {vmovapd zmm(ϕ(dst)){k(1)}, zmm(ϕ(false_val)); self.amd};
        } else if dst == false_val {
            amd! {vmovapd zmm(ϕ(dst)){k(1)}, zmm(ϕ(true_val)); self.amd};
        } else {
            amd! {vmovapd zmm(ϕ(dst)), zmm(ϕ(false_val)); self.amd};
            amd! {vmovapd zmm(ϕ(dst)){k(1)}, zmm(ϕ(true_val)); self.amd};
        }
    }

    /****************** Prologues/Epilogues ********************/

    #[cfg(target_family = "unix")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        amd! {push r(Amd::RBP); self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd};
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};

        for i in 0..count_states {
            amd! {vmovsd [r(MEM) + i * 8], xmm(i as u8); self.amd};
        }
    }

    #[cfg(target_family = "windows")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        // self.amd.push(Amd::RBP);
        amd! {push r(Amd::RBP); self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);

        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd};
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};

        for i in 0..count_states.min(4) {
            amd! {vmovsd [r(MEM) + i * REG_SIZE], xmm(i as u8); self.amd};
        }

        for i in 4..count_states {
            let i = i as u32;
            // the offset of the fifth or eight arguments:
            // +4 for the 32-byte home
            // +1 for the return address in the stack
            // +1 for RBP in the stack
            // -4 for the first four arguments passed in XMM0-XMM3
            amd! {vmovsd xmm(0), [r(MEM) + (frame_size + (i + 2) * REG_SIZE)]; self.amd};
            amd! {vmovsd [r(MEM) + i * REG_SIZE], xmm(0); self.amd};
        }
    }

    fn epilogue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize, idx_ret: i32) {
        amd! {vzeroupper; self.amd};
        amd! {vmovsd xmm(0), [r(MEM) + idx_ret * REG_SIZE as i32]; self.amd};

        let total_size = align_stack(cap as u32 * REG_SIZE)
            + align_stack((count_states + count_obs) as u32 * REG_SIZE);

        amd! {add rsp, total_size; self.amd};
        amd! {pop r(Amd::RBP); self.amd};
        amd! {ret; self.amd};
    }

    /*
     * prologue_indirect generates the stack frame. It works in two modes:
     *  * Direct mode: MEM (state variables + obs) is passed directly as the first
     *      argument. The second argument is null.
     *  * Indirect mode: the second argument is a pointer to an array of pointers
     *      to states and obs. The third argument is the index into these arrays.
     *      MEM is allocated on the stack and filled based on the second and thirds args.
     *
     * Noth that the second argument determines whether it is the direct (args[1] == null) or indirect mode.
     * In both modes, the fourth argument points to an array of params.
     *
     * # Stack Frame Layout:
     *
     * x86-64 stack frame is composed of four segments. The length of each segment should
     *      be a multiple of 16.
     *  1. The return addess + old RBP (16 bytes).
     *  2. General registers area (32 bytes in Linux, uses the home area in Windows).
     *      call to `save_nonvolatile_regs`.
     *  3. Optional mem area to store state variables and observables in vectorized calls.
     *      Of length `frame_size`, which is aligned to 16 by calling `align_stack`.
     *  4. The temporary variables area of size `align_stack(cap * REG_SIZE)`.
     *      It has two sub-segments:
     *  4a. The actual temporary variables area.
     *  4b. A default spill area of `16 * REG_SIZE` bytes. It is generated by
     *      `SymbolTable::new` adding 16 dummy temp variables. The top 10 slots are used
     *      to store callee-saved xmm/zmm registers (`save_used_registers`). The bottom
     *      6 slots are the work area for various call routines, Specifically, the bottom
     *      32 bytes is reserved as the home area for Windows call ABI.
     */
    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        if self.config.symbolica() {
            return self.prologue_symbolica(cap, count_params, count_obs);
        }

        amd! {push r(Amd::RBP); self.amd};
        save_nonvolatile_regs(&mut self.amd);
        amd! {mov r(Amd::RBP), r(Amd::RSP); self.amd};
        amd! {and r(Amd::RSP), 0xffffffc0; self.amd};

        amd! {mov r(MEM), r(ARGS[0]); self.amd}; // first arg = mem if direct mode, otherwise null
        amd! {mov r(STATES), r(ARGS[1]); self.amd}; // second arg = states+obs if indirect mode, otherwise null
        amd! {mov r(IDX), r(ARGS[2]); self.amd}; // third arg = index if indirect mode
        amd! {mov r(PARAMS), r(ARGS[3]); self.amd}; // fourth arg = params

        amd! {or r(STATES), r(STATES); self.amd};
        amd! {jz "@main"; self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);

        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd}; // in indirect mode, MEM is allocated on the stack
                                               // multiply IDX by 4 to convert from f64x4 index to f64 index
        amd! {add r(IDX), r(IDX); self.amd};
        amd! {add r(IDX), r(IDX); self.amd};
        amd! {add r(IDX), r(IDX); self.amd};

        for i in 0..count_states {
            amd! {mov r(Amd::RAX), [r(STATES) + 2 * 8 * i as i32]; self.amd};
            amd! {vmovupd zmm(RET), [r(Amd::RAX) + r(IDX) * 8]; self.amd};
            amd! {vmovupd [r(MEM) + i as u32 * REG_SIZE], zmm(RET); self.amd};
        }

        // may save idx (RDX) as double in RBP + 8/32 * count_states

        self.set_label("@main");
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};
    }

    fn epilogue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) {
        amd! {xor r(Amd::RAX), r(Amd::RAX); self.amd};
        self.set_label("@epilogue");

        if self.config.symbolica() {
            return self.epilogue_symbolica(cap, count_params, count_obs);
        }

        amd! {add rsp, align_stack(cap as u32 * REG_SIZE); self.amd};
        amd! {or r(STATES), r(STATES); self.amd};
        amd! {jz "@done"; self.amd};

        for i in 0..count_obs {
            amd! {mov r(Amd::RCX), [r(STATES) + 2 * 8 * (count_states + i) as i32]; self.amd};
            amd! {vmovupd zmm(RET), [r(MEM) + (count_states + i) as u32 * REG_SIZE]; self.amd};
            amd! {vmovupd [r(Amd::RCX) + r(IDX) * 8], zmm(RET); self.amd};
        }

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);

        amd! {add rsp, frame_size; self.amd};
        self.set_label("@done");
        amd! {vzeroupper; self.amd};

        amd! {mov r(Amd::RSP), r(Amd::RBP); self.amd};
        load_nonvolatile_regs(&mut self.amd);
        amd! {pop r(Amd::RBP); self.amd};
        amd! {ret; self.amd};
    }

    fn save_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.save_stack(reg(*r), *r as u32 + 2);
            }
        }
    }

    fn load_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                self.load_stack(reg(*r), *r as u32 + 2);
            }
        }
    }
}

impl AmdVectorF64x8Generator {
    fn prologue_symbolica(&mut self, cap: usize, count_params: usize, count_obs: usize) {
        amd! {push r(Amd::RBP); self.amd};
        save_nonvolatile_regs(&mut self.amd);
        amd! {mov r(Amd::RBP), r(Amd::RSP); self.amd};
        amd! {and r(Amd::RSP), 0xffffffc0; self.amd};

        amd! {mov r(MEM), r(ARGS[0]); self.amd}; // first arg = mem if direct mode, otherwise null
        amd! {mov r(STATES), r(ARGS[1]); self.amd}; // second arg = states+obs if indirect mode, otherwise null
        amd! {mov r(IDX), r(ARGS[2]); self.amd}; // third arg = index if indirect mode
        amd! {mov r(PARAMS), r(ARGS[3]); self.amd}; // fourth arg = params

        amd! {or r(IDX), r(IDX); self.amd};
        amd! {jz "@main"; self.amd};
        amd! {sub rsp, align_stack(count_params as u32 * REG_SIZE); self.amd};
        amd! {mov r(Amd::RAX), r(PARAMS); self.amd};
        amd! {mov r(PARAMS), r(STACK); self.amd};

        amd! {mov r(Amd::RCX), count_params; self.amd};
        self.set_label("@load");

        for j in 0..NUM_LANES as usize {
            amd! {vmovsd xmm(RET), [r(Amd::RAX) + (8 * j * count_params) as i32]; self.amd};
            amd! {vmovsd [r(PARAMS) + 8 * j as i32], xmm(RET); self.amd};
        }

        amd! {add r(Amd::RAX), 8; self.amd};
        amd! {add r(PARAMS), 8 * NUM_LANES; self.amd};
        amd! {dec r(Amd::RCX); self.amd};
        amd! {jnz "@load"; self.amd};
        amd! {sub r(PARAMS), 8 * NUM_LANES as usize * count_params; self.amd};

        amd! {sub rsp, align_stack(count_obs as u32 * REG_SIZE); self.amd};
        amd! {mov r(STATES), r(MEM); self.amd};
        amd! {mov r(MEM), r(STACK); self.amd};

        self.set_label("@main");

        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};
    }

    fn epilogue_symbolica(&mut self, cap: usize, count_params: usize, count_obs: usize) {
        // add_rsp(&mut self.amd, align_stack(cap as u32 * REG_SIZE));
        amd! {add rsp, align_stack(cap as u32 * REG_SIZE); self.amd};

        amd! {or r(IDX), r(IDX); self.amd};
        amd! {jz "@done"; self.amd};

        amd! {mov r(Amd::RCX), count_obs; self.amd};
        self.set_label("@save");

        for j in 0..NUM_LANES as usize {
            amd! {vmovsd xmm(RET), [r(MEM) + 8 * j as i32]; self.amd};
            amd! {vmovsd [r(STATES) + (8 * j * count_obs) as i32], xmm(0); self.amd};
        }

        amd! {add r(MEM), 8 * NUM_LANES; self.amd};
        amd! {add r(STATES), 8; self.amd};
        amd! {dec r(Amd::RCX); self.amd};
        amd! {jnz "@save"; self.amd};

        let frame_size =
            align_stack(count_params as u32 * REG_SIZE) + align_stack(count_obs as u32 * REG_SIZE);
        amd! {add rsp, frame_size; self.amd};
        self.set_label("@done");

        amd! {vzeroupper; self.amd};

        amd! {mov r(Amd::RSP), r(Amd::RBP); self.amd};
        load_nonvolatile_regs(&mut self.amd);
        amd! {pop r(Amd::RBP); self.amd};
        amd! {ret; self.amd};
    }
}
