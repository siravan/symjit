use crate::code::Func;
use crate::config::{Config, SPILL_AREA};
use crate::generator::{FuncletType, Generator};
use crate::symbol::Loc;
use crate::utils::align_stack;
use crate::utils::{is_external_func, DataType, Reg};
use anyhow::Result;

use super::asm::{Amd, RoundingMode};
use super::*;

const REG_SIZE: u32 = 8;
const T0: u8 = 1; // Reg::Temp
const T1: u8 = 2;
const T2: u8 = 3;

/*
 *  ϕ translates a logical register number (in Reg) to a physical
 *  register number, according to the ABI.
 */
fn ϕ(r: Reg) -> u8 {
    match r {
        Reg::Ret => 0,
        Reg::Temp => 1,
        Reg::Left => 0,
        Reg::Right => 1,
        Reg::Gen(dst) => dst + 4,
        Reg::Static(..) => panic!("passing static registers to codegen"),
    }
}

pub struct AmdComplexGenerator {
    amd: Amd,
    config: Config,
    last_load: usize,
}

impl AmdComplexGenerator {
    pub fn new(config: Config) -> AmdComplexGenerator {
        AmdComplexGenerator {
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
        self.amd.vmovsd_xmm_label(ϕ(dst), label);
    }

    fn call_external(&mut self, op: &str, num_args: usize) -> Result<()> {
        let cap = SPILL_AREA as u32;

        amd! {mov r(ARGS[0]), [&format!("_env_{}_", op)]; self.amd};
        amd! {lea r(ARGS[1]), [r(STACK) + cap * REG_SIZE]; self.amd};
        amd! {mov r(ARGS[2]), num_args; self.amd};
        amd! {lea r(ARGS[3]), [r(STACK) + 4 * REG_SIZE]; self.amd};
        amd! {vzeroupper; self.amd};

        self.amd.call_indirect(&format!("_func_{}_", op));
        self.load_stack(Reg::Ret, 4);

        Ok(())
    }

    fn predefined_consts(&mut self) {
        self.align();
        predefined_consts(&mut self.amd);
    }
}

impl Generator for AmdComplexGenerator {
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
        FuncletType::Real
    }

    fn seal(&mut self) {
        self.predefined_consts();
        self.apply_jumps();
    }

    fn align(&mut self) {
        let mut n = self.amd.a.ip();

        while (n & 7) != 0 {
            amd! {nop ; self.amd};
            n += 1
        }
    }

    fn set_label(&mut self, label: &str) {
        self.amd.a.set_label(label);
    }

    fn branch(&mut self, label: &str) {
        amd! {jmp label; self.amd};
    }

    /// jump to label if cond == is_else
    /// note that `is_else` is not the correct name anymore and should be
    /// changed to `expectation`
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        if is_else {
            amd! {vucomisd xmm(ϕ(cond)), xmm(ϕ(cond)); self.amd};
            amd! {jpe label; self.amd};
        } else {
            amd! {jmp label; self.amd};
        }
    }

    fn fuse_load_math(&mut self) {
        fuse_load_math(&mut self.amd, self.last_load);
    }

    //***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst != s1 {
            self.amd.vmovapd(ϕ(dst), ϕ(s1));
            amd! {vmovapd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        amd! {vxorpd xmm(ϕ(s1)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vxorpd xmm(ϕ(s2)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vxorpd xmm(ϕ(s1)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        let label = format!("_const_{}_", idx);
        amd! {vmovsd xmm(ϕ(dst)), label.as_str(); self.amd};
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        amd! {vmovupd xmm(ϕ(dst)), [r(MEM) + idx * REG_SIZE]; self.amd};
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        amd! {vmovupd [r(MEM) + idx * REG_SIZE], xmm(ϕ(dst)); self.amd};
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        amd! {vmovupd xmm(ϕ(dst)), [r(PARAMS) + idx * REG_SIZE]; self.amd};
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.last_load = self.amd.a.ip();
        if idx < 256 {
            amd! {vmovupd xmm(ϕ(dst)), [r(STACK) + idx * REG_SIZE]; self.amd};
        } else {
            amd! {vmovupd xmm(ϕ(dst)), [r(STATES) + idx * REG_SIZE]; self.amd};
        }
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        if idx < 256 {
            amd! {vmovupd [r(STACK) + idx * REG_SIZE], xmm(ϕ(dst)); self.amd};
        } else {
            amd! {vmovupd [r(STATES) + idx * REG_SIZE], xmm(ϕ(dst)); self.amd};
        }
    }

    fn load_mem_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn save_mem_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {}

    fn load_param_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn load_stack_complex(&mut self, _xd: Reg, _yd: Reg, _idx: u32) {}

    fn save_stack_complex(&mut self, _xs: Reg, _ys: Reg, _idx: u32) {}

    fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_minus_zero_");
        amd! {vunpcklpd xmm(ϕ(Reg::Temp)), xmm(ϕ(Reg::Temp)), xmm(ϕ(Reg::Temp)); self.amd};
        self.xor(dst, s1, Reg::Temp);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        amd! {vmulpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd};
        amd! {vhaddpd xmm(T1), xmm(T1), xmm(T1); self.amd};
        amd! {vsqrtsd xmm(T2), xmm(T1); self.amd};
        amd! {vxorpd xmm(T1), xmm(T1), xmm(T1); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(T2), xmm(T1); self.amd};
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        amd! {vmovq r(Amd::RAX), xmm(ϕ(s1)); self.amd};
        amd! {vmulpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd};
        amd! {vhaddpd xmm(T1), xmm(T1), xmm(T1); self.amd};

        amd! {vsqrtpd xmm(T1), xmm(T1); self.amd};
        amd! {vmovsd xmm(T0), "_minus_zero_"; self.amd};
        amd! {vandnpd xmm(T2), xmm(T0), xmm(ϕ(s1)); self.amd};
        amd! {vaddsd xmm(T1), xmm(T1), xmm(T2); self.amd};
        amd! {vmovsd xmm(T0), "_half_"; self.amd};
        amd! {vmulsd xmm(T1), xmm(T1), xmm(T0); self.amd};
        amd! {vsqrtsd xmm(T1), xmm(T1); self.amd};

        amd! {vunpckhpd xmm(T2), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd};
        amd! {vdivsd xmm(T2), xmm(T2), xmm(T1); self.amd};
        amd! {vmulsd xmm(T2), xmm(T2), xmm(T0); self.amd};

        amd! {vcmpeqsd xmm(T0), xmm(T2), xmm(T2); self.amd};
        amd! {vandpd xmm(T2), xmm(T2), xmm(T0); self.amd};

        amd! {vunpcklpd xmm(ϕ(dst)), xmm(T2), xmm(T1); self.amd};

        let label = format!(".Y{}", self.amd.a.ip());
        amd! {or r(Amd::RAX), r(Amd::RAX); self.amd};
        amd! {js &label; self.amd};
        amd! {vshufpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)), 1; self.amd};
        self.set_label(&label);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        amd! {vxorpd xmm(T1), xmm(T1), xmm(T1); self.amd};
        amd! {vsqrtsd xmm(ϕ(dst)), xmm(ϕ(s1)); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(T1); self.amd};
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        amd! {vshufpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)), 1; self.amd};
        amd! {vxorpd xmm(T2), xmm(T2), xmm(T2); self.amd};
        amd! {vaddsubpd xmm(T1), xmm(T2), xmm(T1); self.amd};
        amd! {vshufpd xmm(T2), xmm(T1), xmm(T1), 1; self.amd};

        amd! {vmulpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd};
        amd! {vhaddpd xmm(T1), xmm(T1), xmm(T1); self.amd};
        amd! {vdivpd xmm(ϕ(dst)), xmm(T2), xmm(T1); self.amd};
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_half_");
        amd! {vunpcklpd xmm(ϕ(Reg::Temp)), xmm(ϕ(Reg::Temp)), xmm(ϕ(Reg::Temp)); self.amd};
        amd! {vmulpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(Reg::Temp)); self.amd};
    }

    fn round(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Round; self.amd};
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Floor; self.amd};
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Ceiling; self.amd};
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        amd! {vroundpd xmm(ϕ(dst)), xmm(ϕ(s1)), RoundingMode::Trunc; self.amd};
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vaddpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vsubpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vunpcklpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate real
        amd! {vunpckhpd xmm(T2), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate imag
        amd! {vmulpd xmm(T1), xmm(T1), xmm(ϕ(s2)); self.amd};
        amd! {vmulpd xmm(T2), xmm(T2), xmm(ϕ(s2)); self.amd};
        amd! {vshufpd xmm(T2), xmm(T2), xmm(T2), 1; self.amd}; // exchange real/imag
        amd! {vaddsubpd xmm(ϕ(dst)), xmm(T1), xmm(T2); self.amd};
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vmulpd xmm(T0), xmm(ϕ(s2)), xmm(ϕ(s2)); self.amd};
        amd! {vhaddpd xmm(T0), xmm(T0), xmm(T0); self.amd};

        amd! {vunpcklpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate real
        amd! {vunpckhpd xmm(T2), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate imag
        amd! {vmulpd xmm(T1), xmm(T1), xmm(ϕ(s2)); self.amd};
        amd! {vmulpd xmm(T2), xmm(T2), xmm(ϕ(s2)); self.amd};
        amd! {vshufpd xmm(T1), xmm(T1), xmm(T1), 1; self.amd}; // exchange real/imag
        amd! {vaddsubpd xmm(ϕ(dst)), xmm(T2), xmm(T1); self.amd};
        amd! {vshufpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)), 1; self.amd};
        amd! {vdivpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(T0); self.amd};
    }

    fn times_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        unreachable!()
    }

    fn divide_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        unreachable!()
    }

    fn support_times2(&self) -> bool {
        self.config.parallel_mul()
    }

    fn times2_loc(&mut self, d1: Reg, s1: Reg, l1: Loc, d2: Reg, s2: Reg, l2: Loc) {
        if d1 == s2 {
            match l1 {
                Loc::Mem(idx) => self.load_mem(Reg::Temp, idx),
                Loc::Param(idx) => self.load_param(Reg::Temp, idx),
                Loc::Stack(idx) => self.load_stack(Reg::Temp, idx),
            }
            self.times(d1, s1, Reg::Temp);

            match l2 {
                Loc::Mem(idx) => self.load_mem(Reg::Temp, idx),
                Loc::Param(idx) => self.load_param(Reg::Temp, idx),
                Loc::Stack(idx) => self.load_stack(Reg::Temp, idx),
            }
            self.times(d2, s2, Reg::Temp);
        } else {
            match l1 {
                Loc::Mem(idx) => {
                    amd! {vmovupd xmm(T0), [r(MEM) + idx * REG_SIZE]; self.amd}
                }
                Loc::Param(idx) => {
                    amd! {vmovupd xmm(T0), [r(PARAMS) + idx * REG_SIZE]; self.amd}
                }
                Loc::Stack(idx) => {
                    if idx < 256 {
                        amd! {vmovupd xmm(T0), [r(STACK) + idx * REG_SIZE]; self.amd}
                    } else {
                        amd! {vmovupd xmm(T0), [r(STATES) + idx * REG_SIZE]; self.amd}
                    }
                }
            }

            match l2 {
                Loc::Mem(idx) => self
                    .amd
                    .vinsertf128_mem(T0, T0, MEM, (idx * REG_SIZE) as i32, 1),
                Loc::Param(idx) => {
                    self.amd
                        .vinsertf128_mem(T0, T0, PARAMS, (idx * REG_SIZE) as i32, 1)
                }
                Loc::Stack(idx) => {
                    if idx < 128 {
                        self.amd
                            .vinsertf128_mem(T0, T0, STACK, (idx * REG_SIZE) as i32, 1)
                    } else {
                        self.amd
                            .vinsertf128_mem(T0, T0, STATES, (idx * REG_SIZE) as i32, 1)
                    }
                }
            }

            self.amd.vinsertf128(ϕ(s1), ϕ(s1), ϕ(s2), 1);

            self.amd.vunpcklpd(T1, ϕ(s1), ϕ(s1)); // duplicate real
            self.amd.vunpckhpd(T2, ϕ(s1), ϕ(s1)); // duplicate imag
            self.amd.vmulpd(T1, T1, T0);
            self.amd.vmulpd(T2, T2, T0);
            self.amd.vshufpd(T2, T2, T2, 5); // exchange real/imag
            self.amd.vaddsubpd(ϕ(d1), T1, T2);

            self.amd.vextractf128(ϕ(d2), ϕ(d1), 1);
        }
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        amd! {vxorpd xmm(T1), xmm(T1), xmm(T1); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(T1); self.amd};
    }

    fn imaginary(&mut self, dst: Reg, s1: Reg) {
        amd! {vxorpd xmm(T1), xmm(T1), xmm(T1); self.amd};
        amd! {vunpckhpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(T1); self.amd};
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        amd! {vxorpd xmm(T1), xmm(T1), xmm(T1); self.amd};
        amd! {vshufpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s1)), 1; self.amd};
        amd! {vaddsubpd xmm(ϕ(dst)), xmm(T1), xmm(ϕ(dst)); self.amd};
        amd! {vshufpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)), 1; self.amd};
    }

    fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpnlesd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)); self.amd};
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpnltsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)); self.amd};
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpltsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)); self.amd};
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmplesd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)); self.amd};
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpeqsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)); self.amd};
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vcmpneqsd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
        amd! {vunpcklpd xmm(ϕ(dst)), xmm(ϕ(dst)), xmm(ϕ(dst)); self.amd};
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vandpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vandnpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vorpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        amd! {vxorpd xmm(ϕ(dst)), xmm(ϕ(s1)), xmm(ϕ(s2)); self.amd};
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.load_const_by_name(Reg::Temp, "_all_ones_");
        amd! {vunpcklpd xmm(ϕ(Reg::Temp)), xmm(ϕ(Reg::Temp)), xmm(ϕ(Reg::Temp)); self.amd};
        self.xor(dst, s1, Reg::Temp);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        amd! {vunpcklpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate real
        amd! {vunpckhpd xmm(T2), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate imag
        amd! {vfmadd132pd xmm(T1), xmm(ϕ(s3)), xmm(ϕ(s2)); self.amd};
        amd! {vmulpd xmm(T2), xmm(T2), xmm(ϕ(s2)); self.amd};
        amd! {vshufpd xmm(T2), xmm(T2), xmm(T2), 1; self.amd}; // exchange real/imag
        amd! {vaddsubpd xmm(ϕ(dst)), xmm(T1), xmm(T2); self.amd};
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        amd! {vunpcklpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate real
        amd! {vunpckhpd xmm(T2), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate imag
        amd! {vfmsub132pd xmm(T1), xmm(ϕ(s3)), xmm(ϕ(s2)); self.amd};
        amd! {vmulpd xmm(T2), xmm(T2), xmm(ϕ(s2)); self.amd};
        amd! {vshufpd xmm(T2), xmm(T2), xmm(T2), 1; self.amd}; // exchange real/imag
        amd! {vaddsubpd xmm(ϕ(dst)), xmm(T1), xmm(T2); self.amd};
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        amd! {vunpcklpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate real
        amd! {vunpckhpd xmm(T2), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate imag
        amd! {vfnmadd132pd xmm(T1), xmm(ϕ(s3)), xmm(ϕ(s2)); self.amd};
        amd! {vmulpd xmm(T2), xmm(T2), xmm(ϕ(s2)); self.amd};
        amd! {vshufpd xmm(T1), xmm(T1), xmm(T1), 1; self.amd}; // exchange real/imag
        amd! {vaddsubpd xmm(T1), xmm(T1), xmm(T2); self.amd};
        amd! {vshufpd xmm(ϕ(dst)), xmm(T1), xmm(T1), 1; self.amd}; // exchange real/imag
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        amd! {vunpcklpd xmm(T1), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate real
        amd! {vunpckhpd xmm(T2), xmm(ϕ(s1)), xmm(ϕ(s1)); self.amd}; // duplicate imag
        amd! {vfnmsub132pd xmm(T1), xmm(ϕ(s3)), xmm(ϕ(s2)); self.amd};
        amd! {vmulpd xmm(T2), xmm(T2), xmm(ϕ(s2)); self.amd};
        amd! {vshufpd xmm(T1), xmm(T1), xmm(T1), 1; self.amd}; // exchange real/imag
        amd! {vaddsubpd xmm(T1), xmm(T1), xmm(T2); self.amd};
        amd! {vshufpd xmm(ϕ(dst)), xmm(T1), xmm(T1), 1; self.amd}; // exchange real/imag
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
        amd! {vzeroupper ; self.amd};
        amd! {call &label; self.amd};

        Ok(())
    }

    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let label = format!("_func_{}_", op);

        if num_args == 2 {
            self.save_stack(Reg::Right, 4);
        }

        amd! {vunpckhpd xmm(1), xmm(0), xmm(0); self.amd};
        amd! {vzeroupper; self.amd};

        if cfg!(target_family = "windows") {
            amd! {lea r(Amd::R8), [r(STACK) + 32]; self.amd};
        } else {
            amd! {lea r(Amd::RDI), [r(STACK) + 32]; self.amd};
        }

        amd! {call &label; self.amd};

        self.load_stack(Reg::Ret, 4);

        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        amd! {call [rip + label]; self.amd};
    }

    fn ret(&mut self) {
        amd! {ret; self.amd};
    }

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        if true_val == false_val {
            self.fmov(dst, true_val);
        } else if dst != false_val {
            self.load_stack(Reg::Temp, idx);
            self.and(dst, Reg::Temp, true_val);
            self.andnot(Reg::Temp, Reg::Temp, false_val);
            self.or(dst, dst, Reg::Temp);
        } else {
            // dst == false_val && dst != true_val
            self.load_stack(Reg::Temp, idx);
            self.andnot(dst, Reg::Temp, false_val);
            self.and(Reg::Temp, Reg::Temp, true_val);
            self.or(dst, dst, Reg::Temp);
        }
    }

    /****************** Prologues/Epilogues ********************/

    #[cfg(target_family = "unix")]
    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize) {
        // self.amd.push(Amd::RBP);
        amd! {push r(Amd::RBP); self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);

        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd};
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};

        for i in 0..count_states {
            self.amd.vmovsd_mem_xmm(MEM, (i * 8) as i32, i as u8);
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
            amd! {vmovsd [r(MEM) + (i as u32 * REG_SIZE)], xmm(i as u8); self.amd};
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
     *      to store callee-saved xmm/ymm registers (`save_used_registers`). The bottom
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

        amd! {mov r(MEM), r(ARGS[0]); self.amd}; // first arg = mem if direct mode, otherwise null
        amd! {mov r(STATES), r(ARGS[1]); self.amd}; // second arg = states+obs if indirect mode, otherwise null
        amd! {mov r(IDX), r(ARGS[2]); self.amd}; // third arg = index if indirect mode
        amd! {mov r(PARAMS), r(ARGS[3]); self.amd}; // fourth arg = params

        amd! {or r(STATES), r(STATES); self.amd};
        amd! {jz "@main"; self.amd};

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        amd! {sub rsp, frame_size; self.amd};
        amd! {mov r(MEM), r(STACK); self.amd}; // in indirect mode, MEM is allocated on the stack

        for i in 0..count_states {
            amd! {mov r(Amd::RAX), [r(STATES) + 2 * 8 * i as i32]; self.amd};
            amd! {vmovsd xmm(RET), [r(Amd::RAX) + r(IDX) * 8]; self.amd};
            amd! {vmovsd [r(MEM) + i as u32 * REG_SIZE], xmm(RET); self.amd};
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
        // self.amd.xor(Amd::RAX, Amd::RAX);
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
            amd! {vmovsd xmm(RET), [r(MEM) + (count_states + i) as u32 * REG_SIZE]; self.amd};
            amd! {vmovsd [r(Amd::RCX) + r(IDX) * 8], xmm(RET); self.amd};
        }

        let frame_size = align_stack((count_states + count_obs) as u32 * REG_SIZE);
        amd! {add rsp, frame_size; self.amd};
        self.set_label("@done");

        amd! {vzeroupper; self.amd};

        load_nonvolatile_regs(&mut self.amd);
        amd! {pop r(Amd::RBP); self.amd};
        amd! {ret; self.amd};
    }

    fn save_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                amd! {vmovsd [r(STACK) + (*r as u32 + 4) as i32], xmm(*r); self.amd};
            }
        }
    }

    fn load_used_registers(&mut self, used: &[u8]) {
        let count_shadows = self.count_shadows();

        for r in used {
            if *r >= count_shadows {
                amd! {vmovsd xmm(*r), [r(STACK) + (*r as u32 + 4) as i32]; self.amd};
            }
        }
    }
}

impl AmdComplexGenerator {
    fn prologue_symbolica(&mut self, cap: usize, _count_params: usize, _count_obs: usize) {
        amd! {push r(Amd::RBP); self.amd};
        save_nonvolatile_regs(&mut self.amd);

        amd! {mov r(MEM), r(ARGS[0]); self.amd}; // first arg = mem if direct mode, otherwise null
        amd! {mov r(STATES), r(ARGS[1]); self.amd}; // second arg = states+obs if indirect mode, otherwise null
        amd! {mov r(IDX), r(ARGS[2]); self.amd}; // third arg = index if indirect mode
        amd! {mov r(PARAMS), r(ARGS[3]); self.amd}; // fourth arg = params

        // prologue_stack(&mut self.amd, cap, REG_SIZE);
        amd! {sub rsp, align_stack(cap as u32 * REG_SIZE); self.amd};
        amd! {mov r(STATES), r(STACK); self.amd};
    }

    fn epilogue_symbolica(&mut self, cap: usize, _count_params: usize, _count_obs: usize) {
        epilogue_stack(&mut self.amd, cap, REG_SIZE);

        amd! {vzeroupper; self.amd};
        load_nonvolatile_regs(&mut self.amd);
        amd! {pop r(Amd::RBP); self.amd};
        amd! {ret; self.amd};
    }
}
