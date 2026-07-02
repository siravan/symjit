use super::asm::{Amd, RoundingMode};

#[allow(dead_code)]
impl Amd {
    pub fn vmovadd(&mut self, reg: u8, rm: u8) {
        self.vex_dd(reg, 0, rm, 0);
        self.append_byte(0x28);
        self.modrm_reg(reg, rm);
    }

    pub fn vbroadcastsd_xmm(&mut self, reg: u8, rm: u8, offset: i32) {
        self.vex3dd(reg, 0, rm, 0, 2);
        self.append_byte(0x19);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vbroadcastsd_xmm_label(&mut self, reg: u8, label: &str) {
        self.vex3dd(reg, 0, 0, 0, 2);
        self.append_byte(0x19);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn vmovdd_xmm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        self.vex_dd(reg, 0, rm, 0);
        self.append_byte(0x10);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vmovdd_xmm_indexed(&mut self, reg: u8, base: u8, index: u8, scale: u8) {
        self.vex_dd(reg, 0, base, index);
        self.append_byte(0x10);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vmovdd_xmm_label(&mut self, reg: u8, label: &str) {
        self.vex_dd(reg, 0, 0, 0);
        self.append_byte(0x10);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn vmovdd_mem_xmm(&mut self, rm: u8, offset: i32, reg: u8) {
        self.vex_dd(reg, 0, rm, 0);
        self.append_byte(0x11);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vmovdd_indexed_xmm(&mut self, base: u8, index: u8, scale: u8, reg: u8) {
        self.vex_dd(reg, 0, base, index);
        self.append_byte(0x11);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vadddd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x58);
        self.modrm_reg(reg, rm);
    }

    pub fn vhadddd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x7c);
        self.modrm_reg(reg, rm);
    }

    pub fn vsubdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x5c);
        self.modrm_reg(reg, rm);
    }

    pub fn vmuldd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x59);
        self.modrm_reg(reg, rm);
    }

    pub fn vdivdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x5e);
        self.modrm_reg(reg, rm);
    }

    pub fn vsqrtdd(&mut self, reg: u8, rm: u8) {
        self.vex_dd(reg, 0, rm, 0);
        self.append_byte(0x51);
        self.modrm_reg(reg, rm);
    }

    pub fn vrounddd(&mut self, reg: u8, rm: u8, mode: RoundingMode) {
        self.vex3dd(reg, 0, rm, 0, 3);
        self.append_byte(0x09);
        self.modrm_reg(reg, rm);
        self.append_byte(match mode {
            RoundingMode::Round => 0,
            RoundingMode::Floor => 1,
            RoundingMode::Ceiling => 2,
            RoundingMode::Trunc => 3,
        });
    }

    pub fn vanddd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x54);
        self.modrm_reg(reg, rm);
    }

    pub fn vandndd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x55);
        self.modrm_reg(reg, rm);
    }

    pub fn vordd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x56);
        self.modrm_reg(reg, rm);
    }

    pub fn vxordd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x57);
        self.modrm_reg(reg, rm);
    }

    pub fn vcmpeqdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(0)
    }

    pub fn vcmpltdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(1);
    }

    pub fn vcmpledd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(2);
    }

    pub fn vcmpunorddd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(3);
    }

    pub fn vcmpneqdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(4);
    }

    pub fn vcmpnltdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(5);
    }

    pub fn vcmpnledd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(6);
    }

    pub fn vcmporddd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(7);
    }

    /*
     * let xmm1 = y1:x1 and xmm2 = y2:x2,
     *
     * vshufdd(0, 1, 2, 0) = vshufpd xmm0, xmm1, xmm2, 0 => xmm0 = x2:x1 = vunpcklpd
     * vshufdd(0, 1, 2, 1) = vshufpd xmm0, xmm1, xmm2, 1 => xmm0 = x2:y1
     * vshufdd(0, 1, 2, 2) = vshufpd xmm0, xmm1, xmm2, 2 => xmm0 = y2:x1
     * vshufdd(0, 1, 2, 3) = vshufpd xmm0, xmm1, xmm2, 1 => xmm0 = y2:y1 = vunpckhpd
     *
     * Specifically,
     *
     * vshufdd(0, 1, 1, 0) = vshufpd xmm0, xmm1, xmm1, 0 => xmm0 = x1:x1 = dup low
     * vshufdd(0, 1, 1, 1) = vshufpd xmm0, xmm1, xmm1, 1 => xmm0 = x1:y1 = flip
     * vshufdd(0, 1, 1, 2) = vshufpd xmm0, xmm1, xmm1, 2 => xmm0 = y1:x1 = ident
     * vshufdd(0, 1, 1, 3) = vshufpd xmm0, xmm1, xmm1, 3 => xmm0 = y1:y1 = dup high
     *
     */

    pub fn vshufdd(&mut self, reg: u8, vreg: u8, rm: u8, imm8: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xc6);
        self.modrm_reg(reg, rm);
        self.append_byte(imm8);
    }

    pub fn vunpckhdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x15);
        self.modrm_reg(reg, rm);
    }

    pub fn vunpckldd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0x14);
        self.modrm_reg(reg, rm);
    }

    pub fn vaddsubdd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_dd(reg, vreg, rm, 0);
        self.append_byte(0xd0);
        self.modrm_reg(reg, rm);
    }

    // imm8 == 0 => reg = vreg[128:256]:rm[0:128]
    // imm8 == 1 => reg = rm[128:256]:vreg[0:128]
    pub fn vinsertf128(&mut self, reg: u8, vreg: u8, rm: u8, imm8: u8) {
        self.vex3pd(reg, vreg, rm, 0, 3);
        self.append_byte(0x18);
        self.modrm_reg(reg, rm);
        self.append_byte(imm8);
    }

    pub fn vinsertf128_mem(&mut self, reg: u8, vreg: u8, rm: u8, offset: i32, imm8: u8) {
        self.vex3pd(reg, vreg, rm, 0, 3);
        self.append_byte(0x18);
        self.modrm_mem(reg, rm, offset);
        self.append_byte(imm8);
    }

    // imm8 == 0 => rm = reg[0:128]
    // imm8 == 1 => rm = reg[128:256]
    pub fn vextractf128(&mut self, rm: u8, reg: u8, imm8: u8) {
        self.vex3pd(reg, 0, rm, 0, 3);
        self.append_byte(0x19);
        self.modrm_reg(reg, rm);
        self.append_byte(imm8);
    }

    pub fn vmovq_reg_xmm(&mut self, rm: u8, reg: u8) {
        self.vex3dd_w1(reg, 0, rm, 0, 1);
        self.append_byte(0x7e);
        self.modrm_reg(reg, rm);
    }

    pub fn movq_xmm_reg(&mut self, reg: u8, rm: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x6e);
        self.modrm_reg(reg, rm);
    }
}
