use super::asm::{Amd, RoundingMode};

#[allow(dead_code)]
impl Amd {
    pub fn vmovsd_xmm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        self.vex_sd(reg, 0, rm, 0);
        self.append_byte(0x10);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vmovsd_xmm_indexed(&mut self, reg: u8, base: u8, index: u8, scale: u8) {
        self.vex_sd(reg, 0, base, index);
        self.append_byte(0x10);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vmovsd_xmm_label(&mut self, reg: u8, label: &str) {
        self.vex_sd(reg, 0, 0, 0);
        self.append_byte(0x10);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn vmovsd_mem_xmm(&mut self, rm: u8, offset: i32, reg: u8) {
        self.vex_sd(reg, 0, rm, 0);
        self.append_byte(0x11);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vmovsd_indexed_xmm(&mut self, base: u8, index: u8, scale: u8, reg: u8) {
        self.vex_sd(reg, 0, base, index);
        self.append_byte(0x11);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vaddsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0x58);
        self.modrm_reg(reg, rm);
    }

    pub fn vsubsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0x5c);
        self.modrm_reg(reg, rm);
    }

    pub fn vmulsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0x59);
        self.modrm_reg(reg, rm);
    }

    pub fn vdivsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0x5e);
        self.modrm_reg(reg, rm);
    }

    pub fn vsqrtsd(&mut self, reg: u8, rm: u8) {
        self.vex_sd(reg, 0, rm, 0);
        self.append_byte(0x51);
        self.modrm_reg(reg, rm);
    }

    pub fn vroundsd(&mut self, reg: u8, rm: u8, mode: RoundingMode) {
        self.vex3pd(reg, reg, rm, 0, 3);
        self.append_byte(0x0b);
        self.modrm_reg(reg, rm);
        self.append_byte(match mode {
            RoundingMode::Round => 0,
            RoundingMode::Floor => 1,
            RoundingMode::Ceiling => 2,
            RoundingMode::Trunc => 3,
        });
    }

    pub fn vcmpeqsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(0)
    }

    pub fn vcmpltsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(1);
    }

    pub fn vcmplesd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(2);
    }

    pub fn vcmpunordsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(3);
    }

    pub fn vcmpneqsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(4);
    }

    pub fn vcmpnltsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(5);
    }

    pub fn vcmpnlesd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(6);
    }

    pub fn vcmpordsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm, 0);
        self.append_byte(0xC2);
        self.modrm_reg(reg, rm);
        self.append_byte(7);
    }

    pub fn vucomisd(&mut self, reg: u8, rm: u8) {
        self.vex_pd(reg, 0, rm, 0);
        self.append_byte(0x2e);
        self.modrm_reg(reg, rm);
    }
}
