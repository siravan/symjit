use super::asm::{Amd, RoundingMode};

#[allow(dead_code)]
impl Amd {
    pub fn vmovapd(&mut self, reg: u8, rm: u8) {
        self.vex_pd(reg, 0, rm, 0);
        self.append_byte(0x28);
        self.modrm_reg(reg, rm);
    }

    pub fn vbroadcastsd_ymm(&mut self, reg: u8, rm: u8, offset: i32) {
        self.vex3pd(reg, 0, rm, 0, 2);
        self.append_byte(0x19);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vbroadcastsd_ymm_label(&mut self, reg: u8, label: &str) {
        self.vex3pd(reg, 0, 0, 0, 2);
        self.append_byte(0x19);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn vmovpd_ymm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        self.vex_pd(reg, 0, rm, 0);
        self.append_byte(0x10);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vmovpd_ymm_indexed(&mut self, reg: u8, base: u8, index: u8, scale: u8) {
        self.vex_pd(reg, 0, base, index);
        self.append_byte(0x10);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vmovpd_ymm_label(&mut self, reg: u8, label: &str) {
        self.vex_pd(reg, 0, 0, 0);
        self.append_byte(0x10);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn vmovpd_mem_ymm(&mut self, rm: u8, offset: i32, reg: u8) {
        self.vex_pd(reg, 0, rm, 0);
        self.append_byte(0x11);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vmovpd_indexed_ymm(&mut self, base: u8, index: u8, scale: u8, reg: u8) {
        self.vex_pd(reg, 0, base, index);
        self.append_byte(0x11);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vaddpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x58);
        self.modrm_reg(reg, rm);
    }

    pub fn vsubpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x5c);
        self.modrm_reg(reg, rm);
    }

    pub fn vmulpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x59);
        self.modrm_reg(reg, rm);
    }

    pub fn vdivpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x5e);
        self.modrm_reg(reg, rm);
    }

    pub fn vsqrtpd(&mut self, reg: u8, rm: u8) {
        self.vex_pd(reg, 0, rm, 0);
        self.append_byte(0x51);
        self.modrm_reg(reg, rm);
    }

    pub fn vroundpd(&mut self, reg: u8, rm: u8, mode: RoundingMode) {
        self.vex3pd(reg, 0, rm, 0, 3);
        self.append_byte(0x09);
        self.modrm_reg(reg, rm);
        self.append_byte(match mode {
            RoundingMode::Round => 0,
            RoundingMode::Floor => 1,
            RoundingMode::Ceiling => 2,
            RoundingMode::Trunc => 3,
        });
    }

    pub fn vandpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x54);
        self.modrm_reg(reg, rm);
    }

    pub fn vandnpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x55);
        self.modrm_reg(reg, rm);
    }

    pub fn vorpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x56);
        self.modrm_reg(reg, rm);
    }

    pub fn vxorpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x57);
        self.modrm_reg(reg, rm);
    }

    pub fn vcmpeqpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(0)
    }

    pub fn vcmpltpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(1);
    }

    pub fn vcmplepd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(2);
    }

    pub fn vcmpunordpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(3);
    }

    pub fn vcmpneqpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(4);
    }

    pub fn vcmpnltpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(5);
    }

    pub fn vcmpnlepd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(6);
    }

    pub fn vcmpordpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(7);
    }

    pub fn vmovmskpd(&mut self, reg: u8, rm: u8) {
        self.vex_pd(reg, 0, rm, 0);
        self.append_byte(0x50);
        self.modrm_reg(reg, rm);
    }

    pub fn vmovq_xmm_reg(&mut self, reg: u8, rm: u8) {
        self.vex3dd_w1(reg, 0, rm, 0, 1);
        self.append_byte(0x6e);
        self.modrm_reg(reg, rm);
    }

    pub fn movq_reg_xmm(&mut self, rm: u8, reg: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x7e);
        self.modrm_reg(reg, rm);
    }

    pub fn vshufpd(&mut self, reg: u8, vreg: u8, rm: u8, imm8: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xc6);
        self.modrm_reg(reg, rm);
        self.append_byte(imm8);
    }

    pub fn vunpckhpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x15);
        self.modrm_reg(reg, rm);
    }

    pub fn vunpcklpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0x14);
        self.modrm_reg(reg, rm);
    }

    pub fn vaddsubpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm, 0);
        self.append_byte(0xd0);
        self.modrm_reg(reg, rm);
    }
}
