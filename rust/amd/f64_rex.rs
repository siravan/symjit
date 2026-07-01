use super::asm::{Amd, RoundingMode};

#[allow(dead_code)]
impl Amd {
    pub fn movapd(&mut self, reg: u8, rm: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x28);
        self.modrm_reg(reg, rm);
    }

    pub fn movsd_xmm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        self.sse_sd(reg, rm);
        self.append_byte(0x10);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn movsd_xmm_indexed(&mut self, reg: u8, base: u8, index: u8, scale: u8) {
        self.sse_sd_index(reg, base, index);
        self.append_byte(0x10);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn movsd_xmm_label(&mut self, reg: u8, label: &str) {
        self.sse_sd(reg, 0);
        self.append_byte(0x10);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn movsd_mem_xmm(&mut self, rm: u8, offset: i32, reg: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0x11);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn movsd_indexed_xmm(&mut self, base: u8, index: u8, scale: u8, reg: u8) {
        self.sse_sd_index(reg, base, index);
        self.append_byte(0x11);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn addsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0x58);
        self.modrm_reg(reg, rm);
    }

    pub fn subsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0x5c);
        self.modrm_reg(reg, rm);
    }

    pub fn mulsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0x59);
        self.modrm_reg(reg, rm);
    }

    pub fn divsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0x5e);
        self.modrm_reg(reg, rm);
    }

    pub fn sqrtsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0x51);
        self.modrm_reg(reg, rm);
    }

    pub fn roundsd(&mut self, reg: u8, rm: u8, mode: RoundingMode) {
        self.sse_pd(reg, rm);
        self.append_bytes(&[0x3a, 0x0b]);
        self.modrm_reg(reg, rm);
        self.append_byte(match mode {
            RoundingMode::Round => 0,
            RoundingMode::Floor => 1,
            RoundingMode::Ceiling => 2,
            RoundingMode::Trunc => 3,
        });
    }

    pub fn cmpeqsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(0)
    }

    pub fn cmpltsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(1);
    }

    pub fn cmplesd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(2);
    }

    pub fn cmpunordsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(3);
    }

    pub fn cmpneqsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(4);
    }

    pub fn cmpnltsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(5);
    }

    pub fn cmpnlesd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(6);
    }

    pub fn cmpordsd(&mut self, reg: u8, rm: u8) {
        self.sse_sd(reg, rm);
        self.append_byte(0xC2);
        self.modrm_reg(reg, rm);
        self.append_byte(7);
    }

    pub fn ucomisd(&mut self, reg: u8, rm: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x2e);
        self.modrm_reg(reg, rm);
    }

    pub fn andpd(&mut self, reg: u8, rm: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x54);
        self.modrm_reg(reg, rm);
    }

    pub fn andnpd(&mut self, reg: u8, rm: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x55);
        self.modrm_reg(reg, rm);
    }

    pub fn orpd(&mut self, reg: u8, rm: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x56);
        self.modrm_reg(reg, rm);
    }

    pub fn xorpd(&mut self, reg: u8, rm: u8) {
        self.sse_pd(reg, rm);
        self.append_byte(0x57);
        self.modrm_reg(reg, rm);
    }
}
