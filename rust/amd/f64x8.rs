use super::asm::{Amd, RoundingMode};

enum Masking {
    Nil,
    Copy(u8),
    Zero(u8),
}

pub struct Prefix {
    mask: Masking,
    reg: u8,
    vreg: u8,
    rm: u8,
    index: u8,
    encoding: u8,
    w: u8,
    len: u32,
    pp: u8,
}

#[allow(unused)]
impl Prefix {
    pub fn new(reg: u8, vreg: u8, rm: u8) -> Prefix {
        assert!(reg < 32 && vreg < 32 && rm < 32);
        Prefix {
            mask: Masking::Nil,
            reg,
            vreg,
            rm,
            index: 0,
            encoding: 1,
            w: 1,
            len: 512,
            pp: 1,
        }
    }

    // packed single
    pub fn ps(&mut self) -> &mut Self {
        self.pp = 0;
        self
    }

    // packed double
    pub fn pd(&mut self) -> &mut Self {
        self.pp = 1;
        self
    }

    pub fn x66(&mut self) -> &mut Self {
        self.pp = 1;
        self
    }

    // scalar single
    // scalar single
    pub fn ss(&mut self) -> &mut Self {
        self.pp = 2;
        self
    }

    pub fn xf3(&mut self) -> &mut Self {
        self.pp = 2;
        self
    }

    // scalar double
    pub fn sd(&mut self) -> &mut Self {
        self.pp = 3;
        self
    }

    pub fn xf2(&mut self) -> &mut Self {
        self.pp = 3;
        self
    }

    fn set_mask(&mut self, maks: Masking) -> &mut Self {
        self.mask = maks;
        self
    }

    pub fn set_index(&mut self, index: u8) -> &mut Self {
        assert!(index < 32);
        self.index = index;
        self
    }

    pub fn set_w(&mut self, w: u8) -> &mut Self {
        assert!(w == 0 || w == 1);
        self.w = w;
        self
    }

    pub fn set_encoding(&mut self, encoding: u8) -> &mut Self {
        assert!(encoding < 8);
        self.encoding = encoding;
        self
    }

    pub fn x0f(&mut self) -> &mut Self {
        self.set_encoding(1)
    }

    pub fn x0f38(&mut self) -> &mut Self {
        self.set_encoding(2)
    }

    pub fn x0f3a(&mut self) -> &mut Self {
        self.set_encoding(2)
    }

    pub fn set_len(&mut self, len: u32) -> &mut Self {
        assert!(len == 128 || len == 256 || len == 512);
        self.len = len;
        self
    }

    pub fn vex(&self, amd: &mut Amd) {
        let r = if self.reg & 8 != 0 { 0 } else { 0x80 };
        let x = if self.index & 8 != 0 { 0 } else { 0x40 };
        let b = if self.rm & 8 != 0 { 0 } else { 0x20 };

        let vvvv = (!self.vreg & 0x0f) << 3;
        let l = if self.len == 256 { 0x20 } else { 0 };

        assert!(matches!(self.mask, Masking::Nil));

        let w = if self.w != 0 { 0x80 } else { 0 };

        amd.append_byte(0xc4);
        amd.append_byte(r | x | b | self.encoding);
        amd.append_byte(w | vvvv | l | self.pp);
    }

    pub fn evex(&self, amd: &mut Amd) {
        let r = if self.reg & 8 != 0 { 0 } else { 0x80 };
        assert!(self.index == 0 || self.rm < 16);
        let x = if self.index & 8 != 0 || self.rm & 0x10 != 0 {
            0
        } else {
            0x40
        };
        let b = if self.rm & 8 != 0 { 0 } else { 0x20 };
        let r_prime = if self.reg & 0x10 != 0 { 0 } else { 0x10 };

        let vvvv = (!self.vreg & 0x0f) << 3;
        let v_prime = if self.vreg & 0x10 != 0 { 0 } else { 8 };

        let l = if self.len == 256 { 0x20 } else { 0 };
        let l_prime = if self.len == 512 { 0x40 } else { 0 };
        let (a, z): (u8, u8) = match self.mask {
            Masking::Nil => (0, 0),
            Masking::Copy(k) => (k, 0),
            Masking::Zero(k) => (k, 0x80),
        };

        let w = if self.w != 0 { 0x80 } else { 0 };
        let br = 0;

        amd.append_byte(0x62);
        amd.append_byte(r | x | b | r_prime | self.encoding);
        amd.append_byte(w | vvvv | 4 | self.pp);
        amd.append_byte(z | l_prime | l | br | v_prime | a);
    }

    pub fn modrm_mem(&mut self, amd: &mut Amd, offset: i32) {
        let n = self.len as i32 / 8;

        // compressed, aka, disp8*N mode
        let compressed = offset & (n - 1) == 0 && offset / n < 128;

        if compressed {
            amd.append_byte(0x40 + ((self.reg & 7) << 3) + (self.rm & 7))
        } else {
            amd.append_byte(0x80 + ((self.reg & 7) << 3) + (self.rm & 7))
        }

        if self.rm == Amd::RSP {
            amd.append_byte(0x24); // SIB byte for RSP
        }

        if compressed {
            amd.append_byte((offset / n) as u8);
        } else {
            amd.append_word(offset as u32);
        }
    }
}

#[allow(dead_code)]
impl Amd {
    pub fn vmovaqd(&mut self, reg: u8, rm: u8) {
        // self.vex_pd(reg, 0, rm, 0);
        Prefix::new(reg, 0, rm).evex(self);
        self.append_byte(0x28);
        self.modrm_reg(reg, rm);
    }

    pub fn vmovaqd_masked(&mut self, reg: u8, rm: u8, k: u8, zero: bool) {
        // self.vex_pd(reg, 0, rm, 0);

        let mask = if zero {
            Masking::Zero(k)
        } else {
            Masking::Copy(k)
        };

        Prefix::new(reg, 0, rm).set_mask(mask).evex(self);
        self.append_byte(0x28);
        self.modrm_reg(reg, rm);
    }

    pub fn vbroadcastsd_zmm(&mut self, reg: u8, rm: u8, offset: i32) {
        // self.vex3pd(reg, 0, rm, 0, 2);
        let mut p = Prefix::new(reg, 0, rm);
        p.set_encoding(2);
        p.evex(self);
        self.append_byte(0x19);
        p.modrm_mem(self, offset);
    }

    pub fn vbroadcastsd_zmm_label(&mut self, reg: u8, label: &str) {
        // self.vex3pd(reg, 0, 0, 0, 2);
        Prefix::new(reg, 0, 0).set_encoding(2).evex(self);
        self.append_byte(0x19);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn vmovqd_zmm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        // self.vex_pd(reg, 0, rm, 0);
        let mut p = Prefix::new(reg, 0, rm);
        p.evex(self);
        self.append_byte(0x10);
        p.modrm_mem(self, offset);
    }

    pub fn vmovqd_zmm_indexed(&mut self, reg: u8, base: u8, index: u8, scale: u8) {
        // self.vex_pd(reg, 0, base, index);
        Prefix::new(reg, 0, base).set_index(index).evex(self);
        self.append_byte(0x10);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vmovqd_zmm_label(&mut self, reg: u8, label: &str) {
        // self.vex_pd(reg, 0, 0, 0);
        Prefix::new(reg, 0, 0).evex(self);
        self.append_byte(0x10);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn vmovqd_mem_zmm(&mut self, rm: u8, offset: i32, reg: u8) {
        // self.vex_pd(reg, 0, rm, 0);
        let mut p = Prefix::new(reg, 0, rm);
        p.evex(self);
        self.append_byte(0x11);
        p.modrm_mem(self, offset);
    }

    pub fn vmovqd_indexed_zmm(&mut self, base: u8, index: u8, scale: u8, reg: u8) {
        // self.vex_pd(reg, 0, base, index);
        Prefix::new(reg, 0, base).set_index(index).evex(self);
        self.append_byte(0x11);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn vaddqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x58);
        self.modrm_reg(reg, rm);
    }

    pub fn vsubqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x5c);
        self.modrm_reg(reg, rm);
    }

    pub fn vmulqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x59);
        self.modrm_reg(reg, rm);
    }

    pub fn vdivqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x5e);
        self.modrm_reg(reg, rm);
    }

    pub fn vsqrtqd(&mut self, reg: u8, rm: u8) {
        // self.vex_pd(reg, 0, rm, 0);
        Prefix::new(reg, 0, rm).evex(self);
        self.append_byte(0x51);
        self.modrm_reg(reg, rm);
    }

    pub fn vroundqd(&mut self, reg: u8, rm: u8, mode: RoundingMode) {
        // self.vex3pd(reg, 0, rm, 0, 3);
        Prefix::new(reg, 0, rm).set_encoding(3).evex(self);
        self.append_byte(0x09);
        self.modrm_reg(reg, rm);
        self.append_byte(match mode {
            RoundingMode::Round => 0,
            RoundingMode::Floor => 1,
            RoundingMode::Ceiling => 2,
            RoundingMode::Trunc => 3,
        });
    }

    pub fn vandqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x54);
        self.modrm_reg(reg, rm);
    }

    pub fn vandnqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x55);
        self.modrm_reg(reg, rm);
    }

    pub fn vorqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x56);
        self.modrm_reg(reg, rm);
    }

    pub fn vxorqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x57);
        self.modrm_reg(reg, rm);
    }

    pub fn vcmpeqqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(0)
    }

    pub fn vcmpltqd(&mut self, k: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(k, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(k, rm);
        self.append_byte(1);
    }

    pub fn vcmpleqd(&mut self, k: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(k, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(k, rm);
        self.append_byte(2);
    }

    pub fn vcmpunordqd(&mut self, k: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(k, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(k, rm);
        self.append_byte(3);
    }

    pub fn vcmpneqqd(&mut self, k: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(k, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(k, rm);
        self.append_byte(4);
    }

    pub fn vcmpnltqd(&mut self, k: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(k, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(k, rm);
        self.append_byte(5);
    }

    pub fn vcmpnleqd(&mut self, k: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(k, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(k, rm);
        self.append_byte(6);
    }

    pub fn vcmpordqd(&mut self, k: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(k, vreg, rm).evex(self);
        self.append_byte(0xc2);
        self.modrm_reg(k, rm);
        self.append_byte(7);
    }

    pub fn vshufqd(&mut self, reg: u8, vreg: u8, rm: u8, imm8: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0xc6);
        self.modrm_reg(reg, rm);
        self.append_byte(imm8);
    }

    pub fn vunpckhqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x15);
        self.modrm_reg(reg, rm);
    }

    pub fn vunpcklqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0x14);
        self.modrm_reg(reg, rm);
    }

    pub fn vaddsubqd(&mut self, reg: u8, vreg: u8, rm: u8) {
        // self.vex_pd(reg, vreg, rm, 0);
        Prefix::new(reg, vreg, rm).evex(self);
        self.append_byte(0xd0);
        self.modrm_reg(reg, rm);
    }

    pub fn kmovw_reg_k(&mut self, reg: u8, k: u8) {
        Prefix::new(reg, 0, k).set_len(128).set_w(0).ps().vex(self);
        self.append_byte(0x93);
        self.modrm_reg(reg, k);
    }

    pub fn kmovw_k_reg(&mut self, k: u8, rm: u8) {
        Prefix::new(k, 0, rm).set_len(128).set_w(0).ps().vex(self);
        self.append_byte(0x92);
        self.modrm_reg(k, rm);
    }

    pub fn knotw(&mut self, k1: u8, k2: u8) {
        Prefix::new(k1, 0, k2).set_len(128).set_w(0).ps().vex(self);
        self.append_byte(0x44);
        self.modrm_reg(k1, k2);
    }

    pub fn vpmovq2m_qd(&mut self, k: u8, rm: u8) {
        Prefix::new(k, 0, rm)
            .set_w(1)
            .set_encoding(2)
            .ss()
            .evex(self);
        self.append_byte(0x39);
        self.modrm_reg(k, rm);
    }

    pub fn vpmovm2q_qd(&mut self, reg: u8, k: u8) {
        Prefix::new(reg, 0, k)
            .set_w(1)
            .set_encoding(2)
            .ss()
            .evex(self);
        self.append_byte(0x38);
        self.modrm_reg(reg, k);
    }
}
