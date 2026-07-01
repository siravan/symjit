use super::asm::Amd;
use crate::utils::DataType;

#[allow(dead_code)]
impl Amd {
    pub const RAX: u8 = 0;
    pub const RCX: u8 = 1;
    pub const RDX: u8 = 2;
    pub const RBX: u8 = 3;
    pub const RSP: u8 = 4;
    pub const RBP: u8 = 5;
    pub const RSI: u8 = 6;
    pub const RDI: u8 = 7;
    pub const R8: u8 = 8;
    pub const R9: u8 = 9;
    pub const R10: u8 = 10;
    pub const R11: u8 = 11;
    pub const R12: u8 = 12;
    pub const R13: u8 = 13;
    pub const R14: u8 = 14;
    pub const R15: u8 = 15;

    pub fn bytes(&self) -> Vec<u8> {
        self.a.bytes()
    }

    pub fn append_byte(&mut self, b: u8) {
        self.a.append_byte(b)
    }

    pub fn append_bytes(&mut self, bs: &[u8]) {
        self.a.append_bytes(bs)
    }

    pub fn append_word(&mut self, u: u32) {
        self.a.append_word(u)
    }

    pub fn append_quad(&mut self, u: u64) {
        self.a.append_quad(u)
    }

    pub fn jump(&mut self, label: &str) {
        self.a.jump(label, 0, |offset, _| (offset - 4) as u32);
    }

    pub fn modrm_reg(&mut self, reg: u8, rm: u8) {
        self.append_byte(0xc0 + ((reg & 7) << 3) + (rm & 7))
    }

    pub fn modrm_sib(&mut self, reg: u8, base: u8, index: u8, scale: u8) {
        self.append_byte(0x04 + ((reg & 7) << 3)); // R/M = 0b100, MOD = 0b00
        let scale = match scale {
            1 => 0,
            2 => 1 << 6,
            4 => 2 << 6,
            8 => 3 << 6,
            _ => {
                panic!("scale in SIB should be 1, 2, 4, or, 8.")
            }
        };
        self.append_byte((scale | (index & 7) << 3) | (base & 7));
    }

    pub fn rex(&mut self, reg: u8, rm: u8) {
        let b = 0x48 + ((rm & 8) >> 3) + ((reg & 8) >> 1);
        self.append_byte(b);
    }

    pub fn rex_index(&mut self, reg: u8, rm: u8, index: u8) {
        let b = 0x48 + ((rm & 8) >> 3) + ((index & 8) >> 2) + ((reg & 8) >> 1);
        self.append_byte(b);
    }

    pub fn modrm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        let small = (-128..128).contains(&offset);

        if small {
            self.append_byte(0x40 + ((reg & 7) << 3) + (rm & 7))
        } else {
            self.append_byte(0x80 + ((reg & 7) << 3) + (rm & 7))
        }

        if rm == Self::RSP {
            self.append_byte(0x24); // SIB byte for RSP
        }

        if small {
            self.append_byte(offset as u8);
        } else {
            self.append_word(offset as u32);
        }
    }

    pub fn vex2pd(&mut self, reg: u8, vreg: u8) {
        // This is the two-byte VEX prefix (VEX2) for packed-double (pd)
        // and 256-bit ymm registers
        let r = (!reg & 8) << 4;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 0, // ps
            DataType::F64 => 1, // pd
        };

        self.append_byte(0xc5);
        self.append_byte(r | vvvv | 4 | pp);
    }

    pub fn vex2sd(&mut self, reg: u8, vreg: u8) {
        // This is the two-byte VEX prefix (VEX2) for scalar-double (sd)
        // and 256-bit ymm registers
        let r = (!reg & 8) << 4;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 2, // ss
            DataType::F64 => 3, // sd
        };

        self.append_byte(0xc5);
        self.append_byte(r | vvvv | pp);
    }

    pub fn vex3pd(&mut self, reg: u8, vreg: u8, rm: u8, index: u8, encoding: u8) {
        // This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        // and 256-bit ymm registers
        // fnault encoding is 1
        let r = (!reg & 8) << 4;
        let x = (!index & 8) << 3;
        let b = (!rm & 8) << 2;
        let w = 0;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 0, // ps
            DataType::F64 => 1, // pd
        };

        self.append_byte(0xc4);
        self.append_byte(r | x | b | encoding);
        self.append_byte(w | vvvv | 4 | pp);
    }

    pub fn vex3pd_w1(&mut self, reg: u8, vreg: u8, rm: u8, index: u8, encoding: u8) {
        // This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        // and 256-bit ymm registers
        // fnault encoding is 1
        let r = (!reg & 8) << 4;
        let x = (!index & 8) << 3;
        let b = (!rm & 8) << 2;
        let w = 0x80;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 0, // ps
            DataType::F64 => 1, // pd
        };

        self.append_byte(0xc4);
        self.append_byte(r | x | b | encoding);
        self.append_byte(w | vvvv | 4 | pp);
    }

    pub fn vex3sd(&mut self, reg: u8, vreg: u8, rm: u8, index: u8, encoding: u8) {
        // This is the three-byte VEX prefix (VEX3) for scalar-double (sd)
        // and 256-bit ymm registers
        // default encoding is 1
        let r = (!reg & 8) << 4;
        let x = (!index & 8) << 3;
        let b = (!rm & 8) << 2;
        let w = 0;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 2, // ss
            DataType::F64 => 3, // sd
        };

        self.append_byte(0xc4);
        self.append_byte(r | x | b | encoding);
        self.append_byte(w | vvvv | pp);
    }

    pub fn vex_sd(&mut self, reg: u8, vreg: u8, rm: u8, index: u8) {
        if rm < 8 && index < 8 {
            self.vex2sd(reg, vreg);
        } else {
            self.vex3sd(reg, vreg, rm, index, 1);
        }
    }

    pub fn vex_pd(&mut self, reg: u8, vreg: u8, rm: u8, index: u8) {
        if rm < 8 && index < 8 {
            self.vex2pd(reg, vreg);
        } else {
            self.vex3pd(reg, vreg, rm, index, 1);
        }
    }

    pub fn vex2dd(&mut self, reg: u8, vreg: u8) {
        // This is the two-byte VEX prefix (VEX2) for packed-double (pd)
        // and 128-bit xmm registers
        let r = (!reg & 8) << 4;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 0, // ps
            DataType::F64 => 1, // pd
        };

        self.append_byte(0xc5);
        self.append_byte(r | vvvv | pp);
    }

    pub fn vex3dd(&mut self, reg: u8, vreg: u8, rm: u8, index: u8, encoding: u8) {
        // This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        // and 128-bit xmm registers
        // default encoding is 1
        let r = (!reg & 8) << 4;
        let x = (!index & 8) << 3;
        let b = (!rm & 8) << 2;
        let w = 0;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 0, // ps
            DataType::F64 => 1, // pd
        };

        self.append_byte(0xc4);
        self.append_byte(r | x | b | encoding);
        self.append_byte(w | vvvv | pp);
    }

    pub fn vex3dd_w1(&mut self, reg: u8, vreg: u8, rm: u8, index: u8, encoding: u8) {
        // This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        // and 128-bit xmm registers
        // default encoding is 1
        let r = (!reg & 8) << 4;
        let x = (!index & 8) << 3;
        let b = (!rm & 8) << 2;
        let w = 0x80;
        let vvvv = (!vreg & 0x0f) << 3;

        let pp = match self.dtype {
            DataType::F32 => 0, // ps
            DataType::F64 => 1, // pd
        };

        self.append_byte(0xc4);
        self.append_byte(r | x | b | encoding);
        self.append_byte(w | vvvv | pp);
    }

    pub fn vex_dd(&mut self, reg: u8, vreg: u8, rm: u8, index: u8) {
        if rm < 8 && index < 8 {
            self.vex2dd(reg, vreg);
        } else {
            self.vex3dd(reg, vreg, rm, index, 1);
        }
    }

    pub fn sse_sd(&mut self, reg: u8, rm: u8) {
        match self.dtype {
            DataType::F32 => self.append_byte(0xf3), // ss
            DataType::F64 => self.append_byte(0xf2), // sd
        };

        self.rex(reg, rm);
        self.append_byte(0x0f);
    }

    pub fn sse_sd_index(&mut self, reg: u8, rm: u8, index: u8) {
        match self.dtype {
            DataType::F32 => self.append_byte(0xf3), // ss
            DataType::F64 => self.append_byte(0xf2), // sd
        };

        self.rex_index(reg, rm, index);
        self.append_byte(0x0f);
    }

    pub fn sse_pd(&mut self, reg: u8, rm: u8) {
        match self.dtype {
            DataType::F32 => {}                      // ps
            DataType::F64 => self.append_byte(0x66), // pd
        };

        self.rex(reg, rm);
        self.append_byte(0x0f);
    }

    /*******************************************/
    pub fn vzeroupper(&mut self) {
        self.append_bytes(&[0xC5, 0xF8, 0x77]);
    }

    // general registers
    pub fn mov(&mut self, reg: u8, rm: u8) {
        self.rex(reg, rm);
        self.append_byte(0x8b);
        self.modrm_reg(reg, rm);
    }

    pub fn mov_reg_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        self.rex(reg, rm);
        self.append_byte(0x8b);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn lea_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        self.rex(reg, rm);
        self.append_byte(0x8d);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn mov_reg_label(&mut self, reg: u8, label: &str) {
        self.rex(reg, 0);
        self.append_byte(0x8b);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.jump(label);
    }

    pub fn mov_mem_reg(&mut self, rm: u8, offset: i32, reg: u8) {
        self.rex(reg, rm);
        self.append_byte(0x89);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn lea_indexed(&mut self, reg: u8, base: u8, index: u8, scale: u8) {
        self.rex(reg, 0);
        self.append_byte(0x8d);
        self.modrm_sib(reg, base, index, scale);
    }

    pub fn movabs(&mut self, rm: u8, imm64: u64) {
        self.rex(0, rm);
        self.append_byte(0xb8 + (rm & 7));
        self.append_word(imm64 as u32);
        self.append_word((imm64 >> 32) as u32);
    }

    pub fn call(&mut self, reg: u8) {
        if reg < 8 {
            self.append_bytes(&[0xff, 0xd0 | reg]);
        } else {
            self.append_bytes(&[0x41, 0xff, 0xd0 | (reg & 7)]);
        }
    }

    pub fn call_indirect(&mut self, label: &str) {
        self.append_bytes(&[0xff, 0x15]);
        self.jump(label);
    }

    // rip-related
    pub fn call_relative(&mut self, label: &str) {
        self.append_byte(0xe8);
        self.jump(label);
    }

    pub fn push(&mut self, reg: u8) {
        if reg < 8 {
            self.append_byte(0x50 | reg);
        } else {
            self.append_bytes(&[0x41, 0x50 | (reg & 7)]);
        }
    }

    pub fn pop(&mut self, reg: u8) {
        if reg < 8 {
            self.append_byte(0x58 | reg);
        } else {
            self.append_bytes(&[0x41, 0x58 | (reg & 7)]);
        }
    }

    pub fn ret(&mut self) {
        self.append_byte(0xc3);
    }

    pub fn add_rsp(&mut self, imm: u32) {
        self.append_bytes(&[0x48, 0x81, 0xc4]);
        self.append_word(imm);
    }

    pub fn sub_rsp(&mut self, imm: u32) {
        self.append_bytes(&[0x48, 0x81, 0xec]);
        self.append_word(imm);
    }

    pub fn or(&mut self, reg: u8, rm: u8) {
        self.rex(reg, rm);
        self.append_byte(0x0b);
        self.modrm_reg(reg, rm);
    }

    pub fn xor(&mut self, reg: u8, rm: u8) {
        self.rex(reg, rm);
        self.append_byte(0x33);
        self.modrm_reg(reg, rm);
    }

    pub fn add(&mut self, reg: u8, rm: u8) {
        self.rex(reg, rm);
        self.append_byte(0x03);
        self.modrm_reg(reg, rm);
    }

    pub fn add_imm(&mut self, rm: u8, imm: u32) {
        self.rex(0, rm);
        self.append_byte(0x81);
        self.modrm_reg(0, rm);
        self.append_word(imm);
    }

    pub fn mov_imm(&mut self, rm: u8, imm: u32) {
        self.rex(0, rm);
        self.append_byte(0xc7);
        self.modrm_reg(0, rm);
        self.append_word(imm);
    }

    pub fn or_imm(&mut self, rm: u8, imm: u32) {
        self.rex(0, rm);
        self.append_byte(0x81);
        self.modrm_reg(1, rm);
        self.append_word(imm);
    }

    pub fn and_imm(&mut self, rm: u8, imm: u32) {
        self.rex(0, rm);
        self.append_byte(0x81);
        self.modrm_reg(4, rm);
        self.append_word(imm);
    }

    pub fn sub_imm(&mut self, rm: u8, imm: u32) {
        self.rex(0, rm);
        self.append_byte(0x81);
        self.modrm_reg(5, rm);
        self.append_word(imm);
    }

    pub fn xor_imm(&mut self, rm: u8, imm: u32) {
        self.rex(0, rm);
        self.append_byte(0x81);
        self.modrm_reg(6, rm);
        self.append_word(imm);
    }

    pub fn cmp_imm(&mut self, rm: u8, imm: u32) {
        self.rex(0, rm);
        self.append_byte(0x81);
        self.modrm_reg(7, rm);
        self.append_word(imm);
    }

    pub fn inc(&mut self, rm: u8) {
        self.rex(0, rm);
        self.append_byte(0xff);
        self.modrm_reg(0, rm);
    }

    pub fn dec(&mut self, rm: u8) {
        self.rex(0, rm);
        self.append_byte(0xff);
        self.modrm_reg(1, rm);
    }

    pub fn jmp(&mut self, label: &str) {
        self.append_byte(0xe9);
        self.jump(label);
    }

    pub fn jz(&mut self, label: &str) {
        self.append_bytes(&[0x0f, 0x84]);
        self.jump(label);
    }

    pub fn jnz(&mut self, label: &str) {
        self.append_bytes(&[0x0f, 0x85]);
        self.jump(label);
    }

    pub fn jpe(&mut self, label: &str) {
        // jump if parity even is true if vucomisd returns
        // an unordered result
        self.append_bytes(&[0x0f, 0x8a]);
        self.jump(label);
    }

    pub fn jpo(&mut self, label: &str) {
        // jump if parity even is true if vucomisd returns
        // an unordered result
        self.append_bytes(&[0x0f, 0x8b]);
        self.jump(label);
    }

    pub fn js(&mut self, label: &str) {
        // jump if sign = 1
        self.append_bytes(&[0x0f, 0x88]);
        self.jump(label);
    }

    pub fn nop(&mut self) {
        self.append_byte(0x90);
    }

    pub fn prefetcht0_ip(&mut self, offset: u32) {
        self.append_bytes(&[0x0f, 0x18, 0x0d]);
        self.append_word(offset);
    }

    pub fn prefetcht1_ip(&mut self, offset: u32) {
        self.append_bytes(&[0x0f, 0x18, 0x15]);
        self.append_word(offset);
    }

    pub fn prefetcht2_ip(&mut self, offset: u32) {
        self.append_bytes(&[0x0f, 0x18, 0x1d]);
        self.append_word(offset);
    }

    pub fn prefetchnta_ip(&mut self, offset: u32) {
        self.append_bytes(&[0x0f, 0x18, 0x05]);
        self.append_word(offset);
    }
}
