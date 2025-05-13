use crate::assembler::Assembler;

pub struct Amd {
    pub a: Assembler,
}

impl Amd {
    pub fn new() -> Amd {
        Amd {
            a: Assembler::new(-4),
        }
    }

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

    pub fn modrm_reg(&mut self, reg: u8, rm: u8) {
        self.append_byte(0xc0 + ((reg & 7) << 3) + (rm & 7))
    }

    pub fn rex(&mut self, reg: u8, rm: u8) {
        self.append_byte(0x48 + ((rm & 8) >> 3) + ((reg & 8) >> 1))
    }

    pub fn modrm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        let small = offset < 128 && offset >= -128;

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
        let r = (reg & 8) << 4;
        let vvvv = vreg << 3;
        self.append_byte(0xc5);
        self.append_byte((r | vvvv | 5) ^ 0xf8);
    }

    pub fn vex2sd(&mut self, reg: u8, vreg: u8) {
        // This is the two-byte VEX prefix (VEX2) for packed-double (pd)
        // and 256-bit ymm registers
        let r = (reg & 8) << 4;
        let vvvv = vreg << 3;
        self.append_byte(0xc5);
        self.append_byte((r | vvvv | 3) ^ 0xf8);
    }

    pub fn vex3pd(&mut self, reg: u8, vreg: u8, rm: u8, encoding: u8) {
        // This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        // and 256-bit ymm registers
        // fnault encoding is 1
        let r = (reg & 8) << 4;
        let b = (rm & 8) << 2;
        let vvvv = vreg << 3;
        self.append_byte(0xc4);
        self.append_byte((r | b | encoding) ^ 0xe0);
        self.append_byte((vvvv | 5) ^ 0x78);
    }

    pub fn vex3sd(&mut self, reg: u8, vreg: u8, rm: u8, encoding: u8) {
        // This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        // and 256-bit ymm registers
        // fnault encoding is 1
        let r = (reg & 8) << 4;
        let b = (rm & 8) << 2;
        let vvvv = vreg << 3;
        self.append_byte(0xc4);
        self.append_byte((r | b | encoding) ^ 0xe0);
        self.append_byte((vvvv | 3) ^ 0x78);
    }

    pub fn vex_sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        if rm < 8 {
            self.vex2sd(reg, vreg);
        } else {
            self.vex3sd(reg, vreg, rm, 1);
        }
    }

    pub fn vex_pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        if rm < 8 {
            self.vex2pd(reg, vreg);
        } else {
            self.vex3pd(reg, vreg, rm, 1);
        }
    }

    // AVX rules!
    pub fn vmovapd(&mut self, reg: u8, rm: u8) {
        self.vex_pd(reg, 0, rm);
        self.append_byte(0x28);
        self.modrm_reg(reg, rm);
    }

    pub fn vmovsd_xmm_mem(&mut self, reg: u8, rm: u8, offset: i32) {
        self.vex_sd(reg, 0, rm);
        self.append_byte(0x10);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vmovsd_xmm_label(&mut self, reg: u8, label: &str) {
        self.vex_sd(reg, 0, 0);
        self.append_byte(0x10);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.a.jump(label, 0);
    }

    pub fn vmovsd_mem_xmm(&mut self, rm: u8, offset: i32, reg: u8) {
        self.vex_sd(reg, 0, rm);
        self.append_byte(0x11);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vbroadcastsd(&mut self, reg: u8, rm: u8, offset: i32) {
        self.vex3pd(reg, 0, rm, 2);
        self.append_byte(0x19);
        self.modrm_mem(reg, rm, offset);
    }

    pub fn vaddsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0x58);
        self.modrm_reg(reg, rm);
    }

    pub fn vsubsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0x5c);
        self.modrm_reg(reg, rm);
    }

    pub fn vmulsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0x59);
        self.modrm_reg(reg, rm);
    }

    pub fn vdivsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0x5e);
        self.modrm_reg(reg, rm);
    }

    pub fn vsqrtsd(&mut self, reg: u8, rm: u8) {
        self.vex_sd(reg, 0, rm);
        self.append_byte(0x51);
        self.modrm_reg(reg, rm);
    }

    pub fn vandpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm);
        self.append_byte(0x54);
        self.modrm_reg(reg, rm);
    }

    pub fn vandnpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm);
        self.append_byte(0x55);
        self.modrm_reg(reg, rm);
    }

    pub fn vorpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm);
        self.append_byte(0x56);
        self.modrm_reg(reg, rm);
    }

    pub fn vxorpd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_pd(reg, vreg, rm);
        self.append_byte(0x57);
        self.modrm_reg(reg, rm);
    }

    pub fn vcmpeqsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(0)
    }

    pub fn vcmpltsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(1);
    }

    pub fn vcmplesd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(2);
    }

    pub fn vcmpunordsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(3);
    }

    pub fn vcmpneqsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(4);
    }

    pub fn vcmpnltsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(5);
    }

    pub fn vcmpnlesd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xc2);
        self.modrm_reg(reg, rm);
        self.append_byte(6);
    }

    pub fn vcmpordsd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vex_sd(reg, vreg, rm);
        self.append_byte(0xC2);
        self.modrm_reg(reg, rm);
        self.append_byte(7);
    }

    pub fn vucomisd(&mut self, reg: u8, rm: u8) {
        self.vex_pd(reg, 0, rm);
        self.append_byte(0x2e);
        self.modrm_reg(reg, rm);
    }

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

    pub fn mov_reg_label(&mut self, reg: u8, label: &str) {
        self.rex(reg, 0);
        self.append_byte(0x8b);
        // modr/m byte with MOD=00 and R/M=101 (RIP-relative address)
        self.append_byte(5 | ((reg & 7) << 3));
        self.a.jump(label, 0);
    }

    pub fn mov_mem_reg(&mut self, rm: u8, offset: i32, reg: u8) {
        self.rex(reg, rm);
        self.append_byte(0x89);
        self.modrm_mem(reg, rm, offset);
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
        self.a.jump(label, 0);
    }

    pub fn jnz(&mut self, label: &str) {
        self.append_bytes(&[0x0f, 0x85]);
        self.a.jump(label, 0);
    }

    pub fn jpe(&mut self, label: &str) {
        // jump if parity even is true if vucomisd returns
        // an unordered result
        self.append_bytes(&[0x0f, 0x8a]);
        self.a.jump(label, 0);
    }

    pub fn quad(&mut self, val: u64) {
        // pseudo-instruction dq
        self.append_word(val as u32);
        self.append_word((val >> 32) as u32);
    }

    pub fn nop(&mut self) {
        self.append_byte(0x90);
    }
}

/*
class AmdSysVStack:
    pub fn __init__(&mut self, mem):
        self.mem = mem
        # shadows are XMM/YMM registers that shadow the stack slots
        self.first_shadow = 2
        self.count_shadows = 14
        self.count_simd_args = 8

    pub fn offset(&mut self, idx):
        ns = self.mem.count_states

        if idx < ns:
            if idx < 8:
                return 8 * (-(1 + idx))
            else:
                return 8 * (idx - 6)
        else:
            return 8 * (-(1 + idx - max(0, ns - 8)))

    pub fn frame_size(self):
        # cap = self.mem.stack_size + min(self.mem.count_states, 8) + self.mem.count_obs
        cap = min(self.mem.count_states, 8) + self.mem.count_obs + self.mem.COUNT_SPILLS
        pad = cap & 1
        return 8 * (cap + pad)


class AmdWindowsStack:
    pub fn __init__(&mut self, mem):
        self.mem = mem
        # shadows are XMM/YMM registers that shadow the stack slots
        self.first_shadow = 2  # XMM2-XMM5
        self.count_shadows = 4
        self.count_simd_args = 4

    pub fn offset(&mut self, idx):
        ns = self.mem.count_states

        if idx < ns:
            return 8 * (idx + 2)
        else:
            return 8 * (-(1 + idx - ns))

    pub fn frame_size(self):
        # cap = self.mem.stack_size + self.mem.count_obs
        cap = self.mem.count_obs + self.mem.COUNT_SPILLS
        pad = cap & 1
        return 8 * (cap + pad)

*/
