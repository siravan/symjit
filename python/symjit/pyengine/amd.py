import sys

from . import assembler
from . import vectorizer


def reg_index(reg):
    if isinstance(reg, int):
        return reg

    if reg == "rax":
        return 0
    elif reg == "rcx":
        return 1
    elif reg == "rdx":
        return 2
    elif reg == "rbx":
        return 3
    elif reg == "rsp":
        return 4
    elif reg == "rbp":
        return 5
    elif reg == "rsi":
        return 6
    elif reg == "rdi":
        return 7
    elif reg == "r8":
        return 8
    elif reg == "r9":
        return 9
    elif reg == "r10":
        return 10
    elif reg == "r11":
        return 11
    elif reg == "r12":
        return 12
    elif reg == "r13":
        return 13
    elif reg == "r14":
        return 14
    elif reg == "r15":
        return 15
    else:
        raise Error("unregnized register")


class Amd(assembler.Assembler):
    def __init__(self):
        super().__init__()
        self.is_win = sys.platform == "win32"

    def apply_jumps(self):
        for label, k in self.jumps:
            target = self.labels[label]
            offset = target - (k + 4)
            self.buf[k] = offset & 0xFF
            self.buf[k + 1] = (offset >> 8) & 0xFF
            self.buf[k + 2] = (offset >> 16) & 0xFF
            self.buf[k + 3] = (offset >> 24) & 0xFF

    def modrm_reg(self, reg, rm):
        reg = reg_index(reg)
        rm = reg_index(rm)
        self.append_byte(0xC0 + ((reg & 7) << 3) + (rm & 7))

    def rex(self, reg, rm):
        reg = reg_index(reg)
        rm = reg_index(rm)
        self.append_byte(0x48 + ((rm & 8) >> 3) + ((reg & 8) >> 1))

    def modrm_mem(self, reg, rm, offset):
        assert offset >= 0
        reg = reg_index(reg)
        rm = reg_index(rm)

        if offset < 128:
            self.append_byte(0x40 + ((reg & 7) << 3) + (rm & 7))
        else:
            self.append_byte(0x80 + ((reg & 7) << 3) + (rm & 7))

        if rm == 4:
            self.append_byte(0x24)  # SIB byte for RSP

        if offset < 128:
            self.append_byte(offset)
        else:
            self.append_word(offset)

    def vex2pd(self, reg, vreg):
        """This is the two-byte VEX prefix (VEX2) for packed-double (pd)
        and 256-bit ymm registers"""
        r = (reg & 8) << 4
        vvvv = vreg << 3
        self.append_byte(0xC5)
        self.append_byte((r | vvvv | 5) ^ 0xF8)

    def vex2sd(self, reg, vreg):
        """This is the two-byte VEX prefix (VEX2) for packed-double (pd)
        and 256-bit ymm registers"""
        r = (reg & 8) << 4
        vvvv = vreg << 3
        self.append_byte(0xC5)
        self.append_byte((r | vvvv | 3) ^ 0xF8)

    def vex3pd(self, reg, vreg, rm, encoding=1):
        """This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        and 256-bit ymm registers"""
        r = (reg & 8) << 4
        b = (rm & 8) << 2
        vvvv = vreg << 3
        self.append_byte(0xC4)
        self.append_byte((r | b | encoding) ^ 0xE0)
        self.append_byte((vvvv | 5) ^ 0x78)

    def vex3sd(self, reg, vreg, rm, encoding=1):
        """This is the three-byte VEX prefix (VEX3) for packed-double (pd)
        and 256-bit ymm registers"""
        r = (reg & 8) << 4
        b = (rm & 8) << 2
        vvvv = vreg << 3
        self.append_byte(0xC4)
        self.append_byte((r | b | encoding) ^ 0xE0)
        self.append_byte((vvvv | 3) ^ 0x78)

    def vex_sd(self, reg, vreg, rm):
        rm = reg_index(rm)
        if rm < 8:
            self.vex2sd(reg, vreg)
        else:
            self.vex3sd(reg, vreg, rm)

    def vex_pd(self, reg, vreg, rm):
        rm = reg_index(rm)
        if rm < 8:
            self.vex2pd(reg, vreg)
        else:
            self.vex3pd(reg, vreg, rm)

    # AVX rules!
    def movapd(self, reg, rm):
        self.vex_pd(reg, 0, rm)
        self.append_byte(0x28)
        self.modrm_reg(reg, rm)
        return self

    def movsd_xmm_mem(self, reg, rm, offset):
        self.vex_sd(reg, 0, rm)
        self.append_byte(0x10)
        self.modrm_mem(reg, rm, offset)
        return self

    def movsd_mem_xmm(self, rm, offset, reg):
        self.vex_sd(reg, 0, rm)
        self.append_byte(0x11)
        self.modrm_mem(reg, rm, offset)
        return self

    def vbroadcastsd(self, reg, rm, offset):
        self.vex3pd(reg, 0, rm, 2)
        self.append_byte(0x19)
        self.modrm_mem(reg, rm, offset)
        return self

    def vaddsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0x58)
        self.modrm_reg(reg, rm)
        return self

    def vsubsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0x5C)
        self.modrm_reg(reg, rm)
        return self

    def vmulsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0x59)
        self.modrm_reg(reg, rm)
        return self

    def vdivsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0x5E)
        self.modrm_reg(reg, rm)
        return self

    def vsqrtsd(self, reg, rm):
        self.vex_sd(reg, 0, rm)
        self.append_byte(0x51)
        self.modrm_reg(reg, rm)
        return self

    def vandpd(self, reg, vreg, rm):
        self.vex_pd(reg, vreg, rm)
        self.append_byte(0x54)
        self.modrm_reg(reg, rm)
        return self

    def vandnpd(self, reg, vreg, rm):
        self.vex_pd(reg, vreg, rm)
        self.append_byte(0x55)
        self.modrm_reg(reg, rm)
        return self

    def vorpd(self, reg, vreg, rm):
        self.vex_pd(reg, vreg, rm)
        self.append_byte(0x56)
        self.modrm_reg(reg, rm)
        return self

    def vxorpd(self, reg, vreg, rm):
        self.vex_pd(reg, vreg, rm)
        self.append_byte(0x57)
        self.modrm_reg(reg, rm)
        return self

    def vcmpeqsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(0)
        return self

    def vcmpltsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(1)
        return self

    def vcmplesd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(2)
        return self

    def vcmpunordsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(3)
        return self

    def vcmpneqsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(4)
        return self

    def vcmpnltsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(5)
        return self

    def vcmpnlesd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(6)
        return self

    def vcmpordsd(self, reg, vreg, rm):
        self.vex_sd(reg, vreg, rm)
        self.append_byte(0xC2)
        self.modrm_reg(reg, rm)
        self.append_byte(7)
        return self

    def vzeroupper(self):
        self.append_byte(0xC5, 0xF8, 0x77)
        return self

    # general registers
    def mov(self, reg, rm):
        self.rex(reg, rm)
        self.append_byte(0x8B)
        self.modrm_reg(reg, rm)
        return self

    def mov_reg_mem(self, reg, rm, offset):
        self.rex(reg, rm)
        self.append_byte(0x8B)
        self.modrm_mem(reg, rm, offset)
        return self

    def mov_mem_reg(self, rm, offset, reg):
        self.rex(reg, rm)
        self.append_byte(0x89)
        self.modrm_mem(reg, rm, offset)
        return self

    def movabs(self, rm, imm64):
        self.rex(0, rm)
        self.append_byte(0xB8 + (reg_index(rm) & 7))
        self.append_word(imm64)
        self.append_word(imm64 >> 32)
        return self

    def call(self, reg):
        reg = reg_index(reg)
        if reg < 8:
            self.append_byte(0xFF, 0xD0 | reg)
        else:
            self.append_byte(0x41, 0xFF, 0xD0 | (reg & 7))
        return self

    def push(self, reg):
        reg = reg_index(reg)
        if reg < 8:
            self.append_byte(0x50 | reg)
        else:
            self.append_byte(0x41, 0x50 | (reg & 7))
        return self

    def pop(self, reg):
        reg = reg_index(reg)
        if reg < 8:
            self.append_byte(0x58 | reg)
        else:
            self.append_byte(0x41, 0x58 | (reg & 7))
        return self

    def ret(self):
        self.append_byte(0xC3)
        return self

    def add_rsp(self, imm):
        self.append_byte(0x48, 0x81, 0xC4)
        self.append_word(imm)
        return self

    def sub_rsp(self, imm):
        self.append_byte(0x48, 0x81, 0xEC)
        self.append_word(imm)
        return self

    def xor(self, reg, rm):
        self.rex(reg, rm)
        self.append_byte(0x33)
        self.modrm_reg(reg, rm)

    def add(self, reg, rm):
        self.rex(reg, rm)
        self.append_byte(0x03)
        self.modrm_reg(reg, rm)

    def add_imm(self, rm, imm):
        self.rex(0, rm)
        self.append_byte(0x81)
        self.modrm_reg(0, rm)
        self.append_word(imm)

    def inc(self, rm):
        self.rex(0, rm)
        self.append_byte(0xFF)
        self.modrm_reg(0, rm)

    def dec(self, rm):
        self.rex(0, rm)
        self.append_byte(0xFF)
        self.modrm_reg(1, rm)

    def jnz(self, label):
        self.append_byte(0x0F, 0x85)
        self.jump(label)


class AmdIR:
    def __init__(self, mem):
        self.amd = Amd()
        self.mem = mem
        # shadows are XMM/YMM registers that shadow the stack slots
        self.first_shadow = 3
        self.count_shadows = 3 if sys.platform == "win32" else 13

    def vectorize(self):
        return vectorizer.vectorize_amd(self.amd, self.mem)

    def buf(self):
        return self.amd.buf

    def load_stack(self, dst, idx):
        self.amd.movsd_xmm_mem(dst, "rsp", 8 * idx)

    def save_stack(self, src, idx):
        self.amd.movsd_mem_xmm("rsp", 8 * idx, src)

    def load_mem(self, dst, idx):
        self.amd.movsd_xmm_mem(dst, "rbp", 8 * idx)

    def save_mem(self, src, idx):
        self.amd.movsd_mem_xmm("rbp", 8 * idx, src)

    def neg(self, dst):
        self.amd.movsd_xmm_mem(1, "rbp", 8 * self.mem.constant(-0.0))
        self.amd.vxorpd(dst, 0, 1)

    def abs(self, dst):
        self.amd.movsd_xmm_mem(1, "rbp", 8 * self.mem.constant(-0.0))
        self.amd.vandnpd(dst, 1, 0)

    def root(self, dst):
        self.amd.vsqrtsd(dst, 0)

    def square(self, dst):
        self.amd.vmulsd(dst, 0, 0)

    def cube(self, dst):
        self.amd.vmulsd(1, 0, 0)
        self.amd.vmulsd(dst, 0, 1)

    def recip(self, dst):
        self.amd.movsd_xmm_mem(1, "rbp", 8 * self.mem.constant(1.0))
        self.amd.vdivsd(dst, 1, 0)

    def plus(self, dst, r):
        self.amd.vaddsd(dst, 0, r)

    def minus(self, dst, r):
        self.amd.vsubsd(dst, 0, r)

    def times(self, dst, r):
        self.amd.vmulsd(dst, 0, r)

    def divide(self, dst, r):
        self.amd.vdivsd(dst, 0, r)

    def gt(self, dst, r):
        self.amd.vcmpnlesd(dst, 0, r)

    def geq(self, dst, r):
        self.amd.vcmpnltsd(dst, 0, r)

    def lt(self, dst, r):
        self.amd.vcmpltsd(dst, 0, r)

    def leq(self, dst, r):
        self.amd.vcmplesd(dst, 0, r)

    def eq(self, dst, r):
        self.amd.vcmpeqsd(dst, 0, r)

    def neq(self, dst, r):
        self.amd.vcmpneqsd(dst, 0, r)

    def boolean_and(self, dst, r):
        self.amd.vandpd(dst, 0, r)

    def boolean_or(self, dst, r):
        self.amd.vorpd(dst, 0, r)

    def boolean_xor(self, dst, r):
        self.amd.vxorpd(dst, 0, r)

    def call_unary(self, dst, idx):
        # Windows 32-byte home area
        if self.amd.is_win:
            self.amd.sub_rsp(32)

        self.amd.mov_reg_mem("rax", "rbx", 8 * idx)
        self.amd.call("rax")

        if dst != 0:
            self.amd.movapd(dst, 0)

        if self.amd.is_win:
            self.amd.add_rsp(32)

    def call_binary(self, dst, r, idx):
        # Windows 32-byte home area
        if self.amd.is_win:
            self.amd.sub_rsp(32)

        if r != 1:
            self.amd.movapd(1, r)

        self.amd.mov_reg_mem("rax", "rbx", 8 * idx)
        self.amd.call("rax")

        if dst != 0:
            self.amd.movapd(dst, 0)

        if self.amd.is_win:
            self.amd.add_rsp(32)

    def ifelse(self, dst, cond, true_reg, false_reg):
        self.amd.vandpd(true_reg, true_reg, cond)
        self.amd.vandnpd(cond, cond, false_reg)
        self.amd.vorpd(dst, true_reg, cond)

    def stack_size(self):
        cap = self.mem.stack_size
        pad = (cap + 1) & 1
        return (cap + pad) * 8

    def prepend_prologue(self):
        # note that we generate the prologue after the main body
        # because we need to know the stack size
        self.amd.begin_prepend()

        self.amd.push("rbp")
        self.amd.push("rbx")

        if self.amd.is_win:
            self.amd.mov("rbp", "rcx")
            self.amd.mov("rbx", "rdx")  # different than the rust code
        else:
            self.amd.mov("rbp", "rdi")
            self.amd.mov("rbx", "rsi")  # different than the rust code

        self.amd.sub_rsp(self.stack_size())
        self.amd.end_prepend()

    def append_epilogue(self):
        self.amd.vzeroupper()
        self.amd.add_rsp(self.stack_size())
        self.amd.pop("rbx")
        self.amd.pop("rbp")
        self.amd.ret()


##########################################################################


def test_avx():
    amd = Amd()
    assert amd.push("rbp").test([0x55])
    assert amd.push("rbx").test([0x53])
    assert amd.mov_reg_reg("rbp", "rdi").test([0x48, 0x8B, 0xEF])
    assert amd.vmovapd_ymm_ymm(1, 5).test([0xC5, 0xFD, 0x28, 0xCD])
    assert amd.vmovsd_xmm_mem(2, "rbp", 0x1234).test(
        [0xC5, 0xFB, 0x10, 0x95, 0x34, 0x12, 0x00, 0x00]
    )
    assert amd.vmovapd_ymm_mem(1, "rbp", 0x1234).test(
        [0xC5, 0xFD, 0x28, 0x8D, 0x34, 0x12, 0x00, 0x00]
    )
    assert amd.vmovupd_mem_ymm("rbp", 0x1234, 0).test(
        [0xC5, 0xFD, 0x11, 0x85, 0x34, 0x12, 0x00, 0x00]
    )
    assert amd.vmulpd(2, 0, 1).test([0xC5, 0xFD, 0x59, 0xD1])
    assert amd.vdivpd(12, 0, 1).test([0xC5, 0x7D, 0x5E, 0xE1])
    assert amd.mov_reg_mem("rax", "rbx", 0x10).test([0x48, 0x8B, 0x43, 0x10])
    assert amd.mov_reg_mem("rbx", "rbx", 0x1234).test(
        [0x48, 0x8B, 0x9B, 0x34, 0x12, 0x00, 0x00]
    )
    assert amd.call("rax").test([0xFF, 0xD0])
    assert amd.vxorpd(4, 0, 1).test([0xC5, 0xFD, 0x57, 0xE1])
    assert amd.vcmpnltpd(15, 0, 1).test([0xC5, 0x7D, 0xC2, 0xF9, 0x05])
    assert amd.vandnpd(1, 13, 1).test([0xC5, 0x95, 0x55, 0xC9])
    assert amd.vandpd(9, 4, 2).test([0xC5, 0x5D, 0x54, 0xCA])
    assert amd.vorpd(7, 4, 5).test([0xC5, 0xDD, 0x56, 0xFD])
    assert amd.vxorpd(7, 0, 1).test([0xC5, 0xFD, 0x57, 0xF9])
    assert amd.vmovapd_ymm_ymm(7, 1).test([0xC5, 0xFD, 0x28, 0xF9])
    assert amd.vaddpd(7, 0, 5).test([0xC5, 0xFD, 0x58, 0xFD])
    assert amd.vsqrtpd(15, 1).test([0xC5, 0x7D, 0x51, 0xF9])
    assert amd.vzeroupper().test([0xC5, 0xF8, 0x77])
    assert amd.pop("rbp").test([0x5D])
    assert amd.ret().test([0xC3])
    assert amd.add_rsp(0x1234).test([0x48, 0x81, 0xC4, 0x34, 0x12, 0x00, 0x00])
    assert amd.sub_rsp(0x4321).test([0x48, 0x81, 0xEC, 0x21, 0x43, 0x00, 0x00])
