from . import assembler
from . import vectorizer


class Arm(assembler.Assembler):
    def __init__(self):
        self.buf = bytearray()
        self.tail = None
        self.labels = {}
        self.jumps = []

    def apply_jumps(self):
        for label, k in self.jumps:
            target = self.labels[label]
            offset = (target - k) >> 2
            imm19 = (offset & 0x0007FFFF) << 5
            self.buf[k] |= imm19 & 0xFF
            self.buf[k + 1] |= (imm19 >> 8) & 0xFF
            self.buf[k + 2] |= (imm19 >> 16) & 0xFF
            self.buf[k + 3] |= (imm19 >> 24) & 0xFF

    def rd(self, x):
        if x == "sp":
            x = 31
        elif x == "lr":
            x = 30
        elif not isinstance(x, int):
            raise ValueError(f"reg {x} not defined")

        assert x < 32
        return x

    def rn(self, x):
        if x == "sp":
            x = 31
        elif x == "lr":
            x = 30
        elif not isinstance(x, int):
            raise ValueError(f"reg {x} not defined")

        assert x < 32
        return x << 5

    def rd2(self, x):
        assert x < 32
        return x << 10

    def rm(self, x):
        assert x < 32
        return x << 16

    def imm(self, x):
        assert x < 4096
        return x << 10

    def imm19(self, x):
        assert abs(x) < 262144
        return x << 10

    def ofs(self, x):
        assert (x & 7 == 0) and (x < 32768)
        return x << 7

    def of7(self, x):
        assert (x & 7 == 0) and (x <= 504)
        return x << 12

    # main rules
    def fmov(self, rd, rn):
        self.append_word(0x1E604000 | self.rd(rd) | self.rn(rn))
        return self

    def mov(self, rd, rm):
        self.append_word(0xAA0003E0 | self.rd(rd) | self.rm(rm))
        return self

    # single register load/store instructions
    def ldr_d(self, rd, rn, ofs):
        # ldr d(rd), [x(rn), ofs]
        self.append_word(0xFD400000 | self.rd(rd) | self.rn(rn) | self.ofs(ofs))
        return self

    def ldr_x(self, rd, rn, ofs):
        # ldr x(rd), [x(rn), ofs]
        self.append_word(0xF9400000 | self.rd(rd) | self.rn(rn) | self.ofs(ofs))
        return self

    def str_d(self, rd, rn, ofs):
        # str d(rd), [x(rn), ofs]
        self.append_word(0xFD000000 | self.rd(rd) | self.rn(rn) | self.ofs(ofs))
        return self

    def str_x(self, rd, rn, ofs):
        # ldr x(rd), [x(rn), ofs]
        self.append_word(0xF9000000 | self.rd(rd) | self.rn(rn) | self.ofs(ofs))
        return self

    # paired-registers load/store instructions
    def ldp_d(self, rd, rd2, rn, of7):
        # ldp d(rd), d(rd2), [x(rn), ofs]
        self.append_word(
            0x6D400000 | self.rd(rd) | self.rd2(rd2) | self.rn(rn) | self.of7(of7)
        )
        return self

    def ldp_x(self, rd, rd2, rn, of7):
        # ldr x(rd), x(rd2), [x(rn), ofs]
        self.append_word(
            0xA9400000 | self.rd(rd) | self.rd2(rd2) | self.rn(rn) | self.of7(of7)
        )
        return self

    def stp_d(self, rd, rd2, rn, of7):
        # stp d(rd), d(rd2), [x(rn), ofs]
        self.append_word(
            0x6D000000 | self.rd(rd) | self.rd2(rd2) | self.rn(rn) | self.of7(of7)
        )
        return self

    def stp_x(self, rd, rd2, rn, of7):
        # stp x(rd), x(rd2), [x(rn), ofs]
        self.append_word(
            0xA9000000 | self.rd(rd) | self.rd2(rd2) | self.rn(rn) | self.of7(of7)
        )
        return self

    # x-registers immediate ops
    def add_imm(self, rd, rn, imm):
        # add x(rd), x(rn), imm
        self.append_word(0x91000000 | self.rd(rd) | self.rn(rn) | self.imm(imm))
        return self

    def sub_imm(self, rd, rn, imm):
        # sub x(rd), x(rn), imm
        self.append_word(0xD1000000 | self.rd(rd) | self.rn(rn) | self.imm(imm))
        return self

    def adds_imm(self, rd, rn, imm):
        # add x(rd), x(rn), imm
        self.append_word(0xB1000000 | self.rd(rd) | self.rn(rn) | self.imm(imm))
        return self

    def subs_imm(self, rd, rn, imm):
        # sub x(rd), x(rn), imm
        self.append_word(0xF1000000 | self.rd(rd) | self.rn(rn) | self.imm(imm))
        return self

    def add(self, rd, rn, rm):
        # add x(rd), x(rn), x(rm)
        self.append_word(0x8B000000 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def sub(self, rd, rn, rm):
        # sub x(rd), x(rn), x(rm)
        self.append_word(0xCB000000 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def add_lsl(self, rd, rn, rm, shift):
        # add x(rd), x(rn), x(rm), LSL #shift
        self.append_word(
            0x8B000000 | self.rd(rd) | self.rn(rn) | self.rm(rm) | self.imm(shift & 3)
        )
        return self

    def sub_lsl(self, rd, rn, rm, shift):
        # sub x(rd), x(rn), x(rm), LSL #shift
        self.append_word(
            0xCB000000 | self.rd(rd) | self.rn(rn) | self.rm(rm) | self.imm(shift & 3)
        )
        return self

    # floating point ops
    def fadd(self, rd, rn, rm):
        # fadd d(rd), d(rn), d(rm)
        self.append_word(0x1E602800 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def fsub(self, rd, rn, rm):
        # fsub d(rd), d(rn), d(rm)
        self.append_word(0x1E603800 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def fmul(self, rd, rn, rm):
        # fmul d(rd), d(rn), d(rm)
        self.append_word(0x1E600800 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def fdiv(self, rd, rn, rm):
        # fdiv d(rd), d(rn), d(rm)
        self.append_word(0x1E601800 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def fsqrt(self, rd, rn):
        # fsqrt d(rd), d(rn)
        self.append_word(0x1E61C000 | self.rd(rd) | self.rn(rn))
        return self

    def fneg(self, rd, rn):
        # fneg d(rd), d(rn)
        self.append_word(0x1E614000 | self.rd(rd) | self.rn(rn))
        return self

    def fabs(self, rd, rn):
        # fabs d(rd), d(rn)
        self.append_word(0x1E60C000 | self.rd(rd) | self.rn(rn))
        return self

    # logical ops
    def vand(self, rd, rn, rm):
        # `vand` instead of `and` because `and` is a reserved word
        # and v(rd).8b, v(rn).8b, v(rm).8b
        self.append_word(0x0E201C00 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def orr(self, rd, rn, rm):
        # orr v(rd).8b, v(rn).8b, v(rm).8b
        self.append_word(0x0EA01C00 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def eor(self, rd, rn, rm):
        # eor v(rd).8b, v(rn).8b, v(rm).8b
        self.append_word(0x2E201C00 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def bsl(self, rd, rn, rm):
        # bitwise select: rd = v(rd) ? v(rn) : v(rm)
        # bsl v(rd).8b, v(rn).8b, v(rm).8b
        self.append_word(0x2E601C00 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def vnot(self, rd, rn):
        # `vnot` instead of `not` because `not` is a reserved word
        # not v(rd).8b, v(rn).8b, v(rm).8b
        self.append_word(0x2E205800 | self.rd(rd) | self.rn(rn))
        return self

    # comparison
    def fcmeq(self, rd, rn, rm):
        # fcmeq d(rd), d(rn), d(rm)
        self.append_word(0x5E60E400 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    # note that rm and rn are exchanged in fcmlt and fcmle
    def fcmlt(self, rd, rn, rm):
        # fcmlt d(rd), d(rn), d(rm)
        self.append_word(0x7EE0E400 | self.rd(rd) | self.rn(rm) | self.rm(rn))
        return self

    def fcmle(self, rd, rn, rm):
        # fcmle d(rd), d(rn), d(rm)
        self.append_word(0x7E60E400 | self.rd(rd) | self.rn(rm) | self.rm(rn))
        return self

    def fcmgt(self, rd, rn, rm):
        # fcmgt d(rd), d(rn), d(rm)
        self.append_word(0x7EE0E400 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    def fcmge(self, rd, rn, rm):
        # fcmge d(rd), d(rn), d(rm)
        self.append_word(0x7E60E400 | self.rd(rd) | self.rn(rn) | self.rm(rm))
        return self

    # misc
    def blr(self, rn):
        # blr x(rd)
        self.append_word(0xD63F0000 | self.rn(rn))
        return self

    def b_eq(self, label):
        self.jump(label, code=0x54000000)

    def b_ne(self, label):
        self.jump(label, code=0x54000001)

    def b_lt(self, label):
        self.jump(label, code=0x5400000B)

    def b_le(self, label):
        self.jump(label, code=0x5400000D)

    def b_gt(self, label):
        self.jump(label, code=0x5400000C)

    def b_ge(self, label):
        self.jump(label, code=0x5400000A)

    def ret(self):
        # ret
        self.append_word(0xD65F03C0)
        return self

    def fmov_const(self, rd, val):
        # fmov d(rd), val
        if val == 0.0:
            self.append_word(0x9E6703E0 | self.rd(rd))
        elif val == 1.0:
            self.append_word(0x1E6E1000 | self.rd(rd))
        elif val == -1.0:
            self.append_word(0x1E7E1000 | self.rd(rd))
        else:
            raise ValueError(f"constant {val} not defined")
        return self


class ArmIR:
    def __init__(self, mem):
        self.arm = Arm()
        self.mem = mem
        # shadows are d(3)-d(7)
        self.first_shadow = 3
        self.count_shadows = 5

    def vectorize(self):
        return vectorizer.vectorize_arm(self.arm, self.mem)

    def buf(self):
        return self.arm.buf

    def load_stack(self, dst, idx):
        self.arm.ldr_d(dst, "sp", 8 * idx)

    def save_stack(self, src, idx):
        self.arm.str_d(src, "sp", 8 * idx)

    def load_mem(self, dst, idx):
        self.arm.ldr_d(dst, 19, 8 * idx)

    def save_mem(self, src, idx):
        self.arm.str_d(src, 19, 8 * idx)

    def neg(self, dst):
        self.arm.fneg(dst, 0)

    def abs(self, dst):
        self.arm.fabs(dst, 0)

    def root(self, dst):
        self.arm.fsqrt(dst, 0)

    def square(self, dst):
        self.arm.fmul(dst, 0, 0)

    def cube(self, dst):
        self.arm.fmul(1, 0, 0)
        self.arm.fmul(dst, 0, 1)

    def recip(self, dst):
        self.arm.fmov_const(1, 1.0)
        self.arm.fdiv(dst, 1, 0)

    def plus(self, dst, r):
        self.arm.fadd(dst, 0, r)

    def minus(self, dst, r):
        self.arm.fsub(dst, 0, r)

    def times(self, dst, r):
        self.arm.fmul(dst, 0, r)

    def divide(self, dst, r):
        self.arm.fdiv(dst, 0, r)

    def gt(self, dst, r):
        self.arm.fcmgt(dst, 0, r)

    def geq(self, dst, r):
        self.arm.fcmge(dst, 0, r)

    def lt(self, dst, r):
        self.arm.fcmlt(dst, 0, r)

    def leq(self, dst, r):
        self.arm.fcmle(dst, 0, r)

    def eq(self, dst, r):
        self.arm.fcmeq(dst, 0, r)

    def neq(self, dst, r):
        self.arm.fcmeq(dst, 0, r)
        self.arm.vnot(0, 0)

    def boolean_and(self, dst, r):
        self.arm.vand(dst, 0, r)

    def boolean_or(self, dst, r):
        self.arm.orr(dst, 0, r)

    def boolean_xor(self, dst, r):
        self.arm.eor(dst, 0, r)

    def call_unary(self, dst, idx):
        self.arm.ldr_x(0, 20, 8 * idx)
        self.arm.blr(0)
        if dst != 0:
            self.arm.fmov(dst, 0)

    def call_binary(self, dst, r, idx):
        if r != 1:
            self.arm.fmov(1, r)
        self.arm.ldr_x(0, 20, 8 * idx)
        self.arm.blr(0)
        if dst != 0:
            self.arm.fmov(dst, 0)

    def ifelse(self, dst, cond, true_reg, false_reg):
        self.arm.bsl(cond, true_reg, false_reg)
        if cond != dst:
            self.arm.fmov(dst, cond)

    def stack_size(self):
        cap = self.mem.stack_size
        pad = cap & 1
        return (cap + pad) * 8

    def prepend_prologue(self):
        # note that we generate the prologue after the main body
        # because we need to know the stack size
        self.arm.begin_prepend()

        n = self.stack_size()
        self.arm.sub_imm("sp", "sp", 32)
        self.arm.str_x("lr", "sp", 0)
        self.arm.stp_x(19, 20, "sp", 16)
        self.arm.sub_imm("sp", "sp", n)
        self.arm.mov(19, 0)
        self.arm.mov(20, 1)  # different than Rust

        self.arm.end_prepend()

    def append_epilogue(self):
        n = self.stack_size()
        self.arm.add_imm("sp", "sp", n)
        self.arm.ldp_x(19, 20, "sp", 16)
        self.arm.ldr_x("lr", "sp", 0)
        self.arm.add_imm("sp", "sp", 32)
        self.arm.ret()


################################################################


def test_arm():
    k = 11
    n = 1000
    arm = Arm()

    assert arm.sub("sp", "sp", 32).test([0xFF, 0x83, 0x00, 0xD1])
    assert arm.str_x(29, "sp", 8).test([0xFD, 0x07, 0x00, 0xF9])
    assert arm.str_x(30, "sp", 16).test([0xFE, 0x0B, 0x00, 0xF9])
    assert arm.str_d(8, "sp", 24).test([0xE8, 0x0F, 0x00, 0xFD])
    assert arm.mov(29, 0).test([0xFD, 0x03, 0x00, 0xAA])
    assert arm.stp_x(29, 30, "sp", 16).test([0xFD, 0x7B, 0x01, 0xA9])
    assert arm.stp_d(8, 9, "sp", 160).test([0xE8, 0x27, 0x0A, 0x6D])
    assert arm.ldp_x(19, 20, "sp", 504).test([0xF3, 0xD3, 0x5F, 0xA9])
    assert arm.ldp_d(k + 1, 13, "sp", 160).test([0xEC, 0x37, 0x4A, 0x6D])
    assert arm.ldr_d(0, 29, 104).test([0xA0, 0x37, 0x40, 0xFD])
    assert arm.fmov(1, 0).test([0x01, 0x40, 0x60, 0x1E])
    assert arm.fadd(0, 0, 1).test([0x00, 0x28, 0x61, 0x1E])
    assert arm.fmul(0, 0, 1).test([0x00, 0x08, 0x61, 0x1E])
    assert arm.fsub(0, 0, 1).test([0x00, 0x38, 0x61, 0x1E])
    assert arm.fcmeq(10, 21, 9).test([0xAA, 0xE6, 0x69, 0x5E])
    assert arm.fcmlt(k, 1, 19).test([0x6B, 0xE6, 0xE1, 0x7E])
    assert arm.fcmle(0, k, 31).test([0xE0, 0xE7, 0x6B, 0x7E])
    assert arm.fcmgt(0, k + 1, 19).test([0x80, 0xE5, 0xF3, 0x7E])
    assert arm.fcmge(17, 30, 3).test([0xD1, 0xE7, 0x63, 0x7E])
    assert arm.fdiv(0, 0, 1).test([0x00, 0x18, 0x61, 0x1E])
    assert arm.str_d(0, 30, 200).test([0xC0, 0x67, 0x00, 0xFD])
    assert arm.ldr_x(29, "sp", 8).test([0xFD, 0x07, 0x40, 0xF9])
    assert arm.ldr_x(30, "sp", 16).test([0xFE, 0x0B, 0x40, 0xF9])
    assert arm.add_imm("sp", "sp", 32).test([0xFF, 0x83, 0x00, 0x91])
    assert arm.vand(2, 5, 22).test([0xA2, 0x1C, 0x36, 0x0E])
    assert arm.orr(1, 0, k + 1).test([0x01, 0x1C, 0xAC, 0x0E])
    assert arm.eor(7, 15, 31).test([0xE7, 0x1D, 0x3F, 0x2E])
    assert arm.vnot(14, 24).test([0x0E, 0x5B, 0x20, 0x2E])
    assert arm.ldr_x("lr", "sp", n).test([0xFE, 0xF7, 0x41, 0xF9])
    assert arm.str_x("lr", "sp", 2 * n).test([0xFE, 0xEB, 0x03, 0xF9])
    assert arm.blr(6).test([0xC0, 0x00, 0x3F, 0xD6])
    assert arm.ret().test([0xC0, 0x03, 0x5F, 0xD6])
    assert arm.fmov_const(5, 0.0).test([0xE5, 0x03, 0x67, 0x9E])
    assert arm.fmov_const(15, 1.0).test([0x0F, 0x10, 0x6E, 0x1E])
    assert arm.fmov_const(k, -1.0).test([0x0B, 0x10, 0x7E, 0x1E])
