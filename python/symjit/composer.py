import numbers
from fractions import Fraction
from typing import Callable, NamedTuple  # , Self


class Slot(NamedTuple):
    loc: str
    idx: int

    def __repr__(self):
        return f"('{self.loc}', {self.idx})"


class Label(NamedTuple):
    id: int


class ComposerNumber:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        frac = Fraction(self.val)
        if frac.is_integer():
            return f"{frac.numerator}"
        else:
            return f"{frac.numerator}/{frac.denominator}"


class Composer:
    def __init__(self, num_params: int, num_outs: int):
        self.num_params = num_params
        self.num_outs = num_outs
        self.count_temp = 0
        self.count_label = 0
        self.constants = []
        self.const_indices = {}
        self.defuns = None
        self.parent = None
        self.ir = []

    def new_block(self):
        block = Composer(self.num_params, self.num_outs)
        block.parent = self
        return block

    def arg(self, id: int) -> Slot:
        if self.parent is None:
            if id < self.num_params:
                return Slot("param", id)
            else:
                raise ValueError(f"param id {id} out of range")
        else:
            return self.parent.arg(id)

    def out(self, id: int) -> Slot:
        if self.parent is None:
            if id < self.num_outs:
                return Slot("out", id)
            else:
                raise ValueError(f"out id {id} out of range")
        else:
            return self.parent.out(id)

    def constant(self, val) -> Slot:
        if self.parent is None:
            if val in self.const_indices:
                return self.const_indices[val]
            elif isinstance(val, numbers.Real):
                x = Slot("const", len(self.constants))
                self.constants.append(ComposerNumber(val))
                self.const_indices[val] = x
                return x
            elif isinstance(val, numbers.Complex):
                idx = len(self.constants)
                x = self.complex(Slot("const", idx), Slot("const", idx + 1))
                self.constants.append(ComposerNumber(val.real))
                self.constants.append(ComposerNumber(val.imag))
                self.const_indices[val] = x
                return x
            else:
                raise ValueError(f"{val} is not an acceptable constant.")
        else:
            return self.parent.constant(val)

    def new_temp(self) -> Slot:
        if self.parent is None:
            t = self.count_temp
            self.count_temp += 1
            return Slot("temp", t)
        else:
            return self.parent.new_temp()

    def new_label(self) -> Label:
        if self.parent is None:
            self.count_label += 1
            return Label(self.count_label)
        else:
            return self.parent.new_label()

    def get_instructions(self):
        return (self.ir, self.count_temp, self.constants)

    def function(self, name: str, *arg: Slot) -> Slot:
        t = self.new_temp()
        self.ir.append(("fun", t, name, [], [*arg], False))
        return t

    def call(self, fun: Callable, *arg: Slot) -> Slot:
        if self.defuns is None:
            name = "composer_func0"
            self.defuns = {name: fun}
        else:
            name = f"composer_func{len(self.defuns)}"
            self.defuns[name] = fun

        t = self.new_temp()
        self.ir.append(("fun", t, name, [], [*arg], True))
        return t

    def assign(self, lhs: Slot, rhs: Slot) -> Slot:
        self.ir.append(("assign", lhs, rhs))
        return lhs

    def fadd(self, x: Slot, y: Slot) -> Slot:
        t = self.new_temp()
        self.ir.append(("add", t, [x, y], 0))
        return t

    def fmul(self, x: Slot, y: Slot) -> Slot:
        t = self.new_temp()
        self.ir.append(("mul", t, [x, y], 0))
        return t

    def fsub(self, x: Slot, y: Slot) -> Slot:
        return self.fadd(x, self.neg(y))

    def fdiv(self, x: Slot, y: Slot) -> Slot:
        return self.fmul(x, self.recip(y))

    def idiv(self, x: Slot, y: Slot) -> Slot:
        q = self.fdiv(x, y)
        return self.floor(q)

    def mod(self, x: Slot, y: Slot) -> Slot:
        return self.fsub(x, self.fmul(y, self.idiv(x, y)))

    def neg(self, arg: Slot) -> Slot:
        return self.function("neg", arg)

    def abs(self, arg: Slot) -> Slot:
        return self.function("abs", arg)

    def sqrt(self, arg: Slot) -> Slot:
        return self.function("root", arg)

    def real_sqrt(self, arg: Slot) -> Slot:
        return self.function("real_root", arg)

    def square(self, arg: Slot) -> Slot:
        return self.function("square", arg)

    def cube(self, arg: Slot) -> Slot:
        return self.function("cube", arg)

    def recip(self, arg: Slot) -> Slot:
        return self.function("recip", arg)

    def round(self, arg: Slot) -> Slot:
        return self.function("round", arg)

    def floor(self, arg: Slot) -> Slot:
        return self.function("floor", arg)

    def ceiling(self, arg: Slot) -> Slot:
        return self.function("ceiling", arg)

    def trunc(self, arg: Slot) -> Slot:
        return self.function("trunc", arg)

    def frac(self, arg: Slot) -> Slot:
        return self.function("frac", arg)

    def powi(self, x: Slot, power: int) -> Slot:
        t = self.new_temp()
        self.ir.append(("pow", t, x, power, False))
        return t

    def powf(self, x: Slot, y: Slot) -> Slot:
        t = self.new_temp()
        self.ir.append(("powf", t, x, y, False))
        return t

    def real(self, arg: Slot) -> Slot:
        return self.function("real", arg)

    def imag(self, arg: Slot) -> Slot:
        return self.function("imaginary", arg)

    def conjugate(self, arg: Slot) -> Slot:
        return self.function("conjugate", arg)

    def complex(self, x: Slot, y: Slot) -> Slot:
        return self.function("complex", x, y)

    # Comparisons

    def lt(self, x: Slot, y: Slot) -> Slot:
        return self.function("lt", x, y)

    def leq(self, x: Slot, y: Slot) -> Slot:
        return self.function("leq", x, y)

    def gt(self, x: Slot, y: Slot) -> Slot:
        return self.function("gt", x, y)

    def geq(self, x: Slot, y: Slot) -> Slot:
        return self.function("geq", x, y)

    def eq(self, x: Slot, y: Slot) -> Slot:
        return self.function("eq", x, y)

    def neq(self, x: Slot, y: Slot) -> Slot:
        return self.function("neq", x, y)

    # Logical

    def and_(self, x: Slot, y: Slot) -> Slot:
        return self.function("and", x, y)

    def or_(self, x: Slot, y: Slot) -> Slot:
        return self.function("or", x, y)

    def xor(self, x: Slot, y: Slot) -> Slot:
        return self.function("xor", x, y)

    def not_(self, arg: Slot) -> Slot:
        return self.function("not", arg)

    def iszero(self, arg: Slot) -> Slot:
        return self.function("iszero", arg)

    def set_label(self, label: Label):
        self.ir.append(("label", label.id))

    def min(self, x: Slot, y: Slot) -> Slot:
        return self.join(self.lt(x, y), x, y)

    def max(self, x: Slot, y: Slot) -> Slot:
        return self.join(self.gt(x, y), x, y)

    def heaviside(self, x: Slot) -> Slot:
        return self.join(
            self.geq(x, self.constant(0)), self.constant(1), self.constant(0)
        )

    def branch(self, label: Label):
        # self.ir.append(("goto", label))
        # note: we use `if_else` instead of `goto` because `goto` can
        # be elided
        self.ir.append(("if_else", self.constant(0), label.id))

    def branch_if(self, cond: Slot, label: Label):
        self.ir.append(("if_else", self.not_(cond), label.id))

    def branch_else(self, cond, label: Label):
        self.ir.append(("if_else", cond, label.id))

    def join(self, cond: Slot, true_val: Slot, false_val: Slot) -> Slot:
        t = self.new_temp()
        self.ir.append(("join", t, cond, true_val, false_val))
        return t

    def append_block(self, block: "Composer"):
        assert block.parent == self
        self.ir.extend(block.ir)

    def append_if_else(self, cond: Slot, block_if: "Composer", block_else: "Composer"):
        label_else = self.new_label()
        label_done = self.new_label()

        self.branch_else(cond, label_else)
        self.append_block(block_if)
        self.branch_if(cond, label_done)
        self.set_label(label_else)
        self.append_block(block_else)
        self.set_label(label_done)

    def append_for(
        self,
        for_var: Slot,
        start: numbers.Number,
        end: numbers.Number,
        block: "Composer",
    ):
        loop = self.new_label()

        self.assign(for_var, self.constant(start))
        self.set_label(loop)
        self.append_block(block)
        self.assign(for_var, self.fadd(for_var, self.constant(1)))
        cond = self.geq(for_var, self.constant(end))
        self.branch_else(cond, loop)

    # Transcendental Functions

    def sin(self, arg: Slot) -> Slot:
        return self.function("sin", arg)

    def cos(self, arg: Slot) -> Slot:
        return self.function("cos", arg)

    def tan(self, arg: Slot) -> Slot:
        return self.function("tan", arg)

    def csc(self, arg: Slot) -> Slot:
        return self.function("csc", arg)

    def sec(self, arg: Slot) -> Slot:
        return self.function("sec", arg)

    def cot(self, arg: Slot) -> Slot:
        return self.function("cot", arg)

    def sinc(self, arg: Slot) -> Slot:
        return self.function("sinc", arg)

    def sinh(self, arg: Slot) -> Slot:
        return self.function("sinh", arg)

    def cosh(self, arg: Slot) -> Slot:
        return self.function("cosh", arg)

    def tanh(self, arg: Slot) -> Slot:
        return self.function("tanh", arg)

    def csch(self, arg: Slot) -> Slot:
        return self.function("csch", arg)

    def sech(self, arg: Slot) -> Slot:
        return self.function("sech", arg)

    def coth(self, arg: Slot) -> Slot:
        return self.function("coth", arg)

    def asin(self, arg: Slot) -> Slot:
        return self.function("arcsin", arg)

    def acos(self, arg: Slot) -> Slot:
        return self.function("arccos", arg)

    def atan(self, arg: Slot) -> Slot:
        return self.function("arctan", arg)

    def asinh(self, arg: Slot) -> Slot:
        return self.function("arcsinh", arg)

    def acosh(self, arg: Slot) -> Slot:
        return self.function("arccosh", arg)

    def atanh(self, arg: Slot) -> Slot:
        return self.function("arctanh", arg)

    def cbrt(self, arg: Slot) -> Slot:
        return self.function("cbrt", arg)

    def exp(self, arg: Slot) -> Slot:
        return self.function("exp", arg)

    def exp2(self, arg: Slot) -> Slot:
        return self.function("exp2", arg)

    def log(self, arg: Slot) -> Slot:
        return self.function("ln", arg)

    def log10(self, arg: Slot) -> Slot:
        return self.function("log", arg)

    def log2(self, arg: Slot) -> Slot:
        return self.function("log2", arg)

    def expm1(self, arg: Slot) -> Slot:
        return self.function("expm1", arg)

    def log1p(self, arg: Slot) -> Slot:
        return self.function("log1p", arg)

    def erf(self, arg: Slot) -> Slot:
        return self.function("erf", arg)

    def erfc(self, arg: Slot) -> Slot:
        return self.function("erfc", arg)

    def gamma(self, arg: Slot) -> Slot:
        return self.function("gamma", arg)

    def loggamma(self, arg: Slot) -> Slot:
        return self.function("loggamma", arg)
