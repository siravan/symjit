from fractions import Fraction


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
    def __init__(self, num_params, num_outs, dtype="float64"):
        assert dtype == "float64" or dtype == "complex128"
        self.dtype = dtype
        self.num_params = num_params
        self.num_outs = num_outs
        self.count_temp = 0
        self.count_label = 0
        self.constants = []
        self.parent = None
        self.ir = []

    def new_block(self):
        block = Composer(self.num_params, self.num_outs, dtype=self.dtype)
        block.parent = self
        return block

    def arg(self, id):
        if self.parent is None:
            if id < self.num_params:
                return ("param", id)
            else:
                raise ValueError(f"param id {id} out of range")
        else:
            return self.parent.arg(id)

    def out(self, id):
        if self.parent is None:
            if id < self.num_outs:
                return ("out", id)
            else:
                raise ValueError(f"out id {id} out of range")
        else:
            return self.parent.out(id)

    def constant(self, val):
        if self.parent is None:
            if self.dtype == "float64" or val.imag == 0:
                try:
                    return ("const", self.constants.index(val))
                except ValueError:
                    self.constants.append(ComposerNumber(val.real))
                    return ("const", len(self.constants) - 1)
            else:
                x = self.constant(val.real)
                y = self.constant(val.imag)
                return self.function("complex", x, y)
        else:
            return self.parent.constant(val)

    def new_temp(self):
        if self.parent is None:
            t = self.count_temp
            self.count_temp += 1
            return ("temp", t)
        else:
            return self.parent.new_temp()

    def new_label(self):
        if self.parent is None:
            self.count_label += 1
            return self.count_label
        else:
            return self.parent.new_label()

    def get_instructions(self):
        return (self.ir, self.count_temp, self.constants)

    def function(self, fun, *arg):
        t = self.new_temp()
        self.ir.append(("fun", t, fun, [], [*arg], False))
        return t

    def assign(self, lhs, rhs):
        self.ir.append(("assign", lhs, rhs))
        return lhs

    def fadd(self, x, y):
        t = self.new_temp()
        self.ir.append(("add", t, [x, y], 0))
        return t

    def fmul(self, x, y):
        t = self.new_temp()
        self.ir.append(("mul", t, [x, y], 0))
        return t

    def fsub(self, x, y):
        t1 = self.new_temp()
        t2 = self.new_temp()
        self.ir.append(("mul", t1, [y, self.constant(-1)], 0))
        self.ir.append(("add", t2, [x, t1], 0))
        return t2

    def fdiv(self, x, y):
        t1 = self.new_temp()
        t2 = self.new_temp()
        self.ir.append(("pow", t1, y, -1, False))
        self.ir.append(("mul", t2, [x, t1], 0))
        return t2

    def neg(self, arg):
        return self.function("neg", arg)

    def abs(self, arg):
        return self.function("abs", arg)

    def sqrt(self, arg):
        return self.function("root", arg)

    def real_sqrt(self, arg):
        return self.function("real_root", arg)

    def square(self, arg):
        return self.function("square", arg)

    def cube(self, arg):
        return self.function("cube", arg)

    def recip(self, arg):
        return self.function("recip", arg)

    def round(self, arg):
        return self.function("round", arg)

    def floor(self, arg):
        return self.function("floor", arg)

    def ceiling(self, arg):
        return self.function("ceiling", arg)

    def trunc(self, arg):
        return self.function("trunc", arg)

    def frac(self, arg):
        return self.function("frac", arg)

    def powi(self, x, power):
        t = self.new_temp()
        self.ir.append(("pow", t, x, power, False))
        return t

    def powf(self, x, y):
        t = self.new_temp()
        self.ir.append(("powf", t, x, y, False))
        return t

    def real(self, arg):
        return self.function("real", arg)

    def imag(self, arg):
        return self.function("imaginary", arg)

    def conjugate(self, arg):
        return self.function("conjugate", arg)

    # Comparisons

    def lt(self, *args):
        return self.function("lt", *args)

    def leq(self, *args):
        return self.function("leq", *args)

    def gt(self, *args):
        return self.function("gt", *args)

    def geq(self, *args):
        return self.function("geq", *args)

    def eq(self, *args):
        return self.function("eq", *args)

    def neq(self, *args):
        return self.function("neq", *args)

    # Logical

    def and_(self, *args):
        return self.function("and", *args)

    def or_(self, *args):
        return self.function("or", *args)

    def xor(self, *args):
        return self.function("xor", *args)

    def not_(self, arg):
        return self.function("not", arg)

    def iszero(self, arg):
        return self.function("iszero", arg)

    def set_label(self, label):
        self.ir.append(("label", label))

    def branch(self, label):
        # self.ir.append(("goto", label))
        # note: we use `if_else` instead of `goto` because `goto` can
        # be elided
        self.ir.append(("if_else", self.constant(0), label))

    def branch_if(self, cond, label):
        self.ir.append(("if_else", self.not_(cond), label))

    def branch_else(self, cond, label):
        self.ir.append(("if_else", cond, label))

    def join(self, cond, true_val, false_val):
        t = self.new_temp()
        self.ir.append(("join", t, cond, true_val, false_val))
        return t

    def select(self, cond, true_val, false_val):
        t = self.new_temp()
        self.ir.append(("join", t, cond, false_val, true_val))
        return t

    def append_block(self, block):
        assert block.parent == self
        self.ir.extend(block.ir)

    def append_if_else(self, cond, block_if, block_else):
        label_else = self.new_label()
        label_done = self.new_label()

        self.branch_else(cond, label_else)
        self.append_block(block_if)
        self.branch_if(cond, label_done)
        self.set_label(label_else)
        self.append_block(block_else)
        self.set_label(label_done)

    def append_for_loop(self, for_var, start, end, block):
        loop = self.new_label()

        self.assign(for_var, self.constant(start))
        self.set_label(loop)
        self.append_block(block)
        s1 = self.fadd(for_var, self.constant(1))
        self.assign(for_var, s1)
        cond = self.geq(for_var, self.constant(end))
        self.branch_else(cond, loop)

    # Transcendental Functions

    def sin(self, arg):
        return self.function("sin", arg)

    def cos(self, arg):
        return self.function("cos", arg)

    def tan(self, arg):
        return self.function("tan", arg)

    def csc(self, arg):
        return self.function("csc", arg)

    def sec(self, arg):
        return self.function("sec", arg)

    def cot(self, arg):
        return self.function("cot", arg)

    def sinc(self, arg):
        return self.function("sinc", arg)

    def sinh(self, arg):
        return self.function("sinh", arg)

    def cosh(self, arg):
        return self.function("cosh", arg)

    def tanh(self, arg):
        return self.function("tanh", arg)

    def csch(self, arg):
        return self.function("csch", arg)

    def sech(self, arg):
        return self.function("sech", arg)

    def coth(self, arg):
        return self.function("coth", arg)

    def asin(self, arg):
        return self.function("arcsin", arg)

    def acos(self, arg):
        return self.function("arccos", arg)

    def atan(self, arg):
        return self.function("arctan", arg)

    def asinh(self, arg):
        return self.function("arcsinh", arg)

    def acosh(self, arg):
        return self.function("arccosh", arg)

    def atanh(self, arg):
        return self.function("arctanh", arg)

    def cbrt(self, arg):
        return self.function("cbrt", arg)

    def exp(self, arg):
        return self.function("exp", arg)

    def exp2(self, arg):
        return self.function("exp2", arg)

    def log(self, arg):
        return self.function("ln", arg)

    def log10(self, arg):
        return self.function("log", arg)

    def log2(self, arg):
        assert self.dtype == "float64"
        return self.function("log2", arg)

    def expm1(self, arg):
        assert self.dtype == "float64"
        return self.function("expm1", arg)

    def log1p(self, arg):
        assert self.dtype == "float64"
        return self.function("log1p", arg)

    def erf(self, arg):
        assert self.dtype == "float64"
        return self.function("erf", arg)

    def erfc(self, arg):
        assert self.dtype == "float64"
        return self.function("erfc", arg)

    def gamma(self, arg):
        assert self.dtype == "float64"
        return self.function("gamma", arg)

    def loggamma(self, arg):
        assert self.dtype == "float64"
        return self.function("loggamma", arg)
