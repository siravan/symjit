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
        self.constants = []
        self.parent = None
        self.ir = []

    def param(self, id):
        if self.parent is None:
            if id < self.num_params:
                return ("param", id)
            else:
                raise ValueError(f"param id {id} out of range")
        else:
            return self.parent.param(id)

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

    def get_instructions(self):
        return (self.ir, self.count_temp, self.constants)

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

    def function(self, fun, *arg):
        t = self.new_temp()
        self.ir.append(("fun", t, fun, [], [*arg], False))
        return t

    def sin(self, arg):
        return self.function("sin", arg)

    def cos(self, arg):
        return self.function("cos", arg)

    def neg(self, arg):
        return self.function("neg", arg)
