import sys
import numbers
from sympy import *

COUNT_TEMP_XMM = 3 if sys.platform == "win32" else 13


class Unary:
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg
        self.is_pure = arg.is_pure and is_pure(op)

    def __repr__(self):
        return f"Unary[{self.is_pure}]('{self.op}', {self.arg})"

    def compile(self, dst, prog, mem, vt):
        self.arg.compile(0, prog, mem, vt)

        if self.op == "neg":
            prog.neg(dst)
        elif self.op == "abs":
            prog.abs(dst)
        elif self.op == "root":
            prog.root(dst)
        elif self.op == "square":
            prog.square(dst)
        elif self.op == "cube":
            prog.cube(dst)
        elif self.op == "recip":
            prog.recip(dst)
        elif self.op == "exp":
            prog.call_unary(dst, vt.find("exp"))
        elif self.op == "ln":
            prog.call_unary(dst, vt.find("log"))
        elif self.op == "sin":
            prog.call_unary(dst, vt.find("sin"))
        elif self.op == "cos":
            prog.call_unary(dst, vt.find("cos"))
        elif self.op == "tan":
            prog.call_unary(dst, vt.find("tan"))
        elif self.op == "sinh":
            prog.call_unary(dst, vt.find("sinh"))
        elif self.op == "cosh":
            prog.call_unary(dst, vt.find("cosh"))
        elif self.op == "tanh":
            prog.call_unary(dst, vt.find("tanh"))
        elif self.op == "arcsin":
            prog.call_unary(dst, vt.find("asin"))
        elif self.op == "arccos":
            prog.call_unary(dst, vt.find("acos"))
        elif self.op == "arctan":
            prog.call_unary(dst, vt.find("atan"))
        elif self.op == "arcsinh":
            prog.call_unary(dst, vt.find("asinh"))
        elif self.op == "arccosh":
            prog.call_unary(dst, vt.find("acosh"))
        elif self.op == "arctanh":
            prog.call_unary(dst, vt.find("atanh"))
        else:
            raise ValueError(f"unary op {self.op} not defined")


class Binary:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.is_pure = left.is_pure and right.is_pure and is_pure(op)

    def __repr__(self):
        return f"Binary[{self.is_pure}]('{self.op}', {self.left}, {self.right})"

    def compile(self, dst, prog, mem, vt):
        sp = mem.push()
        use_reg = sp < prog.count_shadows and self.left.is_pure

        if use_reg:
            self.right.compile(prog.first_shadow + sp, prog, mem, vt)
        else:
            self.right.compile(0, prog, mem, vt)
            prog.save_stack(0, sp)

        self.left.compile(0, prog, mem, vt)
        sp = mem.pop()

        if use_reg:
            r = prog.first_shadow + sp
        else:
            prog.load_stack(1, sp)
            r = 1

        if self.op == "plus":
            prog.plus(dst, r)
        elif self.op == "minus":
            prog.minus(dst, r)
        elif self.op == "times":
            prog.times(dst, r)
        elif self.op == "divide":
            prog.divide(dst, r)
        elif self.op == "power":
            prog.call_binary(dst, r, vt.find("pow"))
        elif self.op == "gt":
            prog.gt(dst, r)
        elif self.op == "geq":
            prog.geq(dst, r)
        elif self.op == "lt":
            prog.lt(dst, r)
        elif self.op == "leq":
            prog.leq(dst, r)
        elif self.op == "eq":
            prog.eq(dst, r)
        elif self.op == "neq":
            prog.neq(dst, r)
        elif self.op == "and":
            prog.boolean_and(dst, r)
        elif self.op == "or":
            prog.boolean_or(dst, r)
        elif self.op == "xor":
            prog.boolean_xor(dst, r)
        else:
            raise ValueError(f"binary op {self.op} not defined")


class IfElse:
    def __init__(self, cond, true_val, false_val):
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val
        self.is_pure = cond.is_pure and true_val.is_pure and false_val.is_pure

    def __repr__(self):
        return f"IfElse[{self.is_pure}]({self.cond}, {self.true_val}, {self.false_val})"

    def compile(self, dst, prog, mem, vt):
        sp = mem.push()
        use_reg_cond = (
            sp < prog.count_shadows and self.true_val.is_pure and self.false_val.is_pure
        )

        if use_reg_cond:
            self.cond.compile(prog.first_shadow + sp, prog, mem, vt)
        else:
            self.cond.compile(0, prog, mem, vt)
            prog.save_stack(0, sp)

        sp = mem.push()
        use_reg_false = sp < COUNT_TEMP_XMM and self.true_val.is_pure

        if use_reg_false:
            self.false_val.compile(prog.first_shadow + sp, prog, mem, vt)
        else:
            self.false_val.compile(0, prog, mem, vt)
            prog.save_stack(0, sp)

        self.true_val.compile(0, prog, mem, vt)

        sp = mem.pop()

        if use_reg_false:
            f = prog.first_shadow + sp
        else:
            prog.load_stack(1, sp)
            f = 1

        sp = mem.pop()

        if use_reg_cond:
            c = 3 + sp
        else:
            prog.load_stack(2, sp)
            c = 2

        prog.ifelse(dst, c, 0, f)


class Const:
    def __init__(self, val):
        self.val = float(val)
        self.is_pure = True

    def __repr__(self):
        return f"Const({self.val})"

    def compile(self, dst, prog, mem, vt):
        prog.load_mem(dst, mem.constant(self.val))


class Var:
    def __init__(self, name):
        self.name = str(name)
        self.is_pure = True

    def __repr__(self):
        return f"Var('{self.name}')"

    def compile(self, dst, prog, mem, vt):
        prog.load_mem(dst, mem.index(self.name))


class Eq:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} = {self.rhs}"

    def compile(self, dst, prog, mem, vt):
        self.rhs.compile(dst, prog, mem, vt)
        prog.save_mem(dst, mem.index(self.lhs.name))


class EqDiff:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"δ{self.lhs} = {self.rhs}"

    def compile(self, dst, prog, mem, vt):
        self.rhs.compile(dst, prog, mem, vt)
        prog.save_mem(dst, mem.index_diff(self.lhs.name))


class Model:
    def __init__(self, iv, states, params, obs, eqs, odes):
        self.iv = iv
        self.states = states
        self.params = params
        self.obs = obs
        self.odes = odes
        self.eqs = eqs

    def __repr__(self):
        return f"""Model(
            iv: {self.iv}
            states: {self.states}
            params: {self.params}
            obs: {self.obs}
            eqs: {self.eqs}
            odes: {self.odes})
        )"""

    def compile(self, dst, prog, mem, vt):
        for eq in self.eqs:
            eq.compile(dst, prog, mem, vt)

        for eq in self.odes:
            eq.compile(dst, prog, mem, vt)

        # we need to prepend and not append prologue
        # because we don't know the stack size until
        # after compiling the body of the function
        prog.prepend_prologue()
        prog.append_epilogue()


def is_pure(op):
    ops = [
        "plus",
        "minus",
        "times",
        "division",
        "square",
        "cube",
        "root",
        "recip",
        "abs",
        "neg",
        "lt",
        "leq",
        "gt",
        "geq",
        "eq",
        "neq",
        "and",
        "or",
        "xor",
    ]
    return op in ops


# ******************** Lowering ***********************


def lower(y):
    if y.is_Number or isinstance(y, numbers.Number):
        return Const(y)
    elif y.is_Symbol:
        return Var(y)
    elif y.is_Relational:
        return relational(y)
    elif y.is_Boolean:
        return boolean(y)
    elif y.is_Piecewise:
        return piecewise(y)
    else:
        return lower_tree(y)


def lower_tree(y):
    if y.is_Add:
        return lower_add(y)
    elif y.is_Mul:
        return lower_mul(y)
    elif y.is_Pow:
        return lower_pow(y)
    elif y.is_Function:
        if len(y.args) == 1:
            return Unary(operation(y.func), lower(y.args[0]))
        elif len(y.args) == 2:
            return Binary(operation(y.func), lower(y.args[0]), lower(y.args[1]))
    raise ValueError(f"unrecognized expression: {y}")


def lower_add(y):
    assert y.is_Add
    if len(y.args) == 1:
        return lower(y.args[0])
    if len(y.args) == 2:
        return Binary("plus", lower(y.args[0]), lower(y.args[1]))
    else:
        return Binary("plus", lower(y.args[0]), lower(y.func(*y.args[1:])))


def lower_mul(y):
    assert y.is_Mul
    if len(y.args) == 2 and y.args[1].is_Pow and y.args[1].args[1] == -1:
        return Binary("divide", lower(y.args[0]), lower(y.args[1].args[0]))
    if len(y.args) == 2:
        return Binary("times", lower(y.args[0]), lower(y.args[1]))
    else:
        return Binary("times", lower(y.args[0]), lower(y.func(*y.args[1:])))


def lower_pow(y):
    assert y.is_Pow
    power = y.args[1]
    arg = y.args[0]

    if power == 2:
        return Unary("square", lower(arg))
    elif power == 3:
        return Unary("cube", lower(arg))
    elif power == -1:
        return Unary("recip", lower(arg))
    elif power == -2:
        return Unary("recip", lower(arg**2))
    elif power == -3:
        return Unary("recip", lower(arg**3))
    elif power == Rational(1, 2):
        return Unary("root", lower(arg))
    elif power == Rational(3, 2):
        return Unary("root", lower(arg**3))
    else:
        return Binary("power", lower(y.args[0]), lower(y.args[1]))


def operation(func):
    op = str(func)
    if func == sqrt:
        op = "root"
    elif func == log:
        op = "ln"  # this is confusing but sympy uses `log` for natural logarithm
    elif func == Abs:
        op = "abs"
    elif func == asin:
        op = "arcsin"
    elif func == acos:
        op = "arccos"
    elif func == atan:
        op = "arctan"
    elif func == asinh:
        op = "arcsinh"
    elif func == acosh:
        op = "arccosh"
    elif func == atanh:
        op = "arctanh"

    return op


def relational(y):
    f = y.func
    op = ""

    if f == LessThan:
        op = "lt"
    elif f == StrictLessThan:
        op = "leq"
    elif f == GreaterThan:
        op = "gt"
    elif f == StrictGreaterThan:
        op = "geq"
    elif f == Equality:
        op = "eq"
    elif f == Unequality:
        op = "neq"
    else:
        raise ValueError("unrecognized relational operator")

    return Binary(op, lower(y.args[0]), lower(y.args[1]))


def boolean(y):
    f = y.func
    op = ""

    if f == And:
        op = "and"
    elif f == Or:
        op = "or"
    elif f == Xor:
        op = "xor"
    else:
        raise ValueError("unrecognized boolean operator")

    return Binary(op, lower(y.args[0]), lower(y.args[1]))


def piecewise(y):
    cond = y.args[0][1]
    x1 = y.args[0][0]

    if len(y.args) == 1:
        return expr(x1)
    if len(y.args) == 2:
        x2 = y.args[1][0]
    else:
        x2 = piecewise(*y.args[1:])

    return IfElse(lower(cond), lower(x1), lower(x2))


def model(states, eqs, params=None, obs=None):
    if not isinstance(states, list):
        states = [states]

    if not isinstance(eqs, list):
        eqs = [eqs]

    if params is None:
        params = []

    if obs is None:
        obs = [f"${i}" for i in range(len(eqs))]

    model = Model(
        Var("$_"),  # iv
        [Var(x) for x in states],  # states
        [Var(x) for x in params],  # params
        [Var(x) for x in obs],  # obs
        [Eq(Var(lhs), lower(rhs)) for (lhs, rhs) in zip(obs, eqs)],  # eqs
        [],  # odes
    )

    return model


def model_ode(iv, states, odes, params=None):
    try:
        states = list(states)
    except TypeError:
        states = [states]

    try:
        odes = list(odes)
    except TypeError:
        odes = [odes]

    assert len(states) == len(odes)

    if params is None:
        params = []

    model = Model(
        Var(iv),
        [Var(x) for x in states],  # states
        [Var(x) for x in params],  # params
        [],  # obs
        [],  # eqs
        [EqDiff(Var(lhs), lower(rhs)) for (lhs, rhs) in zip(states, odes)],  # odes
    )

    return model


def model_jac(iv, states, odes, params=None):
    try:
        states = list(states)
    except TypeError:
        states = [states]

    try:
        odes = list(odes)
    except TypeError:
        odes = [odes]

    assert len(states) == len(odes)

    n = len(states)
    eqs = []
    obs = []

    for i in range(n):
        for j in range(n):
            df = diff(odes[i], states[j])
            eqs.append(df)
            obs.append(f"${i}_{j}")

    if params is None:
        params = []

    model = Model(
        Var(iv),
        [Var(x) for x in states],  # states
        [Var(x) for x in params],  # params
        [Var(x) for x in obs],  # obs
        [Eq(Var(lhs), lower(rhs)) for (lhs, rhs) in zip(obs, eqs)],  # eqs
        [],  # odes
    )

    return model
