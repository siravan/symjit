from sympy import asin, acos, atan, acsc, asec, acot, log, sqrt
from sympy import asinh, acosh, atanh, acsch, asech, acoth
from sympy import Xor, And, Or, Rational, Abs
from sympy import (
    Equality,
    Unequality,
    LessThan,
    StrictLessThan,
    GreaterThan,
    StrictGreaterThan,
)
from sympy import Symbol, diff

import numbers


def tree_node(op, args):
    args = [expr(a) for a in args]
    return {"type": "Tree", "op": op, "args": args}


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


def process_mul(y):
    assert y.is_Mul
    if len(y.args) == 2 and y.args[1].is_Pow and y.args[1].args[1] == -1:
        return tree_node("divide", [y.args[0], y.args[1].args[0]])
    else:
        return tree_node("times", y.args)


def process_pow(y):
    assert y.is_Pow
    power = y.args[1]
    arg = y.args[0]

    if power == 2:
        return tree_node("square", [arg])
    elif power == 3:
        return tree_node("cube", [arg])
    elif power == -1:
        return tree_node("recip", [arg])
    elif power == -2:
        return tree_node("recip", [arg**2])
    elif power == -3:
        return tree_node("recip", [arg**3])
    elif power == Rational(1, 2):
        return tree_node("root", [arg])
    elif power == Rational(3, 2):
        return tree_node("root", [arg**3])
    else:
        return tree_node("power", y.args)


def tree(y):
    if y.is_Add:
        return tree_node("plus", y.args)
    elif y.is_Mul:
        return process_mul(y)
    elif y.is_Pow:
        return process_pow(y)
    elif y.is_Function:
        return tree_node(operation(y.func), y.args)
    else:
        raise ValueError("unreognized tree type")


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

    return tree_node(op, y.args)


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

    return tree_node(op, y.args)


def piecewise(y):
    cond = y.args[0][1]
    x1 = y.args[0][0]

    if len(y.args) == 1:
        return expr(x1)
    if len(y.args) == 2:
        x2 = y.args[1][0]
    else:
        x2 = piecewise(*y.args[1:])

    return tree_node("ifelse", [cond, x1, x2])


def var(sym, val=0.0):
    return {"name": sym.name, "val": float(val)}


def expr(y):
    if y.is_Number or isinstance(y, numbers.Number):
        return {"type": "Const", "val": float(y)}
    elif y.is_Symbol:
        return {"type": "Var", "name": y.name}
    elif y.is_Relational:
        return relational(y)
    elif y.is_Boolean:
        return boolean(y)
    elif y.is_Piecewise:
        return piecewise(y)
    else:
        return tree(y)


def equation(lhs, rhs):
    return {"lhs": lhs, "rhs": rhs}


def ode(y):
    return {
        "type": "Tree",
        "op": "Differential",
        "args": [{"type": "Var", "name": y.name}],
    }


def model(states, eqs, params=None, obs=None):
    if not isinstance(states, list):
        states = [states]

    if not isinstance(eqs, list):
        eqs = [eqs]

    if params is None:
        params = []

    if obs is None:
        obs = [Symbol(f"${i}") for i in range(len(eqs))]

    d = {
        "iv": var(Symbol("$_")),
        "params": [var(x) for x in params],
        "states": [var(x) for x in states],
        "algs": [],
        "odes": [],
        "obs": [equation(expr(lhs), expr(rhs)) for (lhs, rhs) in zip(obs, eqs)],
    }

    return d


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

    d = {
        "iv": var(iv),
        "params": [var(x) for x in params],
        "states": [var(x) for x in states],
        "algs": [],
        "odes": [equation(ode(lhs), expr(rhs)) for (lhs, rhs) in zip(states, odes)],
        "obs": [],
    }

    return d


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

    for i in range(n):
        for j in range(n):
            df = diff(odes[i], states[j])
            eqs.append(df)

    if params is None:
        params = []

    d = {
        "iv": var(iv),
        "params": [var(x) for x in params],
        "states": [var(x) for x in states],
        "algs": [],
        "odes": [],
        "obs": [
            equation(expr(Symbol(f"${i}")), expr(rhs)) for (i, rhs) in enumerate(eqs)
        ],
    }

    return d
