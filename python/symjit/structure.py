from sympy import asin, acos, atan, acsc, asec, acot, log, sqrt
from sympy import asinh, acosh, atanh, acsch, asech, acoth
from sympy import Xor, And, Or, Rational, Abs, Mod
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
    elif func == Mod:
        op = "rem"        
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
    return tree_node("times", y.args)


def process_pow(y):
    assert y.is_Pow
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
        op = "leq"
    elif f == StrictLessThan:
        op = "lt"
    elif f == GreaterThan:
        op = "geq"
    elif f == StrictGreaterThan:
        op = "gt"
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


def piecewise(args):
    cond = args[0][1]
    x1 = args[0][0]

    if len(args) == 1:
        return expr(x1)
    if len(args) == 2:
        x2 = args[1][0]
    else:
        x2 = piecewise(args[1:])

    return tree_node("ifelse", [cond, x1, x2])


def var(sym, val=0.0):
    return {"name": sym.name, "val": float(val)}


def expr(y):
    try:
        if y.is_Number or isinstance(y, numbers.Number):
            return {"type": "Const", "val": float(y)}
        elif y.is_Symbol:
            return {"type": "Var", "name": y.name}
        elif y.is_Relational:
            return relational(y)
        elif y.is_Boolean:
            return boolean(y)
        elif y.is_Piecewise:
            return piecewise(y.args)
        else:
            return tree(y)
    except:
        return y
        

def equation(lhs, rhs):
    return {"lhs": lhs, "rhs": rhs}


def ode(y):
    return {
        "type": "Tree",
        "op": "Differential",
        "args": [{"type": "Var", "name": y.name}],
    }


def model(states, eqs, params=None, obs=None):
    if not hasattr(states, "__iter__"):
        states = [states]

    if not hasattr(eqs, "__iter__"):
        eqs = [eqs]

    if params is None:
        params = []

    if obs is None:
        obs = [Symbol(f"${i}") for i in range(len(eqs))]

    d = {
        "iv": var(Symbol("$_")),
        "params": [var(x) for x in list(params)],
        "states": [var(x) for x in list(states)],
        "algs": [],
        "odes": [],
        "obs": [equation(expr(lhs), expr(rhs)) for (lhs, rhs) in zip(obs, eqs)],
    }

    return d


def model_ode(iv, states, odes, params=None):
    if not hasattr(states, "__iter__"):
        states = [states]

    if not hasattr(odes, "__iter__"):
        odes = [odes]

    assert len(states) == len(odes)

    if params is None:
        params = []

    d = {
        "iv": var(iv),
        "params": [var(x) for x in list(params)],
        "states": [var(x) for x in list(states)],
        "algs": [],
        "odes": [equation(ode(lhs), expr(rhs)) for (lhs, rhs) in zip(states, odes)],
        "obs": [],
    }

    return d


def model_jac(iv, states, odes, params=None):
    if not hasattr(states, "__iter__"):
        states = [states]

    if not hasattr(odes, "__iter__"):
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
        "params": [var(x) for x in list(params)],
        "states": [var(x) for x in list(states)],
        "algs": [],
        "odes": [],
        "obs": [
            equation(expr(Symbol(f"${i}")), expr(rhs)) for (i, rhs) in enumerate(eqs)
        ],
    }

    return d
