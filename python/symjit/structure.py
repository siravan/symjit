from sympy import asin, acos, atan, acsc, asec, acot
from sympy import asinh, acosh, atanh, acsch, asech, acoth
from sympy import Xor, And, Or
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
    if func == asin:
        op = "arcsin"
    elif func == acos:
        op = "arccos"
    elif func == atan:
        op = "arctan"
    elif func == acsc:
        op = "arccsc"
    elif func == asec:
        op = "arcsec"
    elif func == acot:
        op = "arccot"
    elif func == asinh:
        op = "arcsinh"
    elif func == acosh:
        op = "arccosh"
    elif func == atanh:
        op = "arctanh"
    elif func == acsch:
        op = "arccsch"
    elif func == asech:
        op = "arcsech"
    elif func == acoth:
        op = "arccoth"

    return op


def tree(y):
    op = ""

    if y.is_Add:
        op = "plus"
    elif y.is_Mul:
        op = "times"
    elif y.is_Pow:
        op = "power"
    elif y.is_Function:
        op = operation(y.func)
    else:
        raise ValueError("unreognized tree type")

    return tree_node(op, y.args)


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


def model(states, eqs, params=None):
    if not isinstance(states, list):
        states = [states]

    if not isinstance(eqs, list):
        eqs = [eqs]

    if params is None:
        params = []

    d = {
        "iv": var(Symbol("$_")),
        "params": [var(x) for x in params],
        "states": [var(x) for x in states],
        "algs": [],
        "odes": [],
        "obs": [
            equation(expr(Symbol(f"${i}")), expr(rhs)) for (i, rhs) in enumerate(eqs)
        ],
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
        "obs": [equation(expr(Symbol(f"${i}")), expr(rhs)) for (i, rhs) in enumerate(eqs)],
    }

    return d
