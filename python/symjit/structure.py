import numbers
from ast import Expression

from sympy import (
    Abs,
    And,
    Equality,
    GreaterThan,
    Heaviside,
    LessThan,
    Max,
    Min,
    Mod,
    Or,
    Product,
    StrictGreaterThan,
    StrictLessThan,
    Sum,
    Symbol,
    Unequality,
    Xor,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    diff,
    log,
    sqrt,
    symbols,
)
from sympy.printing.conventions import Derivative

from . import expression


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
    elif func == Min:
        op = "min"
    elif func == Max:
        op = "max"
    elif func == Heaviside:
        op = "heaviside"
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


def process_add(y):
    assert y.is_Add
    return tree_node("plus", y.args)


def process_mul(y):
    assert y.is_Mul
    return tree_node("times", y.args)


def process_pow(y):
    assert y.is_Pow
    return tree_node("power", y.args)


def tree(y):
    if y.is_Add:
        return process_add(y)
    elif y.is_Mul:
        return process_mul(y)
    elif y.is_Pow:
        return process_pow(y)
    elif y.is_Function:
        return tree_node(operation(y.func), y.args)
    else:
        raise ValueError("unrecognized tree type")


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


def loops(y):
    args = [y.args[0]]
    args.extend(y.args[1])

    d = {args[1]: symbols(f"${str(args[1])}")}
    args = [eq.subs(d) for eq in args]

    if isinstance(y, Sum):
        return tree_node("Sum", args)
    elif isinstance(y, Product):
        return tree_node("Product", args)
    else:
        raise ValueError("only Sum and Product loops are supported")


def var(sym, val=0.0):
    return {"name": str(sym), "val": float(val)}


def expr(y):
    try:
        if expression.is_expression(y):  # y is a Symbolica Expression?
            return expression.walk_tree(y)
        elif isinstance(y, Sum) or isinstance(y, Product):
            return loops(y)
        elif isinstance(y, numbers.Number) or y.is_number:
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
    except ValueError:
        print(f"fail to convert {y}")
        return y


def equation(lhs, rhs):
    return {"lhs": lhs, "rhs": rhs}


def ode(y):
    return {
        "type": "Tree",
        "op": "Differential",
        "args": [{"type": "Var", "name": y.name}],
    }


def is_singular(y):
    return not hasattr(y, "__iter__") or expression.is_expression(y)


def model(states, eqs, params=None, obs=None):
    if is_singular(states):
        states = [states]

    if is_singular(eqs):
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
    if is_singular(states):
        states = [states]

    if is_singular(odes):
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
    if is_singular(states):
        states = [states]

    if is_singular(odes):
        odes = [odes]

    assert len(states) == len(odes)

    n = len(states)
    eqs = []

    if expression.is_expression(iv):
        for i in range(n):
            for j in range(n):
                df = odes[i].derivative(states[j])
                eqs.append(df)
    else:
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
