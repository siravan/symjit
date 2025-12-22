import numbers
import re

from . import structure

# from symbolica import AtomType, Expression
# symbolica_installed = True

try:
    from symbolica import AtomType, Expression, N, S

    symbolica_installed = True
except ImportError:
    symbolica_installed = False

    class Expression:
        pass

    class AtomType:
        Var = None
        Num = None
        Fn = None
        Mul = None
        Add = None
        Pow = None

    def S(x):
        return x

    def N(x):
        return x


def is_expression(a):
    return isinstance(a, Expression)


def strip_namespace(a):
    match = re.search(r"::(.*)", a.get_name())
    if match is not None:
        return match[1]
    else:
        return ""


def walk_tree(a):
    if not symbolica_installed:
        raise ValueError("Symbolica is not installed")

    if a.get_type() == AtomType.Var:
        return {"type": "Var", "name": strip_namespace(a)}
    elif a.get_type() == AtomType.Num:
        return {"type": "Const", "val": a.evaluate({}, {})}
    elif is_if(a):
        return tree_ifelse(a)
    elif is_Sum(a):
        return tree_Reduce("Sum", a)
    elif is_Product(a):
        return tree_Reduce("Product", a)
    elif a.get_type() == AtomType.Fn:
        return tree_node(operation(a), a)
    elif a.get_type() == AtomType.Mul:
        return tree_node("times", a)
    elif a.get_type() == AtomType.Add:
        return tree_node("plus", a)
    elif a.get_type() == AtomType.Pow:
        return tree_node("power", [a[0], a[1]])
    else:
        raise ValueError("not a valid Symbolica expression")


def is_Sum(a):
    return a.get_type() == AtomType.Fn and strip_namespace(a) == "Sum"


def is_Product(a):
    return a.get_type() == AtomType.Fn and strip_namespace(a) == "Product"


def tree_Reduce(op, a):
    v = S(f"${str(a[1])}")
    a = a.replace(a[1], v)
    return tree_node(op, a)


def is_if(a):
    return a.get_type() == AtomType.Fn and strip_namespace(a) == "if"


bool_ops = ["lt", "leq", "gt", "geq", "eq", "neq", "and", "or", "xor", "not"]


def tree_ifelse(a):
    args = [walk_tree(arg) for arg in a]

    if a.get_type() == AtomType.Fn and strip_namespace(a[0]) in bool_ops:
        return {"type": "Tree", "op": "ifelse", "args": args}
    else:
        # Important! The expression is changed to `if args[0] == 0 ? false_val : true_val`.
        # The reason is that Symjit uses an all-1s mask for true, which is NaN is IEEE754.
        # Therefore, if a boolean value reaches here (it shouldn't, but we cannot rule out
        # all possibilities, a neq comparison with 0.0 will always return false.
        cond = {"type": "Tree", "op": "eq", "args": [args[0], walk_tree(N(0.0))]}
        return {"type": "Tree", "op": "ifelse", "args": [cond, args[2], args[1]]}


def tree_node(op, a):
    args = [walk_tree(arg) for arg in a]
    return {"type": "Tree", "op": op, "args": args}


def operation(a):
    op = strip_namespace(a)

    if op == "<":
        op = "lt"
    elif op == "<=":
        op = "leq"
    elif op == ">":
        op = "gt"
    elif op == ">=":
        op = "geq"
    elif op == "==":
        op = "eq"
    elif op == "!=":
        op = "neq"
    elif op == "sqrt":
        op = "root"
    elif op == "mod":
        op = "rem"
    elif op == "log":
        op = "ln"
    elif op == "asin":
        op = "arcsin"
    elif op == "acos":
        op = "arccos"
    elif op == "atan":
        op = "arctan"
    elif op == "asinh":
        op = "arcsinh"
    elif op == "acosh":
        op = "arccosh"
    elif op == "atanh":
        op = "arctanh"

    return op
