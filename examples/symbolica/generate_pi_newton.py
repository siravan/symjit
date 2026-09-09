import math
import os

import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

x, a = S("x"), S("a")

L = 16

def c(u):
    return sum((-1) ** (i // 2) * u**i / math.factorial(i) for i in range(0, L, 2))


def s(u):
    return (1 - c(u) ** 2).sqrt()


def diff_s(u):
    return s(u).derivative(x) / u.derivative(x)


def expr(x, a, n=4):
    u = x
    for _ in range(n):
        u = u - (s(u) - a) / diff_s(u)
    return 4 * u


ev = expr(x, a).evaluator([x, a], jit_compile=False)
f = compile_evaluator(ev, dtype="complex128", opt_level=2)

SJB = os.path.join(os.path.dirname(__file__), "pi_newton.sjb")

f.save(SJB)
