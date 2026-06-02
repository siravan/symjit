import math

import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

x, a = S("x"), S("a")


def c(u, L=10):
    return sum((-1) ** (i // 2) * u**i / math.factorial(i) for i in range(0, L, 2))


def s(u, L=10):
    return (1 - c(u, L) ** 2).sqrt()


def diff_s(u, L=10):
    return s(u).derivative(x) / u.derivative(x)


def expr(x, a, n=5):
    u = x
    for _ in range(n):
        u = u - (s(u) - a) / diff_s(u)
    return 4 * u


ev = expr(x, a).evaluator({}, {}, [x, a], jit_compile=False)

t = [[0.123456 + 0.5j, math.sqrt(2) / 2]]

print(f"number of instructions = {len(ev.get_instructions()[0])}")

print(f"pi     = {math.pi}")

p1 = ev.evaluate_complex(t)
print(f"eager  = {np.real(p1[0][0])}")

f = compile_evaluator(ev, dtype="complex128")
p2 = ev.evaluate_complex(t)
print(f"symjit = {np.real(p1[0][0])}")
