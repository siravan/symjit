import util

args = util.process_argv()

import math
from random import randint
from sympy import symbols, sqrt
from symjit import compile_func

# calculating pi using Viète's formula (https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula)

N = 21

x = symbols("x")


def viete(x):
    p = 1

    for i in range(N):
        t = x
        for j in range(i):
            t = x + x * sqrt(t)
        p *= sqrt(t)

    return p


f = compile_func([x], [2 / viete(x)], **args)

ps = [f(1 / 2) for _ in range(1000)]
p = ps[randint(0, 999)]

print(p, "?= ", math.pi, "(pi)")
# print(f.dumps())
