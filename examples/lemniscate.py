import util
args = util.process_argv()

import math
from sympy import symbols, sqrt
from symjit import compile_func

# calculating pi using Viète's formula (https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula)

N = 21

x = symbols('x')

def lemniscate(x):
    p = 1

    for i in range(N):
        t = x
        for j in range(i):
            t = x + x / sqrt(t)
        p *= sqrt(t)

    return p


f = compile_func([x], [2 / lemniscate(x)], **args)
p = f(1/2)

print(p, '?= ', 2.622057554292119, '(lemniscate constant)')
# print(f.dumps())
