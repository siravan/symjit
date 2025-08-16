import util
args = util.process_argv()

import math
from sympy import symbols, Piecewise
from symjit import compile_func


def factorial(x, n):
    if n == 0:
        return 1
    else:
        return Piecewise([n, x >= n], [1, True]) * factorial(x, n-1)


x = symbols('x')

p = factorial(x, 20)
f = compile_func([x], p, **args)

# print(f.dumps())

print(f(18), ' ?= ', math.factorial(18))
