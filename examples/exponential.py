import util

args = util.process_argv()

import math

from symjit import compile_func
from sympy import Product, Sum, symbols, Function

# calculating pi using Viète's formula (https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula)

x, y, k = symbols("x y k")

# f = compile_func([x], [2 / viete(x), 2 / lemniscate(x)], **args)
f = compile_func([x], [Sum(x**k / Product(y, (y, 1, k)), (k, 0, 20))], **args)

fact = compile_func([x], Product(y, (y, 1, x)), **args)
F = Function("F")
g = compile_func([x], [Sum(x**k / F(k), (k, 0, 20))], **args, defuns={F: fact})

print(f.dumps())
print(g.dumps())

print(f(1.0), "?= ", math.exp(1.0))
print(g(1.0), "?= ", math.exp(1.0))
