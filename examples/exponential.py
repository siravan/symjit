import util

args = util.process_argv()

import math

from symjit import compile_func
from sympy import Product, Sum, symbols

# calculating pi using Viète's formula (https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula)

x, y, k = symbols("x y k")

# f = compile_func([x], [2 / viete(x), 2 / lemniscate(x)], **args)
f = compile_func([x], [Sum(x**k / Product(y, (y, 1, k)), (k, 0, 10))], **args)

# print(f.dumps())

print(f(1.0), "?= ", math.exp(1.0))
