import util

args = util.process_argv()

from sympy import symbols
from symjit import compile_func

X = symbols("x[0:100]")

for i in range(1, 100):
    f = compile_func(X[:i], sum(X[:i]), **args)
    y0 = f(*range(0, i))
    y1 = i * (i - 1) / 2
    print(i, y0, y0 == y1)
