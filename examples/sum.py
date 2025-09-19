import util

args = util.process_argv()

from sympy import symbols
from symjit import compile_func

N = 100

X = symbols(f"x[0:{N}]")

p = 0

for i in range(N):
    p += X[i]

print(p)

f = compile_func(list(X), p, **args)

print(f(*range(0, N)))
