import util
args = util.process_argv()

import math
from sympy import symbols
from symjit import compile_func

# This is a very slow way to calculate the binomial coefficient!
# The point is to stress the compiler by generating large expression
# trees with easily verifiable results.

N = 7
K = 4

def binom(x, y, n, k):
    if k == 0 or k == n:
        return 1.0
    else:
        return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

x, y = symbols('x y')
f = compile_func([x, y], binom(x, y, N, K), **args)
print(f(1, 1), '?=', math.comb(N, K))
# print(f.dumps())
