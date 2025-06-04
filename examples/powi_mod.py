import util
backend, ty, use_simd = util.process_argv()

print(backend, ty, use_simd)

import math
from sympy import symbols, sin
from symjit import compile_func

N = 8
K = 4
power = 5
modulus = 65537

def binom(x, y, n, k):    
    if k == 0 or k == n:
        return 1.0
    else:
        return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

x, y = symbols('x y')

f = compile_func([x, y], [binom(x, y, N, K)**5 % 65537], backend=backend, ty=ty)
print(f(1, 1)[0], ' ?= ', math.comb(N, K)**power % 65537)
# print(f.dumps())
