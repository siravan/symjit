# This file is generated using claude code
import util

args = util.process_argv()

import numpy as np
from sympy import Matrix, symbols
from symjit import compile_func

N = 10

x = symbols(f"x0:{N}")


def vandermonde(x):
    n = len(x)
    return Matrix(n, n, lambda i, j: x[i] ** j)


# Method 1: build the determinant explicitly (an "unfolded" expression
# obtained through symbolic LU elimination) and compile it directly.
det_expr = vandermonde(x).det(method="lu")
f_explicit = compile_func(list(x), det_expr, **args)

# Method 2: the closed-form formula for the determinant of a Vandermonde
# matrix,
#
#   det(V) = prod_{0 <= i < j < n} (x_j - x_i).
#
# Since n is fixed at compile time, the double product is simply unrolled
# in Python; no explicit `Product` loop (and no complex numbers, unlike
# the circulant case) is needed.
closed_expr = 1
for i in range(N):
    for j in range(i + 1, N):
        closed_expr *= x[j] - x[i]

f_closed = compile_func(list(x), closed_expr, **args)

vals = np.random.randn(N)

d1 = f_explicit(*vals)
d2 = f_closed(*vals)

print("explicit determinant:   ", d1)
print("closed-form determinant:", d2)

np.testing.assert_allclose(d1, d2, rtol=1e-6)
print("ok!")
