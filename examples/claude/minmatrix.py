import util

args = util.process_argv()

import numpy as np
from sympy import Matrix, symbols
from symjit import compile_func

N = 10

x = symbols(f"x0:{N}")


def min_matrix(x):
    # the "min matrix" (the covariance matrix of a standard Wiener
    # process sampled at times x_0 < x_1 < ... < x_{n-1}): entry (i, j)
    # is min(x_i, x_j). Since the x's are assumed to be given in
    # increasing order, min(x_i, x_j) is simply x_{min(i, j)}, which lets
    # the matrix be built with plain symbol lookups instead of sympy's
    # `Min`.
    n = len(x)
    return Matrix(n, n, lambda i, j: x[min(i, j)])


# Method 1: build the determinant explicitly (an "unfolded" expression
# obtained through symbolic LU elimination) and compile it directly.
det_expr = min_matrix(x).det(method="lu")
f_explicit = compile_func(list(x), det_expr, **args)

# Method 2: the closed-form formula for the determinant of the min
# matrix,
#
#   det(M) = x_0 * (x_1 - x_0) * (x_2 - x_1) * ... * (x_{n-1} - x_{n-2}),
#
# i.e. the product of the gaps between consecutive (sorted) sample
# points. As with the Vandermonde example, n is fixed at compile time, so
# the product is simply unrolled in Python.
closed_expr = x[0]
for k in range(1, N):
    closed_expr *= x[k] - x[k - 1]

f_closed = compile_func(list(x), closed_expr, **args)

vals = np.sort(np.random.rand(N))

d1 = f_explicit(*vals)
d2 = f_closed(*vals)

print("explicit determinant:   ", d1)
print("closed-form determinant:", d2)

np.testing.assert_allclose(d1, d2, rtol=1e-6)
print("ok!")
