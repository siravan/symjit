# This file is generated using claude code based on the following prompt:
"""
write an example similar to other programs in the examples directory. The example should take
10 inputs, generate a circuland matrix, and then calculate its determinants in two ways.
First, find the determinant explicitely and generate an unfold expression, which is solved
using compute_func. The second way is using the closed-form formula for the determinant
of a circulant matrix.
"""

import util

args = util.process_argv()

import numpy as np
from sympy import Matrix, Product, cos, pi, sin, symbols
from symjit import compile_func

N = 10

c = symbols(f"c0:{N}")


def circulant(c):
    n = len(c)
    return Matrix(n, n, lambda i, j: c[(j - i) % n])


# Method 1: build the determinant explicitly (an "unfolded" expression
# obtained through symbolic LU elimination) and compile it directly.
det_expr = circulant(c).det(method="lu")

f_explicit = compile_func(list(c), det_expr, **args)

print(f_explicit.measure("ker-scalar-size"))

# Method 2: the closed-form formula for the determinant of a circulant
# matrix,
#
#   det(C) = prod_{j=0}^{n-1} sum_{k=0}^{n-1} c_k * w^(j*k),
#
# where w = exp(2*pi*i/n) is the n-th root of unity. `iu` stands in for
# the imaginary unit `i`: symjit does not accept a literal complex
# constant inside an expression, so it is instead passed in at call time
# as a complex-valued parameter.
j = symbols("j")
iu = symbols("iu")

root = sum(c[k] * (cos(2 * pi * j * k / N) + iu * sin(2 * pi * j * k / N)) for k in range(N))
closed_expr = Product(root, (j, 0, N - 1))

closed_args = {**args, "dtype": "complex128"}
f_closed = compile_func(list(c), closed_expr, params=[iu], **closed_args)

# vals = np.arange(1.0, N + 1.0)
vals = np.random.randn(N)

d1 = f_explicit(*vals)
d2 = f_closed(*vals, 1j).real

print("explicit determinant:   ", d1)
print("closed-form determinant:", d2)

np.testing.assert_allclose(d1, d2, rtol=1e-6)
print("ok!")
