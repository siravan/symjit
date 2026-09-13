import util

args = util.process_argv()

import random

import numpy as np
from sympy import Matrix, cos, cosh, sin, sinh, symbols, tan, tanh
from symjit import compile_func

N = 10

x = symbols("x")

# each entry of the matrix is a randomly chosen trigonometric or
# hyperbolic function of the single argument `x`
FUNCS = [sin, cos, tan, sinh, cosh, tanh]
NP_FUNCS = {sin: np.sin, cos: np.cos, tan: np.tan, sinh: np.sinh, cosh: np.cosh, tanh: np.tanh}

choice = [[random.choice(FUNCS) for _ in range(N)] for _ in range(N)]

M = Matrix(N, N, lambda i, j: choice[i][j](x))

# build the determinant explicitly (an "unfolded" expression obtained
# through symbolic LU elimination) and compile it into a function of the
# single variable `x`. The function is compiled for complex input/output
# so that `xv` below can be an arbitrary complex number.
det_expr = M.det(method="lu")
complex_args = {**args, "dtype": "complex128"}
f = compile_func([x], det_expr, **complex_args)

print(f.measure("ker-scalar-size"))

xv = np.random.randn() + 1j * np.random.randn()

got = f(xv)

# compare to the numerical value: evaluate every entry directly with
# numpy at the same point and take numpy's determinant.
numeric_matrix = np.array([[NP_FUNCS[choice[i][j]](xv) for j in range(N)] for i in range(N)])
want = np.linalg.det(numeric_matrix)

print("symjit determinant:", got)
print("numpy determinant: ", want)

np.testing.assert_allclose(got, want, rtol=1e-6)
print("ok!")
