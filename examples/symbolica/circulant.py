import cmath
import time

import numpy as np
from symbolica import Matrix, S
from symjit import compile_evaluator

# Symbolica version of ../circulant.py: builds a 10x10 circulant matrix
# out of Symbolica expressions and computes its determinant in two ways.

N = 10

c = [S(f"c{i}") for i in range(N)]


def circulant(c):
    n = len(c)
    return [[c[(j - i) % n] for j in range(n)] for i in range(n)]


# Method 1: build the determinant explicitly. Symbolica's `Matrix.det()`
# works over exact rational polynomials and returns the fully expanded
# closed-form polynomial in the c_i, which is then turned back into an
# `Expression`, given a Symbolica evaluator, and compiled.
M = Matrix.from_nested(circulant(c))

reals = np.random.rand(N) < 0.5
reals = reals.astype(int)

t0 = time.time()
det_expr = M.det().to_expression()
t1 = time.time()
print(f"determinant expansion took {t1 - t0:.2f} s")

ev = det_expr.evaluator(c)
ev.set_real_params(reals)
f_explicit = compile_evaluator(ev, dtype="complex128", fast_complex=True)

# Method 2: the closed-form formula for the determinant of a circulant
# matrix,
#
#   det(C) = prod_{j=0}^{n-1} sum_{k=0}^{n-1} c_k * w^(j*k),
#
# where w = exp(2*pi*i/n) is the n-th root of unity. Unlike the SymPy
# bridge (see ../circulant.py), Symbolica expressions accept a literal
# Python `complex` as a coefficient directly, so the roots of unity can
# just be plugged in as numeric constants -- no extra parameter is
# needed to stand in for the imaginary unit.
w = cmath.exp(2j * cmath.pi / N)

closed_expr = 1
for j in range(N):
    term = sum(c[k] * w ** (j * k) for k in range(N))
    closed_expr = closed_expr * term

ev_closed = closed_expr.evaluator(c)
ev_closed.set_real_params(reals)
f_closed = compile_evaluator(ev_closed, dtype="complex128", direct=True, fast_complex=True)

# print(f_explicit.dumps('bytecode'))
# print(f_closed.dumps('bytecode'))

vals = np.random.rand(N) + np.random.rand(N) * 1j
X = np.array([vals])

d1 = f_explicit.evaluate_complex(X)[0, 0]
d2 = f_closed.evaluate_complex(X)[0, 0]

for i in reals:
    X[:,i] = X[:,i].real

d3 = f_explicit.evaluate_complex(X)[0, 0]
d4 = f_closed.evaluate_complex(X)[0, 0]

print("explicit determinant:           ", d1)
print("closed-form determinant:        ", d2)
print("explicit determinant (real):    ", d3)
print("closed-form determinant (real): ", d4)

np.testing.assert_allclose(d1, d2, rtol=1e-6)
print("ok!")
