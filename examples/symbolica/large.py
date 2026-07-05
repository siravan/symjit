import math

import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

DEPTH = 12
NCOLS = 2**DEPTH
NROWS = 107

xs = [S(f"x{i}") for i in range(NCOLS)]


def large(a, b):
    d = b - a
    if d == 1:
        return xs[a]
    elif (round(math.log2(d)) % 2) == 0:
        n = (a + b) // 2
        return large(a, n) + large(n, b)
    else:
        n = (a + b) // 2
        return large(a, n) * large(n, b)


ev = large(0, NCOLS).evaluator(xs, jit_compile=False)
f = compile_evaluator(ev, dtype="complex128", direct=False)

print(f"{len(f.dumps('simd'))} bytes")
# print(f.dumps("simd"))
print(large(0, 16))

X = np.random.rand(NROWS, NCOLS) + np.random.rand(NROWS, NCOLS) * 1j - (0.5 + 0.5j)

A = ev.evaluate_complex(X)
B = f.evaluate_complex(X)

np.testing.assert_array_almost_equal(A, B, verbose=True)
print("pass!")
