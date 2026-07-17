import math

import numpy as np
from symjit import Composer, compile_composer

DEPTH = 13
NCOLS = 2**DEPTH
NROWS = 128

K = np.reshape(np.arange(NROWS * NCOLS), (NROWS, NCOLS))
X = np.cos(K) + np.sin(K)**2 * 1j

def large(cp: Composer, a: int, b: int):
    d = b - a
    if d == 1:
        return cp.arg(a), X[0, a]
    else:
        n = (a + b) // 2
        child_left, val_left = large(cp, a, n)
        child_right, val_right = large(cp, n, b)

        if (round(math.log2(d)) % 2) == 0:
            return cp.fadd(child_left, child_right), val_left + val_right
        else:
            return cp.fdiv(child_left, child_right), val_left / val_right


for depth in range(DEPTH+1):
    print(f"{depth}\t", end='')
    ncols = 2**depth
    cp = Composer(ncols, 1)
    root, val = large(cp, 0, ncols)
    cp.assign(cp.out(0), root)

    f = compile_composer(cp, dtype="complex128")
    B = f.evaluate_complex(X[:, 0:ncols])

    np.testing.assert_array_almost_equal(val, B[0], verbose=True)
    print(f"{B[0]}; pass!")
