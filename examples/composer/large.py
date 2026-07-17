import math

import numpy as np
from symjit import Composer, compile_composer

DEPTH = 16
NCOLS = 2**DEPTH
NROWS = 1

X = np.random.rand(NROWS, NCOLS) + np.random.rand(NROWS, NCOLS) * 1j - (0.5 + 0.5j)

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
    cp = Composer(NCOLS, 1)
    root, val = large(cp, 0, 2**depth)
    cp.assign(cp.out(0), root)

    f = compile_composer(cp, dtype="complex128")
    B = f.evaluate_complex(X)

    np.testing.assert_array_almost_equal(val, B[0], verbose=True)
    print(f"{B[0]}; pass!")
