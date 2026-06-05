import random

import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

N = 1000
M = 25

vars = [S(f"x{i}") for i in range(M)]


def term():
    eq = complex(2 * random.random() - 1, 2 * random.random() - 1)
    # eq = complex(2 * random.random() - 1, 0)
    for _ in range(5):
        eq *= vars[random.randint(0, M - 1)]
    return eq


def poly():
    return sum(term() for _ in range(N))


p = poly()

print(p)

e = p.evaluator(vars)

f1 = compile_evaluator(e, dtype="complex128", opt_level=1)
f2 = compile_evaluator(e, dtype="complex128", opt_level=2)

# print(f1.dumps())

m = len(vars)
X = np.random.random((20, m)) + np.random.random((20, m)) * 1j

Y = e.evaluate_complex(X)
Z2 = f2.evaluate_complex(X)
Z1 = f1.evaluate_complex(X)

print("Y == Z1 => ", np.abs(Y - Z1) < 1e-10)
print("Y == Z2 => ", np.abs(Y - Z2) < 1e-10)
