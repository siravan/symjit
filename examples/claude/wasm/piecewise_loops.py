"""Conditionals and loops in WebAssembly: `Piecewise` (compiled branch-free, like
`Min`/`Max`), and `Sum`/`Product` (compiled to loops, structured by the backend's
control-flow lowering).
"""

import sys

import numpy as np
from sympy import And, Max, Min, Piecewise, Product, Sum, sin, symbols

import wasm_util as wu
from symjit import compile_func

x, y, k = symbols("x y k")

eqs = [
    # a clipped, piecewise-defined profile
    Piecewise((0.0, x < -1), (1 - x * x, And(x >= -1, x <= 1)), (x - 1, True)),
    Min(x, y) + Max(sin(x), y),
    # a polynomial (sum of x^k / (k + 1), the series of -log(1 - x)/x) and a
    # finite Fourier series of a square wave
    Sum(x**k / (k + 1), (k, 0, 12)),
    Sum(sin((2 * k + 1) * x) / (2 * k + 1), (k, 0, 25)),
]


def reference(X, Y):
    return [
        np.where(X < -1, 0.0, np.where(X <= 1, 1 - X * X, X - 1)),
        np.minimum(X, Y) + np.maximum(np.sin(X), Y),
        sum(X**j / (j + 1) for j in range(13)),
        sum(np.sin((2 * j + 1) * X) / (2 * j + 1) for j in range(26)),
    ]


f = compile_func([x, y], eqs, ty="wasm")
path, layout = wu.export(f, "piecewise_loops")

if not wu.have_node():
    print("node not found: the module is written but not run")
    sys.exit()

X = np.linspace(-2.5, 2.5, 2001)
Y = 0.5 * np.sin(5 * X)
got = wu.node(path, layout, "evaluateMany", [X.tolist(), Y.tolist()])
np.testing.assert_allclose(got, reference(X, Y), rtol=1e-12, atol=1e-12)

# a product: 6! / (6 - n)! for n = 0..6 (an empty product for n = 0)
g = compile_func([x], [Product(x - k + 1, (k, 1, 6))], ty="wasm")
path, layout = wu.export(g, "product")
got = wu.node(path, layout, "evaluate", [[6.0], [2.5]])
print(f"Product(x - k + 1, k = 1..6) at x = 6, 2.5: {got}")
np.testing.assert_allclose(got, [[720.0], [np.prod([2.5 - j + 1 for j in range(1, 7)])]], rtol=1e-14)

print(f"{len(X)} points of {len(eqs)} piecewise and looping outputs match numpy")
print("ok!")
