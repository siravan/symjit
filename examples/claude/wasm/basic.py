"""A model compiled to WebAssembly (`ty="wasm"`) and evaluated in Node with host.mjs:
one point at a time (`evaluate`), over arrays (`evaluateMany`), and through the
`fast` export of a single-output model.
"""

import sys

import numpy as np
from sympy import cos, exp, sin, sqrt, symbols

import wasm_util as wu
from symjit import compile_func

x, y = symbols("x y")
eqs = [sin(x) * cos(y), exp(-x * x - y * y), sqrt(1 + x * x) / (1 + y * y)]


def reference(X, Y):
    return [np.sin(X) * np.cos(Y), np.exp(-X * X - Y * Y), np.sqrt(1 + X * X) / (1 + Y * Y)]


f = compile_func([x, y], eqs, ty="wasm")
path, layout = wu.export(f, "basic")

# calls from Python still work (they run on symjit's bytecode interpreter)
np.testing.assert_allclose(f(0.5, 2.0), reference(0.5, 2.0), rtol=1e-14)

if not wu.have_node():
    print("node not found: the module is written but not run")
    sys.exit()

# one point at a time
points = [[0.5, 2.0], [-1.0, 0.25], [3.0, -1.5]]
got = wu.node(path, layout, "evaluate", points)
for p, g in zip(points, got):
    print(f"f{tuple(p)} = {np.round(g, 6).tolist()}")
    np.testing.assert_allclose(g, reference(*p), rtol=1e-14)

# many points: one array per state in, one array per output out
X = np.linspace(-2, 2, 10001)
Y = np.cos(3 * X)
got = wu.node(path, layout, "evaluateMany", [X.tolist(), Y.tolist()])
np.testing.assert_allclose(got, reference(X, Y), rtol=1e-14)
print(f"evaluateMany: {len(X)} points match numpy")

# a single-output model without parameters also exports `fast(x, y)`
g = compile_func([x, y], [sin(x) * y + x / (1 + y * y)], ty="wasm")
path, layout = wu.export(g, "basic_fast")
got = wu.node(path, layout, "fast", points)
np.testing.assert_allclose(got, [np.sin(a) * b + a / (1 + b * b) for a, b in points], rtol=1e-14)
print(f"fast: {got}")

print("ok!")
