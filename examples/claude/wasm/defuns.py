"""User-defined functions in WebAssembly. A Python function passed in `defuns` is
imported by the module under its name, so the host supplies its implementation
(here in JavaScript); from Python the function itself is called.
"""

import sys

import numpy as np
from sympy import Function, symbols

import wasm_util as wu
from symjit import compile_func

x, y = symbols("x y")
softplus = Function("softplus")
hypot = Function("hypot")

python_defs = {
    "softplus": lambda a: np.log1p(np.exp(a)),
    "hypot": lambda a, b: np.hypot(a, b),
}
javascript_defs = {
    "softplus": "(a) => Math.log1p(Math.exp(a))",
    "hypot": "(a, b) => Math.hypot(a, b)",
}

f = compile_func([x, y], [softplus(x) - hypot(x, y)], defuns=python_defs, ty="wasm")
path, layout = wu.export(f, "defuns")

points = [[0.5, 2.0], [-3.0, 4.0], [10.0, 0.0]]
want = [[np.log1p(np.exp(a)) - np.hypot(a, b)] for a, b in points]
np.testing.assert_allclose([f(*p) for p in points], want, rtol=1e-15)  # Python callbacks

if not wu.have_node():
    print("node not found: the module is written but not run")
    sys.exit()

got = wu.node(path, layout, "evaluate", points, host=javascript_defs)
for p, g in zip(points, got):
    print(f"softplus({p[0]}) - hypot{tuple(p)} = {g[0]:.12g}")
np.testing.assert_allclose(got, want, rtol=1e-14)
print("ok!")
