"""A complex model (`dtype="complex128"`) in WebAssembly. Complex values are pairs of
f64 (re, im) in memory, and complex functions are imported as `cplx_<name>`, taking
and returning (re, im); host.mjs implements them (`COMPLEX`).
"""

import sys

import numpy as np
from sympy import exp, sin, sqrt, symbols

import wasm_util as wu
from symjit import compile_func

z, w = symbols("z w")
# (the imaginary unit `I` cannot appear in symjit expressions; complex values come
# from the inputs)
eqs = [z * w + 1, exp(z) * sin(w), sqrt(1 + z * z), z**w]


def reference(Z, W):
    return [Z * W + 1, np.exp(Z) * np.sin(W), np.sqrt(1 + Z * Z), Z**W]


f = compile_func([z, w], eqs, ty="wasm", dtype="complex128")
path, layout = wu.export(f, "complex_numbers")

if not wu.have_node():
    print("node not found: the module is written but not run")
    sys.exit()

rng = np.random.default_rng(1)
Z = rng.uniform(-1.5, 1.5, 1000) + 1j * rng.uniform(-1.5, 1.5, 1000)
W = rng.uniform(-1, 1, 1000) + 1j * rng.uniform(-1, 1, 1000)

# each complex state is two real arrays (re, im); outputs come back the same way
columns = [Z.real, Z.imag, W.real, W.imag]
got = np.array(wu.node(path, layout, "evaluateMany", [c.tolist() for c in columns]))
got = got[0::2] + 1j * got[1::2]
np.testing.assert_allclose(got, reference(Z, W), rtol=1e-12)

print(f"{len(Z)} complex points of {len(eqs)} outputs match numpy")
print("imported functions: cplx_exp, cplx_sin, cplx_power (sqrt is inlined by symjit)")
print("ok!")
