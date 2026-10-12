"""The Mandelbrot set in the browser: compiles the Composer program of
examples/composer/manderbrot.py to WebAssembly (out/mandelbrot.wasm), checks it
against the native compiler in Node, and leaves the module for mandelbrot.html,
which runs it for every pixel and draws the result on a canvas.

For each point c the program iterates z <- z^2 + c (z = 0, 19 iterations) and returns
sqrt(|z|) if |z| < 4, and 0 otherwise (the point escaped).

To view it, serve this directory (browsers do not fetch files from file:// URLs):

    python mandelbrot.py
    python -m http.server 8000
    # open http://localhost:8000/mandelbrot.html
"""

import sys

import numpy as np

import wasm_util as wu
from symjit import Composer, compile_composer


def mandelbrot(**options):
    cp = Composer(1, 1)
    z = cp.new_temp()
    cp.assign(z, cp.constant(0))
    c = cp.arg(0)

    bl = cp.new_block()
    bl.assign(z, bl.fadd(bl.square(z), c))
    cp.append_for(cp.new_temp(), 1, 20, bl)

    cp.assign(z, cp.abs(z))
    t = cp.join(cp.lt(z, cp.constant(4.0)), cp.sqrt(z), cp.constant(0))
    cp.assign(cp.out(0), t)

    return compile_composer(cp, dtype="complex128", **options)


f = mandelbrot(ty="wasm")
path, layout = wu.export(f, "mandelbrot", dtype="complex128")

if not wu.have_node():
    print("node not found: the module is written but not checked")
    sys.exit()

# check a coarse grid against the native compiler; complex inputs and outputs are
# (re, im) pairs
A, B = np.meshgrid(np.linspace(-2, 1, 61), np.linspace(-1.5, 1.5, 61))
C = (A + 1j * B).ravel()
want = mandelbrot().evaluate_complex(C.reshape(-1, 1)).ravel()
got = wu.node(path, layout, "evaluateInputs", [[c.real, c.imag] for c in C])
got = np.array([re + 1j * im for re, im in got])
np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

inside = np.count_nonzero(got.real > 0)
print(f"{len(C)} points match the native compiler ({inside} did not escape)")
print("now serve this directory (python -m http.server 8000) and open /mandelbrot.html")
print("ok!")
