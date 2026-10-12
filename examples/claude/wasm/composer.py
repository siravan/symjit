"""A `Composer` program with an if/else block and a counted loop, compiled to
WebAssembly: a signed square root by Newton's iteration,

    r = sign(a) * sqrt(|a|),  with  t <- (t + |a|/t) / 2  repeated 30 times.

Composer (and Symbolica) modules take their inputs as parameters and write their
outputs to memory: `run(outs, 0, 0, inputs)`, which host.mjs calls `evaluateInputs`.
They cannot be called from Python with ty="wasm"; the native compile checks them here.
"""

import sys

import numpy as np

import wasm_util as wu
from symjit import Composer, compile_composer


def program():
    cp = Composer(1, 2)  # one input, two outputs
    a = cp.arg(0)
    m = cp.abs(a)

    # t = |a| if |a| > 1 else 1: a starting point above sqrt(|a|)
    big = cp.gt(m, cp.constant(1.0))
    b1, b2 = cp.new_block(), cp.new_block()
    t1 = b1.fadd(m, cp.constant(0.0))
    t2 = b2.fadd(cp.constant(1.0), cp.constant(0.0))
    cp.append_if_else(big, b1, b2)
    t = cp.join(big, t1, t2)

    # Newton's iteration in a counted loop
    body = cp.new_block()
    body.assign(t, body.fmul(cp.constant(0.5), body.fadd(t, body.fdiv(m, t))))
    cp.append_for(cp.new_temp(), 0, 30, body)

    neg = cp.lt(a, cp.constant(0.0))
    cp.assign(cp.out(0), cp.join(neg, cp.neg(t), t))
    cp.assign(cp.out(1), cp.fsub(cp.fmul(t, t), m))  # the residual
    return cp


f = compile_composer(program(), ty="wasm")
path, layout = wu.export(f, "composer")

native = compile_composer(program())
inputs = [[2.0], [-9.0], [1e-6], [0.25], [12345.678]]
want = [np.ravel(native(*p)).tolist() for p in inputs]

if not wu.have_node():
    print("node not found: the module is written but not run")
    sys.exit()

got = wu.node(path, layout, "evaluateInputs", inputs)
for p, g in zip(inputs, got):
    print(f"signed sqrt({p[0]}) = {g[0]:.12g}  (residual {g[1]:.1e})")
np.testing.assert_allclose(got, want, rtol=1e-15, atol=1e-15)
np.testing.assert_allclose([g[0] for g in got], [np.sign(p[0]) * np.sqrt(abs(p[0])) for p in inputs], rtol=1e-15)
print("ok!")
