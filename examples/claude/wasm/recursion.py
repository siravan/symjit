"""A recursive `Composer` function in WebAssembly: Fibonacci,

    fib(n) = n if n < 2 else fib(n - 1) + fib(n - 2),

with the recursive call (`call(None, ...)`) in the else block. A recursion too deep
for the module's stack makes `run` return 1 (host.mjs raises an error), and the
module keeps working afterwards; native code has no such check.
"""

import sys

import wasm_util as wu
from symjit import Composer, compile_composer


def fibonacci():
    cp = Composer(1, 1)
    n = cp.arg(0)
    small = cp.lt(n, cp.constant(2.0))
    base, rec = cp.new_block(), cp.new_block()
    r1 = base.fadd(n, cp.constant(0.0))
    r2 = rec.fadd(rec.call(None, rec.fsub(n, cp.constant(1.0))), rec.call(None, rec.fsub(n, cp.constant(2.0))))
    cp.append_if_else(small, base, rec)
    cp.assign(cp.out(0), cp.join(small, r1, r2))
    return cp


f = compile_composer(fibonacci(), ty="wasm")
path, layout = wu.export(f, "recursion")

if not wu.have_node():
    print("node not found: the module is written but not run")
    sys.exit()

ns = [0, 1, 2, 10, 20, 25]
got = wu.node(path, layout, "evaluateInputs", [[n] for n in ns])
print("fib:", {n: g[0] for n, g in zip(ns, got)})
assert [g[0] for g in got] == [0, 1, 1, 55, 6765, 75025]

# 100000 nested calls exhaust the stack region; the next call works again
got = wu.node(path, layout, "evaluateInputs", [[100000], [10]])
print("fib(100000):", got[0])
assert "status 1" in got[0] and got[1] == [55]
print("ok!")
