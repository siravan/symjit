import util
args = util.process_argv()

import numpy as np
from sympy import symbols, lambdify, sin, cos
from symjit import compile_func

x, y = symbols("x y")

def test(p):
    f = compile_func([x, y], p, **args)
    g = lambdify([x, y], p)
    assert(f(1, 2) == g(1, 2))
    assert(f(1.0, 2.0) == g(1.0, 2.0))
    u = np.arange(100)
    v = np.arange(100)
    np.testing.assert_array_almost_equal(f(u, v), g(u, v))

test(x+y)
test([x+y])
test((x+y,))
test([sin(x+y)])
test((sin(x+y),))
test([x+y, x*y])
test((x+y, x*y))
test([sin(x+y), cos(x*y)])
test((sin(x+y), cos(x*y)))

print("ok!")
