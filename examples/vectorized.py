import util
backend, ty, use_simd, use_threads = util.process_argv()

import numpy as np
from sympy import *
from symjit import compile_func

x, y = symbols("x y")
f = compile_func([x, y], [x + y, x * y], backend=backend, ty=ty, use_threads=use_threads, use_simd=use_simd)
g = lambdify([x, y], [x + y, x * y])

u = np.random.rand(10)
w = np.random.rand(10)

np.testing.assert_equal(f(u, w), g(u, w))
print("ok!")
