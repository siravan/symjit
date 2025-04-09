import sys
import numpy as np
from sympy import *
from symjit import compile_func

backend = "python" if len(sys.argv) > 1 and sys.argv[1] == "py" else "rust"

x, y = symbols("x y")
f = compile_func([x, y], [x + y, x * y], backend=backend)
g = lambdify([x, y], [x + y, x * y])

u = np.random.rand(10)
w = np.random.rand(10)

np.testing.assert_equal(f(u, w), g(u, w))
print("ok!")
