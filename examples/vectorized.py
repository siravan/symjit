from sympy import *
import numpy as np
from symjit import compile_func

x, y = symbols('x y')
f = compile_func([x, y], [x+y, x*y])
g = lambdify([x, y], [x+y, x*y])

u = np.random.rand(10)
w = np.random.rand(10)

np.testing.assert_equal(f(u, w), g(u, w))
print('ok!')
