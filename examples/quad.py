import numpy as np
from scipy.integrate import quad
from sympy import symbols, atan, sqrt
from symjit import compile_func
from math import pi

x = symbols("x")
# Ahmed's Integral (Inside Interesting Integrals, 6.2)
f = compile_func([x], [atan(sqrt(2+x**2)) / ((1+x**2)*sqrt(2+x**2))])

sol = quad(f, 0.0, 1.0)

np.testing.assert_approx_equal(sol[0], 5*pi**2/96)

print('ok!')

