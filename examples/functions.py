import sys
from sympy import *
from symjit import *
import numpy as np

backend = "python" if len(sys.argv) > 1 and sys.argv[1] == "py" else "rust"

x, y = symbols('x y')

eqs = [
    [x + y],
    [x - y],
    [x * y],
    [x / y],
    [1 / x],
    [sqrt(x)],
    [x ** 2],
    [x ** 3],
    [x ** 4],
    [x ** (-1)],
    [x ** (-2)],
    [x ** (-3)],
    [x ** (-4)],
    [x ** Rational(1, 2)],
    [x ** Rational(3, 2)],
    [x ** Rational(2, 3)],
    [x ** Rational(4, 3)],
    [x ** y],
    [-x],
    [abs(x - y)],    
    [exp(x)],
    [log(x)],
    [sin(x)],
    [cos(x)],
    [tan(x)],
    # [csc(x)],
    # [sec(x)],
    # [cot(x)],
    [sinh(x)],
    [cosh(x)],
    [tanh(x)],
    # [csch(x)],
    # [sech(x)],
    # [coth(x)],
    [asin(x)],
    [acos(x)],
    [atan(x)],
    # [asinh(x)],
    # [acosh(1 + x)],
    # [atanh(x)],
    [Piecewise((1, x > y), (0, True))],
    [Piecewise((1, x < y), (0, True))],
    [Piecewise((1, x >= y), (0, True))],
    [Piecewise((1, x <= y), (0, True))],
    [Piecewise((1, ((x < y) | (x > sqrt(y)))), (0, True))],
    [Piecewise((1, ((x < y) & (x > sqrt(y)))), (0, True))],
    [Piecewise((1, ((x < y) ^ (x > sqrt(y)))), (0, True))],
]


X = np.random.rand(15) * 0.9 + 0.1
Y = np.random.rand(15) * 0.9 + 0.1

for eq in eqs:
    print('testing ', eq)
    f = compile_func([x, y], eq, use_simd=True, backend=backend)
    g = lambdify([x, y], eq)
    np.testing.assert_array_almost_equal(f(X[0], Y[0]), g(X[0], Y[0]))
    np.testing.assert_array_almost_equal(f(X, Y), g(X, Y))
    
print('ok!')

