import util

args = util.process_argv()

import numpy as np
from symjit import *
from sympy import *

x, y = symbols("x y")

eqs = [
    [x + y],
    [x - y],
    [x * y],
    [x / y],
    [x % y],
    [1 / x],
    [sqrt(x)],
    [cbrt(x)],
    [x**2],
    [x**3],
    [x**4],
    [x ** (-1)],
    [x ** (-2)],
    [x ** (-3)],
    [x ** (-4)],
    [x ** Rational(1, 2)],
    [x ** Rational(3, 2)],
    [x ** Rational(2, 3)],
    [x ** Rational(4, 3)],
    [x**y],
    [-x],
    [Abs(x - y)],
    [Min(x, y)],
    [Max(x, y)],
    [Heaviside(x)],
    [exp(x)],
    [log(x)],
    [sin(x)],
    [cos(x)],
    [tan(x)],
    [sinc(x)],
    [csc(x)],
    [sec(x)],
    [cot(x)],
    [sinh(x)],
    [cosh(x)],
    [tanh(x)],
    # [csch(x)],
    # [sech(x)],
    # [coth(x)],
    [asin(x)],
    [acos(x)],
    [atan(x)],
    [atan2(x, y)],
    [asinh(x)],
    [acosh(1 + x)],
    [atanh(x)],
    [floor(x)],
    [ceiling(x)],
    [frac(x)],
    [Si(x)],
    [Ci(x)],
    # [Shi(x)],
    # [Chi(x)],
    [erf(x)],
    [erfc(x)],
    [gamma(x)],
    [loggamma(x)],
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
    print("testing ", eq)
    try:
        f = compile_func([x, y], eq, **args)
        g = lambdify([x, y], eq)
        np.testing.assert_array_almost_equal(f(X[0], Y[0]), g(X[0], Y[0]))
        np.testing.assert_array_almost_equal(f(X, Y), g(X, Y))
    except ValueError:
        print("operation not implemented")

print("ok!")
