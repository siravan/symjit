import util
args = util.process_argv()

import numpy as np
from scipy.integrate import nquad
from sympy import symbols, exp
from symjit import compile_func

N = 5
t, x = symbols("t x")
f = compile_func([t, x], exp(-t * x) / t**N, **args)

sol = nquad(f, [[1, np.inf], [0, np.inf]])

np.testing.assert_approx_equal(sol[0], 1 / N)

print("ok!")
