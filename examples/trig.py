import util
backend, ty, use_simd, use_threads = util.process_argv()

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from sympy import symbols
from symjit import compile_ode

t, x, y = symbols("t x y")
f = compile_ode(t, (x, y), (y, -x), backend=backend, ty=ty)
t_eval = np.arange(0, 10, 0.01)
sol = scipy.integrate.solve_ivp(f, (0, 10), (0.0, 1.0), t_eval=t_eval)

plt.plot(t_eval, sol.y.T)
plt.show()
