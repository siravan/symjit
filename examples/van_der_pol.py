import util
args = util.process_argv()

import time
from math import sqrt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols
from symjit import compile_func, compile_ode, compile_jac

t, x, y, mu = symbols("t x y mu")

# this is the rescaled Van der Pol equation (see Hairer II 1.5')
ode = [y, mu * ((1 - x * x) * y - x)]

t0 = time.time()

f = compile_ode(t, [x, y], ode, params=[mu], **args)
jac = compile_jac(t, [x, y], ode, params=[mu], **args)

u0 = [0.0, sqrt(3.0)]
t_eval = np.arange(0, 10.0, 0.01)

# non-stiff, can use an explicit method like RK45, i.e., the Explicit Runge-Kutta method of order 5(4)
sol1 = solve_ivp(f, (0, 10.0), u0, method="RK45", t_eval=t_eval, args=[5.0])
# stiff because mu is now 1e6. RK45 fails. It needs an implicit method like backward differentiation formula (BDF)
sol2 = solve_ivp(f, (0, 10.0), u0, method="BDF", t_eval=t_eval, args=[1e6], jac=jac)

print(f"compilation + running time: {1000 * (time.time() - t0)} ms")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(t_eval, sol1.y[0, :])
ax2.plot(t_eval, sol2.y[0, :])
plt.show()
