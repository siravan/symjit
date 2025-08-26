import util
args = util.process_argv()

import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols
from symjit import compile_ode

t, x, y, z = symbols("t x y z")
sigma, rho, beta = symbols("sigma rho beta")

ode = (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)

t0 = time.time()

f = compile_ode(t, (x, y, z), ode, params=(sigma, rho, beta), **args)

# print(f.dumps())

u0 = (1.0, 1.0, 1.0)
p = (10.0, 28.0, 8 / 3)
t_eval = np.arange(0, 100, 0.01)

sol = solve_ivp(f, (0, 100.0), u0, t_eval=t_eval, args=p)

print(f"compilation + running time: {1000 * (time.time() - t0):.1f} ms")

plt.plot(sol.y[0, :], sol.y[2, :])

if __name__ == "__main__":
    plt.show()
