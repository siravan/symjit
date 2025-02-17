import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols

from symjit import compile_ode, compile_jac

t, x, y = symbols("t x y")
alpha, beta, gamma, delta = symbols("alpha beta gamma delta")

ode = (
    alpha*x - beta*x*y,
    -gamma*y + delta*x*y
    )

f = compile_ode(t, (x, y), ode, params=(alpha, beta, gamma, delta))
jac = compile_jac(t, (x, y), ode, params=(alpha, beta, gamma, delta))

u0 = (1.0, 1.0)
p = (2.0, 1.2, 3.0, 1.0)
t_eval = np.arange(0, 100, 0.1)

sol = solve_ivp(f, (0, 100.0), u0, method='BDF', t_eval=t_eval, args=p, jac=jac)

plt.plot(t_eval, sol.y[0, :])
plt.plot(t_eval, sol.y[1, :])
plt.show()
