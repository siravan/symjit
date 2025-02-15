import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols

import symjit

t, x, y = symbols('t x y')
f = symjit.compile_ode(t, (x, y), (y, -x))
t_eval=np.arange(0, 10, 0.01)
sol = scipy.integrate.solve_ivp(f, (0, 10), (0.0, 1.0), t_eval=t_eval)

plt.plot(t_eval, sol.y.T)
plt.show()


