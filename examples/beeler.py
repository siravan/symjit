import os
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

import symjit

path = os.path.join(os.path.dirname(__file__), "cellml", "beeler.json")

with open(path) as fd:
    model = fd.read()

f = symjit.compile_json(model)
u0 = f.get_u0()
p = f.get_p()

t_eval = np.arange(0, 2000, 1.0)
sol = scipy.integrate.solve_ivp(f, (0, 2000), u0, t_eval=t_eval, args=p)

plt.plot(t_eval, sol.y[6, :])
plt.show()
