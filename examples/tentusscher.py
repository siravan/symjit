import util
backend, ty, use_simd, use_threads = util.process_argv()

import os
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

from symjit import compile_json

path = os.path.join(os.path.dirname(__file__), "cellml", "tentusscher.json")

with open(path) as fd:
    model = fd.read()

f = compile_json(model, ty=ty)
u0 = f.get_u0()
p = f.get_p()

t_eval = np.arange(0, 2000, 1.0)
sol = scipy.integrate.solve_ivp(
    f, (0, 2000), u0, t_eval=t_eval, args=p, method="BDF", max_step=0.1
)

plt.plot(t_eval, sol.y[11, :])
plt.show()
