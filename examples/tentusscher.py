import util
args = util.process_argv()

import os
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
import time
from symjit import compile_json

path = os.path.join(os.path.dirname(__file__), "cellml", "tentusscher.json")

with open(path) as fd:
    model = fd.read()

f = compile_json(model, **args)
u0 = f.get_u0()
p = f.get_p()

t_eval = np.arange(0, 2000, 1.0)

t0 = time.perf_counter_ns()
sol = scipy.integrate.solve_ivp(
    f, (0, 2000), u0, t_eval=t_eval, args=p, method="BDF", max_step=0.1
)
t1 = time.perf_counter_ns()

print(f"done in {(t1-t0)*1e-6:.1f} ms")

plt.plot(t_eval, sol.y[11, :])
plt.show()
