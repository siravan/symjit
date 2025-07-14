import util
backend, ty, use_simd, use_threads = util.process_argv()

import time
import os
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

import symjit

path = os.path.join(os.path.dirname(__file__), "cellml", "beeler.json")

with open(path) as fd:
    model = fd.read()

t0 = time.time()

f = symjit.compile_json(model, ty=ty)
u0 = f.get_u0()
p = f.get_p()

t1 = time.time()

t_eval = np.arange(0, 2000, 1.0)
sol = scipy.integrate.solve_ivp(
    f, (0, 2000), u0, t_eval=t_eval, args=p, method="BDF", max_step=0.1
)

print(f"compilation time: {1000 * (t1 - t0):.1f} ms")
print(f"running time: {1000 * (time.time() - t1):.1f} ms")

plt.plot(t_eval, sol.y[6, :])
plt.show()
