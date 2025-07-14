import util
backend, ty, use_simd, use_threads = util.process_argv()

from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from symjit import compile_func

x, sigma = symbols("x sigma")
f = compile_func(
    [x], [exp(-((x - 100) ** 2) / (2 * sigma**2))], params=[sigma], backend=backend, use_threads=use_threads
)

t = np.arange(0, 200)
y = f(t, 25.0)[0]

plt.plot(t, y)
plt.show()
