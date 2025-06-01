import util
backend, ty, use_simd = util.process_argv()

import time
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from symjit import compile_func

x, y, a, b = symbols("x y a b")

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
X = np.zeros_like(A)
Y = np.zeros_like(A)

t0 = time.time()

f = compile_func([a, b, x, y], [x**2 - y**2 + a, 2 * x * y + b], backend=backend, ty=ty, use_simd=use_simd)

t1 = time.time()

for i in range(20):
    X, Y = f(A, B, X, Y)

print(f"compilation time: {1000 * (t1 - t0):.1f} ms")
print(f"running time: {1000 * (time.time() - t1):.1f} ms")

# Z = np.hypot(X, Y)

plt.imshow((np.abs(X) < 2) & (np.abs(Y) < 2))
plt.show()
