import util

args = util.process_argv()

import time

import matplotlib.pyplot as plt
import numpy as np
from symjit import compile_func
from sympy import symbols

x, y, a, b = symbols("x y a b")

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
X = np.zeros_like(A)
Y = np.zeros_like(A)

t0 = time.perf_counter_ns()

f = compile_func([a, b, x, y], [x**2 - y**2 + a, 2 * x * y + b], **args)

# print(f.dumps("bytecode"))
# print(f.dumps("simd"))

t1 = time.perf_counter_ns()

for i in range(20):
    X, Y = f(A, B, X, Y)

t2 = time.perf_counter_ns()

print(f"compilation time: {1e-6 * (t1 - t0):.1f} ms")
print(f"running time: {1e-6 * (t2 - t1):.1f} ms")

# Z = np.hypot(X, Y)

plt.imshow((np.abs(X) < 2) & (np.abs(Y) < 2))

if __name__ == "__main__":
    plt.show()
