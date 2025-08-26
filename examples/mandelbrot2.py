import util
args = util.process_argv()

import time
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, expand
from symjit import compile_func

x, y, a, b = symbols("x y a b")

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))

def quad_map(x, y, a, b):
    return (x**2 - y**2 + a, 2 * x * y + b)


X = 0
Y = 0

for i in range(12):
    X, Y = quad_map(X, Y, a, b)

t0 = time.perf_counter_ns()

f = compile_func([a, b], [X, Y], **args)

t1 = time.perf_counter_ns()

X, Y = f(A, B)

t2 = time.perf_counter_ns()

print(f"compilation time: {1e-6 * (t1 - t0):.1f} ms")
print(f"running time: {1e-6 * (t2 - t1):.1f} ms")

# Z = np.hypot(X, Y)

plt.imshow((np.abs(X) < 2) & (np.abs(Y) < 2))

if __name__ == "__main__":
    plt.show()
