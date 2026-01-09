import util

args = util.process_argv()

import time

import matplotlib.pyplot as plt
import numpy as np
from symjit import compile_func
from sympy import symbols

z, c = symbols("z c")

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
C = A + B * 1j

p = z
for _ in range(20):
    p = p**2 + c

t0 = time.perf_counter_ns()

f = compile_func([z, c], [p], **args)

n = A.shape[0] * A.shape[1]

t1 = time.perf_counter_ns()

Z = np.zeros(C.shape, dtype=np.complex128)
Z = f(Z, C)[0]

t2 = time.perf_counter_ns()

print(f"compilation time: {1e-6 * (t1 - t0):.1f} ms")
print(f"running time: {1e-6 * (t2 - t1):.1f} ms")

X = np.reshape(Z.real, A.shape)
Y = np.reshape(Z.imag, A.shape)

plt.imshow((np.abs(X) < 2) & (np.abs(Y) < 2))

if __name__ == "__main__":
    plt.show()
