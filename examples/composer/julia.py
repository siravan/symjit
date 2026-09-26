import time
import statistics
import matplotlib.pyplot as plt
import gc
import numpy as np
from symjit import Composer, compile_composer

P = 400
L = 12
N = 2**L
M = 100

def tree(cp, k0, k1, z, c):
    if k1 - k0 == 1:
        cp.assign(cp.out(k0), z)
    else:
        t1 = cp.fsub(z, c)
        t2 = cp.sqrt(t1)
        t3 = cp.neg(t2)
        k_mid = (k0 + k1) // 2
        tree(cp, k0, k_mid, t2, c)
        tree(cp, k_mid, k1, t3, c)


def julia():
    cp = Composer(1, N)
    c = cp.arg(0)
    z = cp.constant(np.random.randn() + np.random.randn()*1j)
    tree(cp, 0, N, z, c)
    f = compile_composer(cp, dtype="complex128", direct=True, fast_complex=True, use_simd=False, use_threads=False)
    return f

f = julia()

# print(f.dumps('bytecode'))

X = [[0.25 + 0.25j]]
T = []

for _ in range(1000):
    gc.collect()
    gc.disable()
    t0 = time.perf_counter_ns()
    a = f.evaluate_complex(X).sum()
    t1 = time.perf_counter_ns()
    T.append((t1 - t0) / 1e6)
    gc.enable()

print(f"Time: {statistics.geometric_mean(T):.4f} ms")


Z = f.evaluate_complex([[0.4 + 0.4j]])
plt.plot(Z[0].real, Z[0].imag, ".")
plt.show()
