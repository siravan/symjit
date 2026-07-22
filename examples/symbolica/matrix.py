from symbolica import E, S, Matrix, Expression
import numpy as np
import math
import time
import os

from symjit import compile_evaluator

CONFIG = os.path.join(os.path.dirname(__file__), "symjit.toml")

N = 100

def vars(n, name):
    return [E(f"{name}_{i}_{j}") for i in range(n) for j in range(n)]

def mat(v):
    n = math.isqrt(len(v))
    return Matrix.from_linear(n, n, v)

def run(n):
    v = vars(n, "x")
    m = mat(v)
    u = (m * m * m)
    ev = Expression.evaluator_multiple([u[i,j].to_expression() for i in range(n) for j in range(n)], v)

    A = np.random.randn(n, n) + np.random.randn(n, n) * 1j
    B = np.matmul(A, np.matmul(A, A))

    dt_gen = 0
    dt_run = 0

    f = compile_evaluator(ev, ty=CONFIG)

    for _ in range(N):
        t0 = time.thread_time_ns()
        f = compile_evaluator(ev)
        t1 = time.thread_time_ns()
        C = f.evaluate_complex(A.reshape(1, -1)).reshape(n, n)
        t2 = time.thread_time_ns()
        dt_gen += (t1 - t0) * 1e-6 / N
        dt_run += (t2 - t1) * 1e-6 / N

    np.testing.assert_array_almost_equal(B, C)

    print(f"{n}:\tgeneration in {dt_gen:.3f} ms and execution in {dt_run:.3f} ms")


for i in range(2, 18):
    run(i)
