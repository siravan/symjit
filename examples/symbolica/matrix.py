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
    A = np.random.rand(n, n) + np.random.rand(n, n) * 1j

    dt_np = 0

    for _ in range(N):
        t0 = time.thread_time_ns()
        B = np.linalg.matrix_power(A, 3)
        t1 = time.thread_time_ns()
        dt_np += (t1 - t0) * 1e-6 / N

    B = np.linalg.matrix_power(A, 3)
    A0 = A.reshape(1, -1)
    B0 = B.reshape(1, -1)

    v = vars(n, "x")
    m = mat(v)
    u = (m * m * m)
    ev = Expression.evaluator_multiple([u[i,j].to_expression() for i in range(n) for j in range(n)], v)

    dt_gen = 0
    dt_run = 0

    f = compile_evaluator(ev, ty=CONFIG)

    for _ in range(N):
        t0 = time.thread_time_ns()
        f = compile_evaluator(ev)
        t1 = time.thread_time_ns()

        _ = f.evaluate_complex(A0)

        t2 = time.thread_time_ns()
        C = f.evaluate_complex(A0)
        t3 = time.thread_time_ns()

        dt_gen += (t1 - t0) * 1e-6 / N
        dt_run += (t3 - t2) * 1e-6 / N

        np.testing.assert_array_almost_equal(B0, C)

    print(f"{n}:\tgeneration in {dt_gen:.3f} ms and execution in {dt_run:.3f} ms vs {dt_np:.3f} for np")


for i in range(2, 30):
    run(i)
