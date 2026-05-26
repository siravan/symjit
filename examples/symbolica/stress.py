import random
import time

import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

x = S("x")

x0 = np.random.rand(10000, 1) * 0.0001 + 0j


def stress_fun(n):
    e = x**2 + x
    for _ in range(n):
        if random.random() < 0.5:
            e = e**2 + e
        else:
            e = e**2 - e
    e = e.derivative(x)
    return e


def time_evaluator(f, k):
    r = 0

    t0 = time.perf_counter_ns()
    for _ in range(k):
        r += f.evaluate_complex(x0[i : i + 1, 0:])
    t1 = time.perf_counter_ns()

    return t1 - t0, r


for i in range(15):
    ev = stress_fun(i).evaluator({}, {}, [x])
    compiled_f = ev.compile("stress", "stress.cpp", "stress.so", "complex")
    symjit_f = compile_evaluator(ev, dtype="complex128", direct=True)

    # compiled_dt, compiled_r = time_evaluator(compiled_f, 1000)
    symjit_dt, symjit_r = time_evaluator(symjit_f, 1000)
    eager_dt, eager_r = time_evaluator(ev, 1000)

    print(
        i,
        eager_r[0][0],
        # compiled_r[[0]],
        symjit_r[0][0],
        eager_dt * 1e-6,
        # compiled_dt * 1e-6,
        symjit_dt * 1e-6,
    )
