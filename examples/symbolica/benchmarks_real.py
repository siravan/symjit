import math
import os
import random
import time

import numpy as np
import symjit
from symbolica import E

P = 60
N = 10000

CONFIG = os.path.join(os.path.dirname(__file__), "symjit.toml")


def build_evaluator_poly(num_terms: int, num_factors: int):
    vars = [E(f"x_{i}") for i in range(P)]

    expr = math.prod(vars)

    for _ in range(num_terms):
        random.shuffle(vars)
        expr += random.random() * math.prod(vars[:num_factors])

    ev = expr.evaluator(vars, jit_compile=False, cpe_iterations=0, iterations=0)
    return ev


def run():
    rng = np.random.default_rng(1349)
    inputs = rng.random((N, P)) - 0.5

    for k in range(31):
        print(f"{k}\t", end="")
        num_terms = math.floor(1.5**k)
        ev = build_evaluator_poly(num_terms, 10)
        # ev.jit_compile(False)

        t_start = time.time()
        res_eager = sum(ev.evaluate(inputs))
        t_eager = (time.time() - t_start) * 1000000.0 / N

        f = symjit.compile_evaluator(ev, use_simd=False, use_threads=False)
        t_start = time.time()
        res_symjit_no_simd = sum(f.evaluate(inputs))
        t_symjit_no_simd = (time.time() - t_start) * 1000000.0 / N

        f = symjit.compile_evaluator(
            ev, enable_simd512=False, use_simd=True, use_threads=False
        )
        t_start = time.time()
        res_symjit_simd = sum(f.evaluate(inputs))
        t_symjit_simd = (time.time() - t_start) * 1000000.0 / N

        f = symjit.compile_evaluator(
            ev, use_simd=True, enable_simd512=True, use_threads=False
        )
        t_start = time.time()
        res_symjit_simd512 = sum(f.evaluate(inputs))
        t_symjit_simd512 = (time.time() - t_start) * 1000000.0 / N

        threashold = 1e-14 * math.sqrt(num_terms)

        valid = (
            abs(res_eager - res_symjit_no_simd) < threashold
            and abs(res_eager - res_symjit_simd) < threashold
            and abs(res_eager - res_symjit_simd512) < threashold
        )

        msg = "   pass" if valid else "   fail"

        print(
            f"{k}\t{num_terms}\t{t_eager:7.1f}\t{t_symjit_no_simd:7.1f}\t{t_symjit_simd:7.1f}\t{t_symjit_simd512:7.1f}\t{msg}"
        )


run()
