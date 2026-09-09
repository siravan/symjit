import math
import os
import random

import numpy as np
from symbolica import E

K = 20
P = 60
N = 25000

def build_evaluator_poly(num_terms: int, num_factors: int):
    vars = [E(f"x_{i}") for i in range(P)]

    expr = math.prod(vars)

    for _ in range(num_terms):
        random.shuffle(vars)
        expr += random.random() * math.prod(vars[:num_factors])

    ev = expr.evaluator(vars, jit_compile=False, cpe_iterations=0, iterations=0)
    return ev


rng = np.random.default_rng(1349)
inputs = rng.random((N, P)) + rng.random((N, P)) * 1j - (0.5 + 0.5j)
num_terms = math.floor(1.5**K)

ev = build_evaluator_poly(num_terms, 10)
res_eager = sum(ev.evaluate_complex(inputs))
print(f"res = {res_eager[0].real:.15f} + {res_eager[0].imag:.15f}j")

INSTRUCTIONS = os.path.join(
    os.path.dirname(__file__), f"benchmark_instructions.txt"
)

with open(INSTRUCTIONS, "w") as fd:
    fd.write(str(ev.get_instructions()))
