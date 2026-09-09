import math
import os
import time

import numpy as np
import symjit

K = 20
P = 60
N = 25000

INSTRUCTIONS = os.path.join(
    os.path.dirname(__file__), f"benchmark_instructions.txt"
)

with open(INSTRUCTIONS, "rt", encoding="utf-8") as fd:
    ev = fd.read()


rng = np.random.default_rng(1349)
inputs = rng.random((N, P)) + rng.random((N, P)) * 1j - (0.5 + 0.5j)
num_terms = math.floor(1.5**K)
threashold = 1e-14 * math.sqrt(num_terms)

res_eager = [-0.449997965985545 + -0.313506785844491j]

def run_config(args):
    f = symjit.compile_evaluator(ev, dtype="complex128", **args)

    t_start = time.time()
    res = sum(f.evaluate_complex(inputs))
    t_symjit_cfg = (time.time() - t_start) * 1000.0

    valid = abs(res_eager - res) < threashold
    msg = "\033[32mpass\033[0m" if valid else "\033[31mfail\033[0m"
    print(f"\t{t_symjit_cfg:7.1f}\t{msg}")


def run():
    for use_simd in [False, True]:
        for fast_complex in [False, True]:
                for compress in [False, True]:
                    for opt_level in [0, 1, 2, 3]:
                        args = {"use_simd": use_simd, "fast_complex": fast_complex, "compress": compress, "opt_level": opt_level}
                        print(args)
                        run_config(args)

run()

# args = {"use_simd": False, "fast_complex": True, "compress": True, "opt_level": 1}
# run_config(args)
