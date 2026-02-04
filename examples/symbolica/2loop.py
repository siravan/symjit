import os

# import resource
import time

import numpy as np
from symjit import compile_evaluator, load_func

# resource.setrlimit(resource.RLIMIT_STACK, (16777216, 2 * 16777216))

print("Building symjit evaluator...")
t_start = time.time()

with open(
    os.path.join(os.path.dirname(__file__), "evaluator_instructions_2loop.txt"),
    encoding="utf-8",
) as fd:
    S = fd.read()

f = compile_evaluator(S, dtype="complex128", use_threads=True, use_simd=True)
print(f"completed in {time.time() - t_start:.1f} s.")

f.save("2loop.sjb")

N_SAMPLES = 10000
n = f.count_params // 2

rng = np.random.default_rng(1337)
samples_real = rng.random(N_SAMPLES * n).reshape((N_SAMPLES, n))
samples_imag = rng.random(N_SAMPLES * n).reshape((N_SAMPLES, n))

samples = samples_real + 1j * samples_imag

print("Running symjit evaluator...")

t_start = time.time()
f.evaluate_complex(samples)
print(f"Symjit evaluation: {((time.time() - t_start) * 1000.0 / N_SAMPLES):.3f} ms")

print(f.evaluate_complex(samples[:1, :]).sum())

g = load_func("2loop.sjb")

print(g.evaluate_complex(samples[:1, :]).sum())
