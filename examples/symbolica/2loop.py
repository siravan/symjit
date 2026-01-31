import os
import resource
import time

import numpy as np
from symbolica import E, S  # type: ignore
from symjit import compile_evaluator, load_func

resource.setrlimit(resource.RLIMIT_STACK, (16777216, 2 * 16777216))

print("Building symjit evaluator...")
t_start = time.time()

with open(
    os.path.join(os.path.dirname(__file__), "evaluator_instructions_2loop.txt")
) as fd:
    S = fd.read()

f = compile_evaluator(S, dtype="complex128")
print(f"completed in {time.time() - t_start:.1f} s.")

f.save("2loop.sjb")

rng = np.random.default_rng(1337)
samples_real = rng.random(f.count_params // 2)
samples_imag = rng.random(f.count_params // 2)
samples = samples_real + 1j * samples_imag

N_SAMPLES = 1000

print("Running symjit evaluator...")

t_start = time.time()
for _ in range(N_SAMPLES):
    f.evaluate_complex(samples[None, :])

print(f"Symjit evaluation: {((time.time() - t_start) * 1000.0 / N_SAMPLES):.3f} ms")

print(f.evaluate_complex(samples[None, :]).sum())

g = load_func("2loop.sjb")

print(g.evaluate_complex(samples[None, :]).sum())
