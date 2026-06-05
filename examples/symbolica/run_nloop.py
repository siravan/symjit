import os
import sys
import time

import numpy as np
from symjit import compile_evaluator, load_func

if len(sys.argv) < 2:
    print("use nloop n; where n=1, 2, or 3")

CONFIG = os.path.join(os.path.dirname(__file__), "symjit.toml")
INSTRUCTIONS = os.path.join(
    os.path.dirname(__file__), f"{sys.argv[1]}loop_instructions_2.txt"
)

with open(INSTRUCTIONS, "r") as fd:
    evaluator = fd.read()

print("Building symjit evaluator...")
t_start = time.time()
symjit_f = compile_evaluator(evaluator, dtype="complex128", ty=CONFIG)
print(f"completed in {time.time() - t_start:.1f} s.")

n = symjit_f.complex_compiler.count_params // 2

symjit_f.save(f"loop.sjb")

N_SAMPLES = 1000

rng = np.random.default_rng(1337)
samples_real = rng.random(n)
samples_imag = rng.random(n)
samples = samples_real + 1j * samples_imag

t_start = time.time()
for _ in range(N_SAMPLES):
    symjit_f.evaluate_complex(samples[None, :])
print(f"Symjit evaluation: {((time.time() - t_start) * 1000.0 / N_SAMPLES):.3f} ms")
print(symjit_f.evaluate_complex(samples[None, :]).sum())

g = load_func(f"loop.sjb")

print(g.evaluate_complex(samples[None, :]).sum())
os.remove(f"loop.sjb")
