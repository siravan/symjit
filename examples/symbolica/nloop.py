import os
import random
import sys
import time

import numpy as np
from symbolica import E, S  # type: ignore
from symjit import compile_evaluator, load_func

if len(sys.argv) < 2:
    print("use nloop n; where n=1, 2, or 3")

n = int(sys.argv[1])

INPUT = os.path.join(os.path.dirname(__file__), f"cff_evaluator_inputs_{n}loop.py")
print(f"Running example from {INPUT}")
exec(open(INPUT, "r").read())

print("Building symbolica evaluator...")
t_start = time.time()
evaluator = expression.evaluator(
    constants=constants,
    functions=functions,  # type: ignore
    params=input_params,
    iterations=100,
    n_cores=8,
    external_functions=None,
    conditionals=conditionals,
    cpe_iterations=10,
    verbose=True,
)
print(f"completed in {time.time() - t_start:.1f} s.")

n = len(input_params)
# evaluator.set_real_params(random.sample(range(n), n // 2))

print("Building symjit evaluator...")
t_start = time.time()
symjit_f = compile_evaluator(evaluator, dtype="complex128")
print(f"completed in {time.time() - t_start:.1f} s.")

print("Compiling symbolica evaluator...")
t_start = time.time()
evaluator.compile("test", "./test.cpp", "./test.so", "complex", inline_asm="default")
print(f"completed in {time.time() - t_start:.1f} s.")

print(symjit_f.dumps("bytecode"))
symjit_f.save(f"loop.sjb")

N_SAMPLES = 1000

rng = np.random.default_rng(1337)
samples_real = rng.random(n)
samples_imag = rng.random(n)
samples = samples_real + 1j * samples_imag

t_start = time.time()
for _ in range(N_SAMPLES):
    evaluator.evaluate_complex(samples[None, :])

print(f"Symbolica evaluation: {((time.time() - t_start) * 1000.0 / N_SAMPLES):.3f} ms")
print(evaluator.evaluate_complex(samples[None, :]).sum())

t_start = time.time()
for _ in range(N_SAMPLES):
    symjit_f.evaluate_complex(samples[None, :])
print(f"Symjit evaluation: {((time.time() - t_start) * 1000.0 / N_SAMPLES):.3f} ms")
print(symjit_f.evaluate_complex(samples[None, :]).sum())

g = load_func(f"loop.sjb")

print(g.evaluate_complex(samples[None, :]).sum())
# os.remove(f"loop.sjb")
