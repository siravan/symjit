import os

import numpy as np
from symbolica import E, Expression, S
from symjit import compile_evaluator

file = os.path.join(os.path.dirname(__file__), "1loop_instructions.txt")

with open(file) as fd:
    one_loop_instructions = fd.read()

f_without_simd = compile_evaluator(
    one_loop_instructions, dtype="complex128", use_simd=False, use_threads=False
)

count_params = f_without_simd.complex_compiler.count_params // 2

X = (
    -100000 * np.random.rand(1000, count_params)
    + np.random.rand(1000, count_params) * 1j
)

Y_without_simd = f_without_simd.evaluate_complex(X)

for simd_branch in [False, True]:
    print(f"simd_branch = {simd_branch}...", end="")
    f_with_simd = compile_evaluator(
        one_loop_instructions, dtype="complex128", use_threads=False, simd_branch=True
    )

    Y_with_simd = f_with_simd.evaluate_complex(X)
    assert (Y_with_simd == Y_without_simd).all()
    print("passed")
