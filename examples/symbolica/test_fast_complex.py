import os

import numpy as np
from symbolica import E, Expression, S
from symjit import compile_evaluator

file = os.path.join(os.path.dirname(__file__), "1loop_instructions.txt")

with open(file) as fd:
    one_loop_instructions = fd.read()

f_normal = compile_evaluator(
    one_loop_instructions,
    dtype="complex128",
    use_simd=False,
    use_threads=False,
    fast_complex=False,
)

f_fast = compile_evaluator(
    one_loop_instructions,
    dtype="complex128",
    use_simd=False,
    use_threads=False,
    fast_complex=True,
)

count_params = f_normal.complex_compiler.count_params // 2

N = 10007

for i in range(1):
    # X = np.random.rand(N, count_params) + np.random.rand(N, count_params) * 1j
    X = np.random.rand(1, count_params) + np.random.rand(1, count_params) * 1j
    Y_normal = f_normal.evaluate_complex(X)
    Y_fast = f_fast.evaluate_complex(X)
    print(i, Y_normal, Y_fast)
