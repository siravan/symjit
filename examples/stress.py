import util

args = util.process_argv()

from sympy import *
from symjit import compile_func
import time

x = symbols("x")

if args["dtype"] == "complex128":
    x0 = 0.0001 + 0.0001j
else:
    x0 = 0.0001

print("depth\tlambdify\trust\t\tdt")

for i in range(14):
    e = x**2 + x

    for _ in range(i):
        e = e**2 + e

    ed = e.diff(x)

    fr = compile_func([x], ed, **args)
    fl = lambdify([x], ed)

    t0 = time.perf_counter_ns()
    for _ in range(1000):
        r = fr(x0)
    t1 = time.perf_counter_ns()

    for _ in range(1000):
        l = fl(x0)

    print(f"{i}\t{l:.12f}\t{r:.12f}\t{1e-6 * (t1 - t0):.3f}")
