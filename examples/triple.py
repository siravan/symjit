import util

args = util.process_argv()

import time
import math
import numpy as np
from scipy.integrate import tplquad
from sympy import symbols, cos
from symjit import compile_func

x, y, z = symbols("x y z")
f = compile_func([x, y, z], 1 / (1 - cos(x) * cos(y) * cos(z)), **args)

t0 = time.perf_counter_ns()
u = tplquad(lambda x, y, z: f(x, y, z), 0, math.pi, 0, math.pi, 0, math.pi)
t1 = time.perf_counter_ns()

print(f"{u[0]}\tlambda:\t\tdone in {(t1 - t0) * 1e-6:.1f} ms")

h = f.callable_quad(use_fast=False)
t0 = time.perf_counter_ns()
u = tplquad(h, 0, math.pi, 0, math.pi, 0, math.pi)
t1 = time.perf_counter_ns()

print(f"{u[0]}\tcallable:\tdone in {(t1 - t0) * 1e-6:.1f} ms")

h = f.callable_quad()
print(f.dumps('fast'))
t0 = time.perf_counter_ns()
u = tplquad(h, 0, math.pi, 0, math.pi, 0, math.pi)
t1 = time.perf_counter_ns()

print(f"{u[0]}\tcallable fast:\tdone in {(t1 - t0) * 1e-6:.1f} ms")
