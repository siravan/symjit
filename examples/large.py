from sympy.plotting.series import lambdify
import util
args = util.process_argv()

from sympy import symbols, sin
from symjit import compile_func
from numpy import random
import time
import math

n = 14
N = 2**n

V = symbols(f'v[0:{N}]')
X = random.randn(N)

def tree(level, k):
    if level == 0:
        return V[k], X[k]
    else:
        T_x, x = tree(level-1, k)
        T_y, y = tree(level-1, k+2**(level-1))
        return sin(T_x + T_y), math.sin(x + y)


T, t = tree(n, 0)

if n < 8:
    print(T)

t0 = time.perf_counter_ns()
f = compile_func(V, T)
t1  = time.perf_counter_ns()
print(f'compile_func\t in {1e-6*(t1-t0):.3f} ms')

y_f = 0
y_t = 0

t0 = time.perf_counter_ns()
for _ in range(1000):
    y_f += f(*X)
    y_t += t
t1 = time.perf_counter_ns()

print(f'symjit\t{y_f} in {1e-8*(t1-t0):.3f} ms')
print(f'value\t{y_t}')
# print(f.dumps())
