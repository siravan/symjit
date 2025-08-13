from sympy import *
from symjit import compile_func
import time

x = symbols('x')

x0 = 0.0001

print(f"depth\trust\t\tpython\t\tlambdify")

for i in range(14):
    e = x**2 + x

    for _ in range(i):
        e = e**2 + e

    ed = e.diff(x)

    fr = compile_func([x], [ed], backend='rust', cse=True)
    fp = compile_func([x], [ed], backend='python')
    fl = lambdify([x], ed)

    t0 = time.time()
    y = fr(x0)[0]
    t1 = time.time()

    print(f"{i}\t{y:.12f}\t{fp(x0)[0]:.12f}\t{fl(x0):.12f}\t{1000*(t1-t0)}")
