from sympy import *
from symjit import compile_func

x = symbols('x')

x0 = 0.0001

print(f"depth\trust\t\tpython\t\tlambdify")

for i in range(12):
    e = x**2 + x
    
    for _ in range(i):
        e = e**2 + e

    ed = e.diff(x)

    fr = compile_func([x], [ed], backend='rust')
    fp = compile_func([x], [ed], backend='python')
    fl = lambdify([x], ed)
    
    print(f"{i}\t{fr(x0)[0]:.12f}\t{fp(x0)[0]:.12f}\t{fl(x0):.12f}")

