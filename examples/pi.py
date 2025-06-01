import math
from sympy import symbols, lambdify
from symjit import compile_func

# calculating pi using Machine formula

N = 4

def arctan_series(x):
    s = x

    for i in range(1, N):
        coef = -(1 + 2 * i) if (i & 1 == 1) else 1 + 2 * i
        s += x**abs(coef) / coef

    return s
    

x, y = symbols('x y')
p = 4 * (4 * arctan_series(x) - arctan_series(y))
print(p)

f = compile_func([x, y], p, ty='amd-avx')
g = lambdify([x, y], p)

print(f(1/5, 1/239)[0], '?=', g(1/5, 1/239), '; pi = ', math.pi)
# print(f.dumps())
