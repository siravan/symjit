import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, expand
from symjit import compile_func

x, y, a, b = symbols("x y a b")

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))

def quad_map(x, y, a, b):
    return (x**2 - y**2 + a, 2*x*y + b)
    
X = 0
Y = 0

for i in range(12):
    X, Y = quad_map(X, Y, a, b)    
    
f = compile_func([a, b], [X, Y])

X, Y = f(A, B)

Z = np.hypot(X, Y)    

plt.imshow(Z < 2)
plt.show()

