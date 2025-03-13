import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from symjit import compile_func

x, y, a, b = symbols("x y a b")

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
X = np.zeros_like(A)
Y = np.zeros_like(A)

f = compile_func([a, b, x, y], [x**2 - y**2 + a, 2*x*y + b])

for i in range(20):
    X, Y = f(A, B, X, Y)
 
Z = np.hypot(X, Y)    

plt.imshow(Z < 2)
plt.show()
