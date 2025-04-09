import numpy as np
from symjit import compile_func
from sympy import symbols

x, y = symbols("x y")
f = compile_func([x, y], [x + y, x * y])
assert np.all(f(3, 5) == [8.0, 15.0])
print("Test succeeded")
