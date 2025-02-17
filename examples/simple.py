from sympy import symbols
import numpy as np
import symjit

x, y = symbols("x y")
f = symjit.compile_func([x, y], [x + y, x * y])
assert np.all(f(3, 4) == [7.0, 12.0])
print("ok!")
