import sys
import numpy as np
from sympy import symbols
from symjit import compile_func

backend = "python" if len(sys.argv) > 1 and sys.argv[1] == "py" else "rust"

x, y = symbols("x y")
f = compile_func([x, y], [x + y, x * y], backend=backend)
assert np.all(f(3, 4) == [7.0, 12.0])
print("ok!")
