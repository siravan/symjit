import sys
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from symjit import compile_func

backend = "python" if len(sys.argv) > 1 and sys.argv[1] == "py" else "rust"

x, sigma = symbols("x sigma")
f = compile_func(
    [x], [exp(-((x - 100) ** 2) / (2 * sigma**2))], params=[sigma], backend=backend
)

t = np.arange(0, 200)
y = f(t, 25.0)[0]

plt.plot(t, y)
plt.show()
