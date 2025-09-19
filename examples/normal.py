import util

args = util.process_argv()

from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from symjit import compile_func

x, sigma = symbols("x sigma")
f = compile_func([x], [exp(-((x - 100) ** 2) / (2 * sigma**2))], params=[sigma], **args)

t = np.arange(0, 200)
y = f(t, 25.0)[0]

plt.plot(t, y)

if __name__ == "__main__":
    plt.show()
