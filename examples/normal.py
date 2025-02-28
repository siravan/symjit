from sympy import *
import numpy as np
import matplotlib.pyplot as plt

from symjit import compile_func

x, sigma = symbols('x sigma')
f = compile_func([x], [exp(-(x-100)**2/(2*sigma**2))], params=[sigma])

t = np.arange(0, 200)
y = f(t, 25.)[0]

plt.plot(t, y)
plt.show()
