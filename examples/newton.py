import util

args = util.process_argv()

from sympy import symbols, I, re, im, diff
from sympy.polys.specialpolys import swinnerton_dyer_poly
from symjit import compile_func
from random import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

z = symbols("z")
x, y = symbols("x y", real=True)

# p = random_poly(z, 2, -10, 10)
p = swinnerton_dyer_poly(4, z)
# print(p)

g = compile_func([z], [p], **args)

# for i in range(16):
#     u = (
#         (sqrt(2) if i & 1 != 0 else -sqrt(2))
#         + (sqrt(3) if i & 2 != 0 else -sqrt(3))
#         + (sqrt(5) if i & 4 != 0 else -sqrt(5))
#         + (sqrt(7) if i & 8 != 0 else -sqrt(7))
#     )
#     print(f"analytic root {i} = {u}")

dp = diff(p, z)

p = p.subs({z: x + I * y})
dp = dp.subs({z: x + I * y})

f = compile_func([x, y], [re(p), im(p), re(dp), im(dp)], **args)

x0 = 5 * (random() - 0.5)
y0 = random()

for i in range(10):
    a, b, c, d = f(x0, y0)
    r = c**2 + d**2
    x0 = x0 - (a * c + b * d) / r
    y0 = y0 - (b * c - a * d) / r

print(f"newton root = {x0} + {y0}*im")
a, b, c, d = f(x0, y0)
print(f"residue = {a**2 + b**2}")

X = np.arange(-3, 3, 0.01)
plt.plot(X, np.zeros_like(X), color="red")
plt.plot(X, g(X)[0])
plt.plot([x0], [0], "o", color="red")

if __name__ == "__main__":
    plt.show()
