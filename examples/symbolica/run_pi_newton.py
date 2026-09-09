import os
import math
import numpy as np
from symjit import load_func


t = [[0.123456 + 0.5j, math.sqrt(2) / 2]]

SJB = os.path.join(os.path.dirname(__file__), "pi_newton.sjb")

f = load_func(SJB)

p = f.evaluate_complex(t)
print(f"symjit = {np.real(p[0][0])}")
