import util

args = util.process_argv()

import numpy as np
from symjit import *
from sympy import *

x = symbols("x[0:10]")
X = np.arange(10)

for i in range(1, 11):
    f = compile_func(x[:i], sum(x[:i]), **args)
    b = f.fast_func() is not None
    if b:
        s = f.fast_func()(*X[:i])
    else:
        s = f(*X[:i])
    print(i, s, "Fast" if b else "Regular")
    assert s == sum(X[:i])
