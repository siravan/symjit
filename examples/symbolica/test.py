import numpy as np
from symbolica import *
from symjit import compile_evaluator

x, y = S("x"), S("y")
e = E("x + y^2").evaluator({}, {}, [x, y])
f = compile_evaluator(e)

X = np.array([[4.0, 10.0]])
assert e.evaluate(X) == f.evaluate(X)

print("Test succeeded")
