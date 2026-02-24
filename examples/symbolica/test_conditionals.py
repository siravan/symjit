import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

ev = E("if(y, x + 1, x + 2)").evaluator(
    {}, {}, [S("x"), S("y")], conditionals=[S("if")]
)

f_with_simd = compile_evaluator(ev, use_threads=False, ty="bytecode")

print(f_with_simd.dumps())

X = np.random.rand(1000, 2)
X[:, 1] = X[:, 1] > 0.8

Y = ev.evaluate(X)
Y_with_simd = f_with_simd.evaluate(X)

assert (Y_with_simd == Y).all()

print("passed")
