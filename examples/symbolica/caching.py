import numpy as np
from symbolica import E, Expression, S
from symjit import compile_evaluator

x, y = S("x"), S("y")
ev = E("x + 3*y").evaluator({}, {}, [x, y])
f = compile_evaluator(ev)

xs = [np.random.rand(1, 2) + np.random.rand(1, 2) * 1j for i in range(100)]

print(f.evaluate_complex(xs[10]))

f.start_caching()
for x in xs:
    f.evaluate_complex(x)
f.stop_caching()

print(f.evaluate_complex(xs[10]))

f.start_caching()
for x in xs:
    f.evaluate(x.real)
f.stop_caching()

print(f.evaluate(xs[10].real))
