import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

ev = E("if(q, 2*exp(x), exp(x))").evaluator([S("q"), S("x")])
f = compile_evaluator(ev, cse = True)

assert f.evaluate([[1.0, 2.0]]) == np.exp(2.0) * 2
assert f.evaluate([[0.0, 2.0]]) == np.exp(2.0)

assert f.evaluate_complex([[1.0, 2. + 1j]]) == np.exp(2 + 1j) * 2
assert f.evaluate_complex([[0.0, 2.0 + 1j]]) == np.exp(2 + 1j)
