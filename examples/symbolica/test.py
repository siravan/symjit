from ast import Expression
from builtins import AssertionError

import numpy as np
from symbolica import *
from symjit import compile_evaluator


def assert_verbose(a, b):
    if abs(a - b) > 1e-10:
        raise AssertionError(f"{a} != {b}")


x, y, z = S("x"), S("y"), S("z")

# simple
ev = E("x + y^2").evaluator({}, {}, [x, y])
f = compile_evaluator(ev)

X = np.array([[4.0, 10.0]])
Z = np.array([[4.0 + 2j, 10.0 - 5j]])

assert_verbose(ev.evaluate(X), f.evaluate(X))
assert_verbose(ev.evaluate_complex(X), f.evaluate_complex(X))
assert_verbose(ev.evaluate(Z), f.evaluate(Z))
assert_verbose(ev.evaluate_complex(Z), f.evaluate_complex(Z))

# transcedental
ev = E("x + sqrt(y^2) - sin(x*y+z)").evaluator({}, {}, [x, y, z])
f = compile_evaluator(ev)

X = np.array([[4.0, 10.0, 5.0]])
Z = np.array([[4.0 + 2j, 10.0 - 5j, -1 + 6j]])

assert_verbose(ev.evaluate(X), f.evaluate(X))
assert_verbose(ev.evaluate_complex(X), f.evaluate_complex(X))
assert_verbose(ev.evaluate(Z), f.evaluate(Z))
assert_verbose(ev.evaluate_complex(Z), f.evaluate_complex(Z))

# mixed real/complex
ev = E("x+y+z").evaluator({}, {}, [x, y, z])
f = compile_evaluator(ev, ty="bytecode", dtype="complex128")
# print(f.dumps(dtype="complex128"))

X = np.array([[4.0, 10.0, 5.0]])
Z = np.array([[4.0, 10.0, -1 + 6j]])

assert_verbose(ev.evaluate(X), f.evaluate(X))
assert_verbose(ev.evaluate_complex(Z), f.evaluate_complex(Z))

ev.set_real_params([0, 1])
f = compile_evaluator(ev, ty="bytecode", dtype="complex128")
# print(f.dumps(dtype="complex128"))

assert_verbose(ev.evaluate(X), f.evaluate(X))
assert_verbose(ev.evaluate_complex(Z), f.evaluate_complex(Z))

# real sqrt
ev = E("sqrt(x*y)").evaluator({}, {}, [x, y])
f = compile_evaluator(ev, dtype="complex128")

X = np.array([[4.0, 25.0]])
Z = np.array([[-5 + 12j, 3 - 4j]])

assert_verbose(ev.evaluate(X), f.evaluate(X))
assert_verbose(ev.evaluate_complex(Z), f.evaluate_complex(Z))

ev.set_real_params([0, 1], sqrt_real=True)
f = compile_evaluator(ev, dtype="complex128")

assert_verbose(ev.evaluate(X), f.evaluate(X))
assert_verbose(ev.evaluate_complex(Z), f.evaluate_complex(Z))

# multiple-expression
ev = Expression.evaluator_multiple([E("x+y^2"), E("2*x*y+5")], {}, {}, [x, y])
f = compile_evaluator(ev)

X = np.array([[3.0, 15.0]])

assert (ev.evaluate(X) == f.evaluate(X)).all()

# extra params

ev = E("x+y^2").evaluator({}, {}, [x, y, z])
f = compile_evaluator(ev, num_params=3)

X = np.array([[4.0, 3.0, 2.0]])


print("all tests passed")
