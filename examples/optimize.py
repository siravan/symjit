from sympy import symbols, exp, log
from symjit import compile_func
from sympy.codegen.rewriting import optimize, optims_c99

x = symbols('x')
y = optimize((3*exp(2*x) - 3)*log(3*x+3)*2**x + log(x)/log(2), optims_c99)
print(y)
f = compile_func([x], y)
assert(f(2.1234) == y.evalf(subs={x: 2.1234}))

print('ok')
