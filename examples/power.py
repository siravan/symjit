import math
from sympy import symbols, sin
from symjit import compile_func

# calculating pi using Machine formula

N = 100

x = symbols('x')

p = 0
q = 0
x0 = math.exp(math.log(N) / N)

for i in range(-N, N+1):    
    p += sin(1 + x**i)**2
    q += math.sin(1 + x0**i)**2
    
print(p)    

f = compile_func([x], p)

print('symjit:\t', f(x0))
print('evalf:\t', p.evalf(subs = {x: x0}))
print('python:\t', q)

