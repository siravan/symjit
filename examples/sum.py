from sympy import symbols, sin
from symjit import compile_func

N = 100

X = symbols(f'x[0:{N}]')

p = 0

for i in range(N):    
    p += X[i]
    
print(p)    

f = compile_func(list(X), p)

print(f(*range(0, N))[0])

