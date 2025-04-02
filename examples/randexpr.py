from sympy import *
from random import random, randint
from symjit import compile_func


def generate_random_expr(d, *xs):
    r = random()
    
    if d <= 0:
        return generate_random_terminal(d-1, *xs)
    elif r < 0.1:
        return generate_random_unary(d-1, *xs)
    else:
        return generate_random_binary(d-1, *xs)

        
def generate_random_terminal(d, *xs):
    r = random()
    
    if r < 0.9:
        return xs[randint(0, len(xs)-1)]
    else:
        return randint(1, 10)
        

def generate_random_unary(d, *xs):
    eq = generate_random_expr(d, *xs)
    
    r = random()
    
    if r < 0.25:
        return eq ** randint(-5, 5)
    elif r < 0.4:
        return 1 / eq
    else:        
        u = [sin, cos, tan, sinh, cosh, tanh, exp, sqrt]
        f = u[randint(0, len(u)-1)]
        return f(eq)

    
def generate_random_binary(d, *xs):    
    r = random()
    
    u = generate_random_expr(d, *xs)
    v = generate_random_expr(d, *xs)
    
    if r < 0.3:
        return u + v
    elif r < 0.6:
        return u * v
    elif r < 0.8:
        return u - v
    else:
        return u / v

############################################################

x, y, z = symbols('x y z')

for i in range(100):
    q = generate_random_expr(5, x, y, z)
    print(q)
    
    try:
        f = compile_func([x, y, z], [q])
        g = lambdify([x, y, z], [q])
        
        X = random()
        Y = random()
        Z = random()
        
        F = f(X, Y, Z)
        G = g(X, Y, Z)
        
        print(float(F[0]) - float(G[0]) < 1e-10)
    except:
        print('Error!')      
    
        
        
