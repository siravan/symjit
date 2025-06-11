import time
import numpy as np
from scipy import integrate
import math
import platform
from sympy import symbols, lambdify, sqrt, sin, cos, Piecewise
from symjit import compile_func

def arch():
    if platform.machine() in ["x86_64", "AMD64"]:
        return "amd"
    elif platform.machine() in ["arm64", "aarch64"]:
        return "arm"
    else:
        return None

x, y, z, a, b = symbols("x y z a b")


def func(states, p, backend, ty, use_simd):
    if backend == 'sympy':
        return lambdify(states, p)
    else:        
        return compile_func(states, p, backend=backend, ty=ty, use_simd=use_simd)


def mandelbrot(backend, ty, use_simd):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
    f = func([a, b, x, y], [x**2 - y**2 + a, 2 * x * y + b], backend=backend, ty=ty, use_simd=use_simd)    
    X = np.zeros_like(A)
    Y = np.zeros_like(A)

    for i in range(5):
        X, Y = f(A, B, X, Y)
        
    return X + Y        
    
    
def mandelbrot2(backend, ty, use_simd):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
 
    def quad_map(x, y, a, b):
        return (x**2 - y**2 + a, 2 * x * y + b)

    X = 0
    Y = 0

    for i in range(5):
        X, Y = quad_map(X, Y, a, b)        
      
    f = func([a, b], [X, Y], backend=backend, ty=ty, use_simd=use_simd)
    X, Y = f(A, B)    
    return X + Y   


def pi(backend, ty, use_simd):
    N = 25

    def arctan_series(x):
        s = x
        for i in range(1, N):
            coef = -(1 + 2 * i) if (i & 1 == 1) else 1 + 2 * i
            s += x**abs(coef) / coef
        return s    
    
    p = 4 * (4 * arctan_series(x) - arctan_series(y))        
    f = func([x, y], p, backend=backend, ty=ty, use_simd=use_simd)             
    return f(1/5, 1/239)


def viete(backend, ty, use_simd):
    p = 1

    for i in range(21):
        t = x
        for j in range(i):
            t = x + x * sqrt(t)
        p *= sqrt(t) 
    
    f = func([x], [2 / p], backend=backend, ty=ty, use_simd=use_simd)             
    return f(1 / 2)


def lemniscate(backend, ty, use_simd):
    p = 1

    for i in range(21):
        t = x
        for j in range(i):
            t = x + x / sqrt(t)
        p *= sqrt(t) 
    
    f = func([x], [2 / p], backend=backend, ty=ty, use_simd=use_simd)             
    return f(1 / 2)


def binom(backend, ty, use_simd):
    N = 12
    K = 7

    def binom(x, y, n, k):    
        if k == 0 or k == n:
            return 1.0
        else:
            return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

    f = func([x, y], binom(x, y, N, K), backend=backend, ty=ty, use_simd=use_simd)                     
    return f(1, 1)


def stress(backend, ty, use_simd):
    e = x**2 + x
        
    for _ in range(i):
        e = e**2 + e
        ed = e.diff(x)

    f = func([x], [ed], backend=backend, ty=ty, use_simd=use_simd)                     
    return f(0.001)
    
    
def power(backend, ty, use_simd):    
    N = 150

    p = 0
    x0 = math.exp(math.log(N) / N)

    for i in range(-N, N+1):    
        p += sin(1 + x**i)**2

    f = func([x], [p], backend=backend, ty=ty, use_simd=use_simd)        
    return f(x0)            
    

def powi_mod(backend, ty, use_simd):
    def binom(x, y, n, k):    
        if k == 0 or k == n:
            return 1.0
        else:
            return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

    p = binom(x, y, 7, 4)**5 % 65537 + binom(x, y, 8, 5)**(4**x) % 65537
    
    f = func([x, y], [p], backend=backend, ty=ty, use_simd=use_simd)
    return f(1, 1)
    

def fact(backend, ty, use_simd):
    def factorial(x, n):
        if n == 0:
            return 1
        else:
            return Piecewise([n, x >= n], [1, True]) * factorial(x, n-1)

    p = factorial(x, 20)
    f = func([x], [p], backend=backend, ty=ty, use_simd=use_simd)
    return f(18)  
    

def triple(backend, ty, use_simd):
    p = 1 / (1 - cos(x) * cos(y) * cos(z))    
    f = func([x, y, z], p, backend=backend, ty=ty, use_simd=use_simd)             
    return integrate.tplquad(lambda x, y, z: f(x, y, z), 0, math.pi, 0, math.pi, 0, math.pi)[0] 
    
    
def triple_fast(backend, ty, use_simd):
    p = 1 / (1 - cos(x) * cos(y) * cos(z))        

    if backend == 'sympy':
        f = lambdify([x, y, z], p)
    else:        
        func = compile_func([x, y, z], p, backend=backend, ty=ty, use_simd=use_simd)        
        f = func.fast_func()
    
    return integrate.tplquad(lambda x, y, z: f(x, y, z), 0, math.pi, 0, math.pi, 0, math.pi)[0] 
        

#############################################################################

def test_model(f, label, pyback=True, bytecode=True):
    print(f'testing {label}')

    print('\ttesting sympy lambdify...\t\t', end='')
    t0 = time.time()
    X0 = f('sympy', None, False)
    t1 = time.time()
    print(f'\tdone in {1000 * (t1 - t0):.1f} ms')

    ty = arch()

    print('\ttesting rust backend for amd without simd...', end='')
    t0 = time.time()
    X = f('rust', ty, False)
    t1 = time.time()
    np.testing.assert_array_almost_equal(X0, X)
    print(f'\tpass in {1000 * (t1 - t0):.1f} ms')
    
    if ty == 'amd':
        print('\ttesting rust backend for amd-avx with simd...', end='')
        t0 = time.time()
        X = f('rust', 'amd-avx', True)
        t1 = time.time()
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1000 * (t1 - t0):.1f} ms')
        
        print('\ttesting rust backend for amd-sse...\t', end='')
        t0 = time.time()
        X = f('rust', 'amd-sse', False)
        t1 = time.time()
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1000 * (t1 - t0):.1f} ms')

    if bytecode:
        print('\ttesting rust backend with bytecode...\t', end='')
        t0 = time.time()
        X = f('rust', 'bytecode', False)
        t1 = time.time()
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1000 * (t1 - t0):.1f} ms')

    if pyback:
        print('\ttesting python backend...\t\t', end='')
        t0 = time.time()
        X = f('python', ty, False)   
        t1 = time.time()
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1000 * (t1 - t0):.1f} ms')
    

test_model(mandelbrot, 'mandelbrot')    
test_model(mandelbrot2, 'mandelbrot2')    
test_model(pi, 'pi')    
test_model(viete, 'pi-viete')
test_model(lemniscate, 'lemniscate')
test_model(binom, 'binom')
test_model(binom, 'stress')
test_model(power, 'power')
test_model(powi_mod, 'powi_mod', pyback=False)
test_model(fact, 'fact')
test_model(triple, 'triple')
test_model(triple_fast, 'triple_fast', pyback=False, bytecode=False)
    
        
        
        

