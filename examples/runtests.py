import time
import numpy as np
from scipy import integrate
import math
import platform
from sympy import symbols, lambdify, sqrt, sin, cos, Piecewise
from symjit import compile_func
from random import randint

L = 1000

def arch():
    if platform.machine() in ["x86_64", "AMD64"]:
        return "amd"
    elif platform.machine() in ["arm64", "aarch64"]:
        return "arm"
    else:
        return None

x, y, z, a, b = symbols("x y z a b")


def func(states, p, args):
    t0 = time.perf_counter_ns()

    if args['backend'] == 'sympy':
        f = lambdify(states, p)
    elif args['backend'] == 'python':
        f = compile_func(states, p, backend='python')
    else:
        f = compile_func(states, p, **args)

    t1 = time.perf_counter_ns()

    print(f'compile in {(t1-t0)*1e-6:.1f} ms\t', end='')
    return f


def mandelbrot(args):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
    f = func([a, b, x, y], [x**2 - y**2 + a, 2 * x * y + b], args)
    X = np.zeros_like(A)
    Y = np.zeros_like(A)

    t0 = time.perf_counter_ns()

    for i in range(5):
        X, Y = f(A, B, X, Y)

    t1 = time.perf_counter_ns()

    return X + Y, t1 - t0


def mandelbrot2(args):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))

    def quad_map(x, y, a, b):
        return (x**2 - y**2 + a, 2 * x * y + b)

    X = 0
    Y = 0

    for i in range(5):
        X, Y = quad_map(X, Y, a, b)

    f = func([a, b], [X, Y], args)

    t0 = time.perf_counter_ns()
    X, Y = f(A, B)
    t1 = time.perf_counter_ns()

    return X + Y, t1 - t0


def mandelbrot3(args):
    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
    f = func([x, y, a, b], [x**2 - y**2 + a, 2 * x * y + b], args)

    t0 = time.perf_counter_ns()

    if args['backend'] == 'sympy':
        X = np.zeros_like(A)
        Y = np.zeros_like(A)
        for i in range(5):
            X, Y = f(X, Y, A, BytesPath)
    else:
        n = A.shape[0] * A.shape[1]
        buf = np.zeros((4, n), dtype="double")
        buf[2, :] = A.ravel()
        buf[3, :] = B.ravel()
        for i in range(5):
            f.execute_vectorized(buf)
        X = np.reshape(buf[0,:], A.shape)
        Y = np.reshape(buf[1,:], A.shape)

    t1 = time.perf_counter_ns()

    return X + Y, t1 - t0


def pi(args):
    N = 25

    def arctan_series(x):
        s = x
        for i in range(1, N):
            coef = -(1 + 2 * i) if (i & 1 == 1) else 1 + 2 * i
            s += x**abs(coef) / coef
        return s

    p = 4 * (4 * arctan_series(x) - arctan_series(y))
    f = func([x, y], p, args)

    t0 = time.perf_counter_ns()
    u = [f(1/5, 1/239) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def viete(args):
    p = 1

    for i in range(21):
        t = x
        for j in range(i):
            t = x + x * sqrt(t)
        p *= sqrt(t)

    f = func([x], [2 / p], args)

    t0 = time.perf_counter_ns()
    u = [f(1 / 2) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def lemniscate(args):
    p = 1

    for i in range(21):
        t = x
        for j in range(i):
            t = x + x / sqrt(t)
        p *= sqrt(t)

    f = func([x], [2 / p], args)

    t0 = time.perf_counter_ns()
    u = [f(1 / 2) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def binom(args):
    N = 15
    K = 8

    def binom(x, y, n, k):
        if k == 0 or k == n:
            return 1.0
        else:
            return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

    f = func([x, y], binom(x, y, N, K), args)

    t0 = time.perf_counter_ns()
    u = [f(1, 1) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def stress(args):
    e = x**2 + x

    for _ in range(15):
        e = e**2 + e
        ed = e.diff(x)

    f = func([x], [ed], args)

    t0 = time.perf_counter_ns()
    u = [f(0.001) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def power(args):
    N = 150

    p = 0
    x0 = math.exp(math.log(N) / N)

    for i in range(-N, N+1):
        p += sin(1 + x**i)**2

    f = func([x], [p], args)

    t0 = time.perf_counter_ns()
    u = [f(x0) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def powi_mod(args):
    def binom(x, y, n, k):
        if k == 0 or k == n:
            return 1.0
        else:
            return binom(x, y, n - 1, k) * x + binom(x, y, n - 1, k - 1) * y

    p = binom(x, y, 7, 4)**5 % 65537 + binom(x, y, 8, 5)**(4**x) % 65537

    f = func([x, y], [p], args)

    t0 = time.perf_counter_ns()
    u = [f(1, 1) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def fact(args):
    def factorial(x, n):
        if n == 0:
            return 1
        else:
            return Piecewise([n, x >= n], [1, True]) * factorial(x, n-1)

    p = factorial(x, 20)
    f = func([x], [p], args)

    t0 = time.perf_counter_ns()
    u = [f(18) for _ in range(L)]
    t1 = time.perf_counter_ns()

    return u[randint(0, L-1)], t1 - t0


def triple(args):
    p = 1 / (1 - cos(x) * cos(y) * cos(z))
    f = func([x, y, z], p, args)

    t0 = time.perf_counter_ns()
    u = integrate.tplquad(lambda x, y, z: f(x, y, z), 0, math.pi, 0, math.pi, 0, math.pi)[0]
    t1 = time.perf_counter_ns()

    return u, t1 - t0


def triple_fast(args):
    p = 1 / (1 - cos(x) * cos(y) * cos(z))

    f = func([x, y, z], p, args)

    if hasattr(f, 'fast_func'):
        f = f.fast_func()

    t0 = time.perf_counter_ns()
    u = integrate.tplquad(lambda x, y, z: f(x, y, z), 0, math.pi, 0, math.pi, 0, math.pi)[0]
    t1 = time.perf_counter_ns()

    return u, t1 - t0


#############################################################################

def test_model(f, label, pyback=True, bytecode=True):
    print(f'testing {label}')

    args = {'backend': 'rust', 'ty': 'native', 'use_simd': True, 'use_threads': True, 'cse': True}

    print('\tlambdify...\t', end='')
    args['backend'] = 'sympy'
    X0, dt = f(args)
    print(f'\tdone in {1e-6 * dt:.3f} ms')

    args['backend'] = 'rust'
    args['ty'] = arch()

    print('\trust backend...\t', end='')
    X, dt = f(args)
    np.testing.assert_array_almost_equal(X0, X)
    print(f'\tpass in {1e-6 * dt:.3f} ms')

    print('\tno CSE...\t', end='')
    args['cse'] = False
    X, dt = f(args)
    args['cse'] = True
    np.testing.assert_array_almost_equal(X0, X)
    print(f'\tpass in {1e-6 * dt:.3f} ms')

    print('\tno threads...\t', end='')
    args['use_threads'] = False
    X, dt = f(args)
    args['use_threads'] = True
    np.testing.assert_array_almost_equal(X0, X)
    print(f'\tpass in {1e-6 * dt:.3f} ms')

    if args['ty'] == 'amd':
        print('\tno simd...\t', end='')
        args['use_simd'] = False
        X, dt = f(args)
        args['use_simd'] = True
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1e-6 * dt:.3f} ms')

        print('\tamd-sse...\t', end='')
        args['ty'] = 'amd-sse'
        X, dt = f(args)
        args['ty'] = 'amd'
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1e-6 * dt:.3f} ms')

    if bytecode:
        print('\tbytecode...\t', end='')
        args['ty'] = 'bytecode'
        X, dt = f(args)
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1e-6 * dt:.3f} ms')

        print('\tdebug mode...\t', end='')
        args['ty'] = 'debug'
        X, dt = f(args)
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1e-6 * dt:.3f} ms')

    if pyback:
        print('\tpython backend...', end='')
        args['backend'] = 'python'
        X, dt = f(args)
        np.testing.assert_array_almost_equal(X0, X)
        print(f'\tpass in {1e-6 * dt:.3f} ms')


test_model(mandelbrot, 'mandelbrot')
test_model(mandelbrot2, 'mandelbrot2')
test_model(mandelbrot2, 'mandelbrot3')
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
