import numpy as np
from sympy import symbols, lambdify, sin, cos, exp, log, Min, Abs
from symjit import compile_func

x, y = symbols("x y")


def test(p):
    f = compile_func([x, y], p, **args)
    g = lambdify([x, y], p)
    np.testing.assert_array_almost_equal(f(1, 2), g(1, 2))
    np.testing.assert_array_almost_equal(f(1.0, 2.0), g(1.0, 2.0))
    u = np.random.rand(3)
    v = np.random.rand(3)
    F = f(u, v)
    G = g(u, v)
    np.testing.assert_array_almost_equal(F, G)

for dtype in ["float64", "complex128"]:
    for use_simd in [False, True]:
        for enable_simd512 in [False, True]:
            for fast_complex in [False, True]:
                for fastmath in [False, True]:
                    for opt_level in [0, 1, 2, 3]:
                        args = {"dtype": dtype,  "use_simd": use_simd, "enable_simd512": enable_simd512, "fastmath": fastmath, "fast_complex": fast_complex, "opt_level": opt_level}
                        print(args)
                        test([sin(y), x * sin(y)])  # fuse_save3 bug
                        test([sin(y), sin(y)+cos(y)*sin(y)])
                        test([exp(x), exp(x)/(1+y)])
                        test([-(x-y)**3 + log(Abs(x))])
                        # test([-(-y**3 - y + x)**3 + log(Min(0,-x)**2 + 1)])
                        test([1.135 - y**3])
                        test([cos(1/x) and sin(1/(y**2+1))])

print("ok!")
