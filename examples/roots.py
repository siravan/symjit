import numpy as np

from symjit import compile_func
from sympy import symbols, sqrt

x = symbols('x')

def test(X, use_simd, fast_complex):
    f = compile_func([x], [sqrt(1 + sqrt(1+sqrt(1+sqrt(x))))], use_simd=use_simd, fast_complex=fast_complex, dtype='complex128')
    Y = f(X)[0]
    np.testing.assert_array_almost_equal(Y, np.sqrt(1+np.sqrt(1+np.sqrt(1+np.sqrt(X)))))
    return f

N = 10007
X = np.random.rand(N) + 1j * np.random.rand(N)

test(X, use_simd=False, fast_complex=False)
test(X, use_simd=False, fast_complex=True)
test(X, use_simd=True, fast_complex=False)
f = test(X, use_simd=True, fast_complex=True)

# print(f.dumps('simd'))

print('ok')
