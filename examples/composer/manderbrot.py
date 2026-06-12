import matplotlib.pyplot as plt
import numpy as np
from symjit import Composer, compile_composer


def manderbrot():
    cp = Composer(1, 1)
    z = cp.new_temp()
    cp.assign(z, cp.constant(0))
    c = cp.arg(0)

    bl = cp.new_block()
    bl.assign(z, bl.fadd(bl.square(z), c))
    cp.append_for(cp.new_temp(), 1, 20, bl)

    cp.assign(z, cp.abs(z))
    t = cp.join(cp.lt(z, cp.constant(4.0)), cp.sqrt(z), cp.constant(0))
    cp.assign(cp.out(0), t)

    f = compile_composer(cp, dtype="complex128", use_simd=True)
    return f


A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
C = (A + B * 1j).reshape((-1, 1))

f = manderbrot()

Y = f.evaluate_complex(C).reshape(A.shape)

print(Y.shape)

plt.imshow(Y.real)
plt.show()
