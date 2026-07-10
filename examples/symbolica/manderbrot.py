import matplotlib.pyplot as plt
import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

x, y, c = S("x"), S("y"), S("c")

z = E("c")
for _ in range(20):
    z = z**2 + c

z = z.abs()

ev = z.evaluator([c])
f = compile_evaluator(ev, dtype="complex128", use_simd=True, direct=False)

# print(ev.get_instructions())
# print(f.dumps("bytecode"))

A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
C = (A + B * 1j).reshape((-1, 1))

Y = f.evaluate_complex(C).reshape(A.shape)

print(Y.shape)

plt.imshow(Y.real < 4)
plt.show()
