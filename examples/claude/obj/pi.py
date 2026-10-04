"""Calculating pi in a C program with kernels compiled by symjit (modeled on examples/pi.py).

The script compiles three functions, writes each as a relocatable object file with a C
header (`write_obj`, in out/), then builds test_pi.c with them and runs it:

    pi_series(x, y)  Machin's formula, 4 (4 arctan(x) - arctan(y)), with arctan as its
                     Taylor series (pure arithmetic: no library calls)
    pi_atan(x, y)    Machin's formula with atan from the C math library (link with -lm)
    pi_bbp(n)        the Bailey-Borwein-Plouffe series up to k = n (a loop in the kernel)

test_pi.c checks the results against the known value of pi.

Needs a symjit library built with the `obj` cargo feature (cargo build --release
--features obj) on x86-64 Linux or macOS, and a C compiler (`cc`, or $CC).
"""

import math
import os
import shutil
import subprocess
import sys

from sympy import Sum, atan, symbols

from symjit import compile_func

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# calculating pi using Machin's formula (as in examples/pi.py)

N = 21


def arctan_series(x):
    s = x

    for i in range(1, N):
        coef = -(1 + 2 * i) if (i & 1 == 1) else 1 + 2 * i
        s += x ** abs(coef) / coef

    return s


x, y, n, k = symbols("x y n k")

functions = {
    "pi_series": compile_func([x, y], [4 * (4 * arctan_series(x) - arctan_series(y))]),
    "pi_atan": compile_func([x, y], [4 * (4 * atan(x) - atan(y))]),
    "pi_bbp": compile_func([n], [Sum((4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)) / 16**k, (k, 0, n))]),
}

# the same functions, called from Python
print("from Python:")
for name, f in functions.items():
    v = f(12.0)[0] if name == "pi_bbp" else f(1 / 5, 1 / 239)[0]
    print(f"  {name:10s} {v!r}  (pi = {math.pi!r})")

os.makedirs(OUT, exist_ok=True)
try:
    for name, f in functions.items():
        f.write_obj(os.path.join(OUT, name))
except ValueError as e:
    print(f"cannot write object files: {e}")
    sys.exit(1)
print(f"\nwrote {', '.join(f'out/{name}.o' for name in functions)} and their headers")

cc = os.environ.get("CC", "cc")
if shutil.which(cc) is None:
    print(f"no C compiler ({cc}): the object files are written but not linked")
    sys.exit()

exe = os.path.join(OUT, "test_pi")
objects = [os.path.join(OUT, f"{name}.o") for name in functions]
cmd = [cc, "-Wall", "-O2", "-I", OUT, os.path.join(HERE, "test_pi.c"), *objects, "-lm", "-o", exe]
print(" ".join(os.path.relpath(c, HERE) if os.path.isabs(c) else c for c in cmd), "\n")
sys.stdout.flush()
subprocess.run(cmd, check=True)

run = subprocess.run([exe])
sys.exit(run.returncode)
