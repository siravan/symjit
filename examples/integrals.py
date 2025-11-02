import warnings

warnings.filterwarnings("ignore")

from sympy import *
from symjit import compile_func
import scipy as sp
import os

u = Symbol("u")

a = 0.001
b = 0.999


def test_integrals(sym):
    print(f"trying {sym}")
    print("--------------------------------------")

    name = f"{sym}.txt"
    path = os.path.join(os.path.dirname(__file__), "suite/", name)
    fd = open(path, "r")

    for line in fd.readlines():
        test_integral(line)

    fd.close()


def test_integral(line):
    s = line.split(";")
    eq = eval(s[1])
    print(f"integrating {eq}: ", end="")

    try:
        sol = integrate(eq, u)
        if not (sol is None or sol.has(Integral)):
            F = compile_func([u], sol)
            y0 = F(b) - F(a)

            f1 = lambdify([u], eq)
            y1, ϵ1 = sp.integrate.quad(f1, a, b)

            f2 = compile_func([u], eq)
            y2, ϵ2 = sp.integrate.quad(f2, a, b)

            if abs(y1 - y0) < ϵ1 and abs(y2 - y0) > ϵ2:
                print(f"\033[91mdiscrepency noted!\033[0m")
            else:
                print("pass!")
        else:
            print("\033[92mskip (numerical)!\033[0m")
            pass
    except:
        print("\033[94mskip (symbolic)!\033[0m")
        pass


test_integrals("basic")
test_integrals("heurisch")
test_integrals("Stewart")
