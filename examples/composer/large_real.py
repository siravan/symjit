import math
import platform
import time
import statistics

import numpy as np
from symjit import Composer, compile_composer

def arch():
    if platform.machine() in ["x86_64", "AMD64"]:
        return "amd"
    elif platform.machine() in ["arm64", "aarch64"]:
        return "arm"
    elif platform.machine() == "riscv64":
        return "riscv"
    else:
        return None


N = 10
DEPTH = 12
NCOLS = 2**DEPTH
NROWS = 10000

K = np.reshape(np.arange(NROWS * NCOLS), (NROWS, NCOLS))
X = np.cos(K)

def large(cp: Composer, a: int, b: int):
    d = b - a
    if d == 1:
        return cp.arg(a), X[0, a]
    else:
        n = (a + b) // 2
        child_left, val_left = large(cp, a, n)
        child_right, val_right = large(cp, n, b)

        if (round(math.log2(d)) % 2) == 0:
            return cp.fadd(child_left, child_right), val_left + val_right
        else:
            return cp.fdiv(child_left, child_right), val_left / val_right


cp = Composer(NCOLS, 1)
root, val = large(cp, 0, NCOLS)
cp.assign(cp.out(0), root)

def Ω(b):
    if b:
        return "T"
    else:
        return "F"


def abbr_ty(ty):
    return ty[0]


def abbr_dtype(dtype):
    if dtype == "complex128":
        return "C"
    else:
        return "R"


def cases():
    cases = []

    if arch() == "amd":
        for use_simd in ["f64", "f64x4", "f64x8"]:
            for use_threads in [False, True]:
                for cse in [False, True]:
                    for fastmath in [False, True]:
                        for opt_level in [0, 1, 2, 3]:
                            args = {
                                "backend": "rust",
                                "use_simd": use_simd != "f64",
                                "enable_simd512": use_simd == "f64x8",
                                "use_threads": use_threads,
                                "cse": cse,
                                "fastmath": fastmath,
                                "opt_level": opt_level,
                                "dtype": "float64",
                            }
                            s = f"s={use_simd}:t={Ω(use_threads)}:c={Ω(cse)}:f={Ω(fastmath)},O={opt_level}"
                            cases.append((s, args))
    else:
        for use_threads in [False, True]:
            for cse in [False, True]:
                for fastmath in [False, True]:
                    for opt_level in [0, 1, 2, 3]:
                        args = {
                            "backend": "rust",
                            "use_simd": False,
                            "use_threads": use_threads,
                            "cse": cse,
                            "fastmath": fastmath,
                            "opt_level": opt_level,
                            "dtype": "float64",
                        }
                        s = f"s=F:t={Ω(use_threads)}:c={Ω(cse)}:f={Ω(fastmath)},O={opt_level}"
                        cases.append((s, args))
    return cases


def test_model():
    print("\td: dtype\t\t(R=float64, C=complex128)")
    print("\ty: ty\t\t(n: native, a: amd-sse, b: bytecode, d: debug)")
    print("\ts: simd\t\t(True/False) or (none/avx256/avx512)")
    print("\tt: threads\t(True/False)")
    print("\tc: cse\t\t(True/False)")
    print("\tf: fastmath\t(True/False)")
    print("\tO: opt_level\t(0/1/2/3)")


    for abbr, args in cases():
        dt = []
        f = compile_composer(cp, **args)

        for _ in range(N):
            t0 = time.perf_counter_ns()
            B = f.evaluate(X)
            t1 = time.perf_counter_ns()
            dt.append(t1 - t0)

        t = statistics.median(dt)
        print(f"{abbr}\t", end="")

        try:
            np.testing.assert_array_almost_equal(val, B[0])
            print(f"\tpass in {1e-6 * t:.3f} ms")
        except AssertionError:
            print(f"\t\033[31mfail\033[0m in {1e-6 * t:.3f} ms")

        if args["opt_level"] == 3:
            print()

test_model()
