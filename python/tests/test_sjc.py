"""Tests of sjc, the Symjit compiler driver: functions are saved with `f.save(...)`, compiled
by sjc into object files and C headers, linked into C programs with `cc ... -lm` and compared
with the JIT-compiled functions (the same machine code, so results must be bit-identical).

The sjc binary is `$SJC`, else `target/release/sjc` (build it with
`cargo build --release -p sjc`). Skipped without it, without a C compiler, or on hosts
other than Linux/macOS on x86-64/ARM64. The library used here needs no cargo features: sjc
has its own copy of symjit.
"""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import sympy as sp

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symjit import compile_func, compile_ode

from test_obj import HOST_OK, run_c, same

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SJC = os.environ.get("SJC", os.path.join(ROOT, "target", "release", "sjc"))
HAVE_SJC = os.path.exists(SJC)
HAVE_SYMBOLICA = importlib.util.find_spec("symbolica") is not None
NEEDS = "needs the sjc binary (cargo build --release -p sjc, or $SJC), Linux or macOS, and a C compiler"

x, y, p, t = sp.symbols("x y p t")


def sjc(*args, cwd):
    return subprocess.run([SJC, *args], cwd=cwd, capture_output=True, text=True)


class SjcModel:
    """A saved function compiled by sjc into `<dir>/<name>.o/.h`, with the layout of `test_obj.Model`."""

    def __init__(self, f, name, directory, symbolica=False, args=(), **save_args):
        self.name = name
        self.symbolica = symbolica
        f.save(os.path.join(directory, name + ".sjb"), **save_args)
        res = sjc(name + ".sjb", *args, cwd=directory)
        if res.returncode != 0:
            raise AssertionError(f"sjc failed: {res.stderr}")
        with open(os.path.join(directory, name + ".h")) as fd:
            header = fd.read()
        macros = dict(re.findall(rf"#define {name.upper()}_(\w+) (\d+)", header))
        self.cs = int(macros["COUNT_STATES"])
        self.co = int(macros["COUNT_OBS"])
        self.cp = int(macros["COUNT_PARAMS"])
        self.mem = int(macros["MEM_SIZE"])
        self.fast = f"double {name}_fast(" in header


@unittest.skipUnless(HAVE_SJC and HOST_OK, NEEDS)
class Sjc(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def check(self, f, name, points, params=(), complex_=False):
        m = SjcModel(f, name, self.d)
        res = run_c(self.d, [m], [list(pt) for pt in points], params)[name]
        for pt, (status, obs, fast) in zip(points, res):
            self.assertEqual(status, 0)
            if complex_:
                args = [complex(pt[i], pt[i + 1]) for i in range(0, len(pt), 2)]
                want = np.array(f(*args), dtype=complex)
                want = np.ravel(np.column_stack([want.real, want.imag]))
            else:
                want = np.ravel(f(*pt, *params))
            self.assertTrue(same(obs, want), f"{name} at {pt}: object {obs}, JIT {list(want)}")
            if fast is not None:
                self.assertTrue(same(fast, want[0]), f"{name}_fast at {pt}: {fast} vs {want[0]}")
        return m

    POINTS = [(0.5, 2.0), (-1.25, 0.75), (3.0, -0.5), (0.0, 0.0)]

    def test_functions_and_branches(self):
        for opt in (1, 2, 3):
            f = compile_func(
                [x, y],
                [sp.sin(x) * y + sp.exp(-x * x), sp.Piecewise((sp.cos(y), x > 0), (x * y, True)), sp.atan2(x, y)],
                opt_level=opt,
            )
            self.check(f, f"funcs{opt}", self.POINTS)

    def test_fast_kernel(self):
        m = self.check(compile_func([x, y], [sp.sqrt(x * x + y * y) + sp.tanh(x)]), "fast", self.POINTS)
        self.assertTrue(m.fast)

    def test_parameters(self):
        f = compile_func([x], [x * p + sp.sin(p), sp.exp(x) - p], params=[p])
        self.check(f, "params", [(0.5,), (-2.0,)], params=[0.75])

    def test_complex(self):
        for fast_complex in (True, False):
            f = compile_func(
                [x, y], [x * y + x, x / (1 + y * y), sp.sqrt(x) * y], dtype="complex128", fast_complex=fast_complex
            )
            pts = [(0.5, 0.25, -1.0, 2.0), (1.5, -0.5, 0.75, 0.0), (-2.0, 1.0, 0.3, -1.2)]
            self.check(f, f"cplx{int(fast_complex)}", pts, complex_=True)

    def test_ode(self):
        f = compile_ode(t, [x, y], [y * sp.cos(t), -sp.sin(x) - 0.1 * y])
        m = SjcModel(f, "pendulum", self.d)
        self.assertEqual((m.cs, m.co), (3, 0))  # t, x, y; the derivatives follow in mem
        lines = [
            '#include <stdio.h>', '#include "pendulum.h"', "int main(void) {",
            "    double mem[PENDULUM_MEM_SIZE] = {0.25, 1.0, -0.5};",
            "    int s = pendulum(mem, NULL, 0, NULL);",
            '    printf("%d %a %a\\n", s, mem[PENDULUM_COUNT_STATES], mem[PENDULUM_COUNT_STATES + 1]);',
            "    return 0;", "}",
        ]
        with open(os.path.join(self.d, "ode.c"), "w") as fd:
            fd.write("\n".join(lines) + "\n")
        exe = os.path.join(self.d, "ode")
        subprocess.run(["cc", "-Wall", "-Werror", "ode.c", "pendulum.o", "-lm", "-o", exe], check=True, cwd=self.d)
        words = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(words[0], "0")
        self.assertTrue(same([float.fromhex(w) for w in words[1:]], f(0.25, [1.0, -0.5])))

    @unittest.skipUnless(HAVE_SYMBOLICA, "needs symbolica")
    def test_symbolica_evaluator(self):
        from symbolica import E, S
        from symjit import compile_evaluator

        ev = E("if(y, x^2 + sin(x), cos(x) * y) + exp(x*y)").evaluator([S("x"), S("y")])
        f = compile_evaluator(ev)
        m = SjcModel(f, "symb", self.d, symbolica=True, dtype="float64")
        pts = [[0.5, 0.0], [0.5, 2.0], [-1.25, 0.75]]
        res = run_c(self.d, [m], pts)["symb"]
        want = np.asarray(ev.evaluate(np.array(pts)))
        for (status, obs, _), w in zip(res, want):
            self.assertEqual(status, 0)
            np.testing.assert_allclose(obs, np.ravel(w), rtol=1e-15)

    def test_info_and_wasm(self):
        f = compile_func([x, y], [sp.sin(x) * y], opt_level=2)
        f.save(os.path.join(self.d, "m.sjb"))
        res = sjc("--emit", "info", "m.sjb", cwd=self.d)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("2 states, 0 parameters, 1 outputs", res.stdout)
        self.assertIn("functions:  sin", res.stdout)

        res = sjc("m.sjb", "-o", "m.wasm", cwd=self.d)
        if "without the `wasm` feature" not in res.stderr:
            self.assertEqual(res.returncode, 0, res.stderr)
            with open(os.path.join(self.d, "m.wasm"), "rb") as fd:
                self.assertEqual(fd.read(4), b"\0asm")

    def test_arm_opt_level_3_is_rejected_for_x86(self):
        terms = sum(sp.sin(k * x + y) * sp.exp(-p / k) for k in range(1, 40))
        f = compile_func([x, y, p], [terms], ty="arm", opt_level=3)
        f.save(os.path.join(self.d, "big.sjb"))
        res = sjc("big.sjb", "--target", "x86_64", "-o", "big.o", cwd=self.d)
        self.assertEqual(res.returncode, 1)
        self.assertIn("registers", res.stderr)
        res = sjc("big.sjb", "--target", "aarch64-linux", "-o", "big.o", cwd=self.d)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
