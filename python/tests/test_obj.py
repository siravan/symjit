"""Tests of object files (`write_obj`): compiled functions are written as relocatable
objects with C headers, linked into a generated C program with `cc ... -lm`, run, and
compared with the JIT-compiled functions (the kernels are the same machine code, so
results must be bit-identical).

`ObjFuzz` does this for the random expressions of test_fuzz.py (one C program per
family); SYMJIT_FUZZ_SEEDS and SYMJIT_FUZZ_START work as there.

They run on Linux and macOS, x86-64 and ARM64 (the objects are in the host's format and
architecture); `ObjCross` checks AArch64 objects (`ty="arm"`) on any host. Skipped when
the library was built without the `obj` cargo feature (the default; build with
`--features obj`) or without a C compiler (`cc`).
"""

import importlib.util
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import sympy as sp

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symjit import Composer, compile_composer, compile_func, compile_ode

import test_fuzz as fz

x, y, z, p, t = sp.symbols("x y z p t")


def obj_available():
    """False if the library cannot write object files (built without `obj`)"""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            compile_func([x], [x]).write_obj(os.path.join(tmp, "probe"))
            return True
        except ValueError as e:
            if "without the `obj` feature" in str(e) or "cannot write object files" in str(e):
                return False
            raise


HAVE_OBJ = obj_available()
HOST_OK = (
    (sys.platform.startswith("linux") or sys.platform == "darwin")
    and platform.machine() in ("x86_64", "AMD64", "aarch64", "arm64")
    and shutil.which("cc") is not None
)
HOST_ARCH = "AArch64" if platform.machine() in ("aarch64", "arm64") else "x86-64"
HAVE_SYMBOLICA = importlib.util.find_spec("symbolica") is not None
NEEDS = "needs the `obj` feature (cargo build --features obj), Linux or macOS, and a C compiler"


def c_double(v):
    v = float(v)
    if np.isnan(v):
        return "NAN"
    if np.isinf(v):
        return "INFINITY" if v > 0 else "-INFINITY"
    return repr(v)


class Model:
    """A compiled function written as `<dir>/<name>.o/.h`, with its layout from the header."""

    def __init__(self, f, name, directory, symbolica=False, **write_args):
        self.name = name
        self.symbolica = symbolica
        f.write_obj(os.path.join(directory, name), **write_args)
        with open(os.path.join(directory, name + ".h")) as fd:
            header = fd.read()
        macros = dict(re.findall(rf"#define {name.upper()}_(\w+) (\d+)", header))
        self.cs = int(macros["COUNT_STATES"])
        self.co = int(macros["COUNT_OBS"])
        self.cp = int(macros["COUNT_PARAMS"])
        self.mem = int(macros["MEM_SIZE"])
        self.fast = f"double {name}_fast(" in header


def run_c(directory, models, points, params=()):
    """Links one C program with all `models` and evaluates each at every point (states,
    or inputs for Composer/Symbolica models) in direct mode, and with `_fast` if there is
    one. Returns {name: [(status, outputs, fast output or None), ...]}."""
    width = max(len(pt) for pt in points)
    lines = ["#include <math.h>", "#include <stdio.h>", "#include <string.h>"]
    lines += [f'#include "{m.name}.h"' for m in models]
    rows = ", ".join("{ " + ", ".join(c_double(v) for v in pt) + " }" for pt in points)
    lines.append(f"static const double P[{len(points)}][{width}] = {{ {rows} }};")
    lines.append(f"static const double sj_params_[{len(params) + 1}] = {{ {', '.join(map(c_double, list(params) + [0]))} }};")
    lines.append(f"static double mem[{max(m.mem for m in models) + 64}];")
    lines.append("int main(void) {")
    lines.append("    (void)sj_params_;")
    for m in models:
        lines.append(f"    for (int k = 0; k < {len(points)}; k++) {{")
        lines.append("        memset(mem, 0, sizeof(mem));")
        if m.symbolica:  # inputs are the parameters, outputs go to mem
            lines.append(f'        printf("{m.name} %d", {m.name}(mem, NULL, 0, P[k]));')
            first = 0
        else:
            lines.append(f"        memcpy(mem, P[k], {m.cs} * sizeof(double));")
            lines.append(f'        printf("{m.name} %d", {m.name}(mem, NULL, 0, sj_params_));')
            first = m.cs
        lines.append(f'        for (int i = 0; i < {m.co}; i++) printf(" %a", mem[{first} + i]);')
        if m.fast:
            args = ", ".join(f"P[k][{i}]" for i in range(m.cs))
            lines.append(f'        printf(" %a", {m.name}_fast({args}));')
        lines.append('        printf("\\n");')
        lines.append("    }")
    lines.append("    return 0;")
    lines.append("}")

    src = os.path.join(directory, "main.c")
    with open(src, "w") as fd:
        fd.write("\n".join(lines) + "\n")
    exe = os.path.join(directory, "main")
    objs = [os.path.join(directory, m.name + ".o") for m in models]
    cc = subprocess.run(["cc", "-Wall", "-Werror", src, *objs, "-lm", "-o", exe], capture_output=True, text=True)
    if cc.returncode != 0 or cc.stderr:
        raise AssertionError(f"cc failed:\n{cc.stderr[:3000]}")
    run = subprocess.run([exe], capture_output=True, text=True, timeout=600)
    if run.returncode != 0:
        raise AssertionError(f"the C program failed: {run.stderr}")

    by_model = {m.name: m for m in models}
    out = {m.name: [] for m in models}
    for line in run.stdout.splitlines():
        words = line.split()
        m = by_model[words[0]]
        vals = [float.fromhex(w) for w in words[2:]]
        out[m.name].append((int(words[1]), vals[: m.co], vals[m.co] if m.fast else None))
    return out


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


@unittest.skipUnless(HAVE_OBJ and HOST_OK, NEEDS)
class Obj(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def check(self, f, name, points, params=(), complex_=False):
        m = Model(f, name, self.d)
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

    POINTS = [(0.5, 2.0), (-1.25, 0.75), (3.0, -0.5), (0.0, 0.0), (2.0, 1e-300)]

    def test_pi(self):
        # known values rather than the JIT: pi by Machin's formula (arctan as a Taylor
        # series, and as libm's atan) and by the BBP series (a loop calling pow)
        n, k = sp.symbols("n k")

        def arctan_series(a, terms=21):
            return sum((-1) ** i * a ** (2 * i + 1) / (2 * i + 1) for i in range(terms))

        machin = [compile_func([x, y], [4 * (4 * f(x) - f(y))]) for f in (arctan_series, sp.atan)]
        bbp = compile_func(
            [n], [sp.Sum((4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)) / 16**k, (k, 0, n))]
        )
        models = [Model(machin[0], "pi_series", self.d), Model(machin[1], "pi_atan", self.d)]
        res = run_c(self.d, models, [[1 / 5, 1 / 239]])
        ulp = math.ulp(math.pi)
        for m in models:
            [(status, [v], fast)] = res[m.name]
            self.assertEqual(status, 0)
            self.assertLessEqual(abs(v - math.pi), ulp, m.name)
            self.assertEqual(fast, v)

        res = run_c(self.d, [Model(bbp, "pi_bbp", self.d)], [[float(t)] for t in range(13)])["pi_bbp"]
        errors = [abs(obs[0] - math.pi) for _, obs, _ in res]
        self.assertTrue(all(e1 <= e0 for e0, e1 in zip(errors, errors[1:])))  # converges
        self.assertEqual(errors[10:], [0.0, 0.0, 0.0])  # exact from 10 terms on
        self.assertGreater(errors[0], 0.008)  # 47/15 with one term

    def test_functions_and_branches(self):
        eqs = [
            sp.sin(x) * y + sp.exp(-x * x), sp.log(1 + x * x), sp.atan2(y, x), sp.Abs(x) ** y,
            sp.Piecewise((sp.cos(x), x < y), (y**3, True)), sp.Min(x, y) + sp.Max(x, sp.tanh(y)),
            sp.sqrt(x * x + y * y), sp.floor(x) + sp.ceiling(y),
        ]
        for opt_level in (0, 1, 2, 3):
            m = self.check(compile_func([x, y], eqs, opt_level=opt_level), f"model{opt_level}", self.POINTS)
            self.assertFalse(m.fast)

    def test_fast_kernel(self):
        m = self.check(compile_func([x, y], [sp.sin(x) * y + x / (1 + y * y)]), "fastmodel", self.POINTS)
        self.assertTrue(m.fast)

    def test_parameters(self):
        f = compile_func([x], [x * p + sp.sin(p), sp.exp(x) - p], params=[p])
        self.check(f, "params", [(0.5,), (-2.0,)], params=[0.75])

    def test_complex(self):
        # complex arithmetic (complex functions cannot be called from object files)
        f = compile_func([x, y], [x * y + x, x / (1 + y * y), sp.sqrt(x) * y], dtype="complex128")
        pts = [(0.5, 0.25, -1.0, 2.0), (1.5, -0.5, 0.75, 0.0), (-2.0, 1.0, 0.3, -1.2)]
        self.check(f, "cplx", pts, complex_=True)

    def test_ode(self):
        f = compile_ode(t, [x, y], [y * sp.cos(t), -sp.sin(x) - 0.1 * y])
        m = Model(f, "pendulum", self.d)
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
        subprocess.run(["cc", "-Wall", "-Werror", os.path.join(self.d, "ode.c"), os.path.join(self.d, "pendulum.o"),
                        "-lm", "-o", exe], check=True)
        words = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(words[0], "0")
        self.assertTrue(same([float.fromhex(w) for w in words[1:]], f(0.25, [1.0, -0.5])))

    def test_indirect_mode(self):
        f = compile_func([x, y], [sp.sin(x) * y, x - y * y])
        Model(f, "vec", self.d)
        n = 101
        X, Y = np.linspace(-2, 2, n), np.cos(np.linspace(0, 5, n))
        lines = [
            "#include <stdio.h>", '#include "vec.h"',
            f"static double xs[{n}], ys[{n}], o0[{n}], o1[{n}];", "int main(void) {",
            f"    for (int i = 0; i < {n}; i++) if (scanf(\"%lf %lf\", &xs[i], &ys[i]) != 2) return 2;",
            f"    symjit_slice arrays[VEC_COUNT_STATES + VEC_COUNT_OBS] = {{ {{xs, {n}}}, {{ys, {n}}}, {{o0, {n}}}, {{o1, {n}}} }};",
            f"    for (size_t i = 0; i < {n}; i++) if (vec(NULL, arrays, i, NULL)) return 1;",
            f'    for (int i = 0; i < {n}; i++) printf("%a %a\\n", o0[i], o1[i]);',
            "    return 0;", "}",
        ]
        with open(os.path.join(self.d, "v.c"), "w") as fd:
            fd.write("\n".join(lines) + "\n")
        exe = os.path.join(self.d, "v")
        subprocess.run(["cc", "-Wall", "-Werror", os.path.join(self.d, "v.c"), os.path.join(self.d, "vec.o"),
                        "-lm", "-o", exe], check=True)
        stdin = "\n".join(f"{float(a)!r} {float(b)!r}" for a, b in zip(X, Y))
        out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
        got = np.array([[float.fromhex(w) for w in line.split()] for line in out.splitlines()]).T
        # the scalar kernel, as in the object (vectorized calls use the SIMD kernel)
        want = np.array([np.ravel(f(a, b)) for a, b in zip(X, Y)]).T
        self.assertTrue(same(got, want))

    def test_composer_with_branches_loops_and_recursion(self):
        from test_regressions import recursive_composer
        import test_composer_fuzz as cf

        programs = [("fib", recursive_composer("fibonacci"))]
        for family in ("branches", "loops"):
            for seed in range(3):
                programs.append((f"{family}{seed}", cf.Program(seed, family).compose()))
        models, natives = [], {}
        for name, cp in programs:
            models.append(Model(compile_composer(cp), name, self.d, symbolica=True, dtype="float64"))
            natives[name] = compile_composer(cp)
        pts = [[float(k), 0.5 - k / 7, 1.0 + k / 3] for k in range(12)]
        res = run_c(self.d, models, pts)
        for m in models:
            for pt, (status, obs, _) in zip(pts, res[m.name]):
                args = pt[: m.cp]
                want = np.ravel(natives[m.name](*args))
                self.assertEqual(status, 0)
                self.assertTrue(same(obs, want), f"{m.name} at {args}: {obs} vs {list(want)}")

    @unittest.skipUnless(HAVE_SYMBOLICA, "needs symbolica")
    def test_symbolica_evaluator(self):
        from symbolica import E, S
        from symjit import compile_evaluator

        ev = E("if(y, x^2 + sin(x), cos(x) * y) + exp(x*y)").evaluator([S("x"), S("y")])
        f = compile_evaluator(ev)
        m = Model(f, "symb", self.d, symbolica=True, dtype="float64")
        pts = [[0.5, 0.0], [0.5, 2.0], [-1.25, 0.75]]
        res = run_c(self.d, [m], pts)["symb"]
        want = np.asarray(ev.evaluate(np.array(pts)))
        for (status, obs, _), w in zip(res, want):
            self.assertEqual(status, 0)
            np.testing.assert_allclose(obs, np.ravel(w), rtol=1e-15)

    def test_errors(self):
        path = os.path.join(self.d, "e")
        cases = [
            (compile_func([x], [sp.csc(x)]), "`csc` is not a C math library function"),
            (compile_func([x], [sp.sin(x)], dtype="complex128"), "complex function"),
            (compile_func([x], [sp.Function("g")(x)], defuns={"g": lambda a: 2 * a}), "user-defined function `g`"),
        ]
        for f, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    f.write_obj(path)
        f = compile_func([x], [x])
        with self.assertRaisesRegex(ValueError, "`format` should be"):
            f.write_obj(path, format="coff")
        with self.assertRaises(ValueError):
            f.write_obj(os.path.join(self.d, "not-an-identifier"))

    def test_macho_format(self):
        f = compile_func([x], [sp.sin(x)])
        f.write_obj(os.path.join(self.d, "mac"), format="macho")
        with open(os.path.join(self.d, "mac.o"), "rb") as fd:
            self.assertEqual(fd.read(4), bytes.fromhex("cffaedfe"))
        with open(os.path.join(self.d, "mac.h")) as fd:
            self.assertIn(f"{HOST_ARCH} Mach-O", fd.read())


@unittest.skipUnless(HAVE_OBJ, "needs the `obj` feature (cargo build --features obj)")
class ObjCross(unittest.TestCase):
    """AArch64 objects compiled with ty="arm" on any host (checked, not run)."""

    def test_arm64_objects(self):
        n, k = sp.symbols("n k")
        f = compile_func([x, y], [4 * (4 * sp.atan(x) - sp.atan(y))], ty="arm")
        bbp = compile_func(
            [n], [sp.Sum((4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)) / 16**k, (k, 0, n))],
            ty="arm",
        )
        with tempfile.TemporaryDirectory() as d:
            for g, name in ((f, "pi_atan"), (bbp, "pi_bbp")):
                g.write_obj(os.path.join(d, name), format="macho")
                with open(os.path.join(d, name + ".o"), "rb") as fd:
                    b = fd.read()
                self.assertEqual(b[:4], bytes.fromhex("cffaedfe"))
                self.assertEqual(int.from_bytes(b[4:8], "little"), 0x0100000C)  # CPU_TYPE_ARM64
                with open(os.path.join(d, name + ".h")) as fd:
                    header = fd.read()
                self.assertIn("AArch64 Mach-O", header)
                self.assertIn(f"double {name}_fast(double x0", header)

            f.write_obj(os.path.join(d, "elf"), format="elf")
            with open(os.path.join(d, "elf.o"), "rb") as fd:
                b = fd.read()
            self.assertEqual(b[:4], b"\x7fELF")
            self.assertEqual(int.from_bytes(b[18:20], "little"), 183)  # EM_AARCH64


@unittest.skipIf(HAVE_OBJ, "the library has the `obj` feature")
class ObjFeatureMissing(unittest.TestCase):
    def test_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "without the `obj` feature|cannot write object files"):
                compile_func([x], [x]).write_obj(os.path.join(tmp, "f"))


FUZZ_OPTIONS = [dict(opt_level=0), dict(opt_level=2), dict(opt_level=3, cse=False), dict(opt_level=2, fastmath=False)]


@unittest.skipUnless(HAVE_OBJ and HOST_OK, NEEDS)
class ObjFuzz(unittest.TestCase):
    def run_family(self, family, generator, dtype=float):
        X = fz.inputs(fz.NPOINTS, 7, dtype)
        complex_ = dtype == complex
        pts = [list(np.ravel(np.column_stack([c.real, c.imag]))) if complex_ else list(c) for c in X.T]
        models, funcs, skipped = [], {}, 0

        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            for seed in range(fz.START, fz.START + fz.SEEDS):
                exprs = fz.build_case(family, generator, seed)
                if exprs is None:
                    continue
                for k, options in enumerate(FUZZ_OPTIONS):
                    f = compile_func(fz.V, exprs, dtype="complex128" if complex_ else "float64", **options)
                    name = f"{family}_{seed}_{k}"
                    try:
                        models.append(Model(f, name, d))
                    except ValueError as e:
                        if complex_ and "complex function" in str(e):
                            skipped += 1
                            continue
                        raise AssertionError(f"{family} seed {seed} {options}: {e}\n{exprs}")
                    funcs[name] = (f, seed, options, exprs)

            self.assertGreater(len(models), 0)
            res = run_c(d, models, pts)

            for m in models:
                f, seed, options, exprs = funcs[m.name]
                for c, (status, obs, fast) in zip(X.T, res[m.name]):
                    want = np.array(f(*c), dtype=dtype)
                    if complex_:
                        want = np.column_stack([want.real, want.imag])
                    want = np.ravel(want)
                    self.assertEqual(status, 0)
                    if not same(obs, want) or (fast is not None and not same(fast, want[0])):
                        self.fail(f"{family} seed {seed} {options} at {c}:\nobject {obs} fast {fast}\n"
                                  f"JIT {list(want)}\nexpressions: {exprs}")
        return len(models), skipped

    def test_real(self):
        self.run_family("real", fz.case_real)

    def test_shared(self):
        self.run_family("shared", fz.case_shared)

    def test_piecewise(self):
        self.run_family("piecewise", fz.case_piecewise)

    def test_complex(self):
        # cases calling complex functions are rejected by write_obj and skipped
        self.run_family("complex", fz.case_complex, complex)


if __name__ == "__main__":
    unittest.main()
