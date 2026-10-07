"""Runs symjit's ARM64 and RISC-V code under qemu-user and compares it with this machine.

The symjit library is cross-compiled for aarch64 and riscv64 (cargo), together with a small C
driver (emulated_driver.c) that calls the library's C ABI, the functions engine.py calls:
`compile`, `execute`, `execute_matrix` (SIMD kernels, threads), `fast_func`, and `translate` +
`evaluate_matrix` for Symbolica evaluators. Under qemu, the driver compiles every model of a
corpus written here with the target's JIT and evaluates it; the results are compared with the
bytecode interpreter run on this machine (and, for Symbolica evaluators, with this machine's JIT).

The corpus: the expression families of test_properties (including loops and complex numbers),
random expressions of test_fuzz, a model with parameters, and the 1- and 2-loop evaluators of
examples/symbolica, each with several option sets. ARM64 runs on two CPU models, `max` (with
FCMA: the complex kernels use fcmla/fcadd) and `cortex-a57` (ARMv8.0, without).

Requirements: qemu-user, the cross compilers (gcc-aarch64-linux-gnu, gcc-riscv64-linux-gnu,
with their C libraries) and the Rust targets (rustup target add aarch64-unknown-linux-gnu
riscv64gc-unknown-linux-gnu). The first run builds the library for both targets (a few
minutes; incremental afterwards), so the test only runs when asked:

    SYMJIT_EMULATED=1 python -m unittest python/tests/test_emulated.py -v

Checks, per case: without fastmath, the scalar, SIMD (`execute_matrix`) and fast paths give the
same bits (as test_properties requires on the host), and agree with this machine's bytecode
interpreter to 1e-13 (the C math library may differ in the last bit between architectures);
with fastmath, they agree with the interpreter within the tolerances of test_properties.
Complex results are compared with a tolerance (as on the host, the SIMD kernels use other
operation sequences). The number of cases that match this machine bit for bit is printed.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

from symjit import compile_evaluator, compile_func, engine, structure  # noqa: E402

import test_fuzz  # noqa: E402
import test_properties as props  # noqa: E402

ENABLED = os.environ.get("SYMJIT_EMULATED") == "1"
NPTS = 67  # odd: SIMD tails
TIMEOUT = int(os.environ.get("SYMJIT_EMULATED_TIMEOUT", "600"))  # seconds per target

ARCHES = {
    # name: (rust target, gcc prefix, qemu, qemu CPU models)
    "aarch64": ("aarch64-unknown-linux-gnu", "aarch64-linux-gnu", "qemu-aarch64", ["max", "cortex-a57"]),
    "riscv64": ("riscv64gc-unknown-linux-gnu", "riscv64-linux-gnu", "qemu-riscv64", ["max"]),
}

EXACT = [dict(), dict(opt_level=0), dict(opt_level=1), dict(opt_level=3), dict(cse=False), dict(compact=False),
         dict(compress=True), dict(use_simd=False), dict(use_threads=False)]
FASTMATH = [dict(fastmath=True), dict(fastmath=True, opt_level=3), dict(fastmath=True, use_simd=False)]
COMPLEX = [dict(), dict(fast_complex=False), dict(opt_level=0), dict(opt_level=3), dict(compress=True),
           dict(use_simd=False), dict(parallel_mul=False)]


def missing_tools(arch):
    target, gcc, qemu, _ = ARCHES[arch]
    need = [qemu, f"{gcc}-gcc", "cargo"]
    missing = [t for t in need if shutil.which(t) is None]
    if not os.path.isdir(f"/usr/{gcc}"):
        missing.append(f"/usr/{gcc} (the cross C library)")
    try:
        installed = subprocess.run(["rustup", "target", "list", "--installed"], capture_output=True,
                                   text=True).stdout.split()
        if target not in installed:
            missing.append(f"rust target {target}")
    except FileNotFoundError:
        missing.append("rustup")
    return missing


def build(arch):
    """cross-compiles the library and the driver; returns their paths"""
    target, gcc, _, _ = ARCHES[arch]
    env = dict(os.environ)
    env[f"CARGO_TARGET_{target.upper().replace('-', '_')}_LINKER"] = f"{gcc}-gcc"
    p = subprocess.run(["cargo", "build", "--release", "--lib", "--target", target], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cargo build for {target} failed:\n{p.stderr[-3000:]}")
    lib = os.path.join(ROOT, "target", target, "release", "libsymjit.so")
    driver = os.path.join(ROOT, "target", target, "emulated_driver")
    subprocess.run([f"{gcc}-gcc", "-O1", "-Wall", "-Wno-format-truncation", os.path.join(HERE, "emulated_driver.c"), "-ldl", "-o", driver],
                   check=True)
    return lib, driver


def quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compile_func(*args, **kwargs)


def as_rows(values, complex_):
    """the outputs of a Func call as rows of doubles ((re, im) rows for complex outputs)"""
    rows = []
    for v in values:
        v = np.asarray(v)
        if complex_:
            rows += [v.real, v.imag]
        else:
            rows.append(v.astype(np.float64))
    return np.array(rows)


class Corpus:
    """the cases: models, inputs and this machine's results, written to a directory"""

    def __init__(self, directory):
        self.d = directory
        self.cases = []  # dicts: name, kind, prefix, line, reference, check

    def add_func(self, name, states, exprs, params=(), pvals=(), cols=None, options={}, complex_=False, tag=""):
        kw = dict(fastmath=False, opt_level=2) | options
        prefix = os.path.join(self.d, f"c{len(self.cases)}")
        model = structure.model(states, exprs, params=list(params) or None)
        with open(prefix + ".model", "w") as fd:
            fd.write(json.dumps(model))
        rows = as_rows(cols, complex_)
        with open(prefix + ".in", "wb") as fd:
            fd.write(rows.tobytes())
            fd.write(np.asarray(pvals, dtype=np.float64).tobytes())
        ref = quiet(states, exprs, params=list(params) or None, ty="bytecode", fastmath=False,
                    dtype="complex128" if complex_ else "float64")
        want = ref(*cols, *pvals)
        opt = engine.pack_options(dtype="complex128" if complex_ else "float64", **kw)
        self.cases.append(dict(name=f"{tag} {options}".strip(), kind="func", prefix=prefix,
                               line=f"func native {opt} {NPTS} 1 {prefix}", want=as_rows(want, complex_),
                               fastmath=kw["fastmath"], complex=complex_, tag=tag))

    def add_eval(self, loops, npts=8):
        path = os.path.join(ROOT, "examples", "symbolica", f"{loops}loop_instructions_2.txt")
        with open(path, encoding="utf-8") as fd:
            text = fd.read()
        f = compile_evaluator(text, dtype="complex128")
        n = f.complex_compiler.count_params // 2
        rng = np.random.default_rng(loops)
        X = rng.random((npts, n)) + 1j * rng.random((npts, n))
        want = f.evaluate_complex(X)
        prefix = os.path.join(self.d, f"c{len(self.cases)}")
        with open(prefix + ".model", "w") as fd:
            fd.write(text)
        X.astype(np.complex128).tofile(prefix + ".in")
        opt = engine.pack_options(dtype="complex128", order="c", opt_level=3)
        self.cases.append(dict(name=f"{loops}-loop evaluator", kind="eval", prefix=prefix,
                               line=f"eval native {opt} {npts} 1 {prefix}", want=want, npts=npts))

    def build(self):
        rng = np.random.default_rng(7)
        x, y = props.x, props.y
        for family, exprs in props.REAL.items():
            cols = [rng.uniform(-3, 3, NPTS), rng.uniform(-3, 3, NPTS)]
            for o in EXACT + FASTMATH:
                self.add_func(family, [x, y], exprs, cols=cols, options=o, tag=family)
        n = props.n
        cols = [rng.uniform(-1, 1, NPTS), rng.integers(1, 30, NPTS).astype(float)]
        for o in EXACT[:4]:
            self.add_func("loops", [x, n], props.LOOPS, cols=cols, options=o, tag="loops")
        a, b = sp.symbols("a b")
        cols = [rng.uniform(-3, 3, NPTS), rng.uniform(-3, 3, NPTS)]
        for o in EXACT[:3]:
            self.add_func("params", [x, y], [a * sp.sin(x) + b * y, a * b - x], params=[a, b], pvals=[1.5, -0.25],
                          cols=cols, options=o, tag="params")
        C = [rng.normal(size=NPTS) + 1j * rng.normal(size=NPTS) for _ in range(2)]
        for o in COMPLEX:
            self.add_func("complex", [x, y], props.COMPLEX, cols=C, options=o, complex_=True, tag="complex")
        # random expressions of test_fuzz (three states)
        for seed in range(40):
            r = random.Random(seed)
            exprs = test_fuzz.case_real(r)
            X = test_fuzz.inputs(NPTS, seed, "float64")
            self.add_func("fuzz", test_fuzz.V, exprs, cols=list(X), options=EXACT[seed % len(EXACT)],
                          tag=f"fuzz {seed}")
        for seed in range(15):
            r = random.Random(1000 + seed)
            exprs = test_fuzz.case_complex(r)
            X = test_fuzz.inputs(NPTS, seed, "complex128")
            self.add_func("fuzz complex", test_fuzz.V, exprs, cols=list(X), options=COMPLEX[seed % len(COMPLEX)],
                          complex_=True, tag=f"fuzz complex {seed}")
        self.add_eval(1)
        self.add_eval(2)
        with open(os.path.join(self.d, "manifest"), "w") as fd:
            fd.write("\n".join(c["line"] for c in self.cases) + "\n")


_CORPUS = []


def shared_corpus():
    """the corpus, built once per process (it is the same for every target)"""
    if not _CORPUS:
        tmp = tempfile.TemporaryDirectory()
        corpus = Corpus(tmp.name)
        corpus.build()
        corpus.tmp = tmp  # keeps the directory alive
        _CORPUS.append(corpus)
    return _CORPUS[0]


def same_bits(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


class EmulatedBase:
    arch = None
    cpu = None

    @classmethod
    def setUpClass(cls):
        if not ENABLED:
            raise unittest.SkipTest("set SYMJIT_EMULATED=1 to run (cross-compiles symjit)")
        missing = missing_tools(cls.arch)
        if missing:
            raise unittest.SkipTest(f"missing: {', '.join(missing)}")
        lib, driver = build(cls.arch)
        cls.corpus = shared_corpus()
        # the driver writes its outputs next to the inputs: one copy of the corpus per run
        cls.tmp = tempfile.TemporaryDirectory()
        _, gcc, qemu, _ = ARCHES[cls.arch]
        manifest = os.path.join(cls.tmp.name, "manifest")
        with open(manifest, "w") as fd:
            for c in cls.corpus.cases:
                fd.write(c["line"].replace(cls.corpus.d, cls.tmp.name) + "\n")
        for f in os.listdir(cls.corpus.d):
            if f.endswith((".model", ".in")):
                shutil.copy(os.path.join(cls.corpus.d, f), cls.tmp.name)
        cmd = [qemu, "-cpu", cls.cpu, "-L", f"/usr/{gcc}", driver, lib, manifest]
        try:
            cls.proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired as e:  # e.g. a loop that never ends: the driver's last line names the case
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            cls.proc = subprocess.CompletedProcess(cmd, -1, out + f"\ntimed out after {TIMEOUT} s", "")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def prefix(self, case):
        return case["prefix"].replace(self.corpus.d, self.tmp.name)

    def read(self, case, ext, shape):
        path = f"{self.prefix(case)}.{ext}"
        return np.fromfile(path).reshape(shape) if os.path.exists(path) else None

    def test_driver_finished(self):
        lines = self.proc.stdout.strip().splitlines()
        last = lines[-1] if lines else ""
        if last != "done":
            prefix = next((l for l in reversed(lines) if l.startswith(self.tmp.name)), "")
            case = next((c["name"] for c in self.corpus.cases if self.prefix(c) == prefix), "?")
            self.fail(f"the driver stopped (exit {self.proc.returncode}) in case {case!r} ({prefix}): {last}\n"
                      f"{self.proc.stderr[-2000:]}")

    def test_cases(self):
        exact = 0
        for case in self.corpus.cases:
            with self.subTest(case=case["name"]):
                err = f"{self.prefix(case)}.err"
                if os.path.exists(err):
                    with open(err) as fd:
                        self.fail(f"compilation failed: {fd.read()}")
                if case["kind"] == "eval":
                    got = self.read(case, "eval", None)
                    self.assertIsNotNone(got, "no output (crashed?)")
                    got = got.view(np.complex128).reshape(case["want"].shape)
                    scale = np.max(np.abs(case["want"]))
                    self.assertLess(np.max(np.abs(got - case["want"])) / scale, 1e-12)
                    exact += int(np.array_equal(got, case["want"]))
                    continue
                want = case["want"]
                scalar = self.read(case, "scalar", want.shape)
                matrix = self.read(case, "matrix", want.shape)
                self.assertIsNotNone(scalar, "no output (crashed?)")
                self.assertIsNotNone(matrix, "no SIMD output (crashed?)")
                fast = self.read(case, "fast", (1, want.shape[1]))
                if case["complex"]:
                    for got in (scalar, matrix):
                        ok = same_bits(got, want) | np.isclose(got, want, rtol=1e-13, atol=1e-13)
                        self.assertTrue(ok.all(), f"{got[~ok][:3]} vs {want[~ok][:3]}")
                elif not case["fastmath"]:
                    self.assertTrue(same_bits(scalar, matrix).all(),
                                    f"scalar vs SIMD: {scalar[~same_bits(scalar, matrix)][:3]} vs "
                                    f"{matrix[~same_bits(scalar, matrix)][:3]}")
                    if fast is not None:
                        self.assertTrue(same_bits(fast, scalar[:1]).all(), "fast vs scalar")
                    ok = same_bits(scalar, want) | np.isclose(scalar, want, rtol=1e-13, atol=1e-14)
                    self.assertTrue(ok.all(), f"vs host: {scalar[~ok][:3]} vs {want[~ok][:3]}")
                else:
                    for got in (scalar, matrix) + ((fast,) if fast is not None else ()):
                        ok = same_bits(got, want[: len(got)]) | np.isclose(got, want[: len(got)], rtol=1e-9, atol=1e-12)
                        if case["tag"] == "rounding":
                            ok |= np.abs(got - want[: len(got)]) <= 1
                        self.assertTrue(ok.all(), f"{got[~ok][:3]} vs {want[: len(got)][~ok][:3]}")
                exact += int(same_bits(scalar, want).all() and same_bits(matrix, want).all())
        print(f"\n{self.arch} ({self.cpu}): {len(self.corpus.cases)} cases, {exact} bit-identical to this machine",
              file=sys.stderr)


class Aarch64(EmulatedBase, unittest.TestCase):
    arch, cpu = "aarch64", "max"


class Aarch64NoFcma(EmulatedBase, unittest.TestCase):
    arch, cpu = "aarch64", "cortex-a57"


class Riscv64(EmulatedBase, unittest.TestCase):
    arch, cpu = "riscv64", "max"


class PiObjectsAarch64(unittest.TestCase):
    """examples/claude/obj: the pi kernels written as AArch64 ELF objects (ty="arm"), linked with
    test_pi.c by the cross compiler and run under qemu (needs a host library with the `obj` feature)"""

    def test_pi(self):
        if not ENABLED:
            self.skipTest("set SYMJIT_EMULATED=1 to run")
        missing = [t for t in ("qemu-aarch64", "aarch64-linux-gnu-gcc") if shutil.which(t) is None]
        if missing:
            self.skipTest(f"missing: {', '.join(missing)}")
        x, y, n, k = sp.symbols("x y n k")

        def arctan_series(v):  # as examples/claude/obj/pi.py
            s = v
            for i in range(1, 21):
                coef = -(1 + 2 * i) if i & 1 == 1 else 1 + 2 * i
                s += v ** abs(coef) / coef
            return s

        functions = {
            "pi_series": [4 * (4 * arctan_series(x) - arctan_series(y))],
            "pi_atan": [4 * (4 * sp.atan(x) - sp.atan(y))],
            "pi_bbp": [sp.Sum((4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)) / 16**k,
                              (k, 0, n))],
        }
        with tempfile.TemporaryDirectory() as d:
            for name, exprs in functions.items():
                states = [n] if name == "pi_bbp" else [x, y]
                try:
                    compile_func(states, exprs, ty="arm").write_obj(os.path.join(d, name), format="elf")
                except ValueError as e:
                    if "without the `obj` feature" in str(e) or "cannot write object files" in str(e):
                        self.skipTest("symjit was built without the `obj` feature")
                    raise
            exe = os.path.join(d, "test_pi")
            src = os.path.join(ROOT, "examples", "claude", "obj", "test_pi.c")
            subprocess.run(["aarch64-linux-gnu-gcc", "-Wall", "-O2", "-I", d, src,
                            *[os.path.join(d, f"{name}.o") for name in functions], "-lm", "-o", exe], check=True)
            p = subprocess.run(["qemu-aarch64", "-L", "/usr/aarch64-linux-gnu", exe], capture_output=True, text=True,
                               timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("ok!", p.stdout)


if __name__ == "__main__":
    unittest.main()
