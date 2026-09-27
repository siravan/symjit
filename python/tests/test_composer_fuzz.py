"""Differential fuzz tests for the `Composer` interface.

Random programs are built with `Composer` -- SSA temporaries, in-place `assign` to temporaries,
selects (`join`), if/else blocks, and counted loops (`append_for`) -- and compiled with several
option combinations. The results, in scalar and vectorized (SIMD) calls, are compared with a
numpy interpreter of the same program.

Families:
    straight   straight-line code with in-place updates (compiled with `direct=True` and `direct=False`)
    branches   if/else blocks merged with `join`, and selects on comparisons
    loops      counted loops (`append_for`) that update an accumulator (`direct=True` only: a
               backward branch does not work with `direct=False`)

Every family runs in a subprocess: a bug that aborts the interpreter (a panic in the compiler)
or hangs is reported as a failure of that test instead of killing the whole test run.

    python -m unittest python/tests/test_composer_fuzz.py -v

Environment variables (besides SYMJIT_PYTHON_PATH, which selects the package to test):
    SYMJIT_FUZZ_SEEDS   programs per family (default 60)
    SYMJIT_FUZZ_START   first seed (default 0)
    SYMJIT_FUZZ_TIMEOUT seconds allowed for a family (default 600)
"""

import os
import random
import subprocess
import sys
import unittest
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..")))

from symjit import Composer, compile_composer  # noqa: E402

SEEDS = int(os.environ.get("SYMJIT_FUZZ_SEEDS", "60"))
START = int(os.environ.get("SYMJIT_FUZZ_START", "0"))
TIMEOUT = int(os.environ.get("SYMJIT_FUZZ_TIMEOUT", "600"))

NPARAMS = 3
NPOINTS = 33

# ---------------------------------------------------------------------------- operations
# name -> (emit(builder, *slots), numpy reference). The functions are made total on the
# reals (e.g., division by 1 + y^2) so that the reference is always finite.
BINARY = {
    "add": (lambda b, x, y: b.fadd(x, y), lambda x, y: x + y),
    "sub": (lambda b, x, y: b.fsub(x, y), lambda x, y: x - y),
    "mul": (lambda b, x, y: b.fmul(x, y), lambda x, y: x * y),
    "div": (lambda b, x, y: b.fdiv(x, b.fadd(b.square(y), b.constant(1.0))), lambda x, y: x / (y * y + 1.0)),
    "min": (lambda b, x, y: b.min(x, y), np.minimum),
    "max": (lambda b, x, y: b.max(x, y), np.maximum),
}
UNARY = {
    "neg": (lambda b, x: b.neg(x), lambda x: -x),
    "abs": (lambda b, x: b.abs(x), np.abs),
    "sqrt": (lambda b, x: b.sqrt(b.abs(x)), lambda x: np.sqrt(np.abs(x))),
    "square": (lambda b, x: b.square(x), np.square),
    "cube": (lambda b, x: b.cube(x), lambda x: x**3),
    "recip": (lambda b, x: b.recip(b.fadd(b.square(x), b.constant(1.0))), lambda x: 1.0 / (x * x + 1.0)),
    "sin": (lambda b, x: b.sin(x), np.sin),
    "cos": (lambda b, x: b.cos(x), np.cos),
    "exp": (lambda b, x: b.exp(b.neg(b.square(x))), lambda x: np.exp(-x * x)),
    "floor": (lambda b, x: b.floor(x), np.floor),
}
UPDATE = {"add": BINARY["add"], "sub": BINARY["sub"], "mul": BINARY["mul"]}
COMPARE = {
    "lt": (lambda b, x, y: b.lt(x, y), lambda x, y: x < y),
    "leq": (lambda b, x, y: b.leq(x, y), lambda x, y: x <= y),
    "gt": (lambda b, x, y: b.gt(x, y), lambda x, y: x > y),
    "geq": (lambda b, x, y: b.geq(x, y), lambda x, y: x >= y),
}


# ---------------------------------------------------------------------------- programs
# A program is a list of statements on named values:
#   ("bin", target, op, a, b)          target = op(a, b)
#   ("un", target, op, a)              target = op(a)
#   ("upd", t, op, a)                  t = op(t, a)      (in place)
#   ("sel", target, cmp, a, b, x, y)   target = a cmp b ? x : y
#   ("if", target, cmp, a, b, (opx, x1, x2), (opy, y1, y2))
#                                      if a cmp b: target = opx(x1, x2) else: target = opy(y1, y2)
#   ("for", K, [("upd", acc, op, a), ...])   K iterations (K >= 1)
# Values are ("arg", i), ("const", c), or ("t", k) (the k-th target).


class Program:
    def __init__(self, seed, family):
        rng = random.Random(f"composer-{family}-{seed}")
        self.family = family
        self.vals = [("arg", i) for i in range(NPARAMS)] + [("const", round(rng.uniform(-3, 3), 2)) for _ in range(3)]
        self.steps = []
        self.temps = []
        self.outs = []
        n = rng.randint(6, 30)

        def src():
            return rng.choice(self.vals + self.temps)

        def new_temp():
            t = ("t", len(self.temps))
            self.temps.append(t)
            return t

        for _ in range(n):
            r = rng.random()
            if family == "loops" and r < 0.2 and self.temps:
                acc = rng.choice(self.temps)
                body = [("upd", acc, rng.choice(list(UPDATE)), src()) for _ in range(rng.randint(1, 3))]
                self.steps.append(("for", rng.randint(1, 5), body))
            elif family == "branches" and r < 0.25:
                args = (rng.choice(list(COMPARE)), src(), src(), src(), src())  # the operands first,
                self.steps.append(("sel", new_temp(), *args))  # then the target
            elif family == "branches" and r < 0.5:
                arm = lambda: (rng.choice(list(BINARY)), src(), src())
                args = (rng.choice(list(COMPARE)), src(), src(), arm(), arm())
                self.steps.append(("if", new_temp(), *args))
            elif r < 0.18 and self.temps:
                self.steps.append(("upd", rng.choice(self.temps), rng.choice(list(UPDATE)), src()))
            elif r < 0.6:
                args = (rng.choice(list(BINARY)), src(), src())
                self.steps.append(("bin", new_temp(), *args))
            else:
                args = (rng.choice(list(UNARY)), src())
                self.steps.append(("un", new_temp(), *args))

        self.outs = [rng.choice(self.temps) for _ in range(2)]

    # -- the Composer version
    def compose(self):
        cp = Composer(NPARAMS, len(self.outs))
        slots = {}
        for v in self.vals:
            slots[v] = cp.arg(v[1]) if v[0] == "arg" else cp.constant(v[1])

        for st in self.steps:
            kind = st[0]
            if kind == "bin":
                _, t, op, a, b = st
                slots[t] = BINARY[op][0](cp, slots[a], slots[b])
            elif kind == "un":
                _, t, op, a = st
                slots[t] = UNARY[op][0](cp, slots[a])
            elif kind == "upd":
                _, t, op, a = st
                cp.assign(slots[t], UPDATE[op][0](cp, slots[t], slots[a]))
            elif kind == "sel":
                _, t, cmp, a, b, x, y = st
                slots[t] = cp.join(COMPARE[cmp][0](cp, slots[a], slots[b]), slots[x], slots[y])
            elif kind == "if":
                _, t, cmp, a, b, (opx, x1, x2), (opy, y1, y2) = st
                cond = COMPARE[cmp][0](cp, slots[a], slots[b])
                bx, by = cp.new_block(), cp.new_block()
                ex = BINARY[opx][0](bx, slots[x1], slots[x2])
                ey = BINARY[opy][0](by, slots[y1], slots[y2])
                cp.append_if_else(cond, bx, by)
                slots[t] = cp.join(cond, ex, ey)
            else:  # for
                _, k, body = st
                block = cp.new_block()
                for _, acc, op, a in body:
                    block.assign(slots[acc], UPDATE[op][0](block, slots[acc], slots[a]))
                cp.append_for(cp.new_temp(), 0, k, block)

        for i, o in enumerate(self.outs):
            cp.assign(cp.out(i), slots[o])
        return cp

    # -- the numpy interpreter
    def reference(self, X):
        env = {("arg", i): X[i].copy() for i in range(NPARAMS)}
        for v in self.vals:
            if v[0] == "const":
                env[v] = np.full(X.shape[1], v[1])

        for st in self.steps:
            kind = st[0]
            if kind == "bin":
                _, t, op, a, b = st
                env[t] = BINARY[op][1](env[a], env[b])
            elif kind == "un":
                _, t, op, a = st
                env[t] = UNARY[op][1](env[a])
            elif kind == "upd":
                _, t, op, a = st
                env[t] = UPDATE[op][1](env[t], env[a])
            elif kind == "sel":
                _, t, cmp, a, b, x, y = st
                env[t] = np.where(COMPARE[cmp][1](env[a], env[b]), env[x], env[y])
            elif kind == "if":
                _, t, cmp, a, b, (opx, x1, x2), (opy, y1, y2) = st
                env[t] = np.where(COMPARE[cmp][1](env[a], env[b]), BINARY[opx][1](env[x1], env[x2]),
                                  BINARY[opy][1](env[y1], env[y2]))
            else:
                _, k, body = st
                for _ in range(k):
                    for _, acc, op, a in body:
                        env[acc] = UPDATE[op][1](env[acc], env[a])
        return np.array([env[o] for o in self.outs])


# ---------------------------------------------------------------------------- the worker
CONFIGS = {
    "straight": [dict(), dict(direct=False), dict(use_simd=False), dict(fastmath=False), dict(enable_simd512=True),
                 dict(direct=False, use_simd=False)],
    "branches": [dict(), dict(direct=False), dict(use_simd=False), dict(simd_branch=False), dict(fastmath=False),
                 dict(enable_simd512=True)],
    "loops": [dict(), dict(use_simd=False), dict(simd_branch=False), dict(fastmath=False), dict(enable_simd512=True)],
}


def worker(family, start, count):
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    X = np.random.default_rng(7).uniform(-2, 2, (NPARAMS, NPOINTS))
    failures = 0

    for seed in range(start, start + count):
        program = Program(seed, family)
        ref = program.reference(X)
        if not np.all(np.isfinite(ref)) or np.max(np.abs(ref)) > 1e12:
            continue
        cp = program.compose()

        for options in CONFIGS[family]:
            print(f"# seed {seed} {options}", flush=True)
            try:
                f = compile_composer(cp, **options)
                vec = np.array(f.evaluate(X.T.copy())).reshape(NPOINTS, -1).T
                scalar = np.array([np.ravel(f(*X[:, i])) for i in range(NPOINTS)]).T
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {family} seed {seed} {options}: {type(e).__name__}: {str(e)[:100]}", flush=True)
                failures += 1
                continue
            for kind, got in (("vectorized", vec), ("scalar", scalar)):
                err = np.abs(got - ref) / (1 + np.abs(ref))
                if not np.all(err < 1e-8):
                    print(f"FAIL {family} seed {seed} {options} {kind}: relative error {err.max():.3g}", flush=True)
                    failures += 1
                    break

    print(f"DONE {family} failures={failures}", flush=True)
    return failures


class ComposerFuzz(unittest.TestCase):
    def run_family(self, family):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", family, str(START), str(SEEDS)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            last = [l for l in out.splitlines() if l.startswith("# seed")][-1:]
            self.fail(f"{family}: no result in {TIMEOUT} s (hang?); last case: {last}")

        fails = [l for l in p.stdout.splitlines() if l.startswith("FAIL")]
        cases = [l for l in p.stdout.splitlines() if l.startswith("# seed")]
        if p.returncode != 0 and not fails:
            tail = "\n".join((p.stderr or "").strip().splitlines()[-4:])
            self.fail(f"{family}: the worker died with code {p.returncode} (a compiler panic aborts the "
                      f"process); last case: {cases[-1:]}\n{tail}")
        self.assertFalse(fails, f"{len(fails)} failures:\n" + "\n".join(fails[:10]))
        self.assertIn(f"DONE {family}", p.stdout)

    def test_straight_line(self):
        self.run_family("straight")

    def test_branches(self):
        self.run_family("branches")

    def test_loops(self):
        self.run_family("loops")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.exit(1 if worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])) else 0)
    unittest.main()
