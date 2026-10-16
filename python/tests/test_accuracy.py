"""Accuracy of the compiled elementary functions, measured in ulps against mpmath (60 digits).

Every function is sampled on its domain (`accuracy_cases.CASES`); the scalar and the
vectorized (SIMD) kernels must both stay within the bound of the function and must return
bit-identical results. A function whose accuracy is below the usual 1--4 ulp of a good libm is
kept in the table with a loose bound and a note (and its actual accuracy is in the report:
`python examples/claude/accuracy.py`).

Set SYMJIT_PYTHON_PATH to test another build of the package.
"""

import os
import sys
import unittest
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

from accuracy_cases import CASES, abs_error, mp, ulp_error, x  # noqa: E402
from symjit import compile_func  # noqa: E402

NSAMPLES = int(os.environ.get("SYMJIT_ACCURACY_SAMPLES", "300"))
OPTIONS = [dict(), dict(fastmath=False), dict(use_simd=False, opt_level=0), dict(enable_simd512=True)]


def max_error(name, f, pts, exact, metric):
    err = ulp_error if metric == "ulp" else abs_error
    got = np.ravel(f(pts))
    return max(err(g, e) for g, e in zip(got, exact)), got


class Accuracy(unittest.TestCase):
    def test_accuracy(self):
        warnings.simplefilter("ignore")
        for name, (sym, ref, sampler, bound, metric, note) in CASES.items():
            pts = sampler(NSAMPLES, np.random.default_rng(abs(hash(name)) % 1000 + 1))
            pts = np.array(sorted(set(pts.tolist())))
            exact = [ref(mp.mpf(float(p))) for p in pts]

            results = {}
            for options in OPTIONS:
                with self.subTest(function=name, options=options):
                    f = compile_func([x], [sym(x)], **options)
                    worst, vec = max_error(name, f, pts, exact, metric)
                    scalar = np.array([np.ravel(f(float(p)))[0] for p in pts])
                    results[str(options)] = vec
                    self.assertLessEqual(
                        worst, bound, f"{name} {options}: {worst:.1f} > {bound} ({metric}); {note}"
                    )
                    np.testing.assert_array_equal(scalar, vec, err_msg=f"{name} {options}: scalar != vectorized")

            # the kernels of different SIMD widths and optimization levels agree with each other
            with self.subTest(function=name, check="all kernels agree"):
                first = next(iter(results.values()))
                for other in results.values():
                    np.testing.assert_array_equal(first, other)


if __name__ == "__main__":
    unittest.main()
