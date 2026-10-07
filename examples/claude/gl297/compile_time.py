"""GL297: compilation (code generation) time of the 31 summed evaluators.

For each kernel variant, every evaluator of GL297_study/summed_instructions is compiled
from its instruction text (`compile_evaluator`: parsing, translation, optimization and the
kernels; the Symbolica bridge compiles the SIMD kernel eagerly, so it is included). The
`first call` column is the extra time of the first batched call (8 points) over a warm
one, which would include any lazy compilation (it is ~0 for compile_evaluator).
The best of `--repeats` runs is reported, with the peak resident memory of the process.

    python compile_time.py                       # all variants
    python compile_time.py --variants scalar "SIMD f64x8 compressed"
    python compile_time.py --json out.json

Run it with SYMJIT_PYTHON_PATH pointing at another build to compare versions.
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.environ.get("GL297_STUDY", os.path.join(HERE, "..", "..", "..", "GL297_study"))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..", "..", "..", "python")))

import symjit  # noqa: E402

VARIANTS = {
    "scalar": dict(use_simd=False, compress=False),
    "scalar compressed": dict(use_simd=False, compress=True, opt_level=3),
    "SIMD f64x4": dict(use_simd=True, enable_simd512=False, compress=False),
    "SIMD f64x4 compressed": dict(use_simd=True, enable_simd512=False, compress=True, opt_level=3),
    "SIMD f64x8": dict(use_simd=True, enable_simd512=True, compress=False),
    "SIMD f64x8 compressed": dict(use_simd=True, enable_simd512=True, compress=True, opt_level=3),
}


def load():
    folder = os.path.join(STUDY, "summed_instructions")
    files = sorted(f for f in os.listdir(folder) if f.startswith("instructions_"))
    texts = []
    for f in files:
        with open(os.path.join(folder, f), encoding="utf-8") as fd:
            texts.append(fd.read())
    with open(os.path.join(STUDY, "data", "GL297_input.json")) as fd:
        data = json.load(fd)
    x = np.array([r["re"] + r["im"] * 1j for r in data])
    return texts, x


def measure(texts, x, opts, simd):
    """(seconds for compile_evaluator, seconds for the SIMD kernels) over all evaluators"""
    X = x[None, :] * (1 + 1e-3 * np.arange(8))[:, None]
    t_compile = t_simd = 0.0
    for text in texts:
        t0 = time.perf_counter()
        f = symjit.compile_evaluator(text, **opts)
        t1 = time.perf_counter()
        t_compile += t1 - t0
        if simd:
            f.evaluate_complex(X)  # compiles the SIMD kernel
            t2 = time.perf_counter()
            f.evaluate_complex(X)
            t3 = time.perf_counter()
            t_simd += (t2 - t1) - (t3 - t2)
        del f
    return t_compile, t_simd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()

    texts, x = load()
    print(f"{len(texts)} evaluators, {sum(len(t) for t in texts) / 2**20:.1f} MB of instructions; "
          f"compilation time in seconds (best of {args.repeats})")
    head = f"{'variant':24s} {'compile':>8s} {'1st call':>8s} {'total':>8s}"
    print(head)
    print("-" * len(head))
    results = {}
    for name in args.variants:
        kw = VARIANTS[name]
        opts = dict(dtype="complex128", num_params=x.size, fast_complex=True, use_threads=False) | kw
        best = None
        for _ in range(args.repeats):
            r = measure(texts, x, opts, kw["use_simd"])
            if best is None or sum(r) < sum(best):
                best = r
        results[name] = dict(compile_s=best[0], simd_s=best[1], total_s=sum(best))
        print(f"{name:24s} {best[0]:8.2f} {best[1]:8.2f} {sum(best):8.2f}")
        sys.stdout.flush()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak resident memory: {peak:.0f} MB")
    results["peak_rss_MB"] = peak
    if args.json:
        with open(args.json, "w") as fd:
            json.dump(results, fd, indent=1)


if __name__ == "__main__":
    main()
