"""GL297: the cost per sample point of a sweep over all 31 evaluators, single-threaded.

GammaLoop evaluates the 31 summed evaluators of GL297 for every sample point. When they
are called one after another for each point (point-major, b = 1), their combined code
(13-32 MB) does not stay in the caches. Evaluating each evaluator over a block of b points
before moving on to the next one (evaluator-major) reuses its code b times and, with the
SIMD kernels, evaluates 4 (AVX2) or 8 (AVX-512) points per pass.

    for block of b points:
        for evaluator in evaluators:
            evaluator.evaluate_complex(block)      # one thread (use_threads=False)

This script reports the time per point of such sweeps for the scalar and SIMD kernels,
with and without compression, and checks every variant against the scalar kernel.

    python batch_sweep.py                   # b = 1, 8, 16, 32
    python batch_sweep.py -b 1 -b 64        # other block sizes
    python batch_sweep.py --json out.json   # also write the results

The evaluators and the input point come from GL297_study/ (summed_instructions/,
data/GL297_input.json); the b points of a block are the input point scaled by 1 + 1e-3 k.
"""

import argparse
import json
import os
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
    evaluators = []
    for f in files:
        with open(os.path.join(folder, f), encoding="utf-8") as fd:
            evaluators.append(fd.read())
    with open(os.path.join(STUDY, "data", "GL297_input.json")) as fd:
        data = json.load(fd)
    x = np.array([r["re"] + r["im"] * 1j for r in data])
    return evaluators, x


def block(x, b):
    return x[None, :] * (1 + 1e-3 * np.arange(b))[:, None]


def sweep_time(funcs, X, samples):
    """seconds per point of one evaluator-major sweep over the block X (best of `samples`)"""
    for f in funcs:  # warm-up
        f.evaluate_complex(X)
    best = float("inf")
    for _ in range(samples):
        t0 = time.perf_counter()
        for f in funcs:
            f.evaluate_complex(X)
        best = min(best, time.perf_counter() - t0)
    return best / X.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-b", type=int, action="append", help="block sizes (default 1, 8, 16, 32)")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    sizes = args.b or [1, 8, 16, 32]

    evaluators, x = load()
    n = x.size
    print(f"{len(evaluators)} evaluators, {n} complex inputs; times in ms per point for a sweep over all of them")
    reference = None
    results = {}
    head = f"{'variant':24s} {'code MB':>8s}  " + "  ".join(f"{'b=' + str(b):>7s}" for b in sizes) + "   max rel diff"
    print(head)
    print("-" * len(head))
    for name in args.variants:
        opts = dict(dtype="complex128", num_params=n, fast_complex=True, use_threads=False) | VARIANTS[name]
        t0 = time.perf_counter()
        funcs = [symjit.compile_evaluator(ev, **opts) for ev in evaluators]
        X = block(x, max(sizes))
        outs = np.concatenate([f.evaluate_complex(X) for f in funcs], axis=1)  # also compiles lazily
        compile_s = time.perf_counter() - t0
        if reference is None:
            reference = outs
        diff = float(np.max(np.abs(outs - reference)) / np.max(np.abs(reference)))
        code = sum(f.measure("ker-scalar-size") + f.measure("ker-simd-size") for f in funcs) / 2**20
        times = [1e3 * sweep_time(funcs, block(x, b), args.samples) for b in sizes]
        results[name] = dict(code_MB=code, compile_s=compile_s, ms_per_point=dict(zip(map(str, sizes), times)),
                             max_rel_diff=diff)
        print(f"{name:24s} {code:8.1f}  " + "  ".join(f"{t:7.3f}" for t in times) + f"   {diff:.1e}")
        sys.stdout.flush()
    if args.json:
        with open(args.json, "w") as fd:
            json.dump(results, fd, indent=1)


if __name__ == "__main__":
    main()
