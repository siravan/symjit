#!/usr/bin/env python3
"""
Runs the pyhpc-benchmarks with numpy and symjit and compares the performance.

    python run.py                       # all benchmarks, default sizes
    python run.py eos -s 65536 -s 1048576
    python run.py isoneutral tke --simd512

The benchmarks (equation of state, isoneutral mixing, and turbulent kinetic energy) come
from https://github.com/dionhaefner/pyhpc-benchmarks. The `size` is the approximate number
of grid points of the 3D arrays, as in the original.

For every size, the script first checks that the symjit and numpy results agree, and then
times both implementations (the mean and the standard deviation of `repetitions` calls,
after a warm-up call). Compilation time of the symjit kernels is reported separately.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eos
import isoneutral
import tke

BENCHMARKS = {"eos": eos, "isoneutral": isoneutral, "tke": tke}
DEFAULT_SIZES = [2**n for n in range(12, 21, 2)]


def as_tuple(out):
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def timeit(func, make_inputs, repetitions, target_time):
    """Times `func(*inputs)`; the number of repetitions is estimated if not given"""
    inputs = make_inputs()
    func(*inputs)  # warm-up

    if repetitions is None:
        t0 = time.perf_counter()
        func(*inputs)
        repetitions = int(min(1000, max(3, target_time / (time.perf_counter() - t0))))

    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        func(*inputs)
        times.append(time.perf_counter() - t0)

    return np.mean(times) * 1e3, np.std(times) * 1e3, repetitions


def run_benchmark(module, sizes, options, repetitions, target_time, check):
    print(f"\n{module.NAME}")

    t0 = time.perf_counter()
    run_symjit = module.setup_symjit(**options)
    print(f"  symjit kernels compiled in {1000 * (time.perf_counter() - t0):.0f} ms")

    print(f"  {'size':>9s}  {'grid':>16s}  {'numpy (ms)':>16s}  {'symjit (ms)':>16s}  {'speedup':>8s}")

    for size in sizes:
        shape = module.generate_inputs(size)[0].shape[:3]

        if check:
            want = as_tuple(module.run_numpy(*module.generate_inputs(size)))
            got = as_tuple(run_symjit(*module.generate_inputs(size)))
            for w, g in zip(want, got):
                np.testing.assert_allclose(g, w, **module.TOLERANCE)

        make = lambda: module.generate_inputs(size)
        m_np, s_np, n_np = timeit(module.run_numpy, make, repetitions, target_time)
        m_sj, s_sj, n_sj = timeit(run_symjit, make, repetitions, target_time)

        grid = "x".join(map(str, shape))
        print(
            f"  {size:9d}  {grid:>16s}  {m_np:9.2f} ±{s_np:5.2f}  {m_sj:9.2f} ±{s_sj:5.2f}  {m_np / m_sj:7.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="*", choices=[[]] + list(BENCHMARKS), default=[], metavar="BENCHMARK",
                        help="eos, isoneutral, or tke (default: all)")
    parser.add_argument("-s", "--size", type=int, action="append", help="grid size (repeatable)")
    parser.add_argument("-r", "--repetitions", type=int, help="fixed number of repetitions")
    parser.add_argument("--target-time", type=float, default=1.0, help="seconds per measurement (default 1)")
    parser.add_argument("--no-check", action="store_true", help="skip comparing the results")
    parser.add_argument("--simd512", action="store_true", help="enable AVX512 (if available)")
    parser.add_argument("--no-threads", action="store_true", help="single-threaded symjit kernels")
    parser.add_argument("--opt-level", type=int, default=2, help="symjit optimization level (default 2)")
    args = parser.parse_args()

    options = dict(enable_simd512=args.simd512, use_threads=not args.no_threads, opt_level=args.opt_level)
    names = args.benchmarks or list(BENCHMARKS)

    print(f"numpy {np.__version__}, symjit options: {options}")
    for name in names:
        run_benchmark(BENCHMARKS[name], args.size or DEFAULT_SIZES, options, args.repetitions,
                      args.target_time, not args.no_check)


if __name__ == "__main__":
    main()
