#!/usr/bin/env python3
"""Symjit performance benchmarks with a per-host baseline.

    python bench/run.py                     # run everything, compare with this host's baseline
    python bench/run.py --quick             # smaller problems (about 10x faster)
    python bench/run.py eos nbody           # selected workloads (see --list)
    python bench/run.py --save              # run and store the results as this host's baseline
    python bench/run.py --json out.json     # also write the results
    python bench/run.py --opt-level 3 --simd512 --no-threads   # symjit options

Every workload is checked against a reference implementation (numpy, a Python function under
scipy, the bytecode interpreter, or Symbolica), then timed: the run time is the minimum over
`--samples` samples of about `--target` seconds each, which is far less noisy than the mean.
Reported per workload: compile time, symjit and reference run times, speedup, throughput, and
the size of the generated machine code.

Comparison with the baseline (bench/baselines/<host>.json, where <host> is the machine and
CPU, so x86-64 and Apple silicon numbers are kept apart): a run time more than `--threshold`
(default 20%) slower, or code more than 10% larger, is a regression; the exit status is then 1.
Compile times are compared with a looser threshold (2x) because they are short and noisy.
A baseline only makes sense for the same options and problem sizes, so they are stored with
it and a mismatch is an error.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(HERE, "..", "python")))

import symjit  # noqa: E402

from workloads import WORKLOADS  # noqa: E402

CODE_GROWTH = 0.10
COMPILE_FACTOR = 2.0


def cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        with open("/proc/cpuinfo") as fd:
            for line in fd:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def host_key():
    s = f"{platform.system()}-{platform.machine()}-{cpu_name()}".lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def best_time(fn, samples, target):
    """Minimum time of one call of fn over `samples` samples of about `target` seconds."""
    t0 = time.perf_counter()
    fn()  # warm-up (and lazy compilation)
    once = max(time.perf_counter() - t0, 1e-7)
    inner = max(1, int(target / once))
    best = float("inf")
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter() - t0) / inner)
    return best


def code_size(funcs):
    total = 0
    for f in funcs:
        c = getattr(f, "compiler", f)
        for kernel in ("ker-scalar-size", "ker-simd-size", "ker-fast-size"):
            total += c.measure(kernel)
    return total


def run_workload(name, setup, args, options):
    case = setup(args.quick, options)
    result = {"compile_ms": case.compile_ms, "work": case.work, "unit": case.unit, "ref_name": case.ref_name}
    got = case.run()
    if case.reference is not None:
        want = case.reference()
        if case.check is not None:
            case.check(got, want)
        result["ref_s"] = best_time(case.reference, args.samples, args.target)
    result["symjit_s"] = best_time(case.run, args.samples, args.target)
    if name == "compile-large":
        result["compile_ms"] = 1e3 * result["symjit_s"]
    result["code_bytes"] = code_size(case.funcs)
    return result


def fmt_time(s):
    if s is None:
        return "-"
    for unit, scale in (("s", 1), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if s >= scale:
            return f"{s / scale:.3g} {unit}"
    return f"{s / 1e-9:.3g} ns"


def compare(name, r, base, threshold):
    """Returns (notes, is_regression)."""
    if base is None:
        return "new", False
    notes, bad = [], False
    ratio = r["symjit_s"] / base["symjit_s"]
    if ratio > 1 + threshold:
        notes.append(f"SLOWER {ratio:.2f}x")
        bad = True
    elif ratio < 1 / (1 + threshold):
        notes.append(f"faster {1 / ratio:.2f}x")
    else:
        notes.append(f"{ratio:.2f}x")
    if base.get("code_bytes"):
        growth = r["code_bytes"] / base["code_bytes"] - 1
        if growth > CODE_GROWTH:
            notes.append(f"CODE +{100 * growth:.0f}%")
            bad = True
        elif growth != 0:
            notes.append(f"code {100 * growth:+.1f}%")
    if name != "compile-large" and base.get("compile_ms", 0) > 5:
        if r["compile_ms"] > COMPILE_FACTOR * base["compile_ms"]:
            notes.append(f"COMPILE {r['compile_ms'] / base['compile_ms']:.1f}x")
            bad = True
    return ", ".join(notes), bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workloads", nargs="*", help="workloads to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the workloads")
    parser.add_argument("--quick", action="store_true", help="smaller problems")
    parser.add_argument("--save", action="store_true", help="store the results as this host's baseline")
    parser.add_argument("--baseline", help="baseline file (default: bench/baselines/<host>[-quick].json)")
    parser.add_argument("--json", help="write the results to this file")
    parser.add_argument("--threshold", type=float, default=0.20, help="run-time regression threshold (default 0.2)")
    parser.add_argument("--samples", type=int, default=7, help="timing samples per measurement (default 7)")
    parser.add_argument("--target", type=float, default=0.2, help="seconds per sample (default 0.2)")
    parser.add_argument("--opt-level", type=int, default=2)
    parser.add_argument("--simd512", action="store_true")
    parser.add_argument("--no-threads", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\n".join(WORKLOADS))
        return 0

    unknown = [w for w in args.workloads if w not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workloads {unknown}; see --list")
    names = args.workloads or list(WORKLOADS)

    options = dict(opt_level=args.opt_level, enable_simd512=args.simd512, use_threads=not args.no_threads)
    host = host_key()
    baseline_path = args.baseline or os.path.join(HERE, "baselines", f"{host}{'-quick' if args.quick else ''}.json")
    baseline = None
    if os.path.exists(baseline_path) and not args.save:
        with open(baseline_path) as fd:
            baseline = json.load(fd)
        if baseline["options"] != options:
            sys.exit(f"{baseline_path} was recorded with options {baseline['options']}, not {options}")

    print(f"host {host}, numpy {np.__version__}, options {options}{', quick' if args.quick else ''}")
    print(f"baseline: {baseline_path if baseline else 'none'}\n")
    header = f"{'workload':24s} {'compile':>9s} {'symjit':>10s} {'reference':>10s} {'speedup':>8s}  " \
             f"{'throughput':>17s} {'code':>8s}  vs baseline"
    print(header)
    print("-" * len(header))

    results, regressions, failures = {}, [], []
    for name in names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = run_workload(name, WORKLOADS[name], args, options)
        except Exception as e:  # a failing workload is reported, the others still run
            failures.append(name)
            print(f"{name:24s} FAILED: {type(e).__name__}: {str(e).strip().splitlines()[0][:80] if str(e).strip() else ''}")
            if os.environ.get("SYMJIT_BENCH_TRACEBACK"):
                traceback.print_exc()
            continue
        results[name] = r
        speedup = f"{r['ref_s'] / r['symjit_s']:.2f}x" if "ref_s" in r else "-"
        rate = r["work"] / r["symjit_s"]
        thr = next(f"{rate / k:.3g} {p}{r['unit']}/s" for k, p in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, "")) if rate >= k or k == 1)
        notes, bad = compare(name, r, (baseline or {}).get("results", {}).get(name), args.threshold)
        if bad:
            regressions.append(name)
        code = f"{r['code_bytes'] / 1024:.1f}K" if r["code_bytes"] else "-"
        print(f"{name:24s} {r['compile_ms']:7.1f}ms {fmt_time(r['symjit_s']):>10s} {fmt_time(r.get('ref_s')):>10s} "
              f"{speedup:>8s}  {thr:>17s} {code:>8s}  {notes}")
        sys.stdout.flush()

    print()
    if args.json or args.save:
        doc = {"host": host, "cpu": cpu_name(), "options": options, "quick": args.quick,
               "numpy": np.__version__, "date": time.strftime("%Y-%m-%d"), "results": results}
        if args.json:
            with open(args.json, "w") as fd:
                json.dump(doc, fd, indent=1)
        if args.save:
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            with open(baseline_path, "w") as fd:
                json.dump(doc, fd, indent=1)
            print(f"baseline saved to {baseline_path}")
    if failures:
        print(f"failed: {', '.join(failures)}")
    if regressions:
        print(f"regressions (> {100 * args.threshold:.0f}% slower, code > {100 * CODE_GROWTH:.0f}% larger, "
              f"or compile > {COMPILE_FACTOR:.0f}x): {', '.join(regressions)}")
    return 1 if failures or regressions else 0


if __name__ == "__main__":
    sys.exit(main())
