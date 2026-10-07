"""Runs examples/pendulum.py against several builds of symjit and compares the timings and
the output (the `BENCH {...}` line the example prints last).

    python pendulum_bench.py                                   # the package in python/
    python pendulum_bench.py -v 2.26.1=/path/to/pkg -v current=python/
    python pendulum_bench.py --runs 3 --save base.json         # record a baseline
    python pendulum_bench.py --baseline base.json              # compare with it

Each version is a directory that contains the `symjit` package (e.g. an unpacked wheel or the
`python/` folder of a snapshot); it is put first on sys.path, and the script checks that symjit
was imported from it. Versions run one after another (never in parallel, to keep the timings
clean); with --runs N the best time of N runs is kept. Extra arguments after `--` are passed
to pendulum.py (e.g. `-- --no-simd --opt_level 3`).

With --baseline, a timing more than --tolerance (default 20%) slower than the baseline, or a
changed nfev / energy deviation / final state, is reported, and the exit status is 1. Note that
the 25-link pendulum is chaotic: a change in the last bit of the mass matrix (e.g. a different
order of operations after a CSE change) changes nfev and the energy deviation slightly, without
being an error; pendulum_check.py compares the kernels directly with lambdify.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
REPO = os.path.dirname(EXAMPLES)

# timings compared with the baseline (smaller is better)
TIMINGS = ["compile_sec", "mm_call_us", "force_call_us", "gradient_call_us", "integration_sec", "integration_per_nfev_us"]
# output that should not change
OUTPUTS = ["nfev", "energy_deviation_percent", "final_q"]

RUNNER = """
import runpy, sys
pkg, script = sys.argv[1], sys.argv[2]
sys.path.insert(0, pkg)
sys.path.insert(1, %r)
import symjit
print("SYMJIT_FILE " + symjit.__file__, flush=True)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
""" % EXAMPLES


def run_once(pkg, extra, timeout):
    cmd = [sys.executable, "-c", RUNNER, pkg, os.path.join(EXAMPLES, "pendulum.py"), *extra]
    env = dict(os.environ, MPLBACKEND="Agg")
    p = subprocess.run(cmd, cwd=EXAMPLES, env=env, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"pendulum.py failed with {pkg}:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    lines = p.stdout.splitlines()
    loaded = next(x for x in lines if x.startswith("SYMJIT_FILE "))[len("SYMJIT_FILE ") :]
    if not os.path.realpath(loaded).startswith(os.path.realpath(pkg)):
        raise RuntimeError(f"symjit was imported from {loaded}, not from {pkg}")
    return json.loads(next(x for x in reversed(lines) if x.startswith("BENCH "))[len("BENCH ") :])


def run_version(pkg, extra, runs, timeout):
    best = None
    for _ in range(runs):
        r = run_once(pkg, extra, timeout)
        if best is None:
            best = r
        else:
            for k in TIMINGS:
                best[k] = min(best[k], r[k])
            for k in OUTPUTS:
                if r[k] != best[k]:
                    print(f"  warning: {k} differs between runs: {r[k]} vs {best[k]}")
    return best


def same_output(a, b):
    return a["nfev"] == b["nfev"] and a["final_q"] == b["final_q"] and a["energy_deviation_percent"] == b["energy_deviation_percent"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="append", default=[], metavar="NAME=PATH", help="a version to run (repeatable)")
    parser.add_argument("--runs", type=int, default=1, help="runs per version, the best time is kept (default 1)")
    parser.add_argument("--save", metavar="FILE", help="write the results as a baseline")
    parser.add_argument("--baseline", metavar="FILE", help="compare with a baseline written by --save")
    parser.add_argument("--tolerance", type=float, default=0.20, help="allowed slowdown (default 0.20)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per run (default 3600)")
    parser.add_argument("extra", nargs="*", help="arguments for pendulum.py (after --)")
    args = parser.parse_args()

    versions = [v.split("=", 1) for v in args.version] or [["current", os.path.join(REPO, "python")]]
    results = {}
    for name, pkg in versions:
        print(f"running {name} ({pkg}) ...", flush=True)
        results[name] = run_version(os.path.abspath(pkg), args.extra, args.runs, args.timeout)

    names = list(results)
    w = max(12, *(len(x) for x in names))
    print()
    print(f"{'':{w}} {'compile s':>10} {'MM us':>8} {'force us':>9} {'grad us':>8} {'integr s':>9} {'us/nfev':>8} {'nfev':>7} {'energy dev %':>13}  final q[0]")
    for name in names:
        r = results[name]
        print(
            f"{name:{w}} {r['compile_sec']:10.3f} {r['mm_call_us']:8.1f} {r['force_call_us']:9.1f} {r['gradient_call_us']:8.1f} "
            f"{r['integration_sec']:9.2f} {r['integration_per_nfev_us']:8.1f} {r['nfev']:7d} {r['energy_deviation_percent']:13.6e}  {r['final_q'][0]:.12f}"
        )
    print()
    first = results[names[0]]
    for name in names[1:]:
        print(f"{name}: output {'identical to' if same_output(first, results[name]) else 'DIFFERS from'} {names[0]}")

    status = 0
    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        print(f"\ncompared with {args.baseline}:")
        for name in names:
            if name not in base:
                print(f"  {name}: not in the baseline")
                continue
            r, b = results[name], base[name]
            for k in TIMINGS:
                ratio = r[k] / b[k]
                if ratio > 1 + args.tolerance:
                    print(f"  {name}: {k} {r[k]:.4g} vs {b[k]:.4g} ({(ratio - 1) * 100:+.0f}%) SLOWER")
                    status = 1
            if not same_output(r, b):
                print(f"  {name}: output changed: nfev {b['nfev']} -> {r['nfev']}, energy deviation {b['energy_deviation_percent']:.6e} -> {r['energy_deviation_percent']:.6e} %")
                status = 1
        print("  OK" if status == 0 else "  REGRESSION")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nresults written to {args.save}")
    sys.exit(status)


if __name__ == "__main__":
    main()
