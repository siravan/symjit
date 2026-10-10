# This file is generated using claude code
"""Runs the seven physics examples with a sweep of symjit options and summarizes the results.

The examples (each prints `ok`/`FAIL` lines for its checks and `all checks passed` at the end):

    domino.py          dimer partition function, a sum over domino tilings      real, wide
    kepler.py          50,000 unrolled leapfrog steps of a Kepler orbit           real, deep (Composer)
    resistors.py       resistance of a resistor grid (Kirchhoff's sums)          real, wide
    kepler_series.py   Lagrange's and Bessel's series for Kepler's equation      real
    impedance.py       impedance of an RLC grid                                  complex, wide
    lee_yang.py        Ising partition function, Lee-Yang zeros                  complex, wide
    quantum.py         2,500 unrolled split-operator steps on a lattice          complex, deep (Composer)

Each run is a separate process (a crash or a time-out affects one run only). The sympy
expressions of the examples are built once: their JSON models are cached in a folder
(SYMJIT_EXAMPLE_CACHE, see util.compile_cached), so a sweep spends its time in symjit.

    python run_physics.py                          # all examples, all option sets, full sizes
    python run_physics.py --quick                  # smaller problems (about a minute)
    python run_physics.py --examples domino kepler --options default no-simd simd512
    python run_physics.py --options no-cse         # without CSE (not in the default sweep: slow)
    python run_physics.py --jobs 4 --json out.json # 4 runs at a time; write the results
    SYMJIT_PYTHON_PATH=/path/to/pkg python run_physics.py   # another build of the package

The exit status is 1 if any run fails, crashes or times out. Peak memory is per run (os.wait4);
with --jobs > 1 the runs share the machine, so their times are less meaningful.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # examples/claude/domino -> the repo

# name: (script, full-size arguments, quick arguments, complex)
EXAMPLES = {
    "domino": ("domino.py", [], ["-m", "5", "-n", "6"], False),
    "kepler": ("kepler.py", [], ["-N", "5000", "--orbits", "200"], False),
    "resistors": ("resistors.py", [], ["-m", "3", "-n", "4"], False),
    "kepler_series": ("kepler_series.py", [], ["-K", "120"], False),
    "impedance": ("impedance.py", [], ["-m", "3", "-n", "4"], True),
    "lee_yang": ("lee_yang.py", [], ["-m", "3", "-n", "4"], True),
    "quantum": ("quantum.py", [], ["-S", "12", "-N", "500", "--batch", "200"], True),
}

# name: (symjit options of util.py, complex examples only)
OPTIONS = {
    "default": ([], False),
    "no-simd": (["--no-simd"], False),
    "simd512": (["--simd512"], False),
    "no-threads": (["--no-threads"], False),
    "O0": (["--opt_level", "0"], False),
    "O1": (["--opt_level", "1"], False),
    "O3": (["--opt_level", "3"], False),
    "no-fastmath": (["--no-fastmath"], False),
    "compress": (["--compress"], False),
    "no-cse": (["--no-cse"], False),
    "no-fast-complex": (["--no-fast_complex"], True),
}
# no-cse is not run by default: the code of the wide sums gets 6-8x larger (impedance: 75 MiB, 7 minutes)
DEFAULT_OPTIONS = [o for o in OPTIONS if o != "no-cse"]


def run(example, option, quick, timeout, cache):
    """runs one example with one option set; returns a result dict"""
    script, full, small, _ = EXAMPLES[example]
    cmd = [sys.executable, "-W", "ignore", os.path.join(HERE, script)] + (small if quick else full) + OPTIONS[option][0]
    env = dict(os.environ)
    python_path = os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(REPO, "python"))
    env["PYTHONPATH"] = os.pathsep.join([python_path, HERE] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if cache:
        env["SYMJIT_EXAMPLE_CACHE"] = cache
    else:
        env.pop("SYMJIT_EXAMPLE_CACHE", None)

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, kill)
    timer.start()
    out = proc.stdout.read()
    _, status, usage = os.wait4(proc.pid, 0)  # also the child's peak memory
    timer.cancel()
    proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0

    r = dict(example=example, option=option, wall=wall, max_rss_mb=usage.ru_maxrss / 1024, exit=proc.returncode)
    r["fails"] = [line[4:].split(":")[0].strip() for line in out.splitlines() if line.startswith("FAIL")]
    r["oks"] = sum(line.startswith("ok  ") for line in out.splitlines())
    m = re.search(r"machine code: ([\d.]+) MiB \(scalar\), ([\d.]+) MiB \(SIMD\)", out)
    r["scalar_mb"], r["simd_mb"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"compilation(?: \(complex128\))? ([\d.]+) s", out)
    r["compile_s"] = float(m.group(1)) if m else None
    r["cached"] = "(cached)" in out
    if timed_out.is_set():
        r["status"] = "TIMEOUT"
    elif proc.returncode < 0:
        r["status"] = f"CRASH (signal {-proc.returncode})"
    elif proc.returncode == 0 and "all checks passed" in out:
        r["status"] = "pass"
    elif r["fails"]:
        r["status"] = "FAIL"
    elif "does not support" in out:
        r["status"] = "skip"
    else:
        r["status"] = f"ERROR (exit {proc.returncode})"
    if r["status"] not in ("pass", "skip"):
        r["tail"] = "\n".join(out.splitlines()[-8:])
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", nargs="*", default=list(EXAMPLES), choices=list(EXAMPLES))
    parser.add_argument("--options", nargs="*", default=DEFAULT_OPTIONS, choices=list(OPTIONS),
                        help="option sets (default: all but no-cse, which takes long)")
    parser.add_argument("--quick", action="store_true", help="smaller problems")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per run (default 900)")
    parser.add_argument("--jobs", type=int, default=1, help="runs at a time (default 1; the full sizes need up to 3 GB each)")
    parser.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "symjit-example-cache"),
                        help="folder of the cached models (default: in the temporary directory)")
    parser.add_argument("--no-cache", action="store_true", help="build the sympy expressions in every run")
    parser.add_argument("--clear-cache", action="store_true", help="empty the cache folder first")
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()

    cache = None if args.no_cache else args.cache
    if cache and args.clear_cache and os.path.isdir(cache):
        shutil.rmtree(cache)

    runs = [(e, o) for e in args.examples for o in args.options if EXAMPLES[e][3] or not OPTIONS[o][1]]
    print(f"{len(runs)} runs ({len(args.examples)} examples, {'quick' if args.quick else 'full'} sizes), "
          f"cache {cache or 'off'}, {args.jobs} at a time\n", flush=True)

    head = f"{'example':14s} {'options':16s} {'status':18s} {'wall s':>7s} {'compile s':>9s} {'MB peak':>8s} {'scalar MiB':>10s} {'SIMD MiB':>8s}"
    print(head + "\n" + "-" * len(head), flush=True)

    def show(r):
        f = lambda v, fmt: format(v, fmt) if v is not None else "-"  # noqa: E731
        print(f"{r['example']:14s} {r['option']:16s} {r['status']:18s} {r['wall']:7.1f} {f(r['compile_s'], '9.2f')} "
              f"{r['max_rss_mb']:8.0f} {f(r['scalar_mb'], '10.2f')} {f(r['simd_mb'], '8.2f')}"
              + (f"  failed: {', '.join(r['fails'])}" if r["fails"] else ""), flush=True)

    results = []
    # the first run of each example builds its cache entry; the others may then run in parallel
    first = {}
    for e, o in runs:
        first.setdefault(e, (e, o))
    for e, o in first.values():
        results.append(run(e, o, args.quick, args.timeout, cache))
        show(results[-1])
    rest = [r for r in runs if r not in first.values()]
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for r in pool.map(lambda eo: run(*eo, args.quick, args.timeout, cache), rest):
            results.append(r)
            show(r)

    # a matrix of the statuses
    print("\n" + f"{'':14s} " + " ".join(f"{o[:9]:>9s}" for o in args.options))
    for e in args.examples:
        cells = []
        for o in args.options:
            r = next((r for r in results if r["example"] == e and r["option"] == o), None)
            cells.append(f"{(r['status'].split()[0] if r else '-')[:9]:>9s}")
        print(f"{e:14s} " + " ".join(cells))

    bad = [r for r in results if r["status"] not in ("pass", "skip")]
    for r in bad:
        print(f"\n--- {r['example']} {r['option']}: {r['status']}\n{r.get('tail', '')}")
    print(f"\n{len(results) - len(bad)} of {len(results)} runs passed, total {sum(r['wall'] for r in results):.0f} s")
    if args.json:
        with open(args.json, "w") as fd:
            json.dump(results, fd, indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
