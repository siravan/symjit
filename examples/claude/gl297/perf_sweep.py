"""GL297: hardware counters (perf stat) of the single-threaded evaluator-major sweep of batch_sweep.py.

For each variant and block size b, the 31 evaluators are compiled and warmed up, then
`perf stat` counts only the measured sweeps (it is enabled and disabled through a control
FIFO, so compilation is excluded). Counts are reported per point:

    instr         instructions retired
    IPC           instructions per cycle
    frontend %    dispatch slots left empty because the frontend (instruction fetch/decode)
                  delivered nothing (8 slots per cycle on Zen 5)
    backend %     dispatch slots left empty because the backend was stalled (data, execution)
    IC<-L2        instruction-cache lines (64 B) filled from L2
    IC<-sys       instruction-cache lines filled from beyond L2 (L3 or memory)
    D<-sys        data-cache demand fills from beyond L2

    python perf_sweep.py                            # scalar and SIMD f64x4, b = 1 and 16
    python perf_sweep.py -b 1 -b 8 -b 32 --variants scalar "SIMD f64x8"
    python perf_sweep.py --only 21 4                # only these evaluators (e.g. the smallest, largest)

Needs perf with access to the AMD Zen core events (kernel.perf_event_paranoid <= 2 for
user-space counting). Takes SYMJIT_PYTHON_PATH and GL297_STUDY like batch_sweep.py.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EVENTS = [
    "cycles",
    "instructions",
    "ic_cache_fill_l2",
    "ic_cache_fill_sys",
    "de_no_dispatch_per_slot.no_ops_from_frontend",
    "de_no_dispatch_per_slot.backend_stalls",
    "ls_dmnd_fills_from_sys.all",
]
DISPATCH_WIDTH = 8


def inner(args):
    """runs under perf: compiles, warms up, then counts `--seconds` worth of sweeps"""
    import numpy as np  # noqa: F401
    import batch_sweep as bs
    import symjit

    evaluators, x = bs.load()
    if args.only:
        evaluators = [evaluators[i] for i in args.only]
    opts = dict(dtype="complex128", num_params=x.size, fast_complex=True, use_threads=False) | bs.VARIANTS[args.variant]
    funcs = [symjit.compile_evaluator(ev, **opts) for ev in evaluators]
    X = bs.block(x, args.b)
    t0 = time.perf_counter()
    sweeps = 0
    while sweeps < 3 or time.perf_counter() - t0 < 1.0:  # warm-up (and lazy compilation)
        for f in funcs:
            f.evaluate_complex(X)
        sweeps += 1
    per_sweep = (time.perf_counter() - t0) / sweeps
    count = max(3, int(args.seconds / per_sweep))

    ctl = open(args.ctl, "w")
    ack = open(args.ack, "r")

    def command(c):
        ctl.write(c + "\n")
        ctl.flush()
        ack.readline()

    command("enable")
    t0 = time.perf_counter()
    for _ in range(count):
        for f in funcs:
            f.evaluate_complex(X)
    elapsed = time.perf_counter() - t0
    command("disable")
    print(f"POINTS {count * args.b} {elapsed}", flush=True)


def measure(variant, b, seconds, only=None):
    with tempfile.TemporaryDirectory() as tmp:
        ctl, ack, out = (os.path.join(tmp, x) for x in ("ctl", "ack", "perf.csv"))
        os.mkfifo(ctl)
        os.mkfifo(ack)
        only = ["--only", *map(str, only)] if only else []
        cmd = ["perf", "stat", "-x", ",", "-o", out, "--delay=-1", f"--control=fifo:{ctl},{ack}",
               "-e", ",".join(EVENTS), "--",
               sys.executable, os.path.abspath(__file__), "--inner", "--variant", variant, "-b", str(b),
               "--seconds", str(seconds), "--ctl", ctl, "--ack", ack, *only]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stdout + p.stderr)
        line = next(x for x in p.stdout.splitlines() if x.startswith("POINTS "))
        points, elapsed = int(line.split()[1]), float(line.split()[2])
        counts = {}
        with open(out) as fd:
            for row in fd:
                f = row.strip().split(",")
                if len(f) > 3 and f[2] in EVENTS or (len(f) > 3 and f[2].split(":")[0] in EVENTS):
                    counts[f[2].split(":")[0]] = float(f[0]) if f[0][0].isdigit() else float("nan")
        return points, elapsed, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-b", type=int, action="append", help="block sizes (default 1, 16)")
    parser.add_argument("--variants", nargs="*", default=["scalar", "SIMD f64x4"])
    parser.add_argument("--seconds", type=float, default=5.0, help="measured time per case (default 5)")
    parser.add_argument("--only", type=int, nargs="*", help="sweep only these evaluators (indices 0-30)")
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--ctl", help=argparse.SUPPRESS)
    parser.add_argument("--ack", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inner:
        args.b = args.b[0]
        return inner(args)

    sizes = args.b or [1, 16]
    head = (f"{'variant':22s} {'b':>3s} {'ms/pt':>7s} {'Minstr/pt':>9s} {'IPC':>5s} {'frontend%':>9s} {'backend%':>8s} "
            f"{'kIC<-L2':>8s} {'kIC<-sys':>8s} {'kD<-sys':>8s}")
    which = f"evaluators {args.only}" if args.only else "all 31 evaluators"
    print(f"per point (sweep over {which}); k = thousands of 64-byte lines")
    print(head)
    print("-" * len(head))
    for variant in args.variants:
        for b in sizes:
            points, elapsed, c = measure(variant, b, args.seconds, args.only)
            slots = DISPATCH_WIDTH * c["cycles"]
            print(f"{variant:22s} {b:3d} {1e3 * elapsed / points:7.3f} {c['instructions'] / points / 1e6:9.2f} "
                  f"{c['instructions'] / c['cycles']:5.2f} "
                  f"{100 * c['de_no_dispatch_per_slot.no_ops_from_frontend'] / slots:9.1f} "
                  f"{100 * c['de_no_dispatch_per_slot.backend_stalls'] / slots:8.1f} "
                  f"{c['ic_cache_fill_l2'] / points / 1e3:8.1f} {c['ic_cache_fill_sys'] / points / 1e3:8.1f} "
                  f"{c['ls_dmnd_fills_from_sys.all'] / points / 1e3:8.1f}", flush=True)


if __name__ == "__main__":
    main()
