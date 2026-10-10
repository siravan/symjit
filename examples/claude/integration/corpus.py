"""Runs sni.py on the Rubi integration test suite in Axiom syntax, as shipped with
SymbolicNumericIntegration.jl (test/AxiomSyntaxTestFiles: about 73,000 problems in 9 groups;
the problems come from Rubi, https://rulebasedintegration.org).

    python corpus.py ~/src/SymbolicNumericIntegration.jl/test/AxiomSyntaxTestFiles
    python corpus.py DIR --sample 300 --json base.json    # 300 problems per group, save the results
    python corpus.py DIR --sample 300 --compare base.json # the same sample, show what changed
    python corpus.py DIR --groups 4 6 --sample 0          # all problems of groups 4 and 6

Each line `[integrand, x, steps, answer]` of a file is one problem. As in the Julia loader
(test/axiom.jl), the symbolic constants are replaced by numbers (here the primes 2, 3, 5, ...
in alphabetical order of the names), since sni.py integrates expressions with numeric
coefficients only. A problem counts as *elementary* if Rubi's answer contains only elementary
functions, and as *special* if it contains special functions (polylog, elliptic, hypergeometric,
erf, Ei, ...; sni.py can find a few: Ei, Si, Ci, erfi). An answer of sni.py is accepted if its
derivative equals the integrand at four complex points (see run.check; `approx`:
to a relative error of 1e-5 only, not 1e-8, i.e. floating-point coefficients); integrands
the loader cannot convert to sympy (e.g. Axiom's two-argument Ei and GAMMA) count as `parse`.

The problems run in a process pool (`--workers`, default: the number of CPUs), each with a
time limit (`--timeout`, default 20 s, also for checking the answer: an answer whose check runs out
of time counts as `unchecked`, shown with the timeouts); times are per problem, so the totals
depend on the load.
"""

import argparse
import glob
import json
import os
import random
import re
import signal
import sys
import time
from multiprocessing import Pool

# one thread per worker for numpy's BLAS (lstsq, qr) and symjit's thread pool: with a thread
# pool in every worker process, the workers fight over the cores and problems time out
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sympy as sp  # noqa: E402

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

ELEMENTARY = {sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, sp.sinh, sp.cosh,
              sp.tanh, sp.coth, sp.sech, sp.csch, sp.asin, sp.acos, sp.atan, sp.acot, sp.asec,
              sp.acsc, sp.asinh, sp.acosh, sp.atanh, sp.acoth, sp.asech, sp.acsch, sp.Abs, sp.sign,
              sp.re, sp.im, sp.arg, sp.conjugate}

# Axiom/Rubi names that sympy spells differently, and markers of problems without a
# closed form; unknown functions become undefined sympy functions
LOCALS = {"GAMMA": sp.gamma, "ProductLog": sp.LambertW, "FresnelC": sp.fresnelc,
          "FresnelS": sp.fresnels, "hypergeometric": sp.Function("hypergeometric"),
          "HypergeometricPFQ": sp.Function("HypergeometricPFQ"), "polylog": sp.polylog,
          "elliptic_f": sp.elliptic_f, "elliptic_e": sp.elliptic_e, "elliptic_pi": sp.elliptic_pi,
          "Unintegrable": sp.Function("Unintegrable"), "CannotIntegrate": sp.Function("CannotIntegrate"),
          "Derivative": sp.Function("Derivative_"), "E": sp.E, "I": sp.I, "pi": sp.pi,
          "S": sp.Symbol("S"), "N": sp.Symbol("N"), "O": sp.Symbol("O"), "Q": sp.Symbol("Q"),
          "beta": sp.Symbol("beta"), "gamma": sp.Symbol("gamma"), "zeta": sp.Symbol("zeta"),
          "lambda": sp.Symbol("lamda")}


def split_top(s):
    """splits s at the commas outside brackets and parentheses"""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def to_sympy(s):
    s = s.replace("%pi", "pi").replace("%e", "E").replace("%i", "I")
    s = re.sub(r"\blambda\b", "lamda", s)
    return sp.sympify(s, locals=LOCALS, convert_xor=True)


def load(root, groups=None):
    """the problems as dicts (group, file, line, text of the integrand, variable and answer)"""
    problems = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.input"), recursive=True)):
        group = os.path.relpath(path, root).split(os.sep)[0]
        if groups and not any(group.startswith(g) for g in groups):
            continue
        with open(path, encoding="utf-8", errors="replace") as fd:
            for lineno, line in enumerate(fd, 1):
                line = line.strip().rstrip(",")
                if not line.startswith("["):
                    continue
                if line.endswith("]]"):  # the last problem closes the list
                    line = line[:-1]
                parts = split_top(line[1:-1])
                if len(parts) != 4:
                    continue
                problems.append(dict(group=group, file=os.path.basename(path), line=lineno,
                                     integrand=parts[0].strip(), var=parts[1].strip(), answer=parts[3].strip()))
    return problems


def classify(answer):
    """'elementary', 'special' or 'none' (no closed form), from Rubi's answer"""
    if any(k in answer for k in ("Unintegrable", "CannotIntegrate", "Derivative")):
        return "none"
    try:
        a = to_sympy(answer)
    except Exception:
        return "special"
    funcs = {f.func for f in a.atoms(sp.Function)}
    return "elementary" if funcs <= ELEMENTARY else "special"


def prepare(p):
    """the integrand with numbers for its symbolic constants, and the variable"""
    x = sp.Symbol(p["var"])
    eq = to_sympy(p["integrand"])
    consts = sorted((s for s in eq.free_symbols if s != x), key=lambda s: s.name)
    if len(consts) > len(PRIMES):
        raise ValueError("too many constants")
    eq = eq.subs({s: v for s, v in zip(consts, PRIMES)})
    return eq, x


class Timeout(BaseException):
    """not an Exception, so that `except Exception` in sympy does not swallow it"""


def _alarm(signum, frame):
    raise Timeout()


def set_alarm(seconds):
    """raises Timeout after `seconds`, and again every second if it was caught anyway (0: off)"""
    signal.setitimer(signal.ITIMER_REAL, seconds, 1.0 if seconds else 0.0)


def work(task):
    """runs one problem in a worker process; returns a result dict"""
    import numpy as np

    import run
    import sni

    p, backend, timeout, seed, *rest = task
    trials = dict(num_trials=rest[0]) if rest and rest[0] else {}
    r = dict(key=f"{p['file']}:{p['line']}", group=p["group"], kind=classify(p["answer"]))
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.perf_counter()
    try:
        eq, x = prepare(p)
    except Exception as e:
        r.update(status="parse", error=f"{type(e).__name__}: {str(e)[:200]}", time=0.0)
        return r
    sol = None
    try:
        r["integrand"] = str(eq)
        set_alarm(timeout)
        sol = sni.integrate(eq, x, backend=backend, rng=np.random.default_rng(seed), **trials)
        set_alarm(0)
        r["answer"] = None if sol is None else str(sol)
        if sol is None:
            r["status"] = "none"
        else:
            set_alarm(timeout)  # the 30-digit check of a large answer can be slow too
            r["status"] = run.check(eq, sol, x)
    except Timeout:
        r["status"] = "timeout" if sol is None else "unchecked"
    except Exception as e:
        r["status"] = "error"
        r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        set_alarm(0)
    r["time"] = time.perf_counter() - t0
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="the AxiomSyntaxTestFiles directory")
    parser.add_argument("--groups", nargs="*", help="group prefixes, e.g. 0 4 (default: all)")
    parser.add_argument("--sample", type=int, default=200, help="problems per group (0: all; default 200)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", default="symjit", choices=["symjit", "lambdify"])
    parser.add_argument("--timeout", type=int, default=20, help="seconds per problem (default 20)")
    parser.add_argument("--trials", type=int, help="trials per candidate set (default: sni.integrate's)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--json", help="write the results to this file")
    parser.add_argument("--compare", help="compare with the results in this file")
    args = parser.parse_args()

    problems = load(args.root, args.groups)
    by_group = {}
    for p in problems:
        by_group.setdefault(p["group"], []).append(p)
    rnd = random.Random(args.seed)
    chosen = []
    for g in sorted(by_group):
        ps = by_group[g]
        chosen += ps if args.sample == 0 or args.sample >= len(ps) else rnd.sample(ps, args.sample)
    print(f"{len(problems)} problems in {len(by_group)} groups; running {len(chosen)} "
          f"with {args.workers} workers, backend {args.backend}", flush=True)

    t0 = time.perf_counter()
    tasks = [(p, args.backend, args.timeout, args.seed + i, args.trials) for i, p in enumerate(chosen)]
    with Pool(args.workers, maxtasksperchild=50) as pool:
        results = []
        for k, r in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            results.append(r)
            if k % 200 == 0:
                print(f"  {k}/{len(tasks)} done", flush=True)
                if args.json:  # partial results, in case the run has to be stopped
                    with open(args.json, "w") as fd:
                        json.dump(dict(args=vars(args), results=results, partial=True), fd, indent=0)
    wall = time.perf_counter() - t0

    report(results, wall)
    if args.json:
        with open(args.json, "w") as fd:
            json.dump(dict(args=vars(args), results=results), fd, indent=0)
    if args.compare:
        compare(results, args.compare)


def report(results, wall):
    head = (f"{'group':34s} {'problems':>8s} {'elem.':>6s} {'ok':>5s} {'ok elem.':>8s} {'ok %el.':>7s} "
            f"{'approx':>6s} {'wrong':>5s} {'none':>5s} {'t/o':>4s} {'err':>4s} {'parse':>5s} {'s/prob':>6s}")
    print("\n" + head + "\n" + "-" * len(head))
    groups = sorted({r["group"] for r in results})
    rows = [(g, [r for r in results if r["group"] == g]) for g in groups] + [("all", results)]
    for g, rs in rows:
        n = len(rs)
        el = [r for r in rs if r["kind"] == "elementary"]
        ok = sum(r["status"] == "ok" for r in rs)
        ok_el = sum(r["status"] == "ok" for r in el)
        count = {s: sum(r["status"] == s for r in rs) for s in ("approx", "wrong", "none", "timeout", "error", "parse")}
        count["timeout"] += sum(r["status"] == "unchecked" for r in rs)
        if g == "all":
            print("-" * len(head))
        print(f"{g[:34]:34s} {n:8d} {len(el):6d} {ok:5d} {ok_el:8d} {100 * ok_el / max(1, len(el)):6.1f}% "
              f"{count['approx']:6d} {count['wrong']:5d} {count['none']:5d} {count['timeout']:4d} {count['error']:4d} "
              f"{count['parse']:5d} "
              f"{sum(r['time'] for r in rs) / max(1, n):6.2f}")
    print(f"\nwall time {wall:.0f} s")
    errors = {}
    for r in results:
        if r["status"] in ("error", "parse"):
            errors[r["error"].split(":")[0]] = errors.get(r["error"].split(":")[0], 0) + 1
    if errors:
        print("errors and parse failures: " + ", ".join(f"{k} {v}" for k, v in sorted(errors.items(), key=lambda kv: -kv[1])))


def compare(results, path):
    with open(path) as fd:
        old = {r["key"]: r for r in json.load(fd)["results"]}
    new = {r["key"]: r for r in results}
    common = sorted(set(old) & set(new))
    gained = [k for k in common if new[k]["status"] == "ok" and old[k]["status"] != "ok"]
    lost = [k for k in common if old[k]["status"] == "ok" and new[k]["status"] != "ok"]
    print(f"\ncompared with {path} ({len(common)} common problems): "
          f"{len(gained)} newly solved, {len(lost)} lost")
    for title, keys in (("lost", lost), ("newly solved", gained)):
        for k in keys[:30]:
            print(f"  {title:12s} {k:60s} {new[k].get('integrand', '')[:60]}  ({new[k]['status']})")


if __name__ == "__main__":
    main()
