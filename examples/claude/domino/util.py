import argparse

symjit = True


def use_symjit():
    return symjit


def process_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", help="backend engine", default="rust")
    parser.add_argument("--ty", help="architecture type", default="native")
    parser.add_argument(
        "--threads",
        help="use multi-threading",
        action=argparse.BooleanOptionalAction,
        dest="use_threads",
        default=True,
    )
    parser.add_argument(
        "--simd",
        help="use simd",
        action=argparse.BooleanOptionalAction,
        dest="use_simd",
        default=True,
    )
    parser.add_argument(
        "--simd512",
        help="enable simd512",
        action=argparse.BooleanOptionalAction,
        dest="enable_simd512",
        default=False,
    )
    parser.add_argument(
        "--cse",
        help="apply common subexpression elimination",
        action=argparse.BooleanOptionalAction,
        dest="cse",
        default=True,
    )
    parser.add_argument(
        "--fastmath",
        help="use fastmath operations",
        action=argparse.BooleanOptionalAction,
        dest="fastmath",
        default=True,
    )
    parser.add_argument(
        "--fast_complex",
        help="use SIMD instructions for scalar complex functions",
        action=argparse.BooleanOptionalAction,
        dest="fast_complex",
        default=True,
    )
    parser.add_argument(
        "--compress",
        help="Contract compiled code",
        action=argparse.BooleanOptionalAction,
        dest="compress",
        default=False,
    )
    parser.add_argument("--dtype", help="data type", default="float64")
    parser.add_argument(
        "--opt_level",
        help="optimization level (0, 1, 2, or 3)",
        action="store",
        dest="opt_level",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--symjit",
        help="do not use symjit at all!",
        action=argparse.BooleanOptionalAction,
        dest="symjit",
        default=True,
    )

    args = vars(parser.parse_args())

    global symjit
    symjit = args.pop("symjit")

    # print(f"options: {args}")

    return args



def compile_cached(name, params, build, **options):
    """Compiles the sympy expressions that build() returns, with a cache for sweeps of options.

    build() returns (models, info): models is a list of (states, expressions) pairs, one per
    function to compile, and info a dict of JSON-serializable values the caller needs (counts).
    The expressions are converted to symjit's JSON models (symjit.structure.model, as in
    compile_func) and compiled with compile_json(model, **options). If the environment variable
    SYMJIT_EXAMPLE_CACHE names a folder, the models and info are stored there (gzipped JSON,
    name-params.json.gz) and later runs with the same params skip sympy (run_physics.py).

    Returns (functions, info, seconds building or loading, seconds compiling, cached)."""
    import gzip
    import json
    import os
    import time

    from symjit import compile_json, structure

    folder = os.environ.get("SYMJIT_EXAMPLE_CACHE")
    key = "-".join(str(p).replace("/", "_").replace(" ", "") for p in params)
    path = os.path.join(folder, f"{name}-{key}.json.gz") if folder else None
    t0 = time.perf_counter()
    cached = bool(path) and os.path.exists(path)
    if cached:
        with gzip.open(path, "rt") as fd:
            data = json.load(fd)
    else:
        models, info = build()
        data = dict(models=[structure.model(states, eqs) for states, eqs in models], info=info)
        if path:
            os.makedirs(folder, exist_ok=True)
            with gzip.open(path + ".tmp", "wt", compresslevel=1) as fd:
                json.dump(data, fd)
            os.replace(path + ".tmp", path)  # complete files only, also with parallel runs
    t1 = time.perf_counter()
    functions = [compile_json(model, **options) for model in data["models"]]
    t2 = time.perf_counter()
    return functions, data["info"], t1 - t0, t2 - t1, cached
