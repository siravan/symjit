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

    print(f"options: {args}")

    return args
