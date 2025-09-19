import argparse

symjit = True


def use_symjit():
    return symjit


def process_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", help="backend engine", default="rust")
    parser.add_argument("--ty", help="architecture type", default="native")
    parser.add_argument(
        "--nosimd", help="do not use simd!", action="store_false", dest="use_simd"
    )
    parser.add_argument(
        "--nothreads",
        help="do not use multi-threading",
        action="store_false",
        dest="use_threads",
    )
    parser.add_argument(
        "--nocse",
        help="do not apply common subexpression elimination",
        action="store_false",
        dest="cse",
    )
    parser.add_argument(
        "--fastmath",
        help="use fastmath operations",
        action="store_true",
        dest="fastmath",
    )
    parser.add_argument(
        "--nosymjit",
        help="do not use symjit at all!",
        action="store_false",
        dest="symjit",
    )

    args = vars(parser.parse_args())

    global symjit
    symjit = args.pop("symjit")

    print(f"options: {args}")

    return args
