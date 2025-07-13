import sys

def process_argv():
    argv = sys.argv
    backend = "python" if len(argv) > 1 and argv[1] == "py" else "rust"
    ty = argv[1] if len(argv) > 1 and argv[1] != "py" else "native"
    use_simd = ("simd" in argv[2]) if len(argv) > 2 else True
    use_threads = ("threads" in argv[2]) if len(argv) > 2 else True

    print(f'Generating code for {ty} using {backend} as backend (simd = {use_simd}; threads = {use_threads})')

    return backend, ty, use_simd, use_threads
