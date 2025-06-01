import sys

def process_argv():
    argv = sys.argv
    backend = "python" if len(argv) > 1 and argv[1] == "py" else "rust"
    ty = argv[1] if len(argv) > 1 and argv[1] != "py" else "native"
    use_simd = (argv[2] == "simd") if len(argv) > 2 else True
    
    print(f'Generating code for {ty} using {backend} as backend {'with' if use_simd else 'without'} SIMD')
    
    return backend, ty, use_simd
