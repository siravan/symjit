# sjc, the Symjit compiler driver

`sjc` compiles a saved symjit application (a `.sjb` file), or a `#[kernel]` function of a
Rust file ([Rust input](#rust-input)), into a relocatable object file and C header
([OBJ.md](OBJ.md)), or into a WebAssembly module ([WASM.md](WASM.md)). The model is
compiled once, in Python or Rust, and saved; sjc turns it into code that a C/C++ build or a
WebAssembly host links or loads, with no Python at build time.

```python
from sympy import symbols, sin, exp
from symjit import compile_func

x, y = symbols("x y")
f = compile_func([x, y], [sin(x) * y + exp(-x * x)])
f.save("model.sjb")
```

```bash
sjc model.sjb                                   # model.o + model.h (this machine)
sjc model.sjb -o lib/kern.o --name kern         # lib/kern.o + lib/kern.h, kernel `kern`
sjc model.sjb --target aarch64-apple-darwin     # Mach-O ARM64 object, e.g. for an Apple Mac
sjc model.sjb -o model.wasm                     # WebAssembly module
sjc --emit info model.sjb                       # what is in the file
```

The object and the header are those `f.write_obj("model")` writes, and the module has the
interface of `compile_func(..., ty="wasm")` modules; their use is described in
[OBJ.md](OBJ.md) and [WASM.md](WASM.md).

## Building it

sjc is the workspace member `sjc/`. It turns on the `obj` and `wasm` features of its own
copy of symjit, so the library and the Python package need not be built with them:

```bash
cargo build --release -p sjc            # target/release/sjc
cargo install --path sjc                # into ~/.cargo/bin
cargo build --release -p sjc --no-default-features --features obj   # objects only
```

Without one of the features, the corresponding outputs report "sjc was built without the
`wasm` feature" (or `obj`).

## Options

```
sjc [OPTIONS] <INPUT.sjb | INPUT.rs>
```

| option | meaning |
|---|---|
| `-o <FILE>` | The output file. Its extension selects the output, unless `--emit` is given: `.o`/`.obj` an object, `.wasm` a module, `.h` the header alone. Default: the input's stem with `.o` (`.wasm` for `--emit wasm`), in the current directory. |
| `--emit <KIND>` | `obj`, `wasm`, `header`, `sjb` (Rust input: the kernel as a saved application) or `info`. `info` prints the model's target, options, counts, functions and registers. A `.sjb` output selects `sjb`. |
| `--target <TARGET>` | `x86_64`, `aarch64` (`arm64`), `wasm32` or `native`, or a triple such as `x86_64-unknown-linux-gnu` or `aarch64-apple-darwin`. The system in a triple picks the object format: Linux and the BSDs ELF, Apple Mach-O. Default: the architecture the model was saved for (`native`: this machine's). |
| `--format <FORMAT>` | `elf` or `macho`, if the target does not imply one. Default: this system's. |
| `--name <IDENT>` | The name of the kernel functions (`name`, `name_fast`) and of the header's macros (`NAME_COUNT_STATES`, ...). Default: the output's stem, which must then be a C identifier. |
| `--header <FILE>` | The header. Default: the object with extension `.h`. |
| `--no-header` | Do not write the header. |
| `-v`, `--verbose` | Say what is read and written (on stderr). |
| `-h`, `--help`, `-V`, `--version` | |

The exit status is 0 on success, 1 on an error and 2 on a usage error; messages go to
stderr as `sjc: error: ...`.

For a saved application, the opt level, `fastmath`, complex mode and the other options are
fixed when the model is compiled and saved, and sjc uses them: the options of Rust input
(`-O`, `-ffast-math`, `--kernel`) are rejected for it with a message saying so.

## Targets and opt levels

A `.sjb` holds the model's intermediate code after register allocation, not its
expressions. With opt level 2 or lower, the code uses the same registers on every
machine, so sjc can compile it for any target: a model saved on x86-64 Linux gives ARM64
objects, and the other way round. Opt level 3 gives ARM64 and RISC-V more registers
(27 instead of 14) than x86-64 has. sjc rejects such a model for x86-64:

```
sjc: error: cannot compile `big.sjb`: the model uses 27 registers (it was compiled with opt
level 3 for a machine with more registers), but x86-64 has 14; compile it for this target or
with opt level 2 or lower
```

WebAssembly modules keep registers in locals, so any model can be compiled to wasm.

A model compiled for another target computes the same function, but not always to the last
bit of the target's own compilation: the order of operations follows the saved code (e.g.
the x86-64 memory-operand forms). The results of the object or module are those of the
saved code.

Other limits are those of object files: they hold x86-64 or AArch64 kernels; they call the C
math library (complex functions are C99's `csin`, `cexp`, ...; symjit's `csc`, `sinc`, ... are
composed of C functions; see [OBJ.md](OBJ.md)), and user-defined functions are symbols the
program defines; AArch64 objects hold kernels up to 1 MB. An x86-64 object uses
AVX and FMA if the machine running sjc has them, SSE2 otherwise; the header says which.

## Rust input

A `.rs` input is a Rust file with functions written for the `#[kernel]` macro of the
`composer` feature ([KERNELS.md](KERNELS.md)). sjc compiles one of its kernels without a
Rust compiler:

```rust
// kernels.rs
use num_complex::Complex;
use symjit::kernel::kernel;

#[kernel]
fn escape(c: Complex<f64>, limit: f64) -> f64 {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < limit && z.norm_sqr() <= 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}
```

```bash
sjc kernels.rs                          # escape.o + escape.h (the only kernel of the file)
sjc kernels.rs --kernel escape -O3 -fno-fast-math --target aarch64-apple-darwin
sjc kernels.rs --kernel escape -o escape.wasm
sjc kernels.rs --kernel escape --emit sjb    # escape.sjb, for later sjc runs or load_func
```

| option | meaning |
|---|---|
| `--kernel <NAME>` | The `#[kernel]` function to compile. Default: the only one of the file (an error lists them if there are several). The outputs are named after it by default. |
| `-O0` ... `-O3` | The opt level (default 2). |
| `-ffast-math`, `-fno-fast-math` | Fuse multiplications and additions (the default), or not. |

The other options are symjit's built-in defaults (`Config::builtin`); a `symjit.toml` is not
read. The kernel's inputs are its parameters and its outputs go to `mem`, as for
`compile_evaluator`: `name(outs, NULL, 0, inputs)` (the header says so).

**How.** sjc parses the file with `syn` and interprets the kernel's body against
`symjit::kernel::KernelBuilder`, doing what the code the macro generates does: numbers,
`for` loops, conditions on numbers and closures run while the kernel is built, and kernel
values (`f64`, `Complex<f64>` and `bool` values derived from the parameters) add instructions.
The tests compile the kernels of `sjc/tests/kernels/corpus.rs` both ways (that file is
also compiled by the macro) and require the same intermediate code and constants.

**What works.** Everything the macro accepts in kernels, `#[kernel(inline)]` helpers and calls:
arithmetic and the methods of `f64` and `Complex<f64>` on kernel values; comparisons, `&&`,
`||`, `!`, `^` (both sides are always evaluated); kernel `if`/`else` (also without `else`
when it assigns variables) and kernel `while` loops, with the variables they assign passed
through; recursion; inline helpers with any parameters (`i32`, arrays); `let` with patterns
and types (`let x: f64` makes a kernel value, as `let mut s = 0.0` does); tuples, arrays
(`[x; N]`, indexing, `iter`, `enumerate`, `zip`, `rev`, `map` with a closure), ranges,
`for` loops (unrolled), `loop`/`break`/`continue` in loops run while building; closures that
do not assign outer variables; `const` items of the file and `std::f64::consts`; numbers'
`f64` methods (`f64::max(a, b)`, `x.sqrt()`, ...); `as` casts between numbers.

**Calls.** A kernel that calls other `#[kernel]` kernels (also through inline helpers, and
recursively) is compiled with them, as the macro's `compile` does: each called kernel is
compiled for its caller (as a complex kernel if the caller is complex; a real kernel cannot
call a complex one), and kernels cannot call each other in a cycle. In an object file the
called kernels are functions of the same object, named `<kernel>__<callee>` (also
declared in the header), and the calls are direct calls:

```
$ sjc kernels.rs --kernel iterate -o it.o && nm it.o
                 U damp
0000000000000000 T it
0000000000000280 T it__step
```

`#[kernel(callback)]` functions are Rust, which sjc does not compile: in an object file a
callback is an undefined symbol of its name, which the program defines in C (or any
language) with the same arguments, and the header declares (`double damp(double x0);`). In a
WebAssembly module it is an import, `composer_<name>`, from module `"symjit"`. A callback
named like a C math function (`hypot`, `fmod`, ...) is an error in an object file, since it
would link to the C library's. WebAssembly modules cannot call other kernels yet, and a
kernel with calls cannot be saved with `--emit sjb` (the file cannot hold the called
kernels and callbacks).

**What does not work (yet).** `match`, `if let`, macros (`println!`), structs, traits and
generics, labeled loops. They are errors with the file, line and column:

```
sjc: error: kernels.rs:12:9: macros (`println!`) are not supported by sjc
```

sjc accepts a little more than the macro: the macro also keeps each function as ordinary
Rust, so methods that only kernel values have (`select`, `to_real`, `modulo`) and some
conversions (a number returned where an inline helper returns `f64`) are type errors there;
sjc compiles them.

## From Rust

sjc uses two functions of the library (feature `obj` for the second):

- `Application::load_for(stream, config, target)` loads a saved application for `target`
  (`None`: the saved type), after checking that its registers fit the target. It does not
  prepare the SIMD and fast kernels. `measure("registers")` gives the registers a model
  uses.
- `Application::object_files(symbol, format, object_file, header_file)` returns the object
  and the header in memory; `write_obj` writes them under the names it derives from its
  argument.

## Tests

```bash
cargo test --release -p sjc                        # sjc/tests/cli.rs: x86 objects with cc,
                                                   # AArch64 under qemu, wasm in Node, Rust input;
                                                   # sjc/tests/source.rs: the interpreter against
                                                   # the macro (identical MIR)
cargo test --release --test load_for               # the library's load_for (add --features wasm)
python python/tests/test_sjc.py                    # functions saved from Python ($SJC or target/release/sjc)
```

The steps that need `cc`, `aarch64-linux-gnu-gcc` and `qemu-aarch64`, or `node` are skipped
without them.
