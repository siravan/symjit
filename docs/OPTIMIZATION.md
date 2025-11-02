# Optimization

The Rust backend supports different optimization and parallelization methods, which can be controlled using `compile_func` arguments. The options are:

* `use_simd` (default `True`): generates SIMD instructions if possible (currently supports AVX instructions on X86-64 processors). SIMD code should improve the performance up to 4x for certain tasks (using 256-bit registers that encode and operate on four doubles simultaneously).
* `use_threads` (default `True`): use multi-threading to speed up parallel processing of array operations using [Rayon rust crate](https://docs.rs/rayon/latest/rayon/).
* `cse` (default `True`): New to version 2.4. It performs common-subexpression elimination, i.e., factoring common expressions and sub-expressions.
* `fastmath` (default `False`): New to version 2.5. It rewrites the code to combine multiplication and addition/substraction into various fused-multiply-add instructions.

Note that SIMD and multi-threading optimizations only apply to vectorized calls, but common-subexpression elimination applies to both scalar and vectorized operations.

## Optimization Level

In version 2.6, the `opt_level` option is added to `compile_*` functions. `opt_level` accepts a value of 0, 1, or 2. The default is 1. Broadly, the levels are parallel to -O0, -O1, -O2 options in gcc and clang. Level-0 performs minimum amount of optimization. Level-1 does basic optimization, such as caching and peephole optimization. Level-2 uses an improved graph-coloring algorithm (based on petgraph crate) for better register allocation. However, level-2 may fail with a warning and revert back to level-1.

## Fast Functions

The result of different `compile` functions is a Python object, say `f`, that encapsulates the underlying compiled code. When we call `f(...)`,  `f.__call__` is called with the arguments. Then, `__call__` checks the type of arguments (scalar vs. vector), packages the inputs accordingly, calls the correct compiled routine via the respective Rust routines, and finally, formats the return values. All these actions have an overhead. The overhead is acceptable if the compiled function is large and complex, but it becomes relatively too expensive if the function is simple and lightweight. In this situation, it is faster to call the underlying compiled code directly. If the following conditions hold, it is possible to do so:

1. The output is a single **scalar** expression.
2. There are zero to eight **scalar** input arguments.
3. There is no parameter.

In most cases, *Symjit* can automatically switch a function to a fast one. However, there are situations when using the fast function directly improves performance. For example, this applies when passing functions to Scipy integration functions (`quad`, `nquad`, `dbpquad`, `tplquad`). To assist this, we can access the fast function by calling `f.fast_func()`. The result is a `ctypes.CFUNCTYPE`-generated foreign function. For example, we can rewrite the integration example above as

```python
import numpy as np
from scipy.integrate import nquad
from sympy import symbols, exp
from symjit import compile_func

def integrate():
    N = 5
    t, x = symbols("t x")
    f = compile_func([t, x], exp(-t*x)/t**N)
    fast = f.fast_func()
    return nquad(lambda t, x: fast(t, x), [[1, np.inf], [0, np.inf]])

sol = integrate()
np.testing.assert_approx_equal(sol[0], 1/N)
```

Some points. First, we pass a lambda function to `nquad` because of the peculiarities of `nquad` (and other Scipy integration routines) concerning the expected signature of the foreign functions. We plan to generate the correct signature in a future version. Second, the lifetime of the fast function is linked to `f`. If `f` goes out of the scope and is garbage-collected, the fast function becomes invalid. Therefore, never store the resulting fast function separately from the parent `f`. Thus, in the example above, we had to add the `integrate` function to provide a scope for the fast function.

## `apply` method

The output of `compile_func` function, say `f`, mimics the behavior and signature of Sympy `lambdify` as muas possible. Whereas this works well for many standard cases, it incurs a time penalty when there are too many state variables. Specially if the input to `f` is already in an iterable or numpy array (for example, an array `y`), then when `f` is called as `f(*y)`, `y` is deconstructed first and is then converted to a numpy array by the `f` Python stub before the Rust code is called. For large `y`, this redundant step can dominate the spent time.

As a workaround, in Version 2.5, we have added an `apply` method to `f`, which accept one or two iterables/arrays. Therefore, its signature is `f(y)` or `f(y, params)`, where `y` corresponds to the state variables and `params` to optional arguments.

```python
    X = symbols('X[0:100]')
    f = compile_func(X, sum(X))
    u = np.arange(100, dtype=np.float64)

    assert(f(*u) == np.float64(4950.0))
    assert(f.apply(u) == np.array([4950.]))

    g = lambdify(X, sum(X))
    assert(f(*u) == np.float64(4950.0))
```

Note that the output of `apply` is always a numpy array.

## Callable

`scipy.LowLevelCallable` is a method to speed up certain Scipy functions by passing a compiled function. Currently, the functions using this feature are mainly doing numerical integration (e.g., `quad`, `dblquad`, `tplquad` in `scipy.integrate`) and image filtering (e.g., `generic_filter` in `scipy.ndimage`). The standard way to create a `scipy.LowLevelCallable` is by wrting the function in C and compile it into a shared library. *Symjit* provides an easier option. As of version 2.5, the object returned by `compile_func` (say, `f` of type `symjit.Func`) has two helper functions that return `LowLevelCallable` objects:

* `Func.callable_quad`: returns a `LowLevelCallable` with a type signature `double (int, double *, void *)`, suitable to be passed to various integration functions (e.g., see `examples/integrate.py`).
* `Func.callable_filter`: returns a `LowLevelCallable` with a type signature `int (double *, npy_intp, double *, void *)`, suitable to be passed to image filtering functions (e.g., see `examples/filter.py`).

The following example shows how to use `callable_quad`:

```python
from scipy.integrate import quad
from sympy import symbols
from symjit import compile_func

x = symbols("x")
f = compile_func([x], x**3)
callable = f.callable_quad()
res = quad(callable, 0, 1)

assert(res[0] == 0.25)
```

As the Scipy ecosystem expands to use `LowLevelCallable` in other applications, we will add them to `Func`.

## Exponentiation to an Integer Power and Modular Exponentiation

Polynomial manipulation over various finite and infinite fields, such as &Zopf;p and &Zopf;, is the cornerstone of computer algebra systems. *Symjit* is primarily designed as a bridge between Sympy and numerical libraries (NumPy, SciPy, ...) and, as such, focuses on floating-point calculations. However, to assist with sympy integer calculations, version 2 has the capability of detecting and emitting special codes for integer exponentiation and modular exponentiation. IEEE 754 doubles can represent integers accurately up to 2**53 = 9007199254740992.

The first special form is `x**n`, where `x` is any variable or expression, and `n` is a constant integer. *Symjit* emits the corresponding code directly in the function byte stream using the exponentiation-by-squaring method. This improves performance by allowing for better register allocations.

The second special form is `x**n % p`, where `p` is any expression. Instead of calculating `x**n` first and then applying `%` (which can easily overflow), *Symjit* incorporates modular reduction at each stage of squaring. For example,

```python
from sympy import symbols
from symjit import compile_func

x = symbols("x")
f = compile_func([x], x ** 1000 % 257)
assert(f(10) == 189)
```

Note that `10**1000` can be represented by a double (the max double value is ~1.8*10**308). Therefore, calculating `10**1000` directly would overflow.
