# from engine import Matrix
import json
import warnings

from sympy import Symbol, lambdify

from . import engine, pyengine, structure
from .func import *
from .ode import *


def can_use_rust(backend):
    if backend not in ["python", "rust"]:
        raise ValueError(f"invalide backend: {backend}")
    return backend == "rust" and engine.lib.is_valid


def can_use_python(backend):
    if backend not in ["python", "rust"]:
        raise ValueError(f"invalide backend: {backend}")
    warnings.warn(
        "The Python codegen backend is deprecated and will be removed in a future version.",
        DeprecationWarning,
    )
    return pyengine.can_compile()


def compile_func(
    states,
    eqs,
    params=None,
    obs=None,
    ty="native",
    use_simd=True,
    use_threads=True,
    cse=True,
    fastmath=False,
    backend="rust",
    opt_level=1,
    defuns=None,
    sanitize=True,
    dtype="float64",
):
    """Compile a list of symbolic expression into an executable form.
    compile_func tries to mimic sympy lambdify, but instead of generating
    a standard python funciton, it returns a callable (Func object) that
    is a thin wrapper over compiled machine-c

    Parameters
    ==========

    states: a single symbol or a list/tuple of symbols.
    eqs: a single symbolic expression or a list/tuple of symbolic expressions.
    params (optional): a list/tuple of additional symbols as parameters to the model.
    ty: target architecture. Options are:
        * "amd": generates x86-64 instructions (amd-sse or amd-avx) depending on the processor.
        * "amd-sse": generates x86-64 SSE2 instructions.
        * "amd-avx": generates x86-64 AVX instrcutions.
        * "arm": generates arm aarch64 instructions.
        * "riscv": generates 64-bit RISC-V instructions.
        * "bytecode": bytecode interpreter for testing and running on unsupported hardware.
        * "native" (default): selects the correct mode based on the processor.
        * "debug": runs "native" and "bytecode" codes and throws an exception if different.
    obs (default `None`): a list of symbols to name equations. If obs is not None, its length should
        be the same as eqs. A named prefixed with `__` is considered a hidden observable (temporary
        variable).
    backend (default `rust`): the code-generator backend (`rust`: dynamic library coded
        in rust. `python`: pyengine library coded in plain Python.
    dtype (default `float64`): the data type. Possibilities are `float64` and `complex128`.
    use_simd (default `True`): generates SIMD code for vectorized operations. Currently supports
        AVX on x86-64 and NEON on aarch64 systems.
    use_threads (default `True`): use multi-threading to speed up parallel operations when called
        on numpy arrays.
    cse (default `True`): performs common-subexpression elimination.
    fastmath (default False): use fastmath floating point operations, especially fused multiply-addition.
    opt_level (default 1): optimization level (0, 1, or 2). Broadly the numbers are parallel to -O0, -O1, -O2
        options to gcc and clang. Level-0 performs minimum amount of optimization. Level-1 does peephole optimization.
        Level-2 uses an improved graph-coloring algorithm for better register allocation.
    defuns (default `None`): a dictionary of Symbol => Definition to pass external Python or Symjit-generated functions.

    ==> returns a Func object, is a callable object `f` with signature `f(x_1,...,x_n,p_1,...,p_m)`,
        where `x`s are the state variables and `p`s are the parameters.

    >>> import numpy as np
    >>> from symjit import compile_func
    >>> from sympy import symbols

    >>> x, y = symbols('x y')
    >>> f = compile_func([x, y], [x+y, x*y])
    >>> assert(np.all(f(3, 5) == [8., 15.]))
    >>> assert(np.all(f.apply([3, 5]) == [8., 15.]))
    """
    if backend == "sympy":
        return lambdify(states, eqs)

    if can_use_rust(backend):
        model = structure.model(states, eqs, params=params, obs=obs)
        defuns = engine.Defuns(defuns)
        compiler = engine.RustyCompiler(
            model,
            ty=ty,
            use_simd=use_simd,
            use_threads=use_threads,
            cse=cse,
            fastmath=fastmath,
            opt_level=opt_level,
            defuns=defuns,
            sanitize=sanitize,
            dtype=dtype,
        )
    elif can_use_python(backend):
        model = pyengine.tree.model(states, eqs, params, obs)
        compiler = pyengine.PyCompiler(model, ty=ty)
    else:
        raise ValueError("unsupported platform")

    if dtype == "complex128":
        return FuncComplex(compiler, eqs)
    else:
        return Func(compiler, eqs)


def compile_ode(
    iv,
    states,
    odes,
    params=None,
    ty="native",
    use_simd=True,
    use_threads=True,
    cse=True,
    fastmath=False,
    backend="rust",
    opt_level=1,
    defuns=None,
    sanitize=True,
    dtype="float64",
):
    """Compile a symbolic ODE model into an executable form suitable for
    passung to scipy.integrate.solve_ivp.

    Parameters
    ==========

    iv: a single symbol, the independent variable.
    states: a single symbol or a list/tuple of symbols.
    odes: a single symbolic expression or a list/tuple of symbolic expressions,
        representing the derivative of the state with respect to iv.
    params (optional): a list/tuple of additional symbols as parameters to the model.
    ty (default `native`): see `compile_func` options for details.
    backend (default `rust`): the code-generator backend (`rust`: dynamic library coded
        in rust. `python`: pyengine library coded in plain Python.
    dtype (default `float64`): the data type. Possibilities are `float64` and `complex128`.
    use_simd (default `True`): generates SIMD code for vectorized operations. Currently
        supports AVX on x86-64 and NEON on aarch64 systems.
    use_threads (default `True`): use multi-threading to speed up parallel operations
        when called on numpy arrays.
    cse (default `True`): performs common-subexpression elimination.
    fastmath (default `False`): use fastmath floating point operations, especially fused multiply-addition.
    opt_level (default 1): optimization level (0, 1, or 2). Broadly the numbers are parallel to -O0, -O1, -O2
        options to gcc and clang. Level-0 performs minimum amount of optimization. Level-1 does peephole optimization.
        Level-2 uses an improved graph-coloring algorithm for better register allocation.
    defuns (default `None`): a dictionary of Symbol => Definition to pass external Python or
        Symjit-generated functions.

    Note that compile_ode accepts use_simd and use_threads but in practice ingores them,
        because compile_ode is usually called on scalars only.

    invariant => len(states) == len(odes)

    ==> returns an OdeFunc object, is a callable object `f` with signature `f(t,y,p0,p1,...)`,
        where `t` is the value of the independent variable, `y` is the state (an array of
        state variables), and `p`s are the parameters.

    >>> import scipy.integrate
    >>> import numpy as np
    >>> from sympy import symbols
    >>> from symjit import compile_ode

    >>> t, x, y = symbols('t x y')
    >>> f = compile_ode(t, (x, y), (y, -x))
    >>> t_eval=np.arange(0, 10, 0.01)
    >>> sol = scipy.integrate.solve_ivp(f, (0, 10), (0.0, 1.0), t_eval=t_eval)

    >>> np.testing.assert_allclose(sol.y[0,:], np.sin(t_eval), atol=0.005)
    """
    if can_use_rust(backend):
        model = structure.model_ode(iv, states, odes, params)
        defuns = engine.Defuns(defuns)
        compiler = engine.RustyCompiler(
            model,
            ty=ty,
            use_simd=use_simd,
            use_threads=use_threads,
            cse=cse,
            fastmath=fastmath,
            opt_level=opt_level,
            defuns=defuns,
            sanitize=sanitize,
            dtype=dtype,
        )
    elif can_use_python(backend):
        model = pyengine.tree.model_ode(iv, states, odes, params)
        compiler = pyengine.PyCompiler(model)
    else:
        raise ValueError("unsupported platform")

    if dtype == "complex128":
        return OdeFuncComplex(compiler)
    else:
        return OdeFunc(compiler)


def compile_jac(
    iv,
    states,
    odes,
    params=None,
    ty="native",
    use_simd=True,
    use_threads=True,
    cse=True,
    fastmath=False,
    backend="rust",
    opt_level=1,
    defuns=None,
    sanitize=True,
    dtype="float64",
):
    """Genenrates and compiles Jacobian for an ODE system.
        iv: a single symbol, the independent variable.
        states: a single symbol or a list/tuple of symbols.
        odes: a single symbolic expression or a list/tuple of symbolic expressions,
            representing the derivative of the state with respect to iv.
        params (optional): a list/tuple of additional symbols as parameters to the model
        ty (default `native`): see compile_func options for details.
        backend (default `rust`): the code-generator backend (`rust`: dynamic library coded
            in rust. `python`: pyengine library coded in plain Python.
        dtype (default `float64`): the data type. Possibilities are `float64` and `complex128`.
        use_simd (default `True`): generates SIMD code for vectorized operations. Currently
            supports AVX on x86-64 and NEON on aarch64 systems.
        use_threads (default `True`): use multi-threading to speed up parallel operations when called
            on numpy arrays.
        cse (default `True`): performs common-subexpression elimination.
        fastmath (default `False`): use fastmath floating point operations, especially fused multiply-addition.
        opt_level (default 1): optimization level (0, 1, or 2). Broadly the numbers are parallel to -O0, -O1, -O2
            options to gcc and clang. Level-0 performs minimum amount of optimization. Level-1 does peephole optimization.
            Level-2 uses an improved graph-coloring algorithm for better register allocation.
        defuns (default `None`): a dictionary of Symbol => Definition to pass external Python or
            Symjit-generated functions.

        Note that similar to `compile_ode`, `compile_jac` accepts use_simd and use_threads but in
            practice ingores them, because compile_ode is usually called on scalars only.

    ===> returns an OdeFunc object that has the same signature as
        the results of `compile_ode`, i.e., `f(t,y,p0,p1,...)`.
        However, it returns a n-by-n Jacobian matrix, where n is
        the number of state variables.
    """
    if can_use_rust(backend):
        model = structure.model_jac(iv, states, odes, params)
        defuns = engine.Defuns(defuns)
        compiler = engine.RustyCompiler(
            model,
            ty=ty,
            use_simd=use_simd,
            use_threads=use_threads,
            cse=cse,
            fastmath=fastmath,
            opt_level=opt_level,
            defuns=defuns,
            sanitize=sanitize,
            dtype=dtype,
        )
    elif can_use_python(backend):
        model = pyengine.tree.model_jac(iv, states, odes, params)
        compiler = pyengine.PyCompiler(model)
    else:
        raise ValueError("unsupported platform")

    if dtype == "complex128":
        return JacFuncComplex(compiler)
    else:
        return JacFunc(compiler)


def update_json_model(model):
    """
    Updates json models to comform to API change in ver 2.11,
    where the explicit independent-variable was removed and appended
    to the front of the states.
    """
    if not isinstance(model, dict):
        model = json.loads(model)

    if len(model["odes"]) == 0:
        return model

    model["states"] = [model["iv"]] + model["states"]
    model["iv"] = structure.var(Symbol("$_"))
    return model


def compile_json(
    model,
    ty="native",
    use_simd=True,
    use_threads=True,
    cse=True,
    fastmath=False,
    opt_level=1,
    backend="rust",
    sanitize=True,
    dtype="float64",
):
    """Compiles CellML models
    CellML json files are extracted using CellMLToolkit.jl
    model is already in Json format; hence, `convert = False`
    """

    model = update_json_model(model)

    if can_use_rust("rust"):
        defuns = engine.Defuns(None)
        compiler = engine.RustyCompiler(
            model,
            ty=ty,
            use_simd=use_simd,
            use_threads=use_threads,
            cse=cse,
            fastmath=fastmath,
            opt_level=opt_level,
            defuns=defuns,
            sanitize=sanitize,
            dtype=dtype,
        )

        if compiler.count_diffs == 0:
            if dtype == "complex128":
                return FuncComplex(compiler, [])
            else:
                return Func(compiler, [])
        else:
            if dtype == "complex128":
                return OdeFuncComplex(compiler)
            else:
                return OdeFunc(compiler)
    else:
        raise ValueError("CellML json files only work with the rust backend")
