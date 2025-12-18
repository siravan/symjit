# from engine import Matrix
import numbers
import os
import warnings

import numpy as np
from sympy import lambdify

from . import engine, pyengine, structure


class Func:
    def __init__(self, compiler, eqs):
        self.compiler = compiler
        self.count_states = self.compiler.count_states
        self.count_params = self.compiler.count_params
        self.count_obs = self.compiler.count_obs
        self.f = self.compiler.fast_func()
        self.prepare_fmt(eqs)
        self.prepare_vecfmt(eqs)

    def prepare_fmt(self, eqs):
        if self.f is not None:
            if isinstance(eqs, list):
                self.fmt = lambda args: [self.f(*args)]
            elif isinstance(eqs, tuple):
                self.fmt = lambda args: (self.f(*args),)
            else:
                self.fmt = lambda args: self.f(*args)
        else:
            if isinstance(eqs, list):
                self.fmt = lambda obs: obs.tolist()
            elif isinstance(eqs, tuple):
                self.fmt = lambda obs: tuple(obs.tolist())
            else:
                self.fmt = lambda obs: obs[0]

    def prepare_vecfmt(self, eqs):
        if isinstance(eqs, list):
            self.vecfmt = lambda res: res
        elif isinstance(eqs, tuple):
            self.vecfmt = lambda res: tuple(res)
        else:
            self.vecfmt = lambda res: res[0]

    def __call__(self, *args):
        if len(args) > self.count_states:
            p = np.array(args[self.count_states :], dtype="double")
            self.compiler.params[:] = p

        if isinstance(args[0], numbers.Number):
            if self.f is not None:
                return self.fmt(args)

            u = np.asarray(args[: self.count_states], dtype="double")
            self.compiler.states[:] = u
            self.compiler.execute()
            return self.fmt(self.compiler.obs)
        elif isinstance(self.compiler, pyengine.PyCompiler):
            return self.call_vectorized(*args)
        else:
            # return self.call_vectorized(*args)
            return self.call_matrix(*args)

    def call_vectorized(self, *args):
        assert len(args) >= self.count_states
        shape = args[0].shape
        n = args[0].size
        h = max(self.count_states, self.count_obs)
        buf = np.zeros((h, n), dtype="double")

        for i in range(self.count_states):
            assert args[i].shape == shape
            buf[i, :] = args[i].ravel()

        self.compiler.execute_vectorized(buf)

        res = []
        for i in range(self.count_obs):
            y = buf[i, :].reshape(shape)
            res.append(y)

        return self.vecfmt(res)

    def call_matrix(self, *args):
        assert len(args) >= self.count_states
        shape = args[0].shape

        with engine.Matrix() as states:
            for i in range(self.count_states):
                assert args[i].shape == shape
                states.add_row(args[i])

            res = []

            with engine.Matrix() as obs:
                for i in range(self.count_obs):
                    X = np.zeros(shape, dtype=np.double)
                    res.append(X)
                    obs.add_row(X)

                self.compiler.execute_matrix(states, obs)

        return self.vecfmt(res)

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return dumps(self.compiler, what=what)

    def fast_func(self):
        return self.f

    def execute_vectorized(self, buf):
        self.compiler.execute_vectorized(buf)

    def apply(self, y, p=None):
        y = np.asarray(y, dtype="double")
        self.compiler.states[:] = y

        if p is not None:
            p = np.asarray(p, dtype="double")
            self.compiler.params[:] = p

        self.compiler.execute(0.0)
        # return self.compiler.obs.copy()
        return self.compiler.obs

    def callable_quad(self, use_fast=True):
        return self.compiler.callable_quad(use_fast=use_fast)

    def callable_filter(self):
        return self.compiler.callable_filter()


class OdeFunc:
    def __init__(self, compiler):
        self.compiler = compiler

    def __call__(self, t, y, *args):
        y = np.array(y, dtype="double")
        self.compiler.states[:] = y

        if len(args) > 0:
            p = np.array(args, dtype="double")
            self.compiler.params[:] = p

        self.compiler.execute(t)
        return self.compiler.diffs.copy()

    def get_u0(self):
        return self.compiler.get_u0()

    def get_p(self):
        return self.compiler.get_p()

    def dump(self, name, what="scalar"):
        return self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return dumps(self.compiler, what=what)


class JacFunc:
    def __init__(self, compiler):
        self.compiler = compiler
        self.count_states = self.compiler.count_states

    def __call__(self, t, y, *args):
        y = np.array(y, dtype="double")
        self.compiler.states[:] = y

        if len(args) > 0:
            p = np.array(args, dtype="double")
            self.compiler.params[:] = p

        self.compiler.execute()
        jac = self.compiler.obs.copy()
        return jac.reshape((self.count_states, self.count_states))

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return dumps(self.compiler, what=what)


def dumps(compiler, what="scalar"):
    name = "symjit_dump.bin"
    compiler.dump(name, what=what)
    with open(name, "rb") as fd:
        b = fd.read()
    os.remove(name)
    if b[0] == ord("#") and b[1] == ord("!"):
        return b.decode("utf8")
    else:
        return b.hex()


def can_use_rust(backend):
    if not backend in ["python", "rust"]:
        raise ValueError(f"invalide backend: {backend}")
    return backend == "rust" and engine.lib.is_valid


def can_use_python(backend):
    if not backend in ["python", "rust"]:
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
):
    """Compile a list of symbolic expression into an executable form.
    compile_func tries to mimic sympy lambdify, but instead of generating
    a standard python funciton, it returns a callable (Func object) that
    is a thin wrapper over compiled machine-code.

    Parameters
    ==========

    states: a single symbol or a list/tuple of symbols.
    eqs: a single symbolic expression or a list/tuple of symbolic expressions.
    params (optional): a list/tuple of additional symbols as parameters to the model.
    ty: target architecture. Options are:
        * "amd": generates x64 instructions (amd-sse or amd-avx) depending on the processor
        * "amd-sse": generates x64 SSE2 instructions
        * "amd-avx": generates x64 AVX instrcutions
        * "arm": generates arm aarch64 instructions
        * "bytecode": bytecode interpreter for testing and running on unsupported hardware
        * "native" (default): selects the correct mode based on the processor
        * "debug": runs "native" and "bytecode" codes and throws an exception if different
    obs (optional): a list of symbols to name equations. If obs is not None, its length should
        be the same as eqs.
    backend (default `rust`): the code-generator backend (`rust`: dynamic library coded
        in rust. `python`: pyengine library coded in plain Python.
    use_simd (default True): generates SIMD code for vectorized operations.
        Currently only AVX on x86-64 systems is supported.
    use_threads (default True): use multi-threading to speed up parallel operations when called
        on numpy arrays.
    cse (default True): performs common-subexpression elimination.
    fastmath (default False): use fastmath floating point operations, especially fused multiply-addition.
    opt_level (default 1): optimization level (0, 1, or 2). Broadly the numbers are parallel to -O0, -O1, -O2
        options to gcc and clang. Level-0 performs minimum amount of optimization. Level-1 does peephole optimization.
        Level-2 uses an improved graph-coloring algorithm for better register allocation.
    defuns (default None): a dictionary of Symbol => Definition to pass external Python or Symjit-generated functions.

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
        defuns = engine.Defuns(defuns, eqs)
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
        )
    elif can_use_python(backend):
        model = pyengine.tree.model(states, eqs, params, obs)
        compiler = pyengine.PyCompiler(model, ty=ty)
    else:
        raise ValueError("unsupported platform")

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
):
    """Compile a symbolic ODE model into an executable form suitable for
    passung to scipy.integrate.solve_ivp.

    Parameters
    ==========

    iv: a single symbol, the independent variable.
    states: a single symbol or a list/tuple of symbols.
    odes: a single symbolic expression or a list/tuple of symbolic expressions,
        representing the derivative of the state with respect to iv.
    params (optional): a list/tuple of additional symbols as parameters to the model
    ty (default `native`): see compile_func options for details.
    backend (default `rust`): the code-generator backend (`rust`: dynamic library coded
        in rust. `python`: pyengine library coded in plain Python.
    use_simd (default True): generates SIMD code for vectorized operations.
        Currently only AVX on x86-64 systems is supported.
    use_threads (default True): use multi-threading to speed up parallel operations when called
        on numpy arrays.
    cse (default True): performs common-subexpression elimination.
    fastmath (default False): use fastmath floating point operations, especially fused multiply-addition.
    opt_level (default 1): optimization level (0, 1, or 2). Broadly the numbers are parallel to -O0, -O1, -O2
        options to gcc and clang. Level-0 performs minimum amount of optimization. Level-1 does peephole optimization.
        Level-2 uses an improved graph-coloring algorithm for better register allocation.
    defuns (default None): a dictionary of Symbol => Definition to pass external Python or Symjit-generated functions.

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
        defuns = engine.Defuns(defuns, odes)
        compiler = engine.RustyCompiler(
            model,
            ty=ty,
            use_simd=use_simd,
            cse=cse,
            fastmath=fastmath,
            opt_level=opt_level,
            defuns=defuns,
            sanitize=sanitize,
        )
    elif can_use_python(backend):
        model = pyengine.tree.model_ode(iv, states, odes, params)
        compiler = pyengine.PyCompiler(model)
    else:
        raise ValueError("unsupported platform")

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
        use_simd (default True): generates SIMD code for vectorized operations.
            Currently only AVX on x86-64 systems is supported.
        use_threads (default True): use multi-threading to speed up parallel operations when called
            on numpy arrays.
        cse (default True): performs common-subexpression elimination.
        fastmath (default False): use fastmath floating point operations, especially fused multiply-addition.
        opt_level (default 1): optimization level (0, 1, or 2). Broadly the numbers are parallel to -O0, -O1, -O2
            options to gcc and clang. Level-0 performs minimum amount of optimization. Level-1 does peephole optimization.
            Level-2 uses an improved graph-coloring algorithm for better register allocation.
        defuns (default None): a dictionary of Symbol => Definition to pass external Python or Symjit-generated functions.

        Note that similar to compile_ode, compile_jac accepts use_simd and use_threads but in
            practice ingores them, because compile_ode is usually called on scalars only.

    ===> returns an OdeFunc object that has the same signature as
        the results of `compile_ode`, i.e., `f(t,y,p0,p1,...)`.
        However, it returns a n-by-n Jacobian matrix, where n is
        the number of state variables.
    """
    if can_use_rust(backend):
        model = structure.model_jac(iv, states, odes, params)
        defuns = engine.Defuns(defuns, odes)
        compiler = engine.RustyCompiler(
            model,
            ty=ty,
            use_simd=use_simd,
            cse=cse,
            fastmath=fastmath,
            opt_level=opt_level,
            defuns=defuns,
            sanitize=sanitize,
        )
    elif can_use_python(backend):
        model = pyengine.tree.model_jac(iv, states, odes, params)
        compiler = pyengine.PyCompiler(model)
    else:
        raise ValueError("unsupported platform")

    return JacFunc(compiler)


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
):
    """Compiles CellML models
    CellML json files are extracted using CellMLToolkit.jl
    model is already in Json format; hence, `convert = False`
    """
    if can_use_rust("rust"):
        defuns = engine.Defuns(None, [])
        compiler = engine.RustyCompiler(
            model,
            ty=ty,
            use_simd=use_simd,
            use_threads=use_threads,
            cse=cse,
            fastmath=fastmath,
            opt_level=opt_level,
            convert=False,
            defuns=defuns,
            sanitize=sanitize,
        )
        return OdeFunc(compiler)
    else:
        raise ValueError("CellML json files only work with the rust backend")
