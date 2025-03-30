import numpy as np
import numbers

from . import engine
from . import structure
      

class Func:
    def __init__(self, compiler):
        self.compiler = compiler
        self.count_states = self.compiler.count_states
        self.count_params = self.compiler.count_params
        self.count_obs = self.compiler.count_obs        

    def __call__(self, *args):
        if len(args) > self.count_states:
            p = np.array(args[self.count_states:], dtype="double")
            self.compiler.params[:] = p
    
        if isinstance(args[0], numbers.Number):
            u = np.array(args[:self.count_states], dtype="double")
            self.compiler.states[:] = u
            self.compiler.execute()
            return self.compiler.obs.copy()
        else:
            return self.call_vectorized(*args)
            
    def call_vectorized(self, *args):
        assert(len(args) >= self.count_states)
        shape = args[0].shape
        n = args[0].size
        h = max(self.count_states, self.count_obs)
        buf = np.zeros((h, n), dtype="double")

        for i in range(self.count_states):
            assert(args[i].shape == shape)
            buf[i,:] = args[i].ravel()            
        
        self.compiler.execute_vectorized(buf)
        
        res = []
        for i in range(self.count_obs):            
            y = buf[i,:].reshape(shape)
            res.append(y)
            
        return res

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)


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
        self.compiler.dump(name, what=what)        
        

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


def compile_func(states, eqs, params=None, obs=None, ty="native", use_simd=True):
    """Compile a list of symbolic expression into an executable form.
    compile_func tries to mimic sympy lambdify, but instead of generating
    a standard python funciton, it returns a callable (Func object) that
    is a thin wrapper over compiled machine-code.
    
    Parameters
    ==========
    
    states: a single symbol or a list/tuple of symbols.
    eqs: a single symbolic expression or a list/tuple of symbolic expressions.
    params (optional): a list/tuple of additional symbols as parameters to the model.
    ty: target architecture ("amd", "arm", "bytecode", or "native").
    obs (optional): a list of symbols to name equations. If obs is not None, its length should 
        be the same as eqs.
    use_simd (default True): generates SIMD code for vectorized operations. 
        Currently only AVX on X64 systems is supported.
    
    ==> returns a Func object, is a callable object `f` with signature `f(x_1,...,x_n,p_1,...,p_m)`, 
        where `x`s are the state variables and `p`s are the parameters.
    
    >>> import numpy as np
    >>> from symjit import compile_func
    >>> from sympy import symbols
    
    >>> x, y = symbols('x y')
    >>> f = compile_func([x, y], [x+y, x*y])
    >>> assert(np.all(f(3, 5) == [8., 15.]))
    """
    model = structure.model(states, eqs, params=params, obs=obs)
    compiler = engine.RustyCompiler(model, ty=ty, use_simd=use_simd)
    return Func(compiler)


def compile_ode(iv, states, odes, params=None, ty="native", use_simd=False):
    """Compile a symbolic ODE model into an executable form suitable for 
    passung to scipy.integrate.solve_ivp.    
    
    Parameters
    ==========
    
    iv: a single symbol, the independent variable.
    states: a single symbol or a list/tuple of symbols.
    odes: a single symbolic expression or a list/tuple of symbolic expressions,
        representing the derivative of the state with respect to iv.
    params (optional): a list/tuple of additional symbols as parameters to the model
    ty: target architecture ("amd", "arm", "bytecode", or "native").
    use_simd (default False): generates SIMD code for vectorized operations. 
        Currently only AVX on X64 systems is supported.
    
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
    model = structure.model_ode(iv, states, odes, params)
    compiler = engine.RustyCompiler(model, ty=ty, use_simd=use_simd)
    return OdeFunc(compiler)
    
def compile_jac(iv, states, odes, params=None, ty="native", use_simd=False):
    """Genenrates and compiles Jacobian for an ODE system.
        It accepts the same arguments as `compile_ode`.
        
    ===> returns an OdeFunc object that has the same signature as 
        the results of `compile_ode`, i.e., `f(t,y,p0,p1,...)`. 
        However, it returns a n-by-n Jacobian matrix, where n is
        the number of state variables. 
    """
    model = structure.model_jac(iv, states, odes, params)
    compiler = engine.RustyCompiler(model, ty=ty, use_simd=use_simd)
    return JacFunc(compiler)

def compile_json(model, ty="native", use_simd=True):
    """Compiles CellML models
        CellML json files are extracted using CellMLToolkit.jl
        model is already in Json format; hence, `convert = False`
    """
    compiler = engine.RustyCompiler(model, ty=ty, use_simd=use_simd, convert=False)
    return OdeFunc(compiler)
