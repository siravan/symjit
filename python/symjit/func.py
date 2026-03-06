import numbers
from asyncio import SelectorEventLoop

import numpy as np

from . import engine, pyengine


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
            p = np.array(args[self.count_states :])
            self.compiler.params[:] = p

        if isinstance(args[0], numbers.Number):
            if self.f is not None:
                return self.fmt(args)

            u = np.asarray(args[: self.count_states])
            self.compiler.states[:] = u
            self.compiler.execute()
            return self.fmt(self.compiler.obs)
        elif isinstance(self.compiler, pyengine.PyCompiler):
            return self.call_vectorized(*args)
        else:
            return self.call_matrix(*args)

    def call_vectorized(self, *args):
        assert len(args) >= self.count_states
        shape = args[0].shape
        n = args[0].size
        h = max(self.count_states, self.count_obs)
        buf = np.zeros((h, n), dtype=np.float64)

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
                    X = np.zeros(shape, dtype=np.float64)
                    res.append(X)
                    obs.add_row(X)

                self.compiler.execute_matrix(states, obs)

        return self.vecfmt(res)

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return self.compiler.dumps(what=what)

    def fast_func(self):
        return self.f

    def execute_vectorized(self, buf):
        self.compiler.execute_vectorized(buf)

    def apply(self, y, p=None):
        y = np.asarray(y, dtype=np.float64)
        self.compiler.states[:] = y

        if p is not None:
            p = np.asarray(p, dtype=np.float64)
            self.compiler.params[:] = p

        self.compiler.execute()
        return self.compiler.obs

    def callable_quad(self, use_fast=True):
        return self.compiler.callable_quad(use_fast=use_fast)

    def callable_filter(self):
        return self.compiler.callable_filter()

    def save(self, file):
        self.compiler.save(file)


class FuncComplex:
    def __init__(self, compiler, eqs):
        self.compiler = compiler
        self.count_states = self.compiler.count_states
        self.count_params = self.compiler.count_params
        self.count_obs = self.compiler.count_obs
        self.prepare_fmt(eqs)
        self.prepare_vecfmt(eqs)

    def prepare_fmt(self, eqs):
        if isinstance(eqs, list):
            self.fmt = lambda obs: np.frombuffer(obs, dtype=np.complex128).tolist()
        elif isinstance(eqs, tuple):
            self.fmt = lambda obs: tuple(
                np.frombuffer(obs, dtype=np.complex128).tolist()
            )
        else:
            self.fmt = lambda obs: obs[0] + obs[1] * 1j

    def prepare_vecfmt(self, eqs):
        if isinstance(eqs, list):
            self.vecfmt = lambda res: res
        elif isinstance(eqs, tuple):
            self.vecfmt = lambda res: tuple(res)
        else:
            self.vecfmt = lambda res: res[0]

    def __call__(self, *args):
        if isinstance(args[0], numbers.Number):
            u = np.frombuffer(
                np.asarray(args, dtype=np.complex128),
                dtype=np.float64,
            )
            self.compiler.params[: self.count_params] = u[self.count_states :]
            self.compiler.states[:] = u[: self.count_states]
            self.compiler.execute()
            return self.fmt(self.compiler.obs)
        else:
            return self.call_matrix(*args)

    def call_matrix(self, *args):
        if len(args) > self.count_states // 2:
            p = np.frombuffer(
                np.asarray(args[self.count_states // 2 :], dtype=np.complex128),
                dtype=np.float64,
            )
            self.compiler.params[:] = p

        shape = args[0].shape

        with engine.Matrix() as states:
            for i in range(self.count_states // 2):
                assert args[i].shape == shape
                v = np.ascontiguousarray(args[i], dtype=np.complex128)
                states.add_row(v.real)
                states.add_row(v.imag)

            res = []

            with engine.Matrix() as obs:
                AB = []

                for i in range(self.count_obs // 2):
                    a = np.empty(shape, dtype=np.float64)
                    b = np.empty(shape, dtype=np.float64)
                    obs.add_row(a)
                    obs.add_row(b)
                    AB.append((a, b))

                self.compiler.execute_matrix(states, obs)

                for a, b in AB:
                    z = np.empty(shape, dtype=np.complex128)
                    z.real = a
                    z.imag = b
                    res.append(z)

        return self.vecfmt(res)

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return self.compiler.dumps(what=what)

    def fast_func(self):
        return None

    def execute_vectorized(self, buf):
        print("`execute_vectorized` is not implemented for complex functions.")
        pass

    def apply(self, y, p=None):
        pass

    def callable_quad(self, use_fast=True):
        pass

    def callable_filter(self):
        pass

    def save(self, file):
        self.compiler.save(file)


############################################################################


class SymbolicaFunc:
    def __init__(self, model, dtype="float64", **args):
        self.model = model
        self.args = args
        self.cache = {}
        self.cache_mode = False
        self.cache_complex = False
        self.samples = None

        if model is None:
            self.compiler = None
            self.complex_compiler = None
            self.args = {}
            return

        if dtype == "complex128":
            self.compile_complex()
            self.compiler = None
        else:
            self.compile_real()
            self.complex_compiler = None

    def compile_real(self):
        compiler = engine.RustyCompiler(self.model, dtype="float64", **self.args)
        self.compiler = compiler

    def compile_complex(self):
        compiler = engine.RustyCompiler(self.model, dtype="complex128", **self.args)
        self.complex_compiler = compiler

    def evaluate(self, inputs):
        if self.compiler is None:
            self.compile_real()

        if len(self.cache) > 0:
            h = hash(inputs.sum())
            if h in self.cache:
                return self.cache[h]

        c = self.compiler
        outs = np.zeros((inputs.shape[0], c.count_obs), dtype=np.float64)

        if self.cache_mode and inputs.shape[0] == 1:
            self.samples.append(inputs)
            return outs

        args = np.ascontiguousarray(inputs[:, : c.count_params].real, dtype=np.float64)
        c.evaluate_matrix(args, outs)
        return outs

    def evaluate_complex(self, inputs):
        if self.complex_compiler is None:
            self.compile_complex()

        if len(self.cache) > 0:
            h = hash(inputs.sum())
            if h in self.cache:
                return self.cache[h]

        c = self.complex_compiler
        assert inputs.shape[1] == c.count_params // 2
        outs = np.zeros((inputs.shape[0], c.count_obs // 2), dtype=np.complex128)

        if self.cache_mode and inputs.shape[0] == 1:
            self.cache_complex = True
            self.samples.append(inputs)
            return outs

        args = np.ascontiguousarray(inputs, dtype=np.complex128)
        c.evaluate_matrix(args, outs, 2)
        return outs

    def start_caching(self):
        self.cache = {}
        self.cache_mode = True
        self.cache_complex = False
        self.samples = []

    def stop_caching(self):
        self.cache_mode = False

        if len(self.samples) == 0:
            return

        inputs = np.concatenate(self.samples)

        if self.cache_complex:
            outs = self.evaluate_complex(inputs)
        else:
            outs = self.evaluate(inputs)

        for i, s in enumerate(self.samples):
            h = hash(s.sum())
            self.cache[h] = outs[i, None]

        self.samples = None

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)

    def dumps(self, what="scalar", dtype="complex128"):
        if dtype == "complex128" and self.complex_compiler is not None:
            return self.complex_compiler.dumps(what=what)
        elif self.compiler is not None:
            return self.compiler.dumps(what=what)

    def save(self, file, dtype="complex128"):
        if dtype == "complex128":
            self.compile_complex()
            self.complex_compiler.save(file)
        else:
            self.compile_real()
            self.compiler.save(file)


class Bridge:
    def __init__(self, evaluator):
        if isinstance(evaluator, str):
            s = evaluator.replace("𝑖", "j")
            a, b, c = eval(s)
        else:
            a, b, c = evaluator.get_instructions()

        self.instructions = a
        self.count_temps = b
        self.consts = c

    def translate(self):
        p = []

        for q in self.instructions:
            op = q[0]
            args = q[1:]

            if op == "add":
                p.append(self.add(*args))
            elif op == "mul":
                p.append(self.mul(*args))
            elif op == "pow":
                p.append(self.pow(*args))
            elif op == "powf":
                p.append(self.powf(*args))
            elif op == "fun":
                p.append(self.fun(*args))
            elif op == "external_fun":
                p.append(self.external_fun(*args))
            elif op == "assign":
                p.append(self.assign(*args))
            elif op == "if_else":
                p.append(self.if_else(*args))
            elif op == "goto":
                p.append(self.goto(*args))
            elif op == "label":
                p.append(self.label(*args))
            elif op == "join":
                p.append(self.join(*args))
            else:
                raise ValueError("undefined instruction")

        consts = [self.num(x) for x in self.consts]
        return (p, self.count_temps, consts)

    def add(self, dst, args, n):
        return {"Add": [self.slot(dst), self.slot_list(args), n]}

    def mul(self, dst, args, n):
        return {"Mul": [self.slot(dst), self.slot_list(args), n]}

    def pow(self, dst, arg, power, is_real):
        return {"Pow": [self.slot(dst), self.slot(arg), power, is_real]}

    def powf(self, dst, arg, power, is_real):
        return {"Powf": [self.slot(dst), self.slot(arg), self.slot(power), is_real]}

    def fun(self, dst, f, arg, is_real):
        name = f.get_name().split("::")[1]

        if name == "exp":
            f = 2
        elif name == "ln":
            f = 3
        elif name == "sin":
            f = 4
        elif name == "cos":
            f = 5
        elif name == "sqrt":
            f = 6
        elif name == "conjugate":
            f = 7
        else:
            raise ValueError(f"fun {name} is not defined.")

        return {"Fun": [self.slot(dst), f, self.slot(arg), is_real]}

    def external_fun(self, dst, name, args):
        return {"ExternalFun": [self.slot(dst), name, self.slot_list(args)]}

    def assign(self, dst, arg):
        return {"Assign": [self.slot(dst), self.slot(arg)]}

    def if_else(self, dst, lbl):
        return {"IfElse": [self.slot(dst), lbl]}

    def goto(self, lbl):
        return {"Goto": lbl}

    def label(self, lbl):
        return {"Label": lbl}

    def join(self, dst, cond, t, f):
        return {"Join": [self.slot(dst), self.slot(cond), self.slot(t), self.slot(f)]}

    def slot(self, item):
        name = item[0]
        idx = item[1]

        if name == "param":
            return {"Param": idx}
        elif name == "out":
            return {"Out": idx}
        elif name == "temp":
            return {"Temp": idx}
        elif name == "const":
            return {"Const": idx}
        else:
            raise ValueError(f"undefined Slot type: {name}")

    def slot_list(self, item):
        return [self.slot(s) for s in item]

    def num(self, x):
        if isinstance(x, numbers.Number):
            val = complex(x, 0)
        else:
            val = x.evaluate_complex({}, {})

        return {
            "re": {"numerator": {"Single": val.real}, "denominator": {"Single": 1}},
            "im": {"numerator": {"Single": val.imag}, "denominator": {"Single": 1}},
        }
