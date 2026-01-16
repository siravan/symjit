import numbers

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
            self.compiler.params[:] = u[self.count_states :]
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
