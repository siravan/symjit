import numpy as np


class OdeFunc:
    def __init__(self, compiler):
        self.compiler = compiler

    def __call__(self, t, y, *args):
        # y = np.array(y, dtype="double")
        self.compiler.states[0] = t
        self.compiler.states[1:] = y

        if len(args) > 0:
            # p = np.array(args, dtype="double")
            self.compiler.params[:] = args

        self.compiler.execute()
        return self.compiler.diffs.copy()

    def get_u0(self):
        return self.compiler.get_u0()

    def get_p(self):
        return self.compiler.get_p()

    def dump(self, name, what="scalar"):
        return self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return self.compiler.dumps(what=what)


class OdeFuncComplex:
    def __init__(self, compiler):
        self.compiler = compiler

    def __call__(self, t, y, *args):
        self.compiler.states[0] = t.real
        self.compiler.states[1] = t.imag

        u = np.frombuffer(
            np.asarray(y, dtype=np.complex128),
            dtype=np.float64,
        )

        self.compiler.states[2:] = u

        if len(args) > 0:
            p = np.frombuffer(
                np.asarray(args, dtype=np.complex128),
                dtype=np.float64,
            )
            self.compiler.params[:] = p

        self.compiler.execute()

        z = np.empty(self.compiler.count_diffs // 2, dtype=np.complex128)
        z.real = self.compiler.diffs[::2]
        z.imag = self.compiler.diffs[1::2]
        return z

    def get_u0(self):
        return self.compiler.get_u0()

    def get_p(self):
        return self.compiler.get_p()

    def dump(self, name, what="scalar"):
        return self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return self.compiler.dumps(what=what)


class JacFunc:
    def __init__(self, compiler):
        self.compiler = compiler
        self.count_states = self.compiler.count_states

    def __call__(self, t, y, *args):
        y = np.array(y, dtype="double")
        self.compiler.states[0] = t
        self.compiler.states[1:] = y

        if len(args) > 0:
            p = np.array(args, dtype="double")
            self.compiler.params[:] = p

        self.compiler.execute()
        jac = self.compiler.obs.copy()
        return jac.reshape((self.count_states - 1, self.count_states - 1))

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return self.compiler.dumps(what=what)


class JacFuncComplex:
    def __init__(self, compiler):
        self.compiler = compiler
        self.count_states = self.compiler.count_states

    def __call__(self, t, y, *args):
        self.compiler.states[0] = t.real
        self.compiler.states[1] = t.imag

        u = np.frombuffer(
            np.asarray(y, dtype=np.complex128),
            dtype=np.float64,
        )

        self.compiler.states[2:] = u

        if len(args) > 0:
            p = np.frombuffer(
                np.asarray(args, dtype=np.complex128),
                dtype=np.float64,
            )
            self.compiler.params[:] = p

        self.compiler.execute()

        z = np.empty(self.compiler.count_obs // 2, dtype=np.complex128)
        z.real = self.compiler.obs[::2]
        z.imag = self.compiler.obs[1::2]

        return z.reshape((self.count_states // 2 - 1, self.count_states // 2 - 1))

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)

    def dumps(self, what="scalar"):
        return self.compiler.dumps(what=what)
