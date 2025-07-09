import os
import sys
import json
import ctypes
import platform
import numpy as np


class Engine:
    def __init__(self):
        dll_name = None

        if sys.platform == "linux" and platform.machine() == "x86_64":
            dll_name = self.find_dll("x86_64-linux")
        if sys.platform == "linux" and platform.machine() == "aarch64":
            dll_name = self.find_dll("aarch64-linux")
        if sys.platform == "darwin":
            dll_name = self.find_dll("darwin")
        elif sys.platform == "win32":
            dll_name = self.find_dll("win_amd64")

        if dll_name is None:
            self.is_valid = False
            return

        try:
            dll_path = os.path.join(os.path.dirname(__file__), dll_name)
            self.dll = ctypes.CDLL(dll_path)
            self.populate()
            self.is_valid = True
        except:
            self.is_valid = False

    def populate(self):
        self._info = self.dll.info
        self._info.argtypes = []
        self._info.restype = ctypes.c_char_p

        self._check_status = self.dll.check_status
        self._check_status.argtypes = [ctypes.c_void_p]
        self._check_status.restype = ctypes.c_char_p

        self._count_states = self.dll.count_states
        self._count_states.argtypes = [ctypes.c_void_p]
        self._count_states.restype = ctypes.c_size_t

        self._count_params = self.dll.count_params
        self._count_params.argtypes = [ctypes.c_void_p]
        self._count_params.restype = ctypes.c_size_t

        self._count_obs = self.dll.count_obs
        self._count_obs.argtypes = [ctypes.c_void_p]
        self._count_obs.restype = ctypes.c_size_t

        self._count_diffs = self.dll.count_diffs
        self._count_diffs.argtypes = [ctypes.c_void_p]
        self._count_diffs.restype = ctypes.c_size_t

        self._run = self.dll.run
        self._run.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # du
            ctypes.POINTER(ctypes.c_double),  # u
            ctypes.c_size_t,  # ns
            ctypes.POINTER(ctypes.c_double),  # p
            ctypes.c_size_t,  # np
            ctypes.c_double,  # t
        ]
        self._run.restype = ctypes.c_bool

        self._execute = self.dll.execute
        self._execute.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_double,  # t
        ]
        self._execute.restype = ctypes.c_bool

        self._execute_vectorized = self.dll.execute_vectorized
        self._execute_vectorized.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # buf
            ctypes.c_size_t,  # n
        ]
        self._execute_vectorized.restype = ctypes.c_bool

        self._compile = self.dll.compile
        self._compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
        self._compile.restype = ctypes.c_void_p

        self._dump = self.dll.dump
        self._dump.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self._dump.restype = ctypes.c_bool

        self._finalize = self.dll.finalize
        self._finalize.argtypes = [ctypes.c_void_p]
        self._finalize.restype = None

        self._ptr_states = self.dll.ptr_states
        self._ptr_states.argtypes = [ctypes.c_void_p]
        self._ptr_states.restype = ctypes.POINTER(ctypes.c_double)

        self._ptr_params = self.dll.ptr_params
        self._ptr_params.argtypes = [ctypes.c_void_p]
        self._ptr_params.restype = ctypes.POINTER(ctypes.c_double)

        self._ptr_obs = self.dll.ptr_obs
        self._ptr_obs.argtypes = [ctypes.c_void_p]
        self._ptr_obs.restype = ctypes.POINTER(ctypes.c_double)

        self._ptr_diffs = self.dll.ptr_diffs
        self._ptr_diffs.argtypes = [ctypes.c_void_p]
        self._ptr_diffs.restype = ctypes.POINTER(ctypes.c_double)

        self._fast_func = self.dll.fast_func
        self._fast_func.argtypes = [ctypes.c_void_p]
        self._fast_func.restype = ctypes.c_void_p

        ######################################################

        self._create_matrix = self.dll.create_matrix
        self._create_matrix.argtypes = []
        self._create_matrix.restype = ctypes.c_void_p

        self._add_row = self.dll.add_row
        self._add_row.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t
        ]
        self._add_row.restype = None

        self._finalize_matrix = self.dll.finalize_matrix
        self._finalize_matrix.argtypes = [ctypes.c_void_p]
        self._finalize_matrix.restype = None

        self._execute_matrix = self.dll.execute_matrix
        self._execute_matrix.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # states
            ctypes.c_void_p,  # obs
        ]
        self._execute_matrix.restype = ctypes.c_bool

    def info(self):
        return self._info()

    def find_dll(self, substr):
        files = os.listdir(os.path.dirname(__file__))
        matches = list(filter(lambda s: s.find(substr) >= 0, files))
        if len(matches) == 0:
            return None
        else:
            return matches[0]


#################################################################

lib = Engine()  # interface to the rust codegen engine


def from_raw_parts(ptr, count):
    return np.ctypeslib.as_array(ptr, shape=(count,))


class Matrix:
    def __init__(self):
        self.p = lib._create_matrix()
        self.rows = []  # the list of new rows owned by self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        lib._finalize_matrix(self.p)

    def add_row(self, row):
        v = np.ascontiguousarray(row, dtype=np.double)

        # if v is a different array than row, then it needs to be
        # preserved for the lifetime of the Matrix
        if v is not row:
            self.rows.append(v)

        ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n = v.size
        lib._add_row(self.p, ptr, n)


class RustyCompiler:
    def __init__(self, model, ty="native", use_simd=True, convert=True):
        if convert:
            model = json.dumps(model)
        self.p = lib._compile(model.encode("utf-8"), ty.encode("utf8"), use_simd)
        status = lib._check_status(self.p)
        if status != b"Success":
            raise ValueError(status.decode())
        self.model = model
        self.json_model = None
        self.ty = ty
        self.populate()

    def __del__(self):
        if hasattr(self, "p"):
            lib._finalize(self.p)

    def get_u0(self):
        if self.json_model is None:
            self.json_model = json.loads(self.model)
        return [x["val"] for x in self.json_model["states"]]

    def get_p(self):
        if self.json_model is None:
            self.json_model = json.loads(self.model)
        return [x["val"] for x in self.json_model["params"]]

    def populate(self):
        self.count_states = lib._count_states(self.p)
        self.count_params = lib._count_params(self.p)
        self.count_obs = lib._count_obs(self.p)
        self.count_diffs = lib._count_diffs(self.p)

        self.states = from_raw_parts(lib._ptr_states(self.p), self.count_states)
        self.params = from_raw_parts(lib._ptr_params(self.p), self.count_params)
        self.obs = from_raw_parts(lib._ptr_obs(self.p), self.count_obs)
        self.diffs = from_raw_parts(lib._ptr_diffs(self.p), self.count_diffs)

    def dump(self, name, what="scalar"):
        if not lib._dump(self.p, name.encode("utf-8"), what.encode("utf-8")):
            raise ValueError("cannot dump the requested code")
        with open(name, "rb") as fd:
            buf = fd.read()
            return buf

    def execute(self, t=0.0):
        if not lib._execute(self.p, t):
            raise ValueError("cannot execute the model")

    def execute_vectorized(self, buf):
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n = buf.shape[1]
        if not lib._execute_vectorized(self.p, ptr, n):
            raise ValueError("cannot execute the model")

    def execute_matrix(self, states, obs):
        if not lib._execute_matrix(self.p, states.p, obs.p):
            raise ValueError("cannot execute the model")

    def fast_func(self):
        if self.ty == "bytecode":
            return None

        f = lib._fast_func(self.p)

        if f is None:
            return None

        sig = [ctypes.c_double for _ in range(self.count_states + 1)]
        fac = ctypes.CFUNCTYPE(*sig)
        return fac(f)
