import ctypes
import json
import os
import platform
import sys

import numpy as np


class Engine:
    def __init__(self):
        dll_name = None

        if sys.platform == "linux" and platform.machine() == "x86_64":
            dll_name = self.find_dll("x86_64-linux")
        if sys.platform == "linux" and platform.machine() == "aarch64":
            dll_name = self.find_dll("aarch64-linux")
        if sys.platform == "linux" and platform.machine() == "riscv64":
            dll_name = self.find_dll("riscv64-linux")
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
        except AttributeError as e:
            print(e)
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
        ]
        self._execute.restype = ctypes.c_bool

        self._evaluate = self.dll.evaluate
        self._evaluate.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # args
            ctypes.c_size_t,  # nargs
            ctypes.POINTER(ctypes.c_double),  # outs
            ctypes.c_size_t,  # nouts
        ]
        self._evaluate.restype = ctypes.c_bool

        self._evaluate_matrix = self.dll.evaluate_matrix
        self._evaluate_matrix.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # args
            ctypes.c_size_t,  # nargs
            ctypes.POINTER(ctypes.c_double),  # outs
            ctypes.c_size_t,  # nouts
        ]
        self._evaluate_matrix.restype = ctypes.c_bool

        self._execute_vectorized = self.dll.execute_vectorized
        self._execute_vectorized.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # buf
            ctypes.c_size_t,  # n
        ]
        self._execute_vectorized.restype = ctypes.c_bool

        self._compile = self.dll.compile
        self._compile.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._compile.restype = ctypes.c_void_p

        self._translate = self.dll.translate
        self._translate.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._translate.restype = ctypes.c_void_p

        self._save = self.dll.save
        self._save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._save.restype = ctypes.c_bool

        self._load = self.dll.load
        self._load.argtypes = [ctypes.c_char_p]
        self._load.restype = ctypes.c_void_p

        self._get_config = self.dll.get_config
        self._get_config.argtypes = [ctypes.c_void_p]
        self._get_config.restype = ctypes.c_size_t

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
            ctypes.c_size_t,
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

        self._callable_quad = self.dll.callable_quad
        self._callable_quad.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
        ]
        self._callable_quad.restype = ctypes.c_double

        self._callable_quad_fast = self.dll.callable_quad_fast
        self._callable_quad_fast.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
        ]
        self._callable_quad_fast.restype = ctypes.c_double

        self._callable_filter = self.dll.callable_filter
        self._callable_filter.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
        ]
        self._callable_filter.restype = ctypes.c_int64

        self._create_defuns = self.dll.create_defuns
        self._create_defuns.argtypes = []
        self._create_defuns.restype = ctypes.c_void_p

        self._add_func = self.dll.add_func
        self._add_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._add_func.restype = None

        self._finalize_defuns = self.dll.finalize_defuns
        self._finalize_defuns.argtypes = [ctypes.c_void_p]
        self._finalize_defuns.restype = None

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
    if count == 0:
        return np.zeros(1)
    else:
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


class Defuns:
    def __init__(self, defuns):
        self.p = lib._create_defuns()
        self.funcs = {}

        fac1 = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
        fac2 = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)

        if defuns is not None:
            for sym, f in defuns.items():
                if hasattr(f, "fast_func"):
                    f = f.fast_func()

                if hasattr(f, "argtypes"):  # f is a CFUNCTION
                    degree = len(f.argtypes)
                else:  # f is a Python bytecode function (normal or lambda)
                    degree = f.__code__.co_argcount
                    if degree == 1:
                        f = fac1(f)
                    elif degree == 2:
                        f = fac2(f)
                    else:
                        raise ValueError(
                            "User-defined functions can have only 1 or 2 arguments"
                        )

                name = str(sym)
                self.funcs[name] = (f, degree)
                lib._add_func(self.p, name.encode("utf8"), f, degree)

    def __del__(self):
        if hasattr(self, "p"):
            lib._finalize_defuns(self.p)


class RustyCompiler:
    def __init__(
        self,
        model,
        ty="native",
        use_simd=True,
        use_threads=True,
        cse=True,
        fastmath=False,
        opt_level=1,
        convert=True,
        defuns=None,
        sanitize=True,
        dtype="float64",
        action="compile",
        file="",
    ):
        if convert:
            model = json.dumps(model, indent=4)

        dtype = str(dtype)
        if dtype not in ["float64", "complex128"]:
            raise ValueError("`dtype` should be `float64` or `complex128`")

        opt = (
            (0x01 if use_simd else 0)
            | (0x02 if use_threads else 0)
            | (0x04 if cse else 0)
            | (0x08 if fastmath else 0)
            | (0x10 if sanitize else 0)
            | (0x20 if dtype == "complex128" else 0)
            | ((opt_level & 0x0F) << 8)
        )

        self.dtype = dtype
        self.defuns = defuns
        self.ty = ty

        if action == "compile":
            self.p = lib._compile(
                model.encode("utf-8"), ty.encode("utf8"), opt, self.defuns.p
            )
            self.symbolica = False
        elif action == "translate":
            self.p = lib._translate(
                model.encode("utf-8"), ty.encode("utf8"), opt, self.defuns.p
            )
            self.symbolica = True
        elif action == "load":
            self.load(file)
        else:
            raise ValueError(f"action {action} not defined")

        status = lib._check_status(self.p)
        if status != b"Success":
            raise ValueError(status.decode())

        self.model = model
        self.json_model = None
        self.populate()

    def __del__(self):
        if hasattr(self, "p"):
            lib._finalize(self.p)

    def save(self, file):
        lib._save(self.p, file.encode("utf-8"))

    def load(self, file):
        self.p = lib._load(file.encode("utf-8"))

        opt = lib._get_config(self.p)
        self.symbolica = opt & 0x40 != 0

        if opt & 0x20 != 0:
            self.dtype = "complex128"
        else:
            self.dtype = "float64"

        t = opt >> 32
        if t == 0:
            self.ty = "native"
        elif t == 1:
            self.ty = "amd"
        elif t == 2:
            self.ty = "amd-avx"
        elif t == 3:
            self.ty = "amd-sse"
        elif t == 4:
            self.ty = "arm"
        elif t == 5:
            self.ty = "risvc"
        elif t == 6:
            self.ty = "bytecode"
        elif t == 7:
            self.ty = "debug"

    def get_u0(self):
        if self.json_model is None:
            self.json_model = json.loads(self.model)
        return [x["val"] for x in self.json_model["states"][1:]]

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

    def dumps(self, what="scalar"):
        name = "symjit_dump.bin"
        self.dump(name, what=what)
        with open(name, "rb") as fd:
            b = fd.read()
        os.remove(name)

        if b[0] == ord("#") and b[1] == ord("!"):
            return b.decode("utf8")
        else:
            return b.hex()

    def execute(self):
        if not lib._execute(self.p):
            raise ValueError("cannot execute the model")

    def execute_vectorized(self, buf):
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n = buf.shape[1]
        if not lib._execute_vectorized(self.p, ptr, n):
            raise ValueError("cannot execute the model")

    def execute_matrix(self, states, obs):
        if not lib._execute_matrix(self.p, states.p, obs.p):
            raise ValueError("cannot execute the model")

    def evaluate(self, args, outs):
        pargs = args.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nargs = args.size
        pouts = outs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nouts = outs.size

        if not lib._evaluate(self.p, pargs, nargs, pouts, nouts):
            raise ValueError("cannot evaluate the model")

    def evaluate_matrix(self, args, outs, k=1):
        pargs = args.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nargs = args.size
        pouts = outs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nouts = outs.size

        if not lib._evaluate_matrix(self.p, pargs, nargs * k, pouts, nouts * k):
            raise ValueError("cannot evaluate the model")

    def fast_func(self):
        if self.ty == "bytecode":
            return None

        f = lib._fast_func(self.p)

        if f is None:
            return None

        sig = [ctypes.c_double for _ in range(self.count_states + 1)]
        fac = ctypes.CFUNCTYPE(*sig)
        return fac(f)

    def callable_quad(self, use_fast=True):
        f = lib._fast_func(self.p)

        try:
            from scipy import LowLevelCallable

            if f is not None and use_fast:
                return LowLevelCallable(
                    lib._callable_quad_fast,
                    user_data=ctypes.c_void_p(f),
                    signature="double (int, double *, void *)",
                )
            else:
                return LowLevelCallable(
                    lib._callable_quad,
                    user_data=ctypes.c_void_p(self.p),
                    signature="double (int, double *, void *)",
                )
        except:
            return None

    def callable_filter(self, use_fast=True):
        try:
            from scipy import LowLevelCallable

            return LowLevelCallable(
                lib._callable_filter,
                user_data=ctypes.c_void_p(self.p),
                signature="int (double *, npy_intp, double *, void *)",
            )

        except:
            return None
