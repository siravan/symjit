import ctypes
import json
import os
import platform
import sys
import warnings

import numpy as np
from numpy.typing import NDArray
from typing import Dict


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
        self.info = self.dll.info
        self.info.argtypes = []
        self.info.restype = ctypes.c_char_p

        self.check_status = self.dll.check_status
        self.check_status.argtypes = [ctypes.c_void_p]
        self.check_status.restype = ctypes.c_char_p

        self.count_states = self.dll.count_states
        self.count_states.argtypes = [ctypes.c_void_p]
        self.count_states.restype = ctypes.c_size_t

        self.count_params = self.dll.count_params
        self.count_params.argtypes = [ctypes.c_void_p]
        self.count_params.restype = ctypes.c_size_t

        self.count_obs = self.dll.count_obs
        self.count_obs.argtypes = [ctypes.c_void_p]
        self.count_obs.restype = ctypes.c_size_t

        self.count_diffs = self.dll.count_diffs
        self.count_diffs.argtypes = [ctypes.c_void_p]
        self.count_diffs.restype = ctypes.c_size_t

        self.run = self.dll.run
        self.run.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # du
            ctypes.POINTER(ctypes.c_double),  # u
            ctypes.c_size_t,  # ns
            ctypes.POINTER(ctypes.c_double),  # p
            ctypes.c_size_t,  # np
            ctypes.c_double,  # t
        ]
        self.run.restype = ctypes.c_bool

        self.execute = self.dll.execute
        self.execute.argtypes = [
            ctypes.c_void_p,  # handle
        ]
        self.execute.restype = ctypes.c_bool

        self.evaluate = self.dll.evaluate
        self.evaluate.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # args
            ctypes.c_size_t,  # nargs
            ctypes.POINTER(ctypes.c_double),  # outs
            ctypes.c_size_t,  # nouts
        ]
        self.evaluate.restype = ctypes.c_bool

        self.evaluate_matrix = self.dll.evaluate_matrix
        self.evaluate_matrix.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # args
            ctypes.c_size_t,  # nargs
            ctypes.POINTER(ctypes.c_double),  # outs
            ctypes.c_size_t,  # nouts
        ]
        self.evaluate_matrix.restype = ctypes.c_bool

        self.execute_vectorized = self.dll.execute_vectorized
        self.execute_vectorized.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # buf
            ctypes.c_size_t,  # n
        ]
        self.execute_vectorized.restype = ctypes.c_bool

        self.compile = self.dll.compile
        self.compile.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.compile.restype = ctypes.c_void_p

        self.translate = self.dll.translate
        self.translate.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.translate.restype = ctypes.c_void_p

        self.save = self.dll.save
        self.save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.save.restype = ctypes.c_bool

        self.load = self.dll.load
        self.load.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.load.restype = ctypes.c_void_p

        self.get_config = self.dll.get_config
        self.get_config.argtypes = [ctypes.c_void_p]
        self.get_config.restype = ctypes.c_size_t

        self.dump = self.dll.dump
        self.dump.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.dump.restype = ctypes.c_bool

        self.measure = self.dll.measure
        self.measure.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.measure.restype = ctypes.c_size_t

        self.finalize = self.dll.finalize
        self.finalize.argtypes = [ctypes.c_void_p]
        self.finalize.restype = None

        self.ptr_states = self.dll.ptr_states
        self.ptr_states.argtypes = [ctypes.c_void_p]
        self.ptr_states.restype = ctypes.POINTER(ctypes.c_double)

        self.ptr_params = self.dll.ptr_params
        self.ptr_params.argtypes = [ctypes.c_void_p]
        self.ptr_params.restype = ctypes.POINTER(ctypes.c_double)

        self.ptr_obs = self.dll.ptr_obs
        self.ptr_obs.argtypes = [ctypes.c_void_p]
        self.ptr_obs.restype = ctypes.POINTER(ctypes.c_double)

        self.ptr_diffs = self.dll.ptr_diffs
        self.ptr_diffs.argtypes = [ctypes.c_void_p]
        self.ptr_diffs.restype = ctypes.POINTER(ctypes.c_double)

        self.fast_func = self.dll.fast_func
        self.fast_func.argtypes = [ctypes.c_void_p]
        self.fast_func.restype = ctypes.c_void_p

        ######################################################

        self.create_matrix = self.dll.create_matrix
        self.create_matrix.argtypes = []
        self.create_matrix.restype = ctypes.c_void_p

        self.add_row = self.dll.add_row
        self.add_row.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.add_row.restype = None

        self.finalize_matrix = self.dll.finalize_matrix
        self.finalize_matrix.argtypes = [ctypes.c_void_p]
        self.finalize_matrix.restype = None

        self.execute_matrix = self.dll.execute_matrix
        self.execute_matrix.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # states
            ctypes.c_void_p,  # obs
        ]
        self.execute_matrix.restype = ctypes.c_bool

        self.callable_quad = self.dll.callable_quad
        self.callable_quad.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
        ]
        self.callable_quad.restype = ctypes.c_double

        self.callable_quad_fast = self.dll.callable_quad_fast
        self.callable_quad_fast.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
        ]
        self.callable_quad_fast.restype = ctypes.c_double

        self.callable_filter = self.dll.callable_filter
        self.callable_filter.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
        ]
        self.callable_filter.restype = ctypes.c_int64

        self.create_defuns = self.dll.create_defuns
        self.create_defuns.argtypes = []
        self.create_defuns.restype = ctypes.c_void_p

        self.add_func = self.dll.add_func
        self.add_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.add_func.restype = None

        self.finalize_defuns = self.dll.finalize_defuns
        self.finalize_defuns.argtypes = [ctypes.c_void_p]
        self.finalize_defuns.restype = None

    def info(self):
        return self.info()

    def find_dll(self, substr):
        files = os.listdir(os.path.dirname(__file__))
        matches = list(filter(lambda s: s.find(substr) >= 0, files))
        if len(matches) == 0:
            return None
        else:
            return matches[0]


#################################################################

lib = Engine()  # interface to the rust codegen engine


def from_raw_parts(ptr, count: int) -> NDArray:
    if count == 0:
        return np.zeros(1)
    else:
        return np.ctypeslib.as_array(ptr, shape=(count,))


class Matrix:
    def __init__(self):
        self.p: int = lib.create_matrix()
        self.rows: list[NDArray] = []  # the list of new rows owned by self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        lib.finalize_matrix(self.p)

    def add_row(self, row):
        v = np.ascontiguousarray(row, dtype=np.double)

        # if v is a different array than row, then it needs to be
        # preserved for the lifetime of the Matrix
        if v is not row:
            self.rows.append(v)

        ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n = v.size
        lib.add_row(self.p, ptr, n)


class Defuns:
    def __init__(self, defuns):
        self.p: int = lib.create_defuns()
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
                lib.add_func(self.p, name.encode("utf8"), f, degree)

    def __del__(self):
        if hasattr(self, "p"):
            lib.finalize_defuns(self.p)


class RustyCompiler:
    def __init__(
        self,
        model,
        ty: str="native",
        use_simd: bool=True,
        enable_simd512: bool=False,
        use_threads: bool=True,
        cse: bool=True,
        fastmath: bool=True,
        opt_level: int=1,
        convert: bool=True,
        defuns: Defuns | None=None,
        dtype: str="float64",
        action: str="compile",
        file: str="",
        num_params: int=1,
        order: str="fortran",
        simd_branch: bool=False,
        fast_complex: bool=True,
        direct: bool=False,
        compact: bool=True,
        compress: bool=False,
        huge: bool=False,
        parallel_mul: bool=True,
    ):
        if convert:
            model = json.dumps(model)

        dtype = str(dtype)
        if dtype not in ["float64", "complex128"]:
            raise ValueError("`dtype` should be `float64` or `complex128`")

        if order not in ["c", "fortran"]:
            raise ValueError("`order` should be either `c` or `fortran`")

        if ty == "amd-sse":
            warnings.warn(
                "`ty = amd-sse` (using x86-64 SSE instructions) is deprecated and will be removed in a future version.",
                DeprecationWarning,
            )

        opt = (
            (0x01 if use_simd else 0)
            | (0x00000002 if use_threads else 0)
            | (0x00000004 if cse else 0)
            | (0x00000008 if fastmath else 0)
            | (0x00000010 if enable_simd512 else 0)
            | (0x00000020 if dtype == "complex128" else 0)
            | (0x00000040 if order == "c" else 0)
            | (0x00000080 if simd_branch else 0)
            | (0x00001000 if compact else 0)
            | (0x00002000 if compress else 0)
            | (0x00004000 if direct else 0)
            | (0x00008000 if fast_complex else 0)
            | (0x00100000 if huge else 0)
            | (0x00200000 if parallel_mul else 0)
            | ((opt_level & 0x0F) << 8)
        )

        self.p: int = 0
        self.dtype: str = dtype
        self.defuns: Defuns = Defuns(defuns)
        self.ty: str = ty

        if action == "compile":
            self.p = lib.compile(
                model.encode("utf-8"), ty.encode("utf8"), opt, self.defuns.p
            )
            self.symbolica: bool = False
        elif action == "translate":
            self.p = lib.translate(
                model.encode("utf-8"), ty.encode("utf8"), opt, self.defuns.p, num_params
            )
            self.symbolica: bool = True
        elif action == "load":
            self.load(file)
        else:
            raise ValueError(f"action {action} not defined")

        status = lib.check_status(self.p)
        if status != b"Success":
            raise ValueError(status.decode())

        self.model = model
        self.json_model = None
        self.populate()

    def __del__(self):
        if hasattr(self, "p"):
            lib.finalize(self.p)

    def save(self, file: str):
        lib.save(self.p, file.encode("utf-8"))

    def load(self, file: str):
        self.p = lib.load(file.encode("utf-8"), self.defuns.p)

        opt = lib.get_config(self.p)
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
        self.count_states: int = lib.count_states(self.p)
        self.count_params: int = lib.count_params(self.p)
        self.count_obs: int = lib.count_obs(self.p)
        self.count_diffs: int = lib.count_diffs(self.p)

        self.states = from_raw_parts(lib.ptr_states(self.p), self.count_states)
        self.params = from_raw_parts(lib.ptr_params(self.p), self.count_params)
        self.obs = from_raw_parts(lib.ptr_obs(self.p), self.count_obs)
        self.diffs = from_raw_parts(lib.ptr_diffs(self.p), self.count_diffs)

    def dump(self, name: str, what: str="scalar"):
        if not lib.dump(self.p, name.encode("utf-8"), what.encode("utf-8")):
            raise ValueError("cannot dump the requested code")
        with open(name, "rb") as fd:
            buf = fd.read()
            return buf

    def dumps(self, what: str="scalar"):
        name = "symjit_dump.bin"
        _ = self.dump(name, what=what)
        with open(name, "rb") as fd:
            b = fd.read()
        os.remove(name)

        if b[0] == ord("#") and b[1] == ord("!"):
            return b.decode("utf8")
        else:
            return b.hex()

    def measure(self, what: str) -> int:
        return lib.measure(self.p, what.encode("utf-8"))

    def execute(self):
        if not lib.execute(self.p):
            raise ValueError("cannot execute the model")

    def execute_vectorized(self, buf: NDArray):
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n = buf.shape[1]
        if not lib.execute_vectorized(self.p, ptr, n):
            raise ValueError("cannot execute the model")

    def execute_matrix(self, states: Matrix, obs: Matrix):
        if not lib.execute_matrix(self.p, states.p, obs.p):
            raise ValueError("cannot execute the model")

    def evaluate(self, args: NDArray, outs: NDArray):
        pargs = args.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nargs = args.size
        pouts = outs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nouts = outs.size

        if not lib.evaluate(self.p, pargs, nargs, pouts, nouts):
            raise ValueError("cannot evaluate the model")

    def evaluate_matrix(self, args: NDArray, outs: NDArray, k: int=1):
        pargs = args.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nargs = args.size
        pouts = outs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nouts = outs.size

        if not lib.evaluate_matrix(self.p, pargs, nargs * k, pouts, nouts * k):
            raise ValueError("cannot evaluate the model")

    def fast_func(self):
        if self.ty == "bytecode":
            return None

        f = lib.fast_func(self.p)

        if f is None:
            return None

        sig = [ctypes.c_double for _ in range(self.count_states + 1)]
        fac = ctypes.CFUNCTYPE(*sig)
        return fac(f)

    def callable_quad(self, use_fast: bool=True):
        f = lib.fast_func(self.p)

        try:
            from scipy import LowLevelCallable

            if f is not None and use_fast:
                return LowLevelCallable(
                    lib.callable_quad_fast,
                    user_data=ctypes.c_void_p(f),
                    signature="double (int, double *, void *)",
                )
            else:
                return LowLevelCallable(
                    lib.callable_quad,
                    user_data=ctypes.c_void_p(self.p),
                    signature="double (int, double *, void *)",
                )
        except:
            return None

    def callable_filter(self, use_fast=True):
        try:
            from scipy import LowLevelCallable

            return LowLevelCallable(
                lib.callable_filter,
                user_data=ctypes.c_void_p(self.p),
                signature="int (double *, npy_intp, double *, void *)",
            )

        except:
            return None
