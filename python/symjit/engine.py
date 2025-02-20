import os
import sys
import ctypes
import platform

def find_dll(substr):
    files = os.listdir(os.path.dirname(__file__))
    matches = list(filter(lambda s: s.find(substr) >= 0, files))
    if len(matches) == 0:
        return None
    else:
        return matches[0]


dll_name = None

if sys.platform == "linux" and platform.machine() == "x86_64":
    dll_name = find_dll("x86_64-linux")
if sys.platform == "linux" and platform.machine() == "aarch64":
    dll_name = find_dll("aarch64-linux")
if sys.platform == "darwin" and platform.machine() == "x86_64":
    dll_name = find_dll("darwin")
elif sys.platform == "win32":
    dll_name = find_dll("win_amd64")

if dll_name is None:
    raise ValueError("unsupported platform")

print(dll_name)

dll_path = os.path.join(os.path.dirname(__file__), dll_name)
dll = ctypes.CDLL(dll_path)


class Engine:
    def __init__(self):
        self._info = dll.info
        self._info.argtypes = []
        self._info.restype = ctypes.c_char_p

        self._check_status = dll.check_status
        self._check_status.argtypes = [ctypes.c_void_p]
        self._check_status.restype = ctypes.c_char_p

        self._count_states = dll.count_states
        self._count_states.argtypes = [ctypes.c_void_p]
        self._count_states.restype = ctypes.c_size_t

        self._count_params = dll.count_params
        self._count_params.argtypes = [ctypes.c_void_p]
        self._count_params.restype = ctypes.c_size_t

        self._count_obs = dll.count_obs
        self._count_obs.argtypes = [ctypes.c_void_p]
        self._count_obs.restype = ctypes.c_size_t

        self._count_diffs = dll.count_diffs
        self._count_diffs.argtypes = [ctypes.c_void_p]
        self._count_diffs.restype = ctypes.c_size_t

        self._run = dll.run
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

        self._execute = dll.execute
        self._execute.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_double,  # t
        ]
        self._execute.restype = ctypes.c_bool

        self._fill_u0 = dll.fill_u0
        self._fill_u0.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # u0
            ctypes.c_size_t,  # ns
        ]
        self._fill_u0.restype = ctypes.c_bool

        self._fill_p = dll.fill_p
        self._fill_p.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.POINTER(ctypes.c_double),  # p
            ctypes.c_size_t,  # np
        ]
        self._fill_p.restype = ctypes.c_bool

        self._compile = dll.compile
        self._compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._compile.restype = ctypes.c_void_p

        self._dump = dll.dump
        self._dump.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._dump.restype = None

        self._finalize = dll.finalize
        self._finalize.argtypes = [ctypes.c_void_p]
        self._finalize.restype = None

        self._ptr_states = dll.ptr_states
        self._ptr_states.argtypes = [ctypes.c_void_p]
        self._ptr_states.restype = ctypes.POINTER(ctypes.c_double)

        self._ptr_params = dll.ptr_params
        self._ptr_params.argtypes = [ctypes.c_void_p]
        self._ptr_params.restype = ctypes.POINTER(ctypes.c_double)

        self._ptr_obs = dll.ptr_obs
        self._ptr_obs.argtypes = [ctypes.c_void_p]
        self._ptr_obs.restype = ctypes.POINTER(ctypes.c_double)

        self._ptr_diffs = dll.ptr_diffs
        self._ptr_diffs.argtypes = [ctypes.c_void_p]
        self._ptr_diffs.restype = ctypes.POINTER(ctypes.c_double)

    def info(self):
        return self._info()

