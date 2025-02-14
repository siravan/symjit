import os
import ctypes
import json
import numpy as np

from . import structure

dll_path = os.path.join(os.path.dirname(__file__), '_lib.cpython-310-x86_64-linux-gnu.so')
dll = ctypes.CDLL(dll_path)    
    
class Library:
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
            ctypes.c_void_p,                    # handle
            ctypes.POINTER(ctypes.c_double),    # du
            ctypes.POINTER(ctypes.c_double),    # u
            ctypes.c_size_t,                    # ns 
            ctypes.POINTER(ctypes.c_double),    # p
            ctypes.c_size_t,                    # np
            ctypes.c_double                     # t
        ]
        self._run.restype = ctypes.c_bool
        
        self._run_py = dll.run_py
        self._run_py.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.POINTER(ctypes.c_double),    # du
            ctypes.c_size_t,                    # nd 
            ctypes.POINTER(ctypes.c_double),    # u
            ctypes.c_size_t,                    # ns 
            ctypes.c_double                     # t
        ]
        self._run_py.restype = ctypes.c_bool
        
        self._execute = dll.execute
        self._execute.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.c_double                     # t
        ]
        self._execute.restype = ctypes.c_bool
        
        self._fill_u0 = dll.fill_u0
        self._fill_u0.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.POINTER(ctypes.c_double),    # u0
            ctypes.c_size_t                     # ns
        ]
        self._fill_u0.restype = ctypes.c_bool        
        
        self._fill_p = dll.fill_p
        self._fill_p.argtypes = [                
            ctypes.c_void_p,                    # handle
            ctypes.POINTER(ctypes.c_double),    # p
            ctypes.c_size_t                     # np
        ]
        self._fill_p.restype = ctypes.c_bool
        
        self._compile = dll.compile
        self._compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._compile.restype = ctypes.c_void_p
        
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
        
lib = Library()     

def from_raw_parts(ptr, count):
    return np.ctypeslib.as_array(ptr, shape=(count,))
    
class BaseFunc:
    def __init__(self, model):
        self.p = lib._compile(model.encode('utf-8'), b'native')        
        status = lib._check_status(self.p)    
        if status != b'Success':
            raise ValueError(status)                    
        self.populate()
        
    def __del__(self):
        lib._finalize(self.p)     
        
    def get_u0(self):
        u0 = np.zeros(self.count_states, dtype='double')
        lib._fill_u0(self.p, np.ctypeslib.as_ctypes(u0), self.count_states)
        return u0
        
    def get_p(self):
        p = np.zeros(self.count_params, dtype='double')
        lib._fill_p(self.p, np.ctypeslib.as_ctypes(p), self.count_params)
        return p
        
    def populate(self):    
        self.count_states = lib._count_states(self.p)
        self.count_params = lib._count_params(self.p)
        self.count_obs = lib._count_obs(self.p)
        self.count_diffs = lib._count_diffs(self.p)
        
        self._states = from_raw_parts(lib._ptr_states(self.p), self.count_states)
        self._params = from_raw_parts(lib._ptr_params(self.p), self.count_params)
        self._obs = from_raw_parts(lib._ptr_obs(self.p), self.count_obs)
        self._diffs = from_raw_parts(lib._ptr_diffs(self.p), self.count_diffs)                


class Func(BaseFunc):
    def __init__(self, model):
        super().__init__(model)
        
    def __call__(self, *args):
        u = np.array(args, dtype='double')        
        self._states[:] = u
        status = lib._execute(self.p, 0.0)
        
        if not status:
            raise ValueError('cannot execute the model')
            
        return self._obs.copy()
        

class OdeFunc(BaseFunc):
    def __init__(self, model):
        super().__init__(model)
        
    def __call__(self, t, y, *args):
        y = np.array(y, dtype='double')
        self._states[:] = y
        
        if len(args) > 0:        
            p = np.array(args, dtype='double')
            self._params[:] = p
        
        status = lib._execute(self.p, t)
        
        if not status:
            raise ValueError('cannot execute the model')
            
        return self._diffs.copy()    

    
def compile_func(states, eqs):
    model = structure.model(states, eqs)
    return Func(json.dumps(model))

def compile_ode(iv, states, odes, params=None):
    model = structure.model_ode(iv, states, odes, params)
    return OdeFunc(json.dumps(model))
        
def compile_json(model):
    return OdeFunc(model)


