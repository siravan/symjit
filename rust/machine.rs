use std::fs;
use std::io::Write;

use super::memory::*;
use super::utils::*;

pub struct MachineCode<T> {
    machine_code: Vec<u8>,
    #[allow(dead_code)]
    code: Memory, // code needs to be here for f to stay valid
    f: CompiledFunc<T>,
    _mem: Vec<T>,
    leaky: bool,
}

impl<T> MachineCode<T> {
    pub fn new(arch: &str, machine_code: Vec<u8>, _mem: Vec<T>, leaky: bool) -> MachineCode<T> {
        let valid = (cfg!(target_arch = "x86_64") && arch == "x86_64")
            || (cfg!(target_arch = "aarch64") && arch == "aarch64")
            || (cfg!(target_arch = "riscv64") && arch == "riscv64");

        let size = machine_code.len();

        let mut code = Memory::new(BranchProtection::None);

        // alignment is set to 4096 to allow for using adrp instruction in arm64
        let p: *mut u8 = code.allocate(size, 4096).unwrap();

        let v = unsafe { std::slice::from_raw_parts_mut(p, size) };
        v.copy_from_slice(&machine_code[..]);

        code.set_readable_and_executable().unwrap();

        let f: CompiledFunc<T> = if valid {
            unsafe {
                std::mem::transmute::<*mut u8, fn(*const T, *const *mut T, usize, *const T)>(p)
            }
        } else {
            Self::invalid
        };

        MachineCode {
            machine_code,
            code,
            f,
            _mem,
            leaky,
        }
    }

    fn invalid(_a: *const T, _b: *const *mut T, _c: usize, _d: *const T) {
        if cfg!(target_arch = "x86_64") {
            panic!("invalid processor architecture; expect x86_64");
        } else if cfg!(target_arch = "aarch64") {
            panic!("invalid processor architecture; expect aarch64");
        } else if cfg!(target_arch = "riscv64") {
            panic!("invalid processor architecture; expect riscv64");
        } else {
            panic!("invalid processor architecture; unknown");
        }
    }
}

impl<T> Drop for MachineCode<T> {
    fn drop(&mut self) {
        if !self.leaky {
            unsafe {
                self.code.free_memory();
            }
        }
    }
}

impl<T> Compiled<T> for MachineCode<T> {
    #[inline]
    fn exec(&mut self, params: &[T]) {
        let p = self._mem.as_ptr();
        let q = params.as_ptr();
        (self.f)(p, std::ptr::null(), 0, q);
    }

    #[inline]
    fn mem(&self) -> &[T] {
        &self._mem[..]
    }

    #[inline]
    fn mem_mut(&mut self) -> &mut [T] {
        &mut self._mem[..]
    }

    fn dump(&self, name: &str) {
        let mut fs = fs::File::create(name).unwrap();
        let _ = fs.write(&self.machine_code[..]);
    }

    fn dumps(&self) -> Vec<u8> {
        self.machine_code.clone()
    }

    fn func(&self) -> CompiledFunc<T> {
        self.f
    }

    fn support_indirect(&self) -> bool {
        true
    }
}

unsafe impl<T> Sync for MachineCode<T> {}
