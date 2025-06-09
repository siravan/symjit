use std::fs;
use std::io::Write;

use super::memory::*;
use super::utils::*;

pub struct MachineCode<T> {
    machine_code: Vec<u8>,
    #[allow(dead_code)]
    code: Memory, // code needs to be here for f to stay valid
    f: fn(&[T]),
    _mem: Vec<T>,
}

impl<T> MachineCode<T> {
    pub fn new(arch: &str, machine_code: Vec<u8>, _mem: Vec<T>) -> MachineCode<T> {
        #[cfg(target_arch = "x86_64")]
        if arch != "x86_64" {
            panic!("cannot run {:?} code", arch);
        }

        #[cfg(target_arch = "aarch64")]
        if arch != "aarch64" {
            panic!("cannot run {:?} code", arch);
        }

        let size = machine_code.len();

        let mut code = Memory::new(BranchProtection::None);
        let p: *mut u8 = code.allocate(size, 64).unwrap();

        let v = unsafe { std::slice::from_raw_parts_mut(p, size) };
        v.copy_from_slice(&machine_code[..]);

        code.set_readable_and_executable().unwrap();

        let f: fn(&[T]) = unsafe { std::mem::transmute(p) };

        MachineCode {
            machine_code,
            code,
            f,
            _mem,
        }
    }
}

impl<T> Compiled<T> for MachineCode<T> {
    #[inline]
    fn exec(&mut self) {
        (self.f)(&mut self._mem);
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

    fn func(&self) -> fn(&[T]) {
        self.f
    }
}
