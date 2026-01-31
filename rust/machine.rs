use anyhow::{anyhow, Result};
use std::fs;
use std::io::{Read, Write};

use super::memory::*;
use super::utils::*;

pub struct MachineCode<T: Default> {
    machine_code: Vec<u8>,
    #[allow(dead_code)]
    code: Memory, // code needs to be here for f to stay valid
    f: CompiledFunc<T>,
    _mem: Vec<T>,
    leaky: bool,
    lanes: usize,
}

impl<T: Clone + Default> MachineCode<T> {
    const MAGIC: usize = 0x6a7c68ec7656d6d;

    pub fn new(
        arch: &str,
        machine_code: Vec<u8>,
        _mem: Vec<T>,
        leaky: bool,
        lanes: usize,
    ) -> MachineCode<T> {
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
            lanes,
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

impl<T: Clone + Default> Storage for MachineCode<T> {
    fn load(stream: &mut impl Read) -> Result<MachineCode<T>> {
        let mut bytes: [u8; 8] = [0; 8];

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != Self::MAGIC {
            return Err(anyhow!("invalid magic number"));
        }

        stream.read_exact(&mut bytes)?;
        let header = usize::from_le_bytes(bytes);

        let lanes = header & 0xff;
        let leaky = (header & 0x010000) != 0;
        let arch = match (header >> 24) & 0x0f {
            1 => "x86_64",
            2 => "aarch64",
            3 => "riscv64",
            _ => return Err(anyhow!("invalid arch")),
        };

        stream.read_exact(&mut bytes)?;
        let mem_size = usize::from_le_bytes(bytes);
        let _mem: Vec<T> = vec![T::default(); mem_size];

        stream.read_exact(&mut bytes)?;
        let size = usize::from_le_bytes(bytes);
        let mut machine_code: Vec<u8> = vec![0; size];
        stream.read_exact(&mut machine_code)?;

        Ok(Self::new(arch, machine_code, _mem, leaky, lanes))
    }

    fn save(&self, stream: &mut impl Write) -> Result<()> {
        #[cfg(target_arch = "x86_64")]
        let arch = 1;
        #[cfg(target_arch = "aarch64")]
        let arch = 2;
        #[cfg(target_arch = "riscv64")]
        let arch = 3;

        let header: usize = self.lanes | (if self.leaky { 0x010000 } else { 0 }) | arch << 24;
        let mem_size: usize = self._mem.len();
        let size: usize = self.machine_code.len();

        stream.write_all(&Self::MAGIC.to_le_bytes())?;
        stream.write_all(&header.to_le_bytes())?;
        stream.write_all(&mem_size.to_le_bytes())?;
        stream.write_all(&size.to_le_bytes())?;
        stream.write_all(&self.machine_code)?;

        Ok(())
    }
}

impl<T: Default> Drop for MachineCode<T> {
    fn drop(&mut self) {
        if !self.leaky {
            unsafe {
                self.code.free_memory();
            }
        }
    }
}

impl<T: Sized + Copy + Default> Compiled<T> for MachineCode<T> {
    #[inline]
    fn exec(&mut self, params: &[T]) {
        let p = self._mem.as_ptr();
        let q = params.as_ptr();
        (self.f)(p, std::ptr::null(), 0, q);
    }

    /// Generic evaluate function for compiled Symbolica expressions
    #[inline]
    fn evaluate(&mut self, args: &[T], outs: &mut [T]) {
        (self.f)(outs.as_ptr(), std::ptr::null(), 0, args.as_ptr());
    }

    /// Generic evaluate_single function for compiled Symbolica expressions
    #[inline]
    fn evaluate_single(&mut self, args: &[T]) -> T {
        let outs = [T::default(); 1];
        (self.f)(outs.as_ptr(), std::ptr::null(), 0, args.as_ptr());
        outs[0]
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

    fn count_lanes(&self) -> usize {
        self.lanes
    }

    fn as_machine(&self) -> Option<&MachineCode<T>> {
        Some(self)
    }
}

unsafe impl<T: Default> Sync for MachineCode<T> {}
