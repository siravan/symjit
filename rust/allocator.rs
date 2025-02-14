/*
    The JIT memory allocator is based on

    https://github.com/dinfuehr/dora/blob/main/dora-runtime/src/os/allocator.rs

    which is the memory allocator for a JIT-compiler for the programming language
    Dora implemented in Rust.

    The code is abbreviated and modified. Specially, we don't need to reserve and
    commit memory separately. Instead, we do both at the same time.
*/

use std::ptr;

const BLOCK_SIZE: usize = 1 << 16;

#[derive(Debug)]
pub struct Allocation {
    pub ptr: *mut u8,
    pub size: usize,
    pub block_size: usize,
}

impl Allocation {
    #[cfg(target_family = "unix")]
    pub fn alloc(size: usize) -> Allocation {
        let block_size = BLOCK_SIZE * (size / BLOCK_SIZE + 1);

        let mut flags = libc::MAP_PRIVATE | libc::MAP_ANON;

        #[cfg(target_os = "macos")]
        let prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC | libc::MAP_JIT;
        #[cfg(not(target_os = "macos"))]
        let prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;

        let ptr = unsafe {
            libc::mmap(ptr::null_mut(), block_size, prot, flags, -1, 0) as *mut libc::c_void
        };

        if ptr == libc::MAP_FAILED {
            panic!("reserving memory with mmap() failed");
        }

        Allocation {
            ptr: ptr as *mut u8,
            size,
            block_size,
        }
    }

    #[cfg(target_family = "windows")]
    pub fn alloc(size: usize) -> Allocation {
        let block_size = BLOCK_SIZE * (size / BLOCK_SIZE + 1);

        use windows_sys::Win32::System::Memory::VirtualAlloc;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE};

        let ptr = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                block_size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            )
        };

        if ptr.is_null() {
            panic!("VirtualAlloc failed");
        }

        Allocation {
            ptr: ptr as *mut u8,
            size,
            block_size,
        }
    }

    pub fn as_mem(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mem_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[cfg(target_family = "unix")]
    pub fn free(&mut self) {
        let result = unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.block_size) };

        if result != 0 {
            panic!("munmap() failed");
        }
    }

    #[cfg(target_family = "windows")]
    pub fn free(&mut self) {
        use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};

        let result = unsafe { VirtualFree(self.ptr as *mut libc::c_void, 0, MEM_RELEASE) };

        if result == 0 {
            panic!("VirtualFree failed");
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.free();
    }
}
