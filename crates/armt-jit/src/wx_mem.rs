use armt_core::{ArmtError, Result};
use std::ptr::NonNull;

/// W^X executable code buffer for emitted x86-64 JIT machine code.
///
/// Allocates `RW` memory, copies code bytes, then switches page permissions to `RX`
/// and flushes the instruction cache before execution.
pub struct CodeBuffer {
    ptr: NonNull<u8>,
    capacity: usize,
    len: usize,
}

unsafe impl Send for CodeBuffer {}

impl CodeBuffer {
    pub fn from_code(code: &[u8]) -> Result<Self> {
        let cap = code.len().max(4096).next_multiple_of(4096);
        let ptr = alloc_rw(cap)?;
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr.as_ptr(), code.len());
        }
        protect_rx(ptr, cap)?;
        flush_icache(ptr, code.len());
        Ok(Self {
            ptr,
            capacity: cap,
            len: code.len(),
        })
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        free_mem(self.ptr, self.capacity);
    }
}

#[cfg(windows)]
mod platform {
    use super::*;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 0x04;
    const PAGE_EXECUTE_READ: u32 = 0x20;

    extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
        fn VirtualProtect(
            addr: *mut u8,
            size: usize,
            new_protect: u32,
            old_protect: *mut u32,
        ) -> i32;
        fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
        fn GetCurrentProcess() -> isize;
        fn FlushInstructionCache(process: isize, base: *const u8, size: usize) -> i32;
    }

    pub fn alloc_rw(size: usize) -> Result<NonNull<u8>> {
        let raw = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        NonNull::new(raw).ok_or_else(|| ArmtError::Jit("VirtualAlloc RW failed".into()))
    }

    pub fn protect_rx(ptr: NonNull<u8>, size: usize) -> Result<()> {
        let mut old = 0u32;
        let ok = unsafe { VirtualProtect(ptr.as_ptr(), size, PAGE_EXECUTE_READ, &mut old) };
        if ok == 0 {
            return Err(ArmtError::Jit("VirtualProtect RX failed".into()));
        }
        Ok(())
    }

    pub fn flush_icache(ptr: NonNull<u8>, size: usize) {
        unsafe {
            let _ = FlushInstructionCache(GetCurrentProcess(), ptr.as_ptr(), size);
        }
    }

    pub fn free_mem(ptr: NonNull<u8>, _size: usize) {
        unsafe {
            let _ = VirtualFree(ptr.as_ptr(), 0, MEM_RELEASE);
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use super::*;

    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    const PROT_EXEC: i32 = 0x4;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;

    extern "C" {
        fn mmap(
            addr: *mut u8,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: isize,
        ) -> *mut u8;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }

    pub fn alloc_rw(size: usize) -> Result<NonNull<u8>> {
        let raw = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw as isize == -1 {
            return Err(ArmtError::Jit("mmap RW failed".into()));
        }
        NonNull::new(raw).ok_or_else(|| ArmtError::Jit("mmap returned null".into()))
    }

    pub fn protect_rx(ptr: NonNull<u8>, size: usize) -> Result<()> {
        let rc = unsafe { mprotect(ptr.as_ptr(), size, PROT_READ | PROT_EXEC) };
        if rc != 0 {
            return Err(ArmtError::Jit("mprotect RX failed".into()));
        }
        Ok(())
    }

    pub fn flush_icache(_ptr: NonNull<u8>, _size: usize) {}

    pub fn free_mem(ptr: NonNull<u8>, size: usize) {
        unsafe {
            let _ = munmap(ptr.as_ptr(), size);
        }
    }
}

use platform::{alloc_rw, flush_icache, free_mem, protect_rx};
