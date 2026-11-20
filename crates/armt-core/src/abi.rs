use crate::cpu::CpuState;
use crate::error::Result;

/// Host / guest ABI calling convention identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallConvention {
    /// Windows x86-64 ABI (`RCX, RDX, R8, R9` + stack, 32-byte shadow space).
    Win64,
    /// System V AMD64 ABI (`RDI, RSI, RDX, RCX, R8, R9` + stack).
    SysV64,
    /// ARM64 Procedure Call Standard (`X0..X7` + stack).
    Aapcs64,
}

impl CallConvention {
    pub const fn host_native() -> Self {
        if cfg!(windows) {
            CallConvention::Win64
        } else {
            CallConvention::SysV64
        }
    }
}

/// Up to 8 integer arguments marshaled between x86-64 host and ARM64 guest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostCallArgs {
    pub args: [u64; 8],
    pub arg_count: usize,
}

impl HostCallArgs {
    pub fn from_slice(slice: &[u64]) -> Self {
        let mut args = [0u64; 8];
        let count = slice.len().min(8);
        args[..count].copy_from_slice(&slice[..count]);
        Self {
            args,
            arg_count: count,
        }
    }

    pub fn apply_to_arm64(&self, cpu: &mut CpuState) {
        for i in 0..self.arg_count.min(8) {
            cpu.x[i] = self.args[i];
        }
    }

    pub fn from_arm64(cpu: &CpuState, count: usize) -> Self {
        let mut args = [0u64; 8];
        let c = count.min(8);
        args[..c].copy_from_slice(&cpu.x[..c]);
        Self { args, arg_count: c }
    }
}

/// Return value pair (`X0`, `X1` in AAPCS64; `RAX`, `RDX` in SysV / `RAX` in Win64).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostCallResult {
    pub x0: u64,
    pub x1: u64,
}

/// Snapshot of a host x64 -> guest ARM64 trampoline call frame.
#[derive(Debug, Clone, Copy)]
pub struct X64ToArm64Frame {
    pub host_conv: CallConvention,
    pub target_pc: u64,
    pub return_sentinel_lr: u64,
    pub guest_sp: u64,
    pub args: HostCallArgs,
}

/// An intercepted ARM64 guest `SVC` / `HVC` call marshaled into host coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arm64Syscall {
    pub svc_imm: u16,
    pub syscall_num: u64,
    pub args: [u64; 8],
}

impl Arm64Syscall {
    pub fn capture(cpu: &CpuState, svc_imm: u16) -> Self {
        // Darwin arm64 places syscall number in X16 (or X8 on Linux arm64).
        let num = if cpu.x[16] != 0 { cpu.x[16] } else { cpu.x[8] };
        let mut args = [0u64; 8];
        args.copy_from_slice(&cpu.x[..8]);
        Self {
            svc_imm,
            syscall_num: num,
            args,
        }
    }
}

/// Outcome returned by a host ABI bridge handler for a guest `SVC` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallOutcome {
    Handled { ret0: u64, ret1: u64, error: bool },
    Exit(i64),
    Unhandled,
}

/// Bridge trait translating between ARM64 AAPCS64 syscalls and host x86-64 services.
pub trait AbiBridge {
    fn handle_syscall(&mut self, call: Arm64Syscall, cpu: &mut CpuState) -> Result<SyscallOutcome>;
}
