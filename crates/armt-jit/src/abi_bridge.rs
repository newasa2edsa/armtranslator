use crate::dynarec::JitEngine;
use crate::interp::StepOutcome;
use crate::mmu::Mmu;
use armt_core::{
    AbiBridge, Arm64Syscall, ArmtError, CpuState, GuestMemory, HostCallArgs, HostCallResult,
    Result, SyscallOutcome, SysRegBus,
};
use std::collections::HashMap;

pub const SENTINEL_RETURN_PC: u64 = 0xFFFF_DEAD_0000_0000;

pub type HostServiceCallback = Box<dyn FnMut(&[u64], &mut CpuState) -> HostCallResult>;

/// Host x86-64 <-> Guest ARM64 ABI translation table for `SVC` / `HVC` host calls.
pub struct HostServiceTable {
    handlers: HashMap<u64, HostServiceCallback>,
}

impl Default for HostServiceTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HostServiceTable {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    pub fn register<F>(&mut self, syscall_num: u64, cb: F)
    where
        F: FnMut(&[u64], &mut CpuState) -> HostCallResult + 'static,
    {
        self.handlers.insert(syscall_num, Box::new(cb));
    }
}

impl AbiBridge for HostServiceTable {
    fn handle_syscall(&mut self, call: Arm64Syscall, cpu: &mut CpuState) -> Result<SyscallOutcome> {
        if let Some(cb) = self.handlers.get_mut(&call.syscall_num) {
            let res = cb(&call.args, cpu);
            Ok(SyscallOutcome::Handled {
                ret0: res.x0,
                ret1: res.x1,
                error: false,
            })
        } else {
            Ok(SyscallOutcome::Unhandled)
        }
    }
}

/// Invoke an ARM64 guest function from host x86-64 code using AAPCS64 calling convention.
///
/// Marshals `args` into `X0..X7`, sets `LR = SENTINEL_RETURN_PC`, runs via `JitEngine`,
/// dispatches any guest `SVC` callbacks through `bridge`, and returns `X0, X1` once guest returns.
pub fn invoke_arm64_fn<M, B>(
    jit: &mut JitEngine,
    cpu: &mut CpuState,
    mem: &mut M,
    mmu: &mut Mmu,
    bridge: &mut B,
    fn_addr: u64,
    args: HostCallArgs,
    max_insns: u64,
) -> Result<HostCallResult>
where
    M: GuestMemory + SysRegBus + ?Sized,
    B: AbiBridge,
{
    args.apply_to_arm64(cpu);
    cpu.x[30] = SENTINEL_RETURN_PC;
    cpu.pc = fn_addr;

    let start_insns = cpu.insn_count;
    loop {
        if cpu.pc == SENTINEL_RETURN_PC {
            return Ok(HostCallResult {
                x0: cpu.x[0],
                x1: cpu.x[1],
            });
        }
        let spent = cpu.insn_count - start_insns;
        if spent >= max_insns {
            return Err(ArmtError::Abi("ARM64 guest function exceeded step budget".into()));
        }
        let outcome = jit.run_until(cpu, mem, mmu, max_insns - spent, Some(SENTINEL_RETURN_PC))?;
        match outcome {
            StepOutcome::Continue => {
                if cpu.pc == SENTINEL_RETURN_PC {
                    return Ok(HostCallResult {
                        x0: cpu.x[0],
                        x1: cpu.x[1],
                    });
                }
            }
            StepOutcome::Svc(imm) => {
                let call = Arm64Syscall::capture(cpu, imm);
                match bridge.handle_syscall(call, cpu)? {
                    SyscallOutcome::Handled { ret0, ret1, error } => {
                        cpu.x[0] = ret0;
                        cpu.x[1] = ret1;
                        if error {
                            cpu.set_nzcv(cpu.n(), cpu.z(), true, cpu.v());
                        } else {
                            cpu.set_nzcv(cpu.n(), cpu.z(), false, cpu.v());
                        }
                        cpu.pc = cpu.pc.wrapping_add(4);
                    }
                    SyscallOutcome::Exit(code) => {
                        return Ok(HostCallResult {
                            x0: code as u64,
                            x1: 0,
                        });
                    }
                    SyscallOutcome::Unhandled => {
                        return Err(ArmtError::Abi(format!(
                            "Unhandled guest SVC #0x{imm:x} (num=0x{:x})",
                            call.syscall_num
                        )));
                    }
                }
            }
            StepOutcome::Hlt(_) => {
                return Ok(HostCallResult {
                    x0: cpu.x[0],
                    x1: cpu.x[1],
                });
            }
            other => {
                return Err(ArmtError::Abi(format!(
                    "Unexpected outcome in invoke_arm64_fn: {other:?}"
                )));
            }
        }
    }
}
