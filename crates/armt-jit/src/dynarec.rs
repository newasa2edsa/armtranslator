use crate::interp::{strip_pac, Interpreter, StepOutcome};
use crate::mmu::Mmu;
use crate::wx_mem::CodeBuffer;
use crate::x64::{Gpr, X64Cc, X64Emitter};
use armt_core::isa::{AddrMode, Condition, Instruction, PacOp, Reg, ShiftType, Width};
use armt_core::{ArmtError, CpuState, GuestMemory, Result, SysRegBus};
use armt_decoder::decode;
use std::collections::HashMap;
use std::mem::offset_of;

const MAX_BLOCK_INSNS: usize = 64;

const EXIT_CONTINUE: u64 = 0;
const EXIT_HLT: u64 = 1;
const EXIT_SVC: u64 = 2;
const EXIT_HVC: u64 = 3;
const EXIT_SMC: u64 = 4;
const EXIT_BRK: u64 = 5;
const EXIT_WFI: u64 = 6;
const EXIT_ERET: u64 = 7;
const EXIT_UNKNOWN: u64 = 8;
const EXIT_FAULT: u64 = 9;
const EXIT_IRQ_YIELD: u64 = 11;
const EXIT_WFE: u64 = 12;

#[inline]
const fn pack_exit(tag: u64, payload: u32) -> u64 {
    tag | ((payload as u64) << 8)
}

#[inline]
fn off_x(r: Reg) -> i32 {
    (offset_of!(CpuState, x) + r.index() * 8) as i32
}

#[inline]
fn off_sp() -> i32 {
    offset_of!(CpuState, sp) as i32
}

#[inline]
fn off_pc() -> i32 {
    offset_of!(CpuState, pc) as i32
}

#[inline]
fn off_nzcv() -> i32 {
    offset_of!(CpuState, nzcv) as i32
}

#[inline]
fn off_insn_count() -> i32 {
    offset_of!(CpuState, insn_count) as i32
}

#[repr(C)]
pub struct JitRuntimeCtx {
    pub cpu_ptr: *mut CpuState,
    pub ram_base: u64,
    pub ram_ptr: *mut u8,
    pub ram_len: u64,
    pub mmu_enabled: u64,
    pub fallback_fn: extern "C" fn(*mut CpuState, *mut JitRuntimeCtx, u64, u32) -> u64,
    pub mem_read_fn: extern "C" fn(*mut JitRuntimeCtx, u64, u8) -> u64,
    pub mem_write_fn: extern "C" fn(*mut JitRuntimeCtx, u64, u8, u64) -> u64,
    pub host_state: *mut (),
    pub fallback_vtable: unsafe fn(*mut (), *mut CpuState, u64, u32) -> u64,
    pub mem_read_vtable: unsafe fn(*mut (), *mut CpuState, u64, u8) -> std::result::Result<u64, u64>,
    pub mem_write_vtable: unsafe fn(*mut (), *mut CpuState, u64, u8, u64) -> std::result::Result<(), u64>,
    pub fault_pending: u64,
    pub fault_addr: u64,
    pub interp_fallbacks: u64,
    pub cache_invalidate_pending: u64,
}

struct HostStateRef<'a, M: GuestMemory + SysRegBus + ?Sized> {
    mem: &'a mut M,
    mmu: &'a mut Mmu,
    last_error: Option<ArmtError>,
}

unsafe fn vtable_fallback<M: GuestMemory + SysRegBus + ?Sized>(
    host_ptr: *mut (),
    cpu_ptr: *mut CpuState,
    pc: u64,
    word: u32,
) -> u64 {
    let host = &mut *(host_ptr as *mut HostStateRef<M>);
    let cpu = &mut *cpu_ptr;
    let prev_daif = cpu.daif;
    match Interpreter::exec_word(cpu, host.mem, host.mmu, pc, word) {
        Ok(StepOutcome::Continue) => {
            // Yield back to the VM loop immediately when IRQ/FIQ becomes unmasked
            if ((prev_daif & 3) & !(cpu.daif & 3)) != 0 {
                cpu.pc = pc.wrapping_add(4);
                EXIT_IRQ_YIELD
            } else {
                EXIT_CONTINUE
            }
        }
        Ok(StepOutcome::Branch(target)) => {
            cpu.pc = target;
            pack_exit(10, 0) // early branch exit from fallback
        }
        Ok(StepOutcome::Hlt(imm)) => {
            cpu.pc = pc;
            pack_exit(EXIT_HLT, imm as u32)
        }
        Ok(StepOutcome::Svc(imm)) => {
            cpu.pc = pc;
            pack_exit(EXIT_SVC, imm as u32)
        }
        Ok(StepOutcome::Hvc(imm)) => {
            cpu.pc = pc;
            pack_exit(EXIT_HVC, imm as u32)
        }
        Ok(StepOutcome::Smc(imm)) => {
            cpu.pc = pc;
            pack_exit(EXIT_SMC, imm as u32)
        }
        Ok(StepOutcome::Brk(imm)) => {
            cpu.pc = pc;
            pack_exit(EXIT_BRK, imm as u32)
        }
        Ok(StepOutcome::Wfi) => {
            cpu.pc = pc;
            EXIT_WFI
        }
        Ok(StepOutcome::Wfe) => {
            cpu.pc = pc;
            EXIT_WFE
        }
        Ok(StepOutcome::Eret) => {
            Mmu::eret(cpu);
            EXIT_ERET
        }
        Ok(StepOutcome::Unknown(w)) => {
            cpu.pc = pc;
            pack_exit(EXIT_UNKNOWN, w)
        }
        Ok(StepOutcome::Exception { .. }) => EXIT_FAULT,
        Err(e) => {
            cpu.pc = pc;
            host.last_error = Some(e);
            EXIT_FAULT
        }
    }
}

unsafe fn vtable_mem_read<M: GuestMemory + SysRegBus + ?Sized>(
    host_ptr: *mut (),
    cpu_ptr: *mut CpuState,
    va: u64,
    size: u8,
) -> std::result::Result<u64, u64> {
    let host = &mut *(host_ptr as *mut HostStateRef<M>);
    let cpu = &*cpu_ptr;
    let res = match size {
        1 => host.mmu.read_mem_u8(cpu, host.mem, va).map(|v| v as u64),
        2 => host.mmu.read_mem_u16(cpu, host.mem, va).map(|v| v as u64),
        4 => host.mmu.read_mem_u32(cpu, host.mem, va).map(|v| v as u64),
        _ => host.mmu.read_mem_u64(cpu, host.mem, va),
    };
    match res {
        Ok(v) => Ok(v),
        Err(e) => {
            host.last_error = Some(e);
            Err(va)
        }
    }
}

unsafe fn vtable_mem_write<M: GuestMemory + SysRegBus + ?Sized>(
    host_ptr: *mut (),
    cpu_ptr: *mut CpuState,
    va: u64,
    size: u8,
    val: u64,
) -> std::result::Result<(), u64> {
    let host = &mut *(host_ptr as *mut HostStateRef<M>);
    let cpu = &*cpu_ptr;
    let res = match size {
        1 => host.mmu.write_mem_u8(cpu, host.mem, va, val as u8),
        2 => host.mmu.write_mem_u16(cpu, host.mem, va, val as u16),
        4 => host.mmu.write_mem_u32(cpu, host.mem, va, val as u32),
        _ => host.mmu.write_mem_u64(cpu, host.mem, va, val),
    };
    match res {
        Ok(()) => Ok(()),
        Err(e) => {
            host.last_error = Some(e);
            Err(va)
        }
    }
}

extern "C" fn jit_fallback_helper(
    cpu: *mut CpuState,
    ctx: *mut JitRuntimeCtx,
    pc: u64,
    word: u32,
) -> u64 {
    unsafe {
        let c = &mut *ctx;
        c.interp_fallbacks += 1;
        let res = (c.fallback_vtable)(c.host_state, cpu, pc, word);
        // Refresh cached mmu_enabled & invalidate JIT blocks on IC maintenance or SCTLR.M change
        let new_mmu = if (*cpu).mmu_enabled() { 1 } else { 0 };
        let is_ic_maint = (word & 0xFFF8_F000) == 0xD508_7000
            && matches!((word >> 8) & 0xF, 1 | 5);
        if new_mmu != c.mmu_enabled || is_ic_maint {
            c.cache_invalidate_pending = 1;
        }
        c.mmu_enabled = new_mmu;
        res
    }
}

extern "C" fn jit_mem_read_helper(
    ctx: *mut JitRuntimeCtx,
    va: u64,
    size: u8,
) -> u64 {
    unsafe {
        let c = &mut *ctx;
        match (c.mem_read_vtable)(c.host_state, c.cpu_ptr, va, size) {
            Ok(v) => v,
            Err(addr) => {
                c.fault_pending = 1;
                c.fault_addr = addr;
                0
            }
        }
    }
}

extern "C" fn jit_mem_write_helper(
    ctx: *mut JitRuntimeCtx,
    va: u64,
    size: u8,
    val: u64,
) -> u64 {
    unsafe {
        let c = &mut *ctx;
        match (c.mem_write_vtable)(c.host_state, c.cpu_ptr, va, size, val) {
            Ok(()) => 0,
            Err(addr) => {
                c.fault_pending = 1;
                c.fault_addr = addr;
                1
            }
        }
    }
}

type JitBlockFn = unsafe extern "C" fn(*mut CpuState, *mut JitRuntimeCtx) -> u64;

struct CompiledBlock {
    _buf: CodeBuffer,
    entry: JitBlockFn,
    _guest_insns: usize,
}

#[derive(Debug, Clone, Default)]
pub struct JitStats {
    pub blocks_compiled: u64,
    pub jit_hits: u64,
    pub jit_misses: u64,
    pub interp_fallbacks: u64,
    pub insns_executed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BlockKey {
    pc: u64,
    mmu_enabled: bool,
    ttbr0: u64,
    ttbr1: u64,
    el: u8,
    max_insns_cap: u8,
}

/// High-speed AArch64 -> x86-64 Dynamic Binary Translator (Dynarec JIT).
pub struct JitEngine {
    cache: HashMap<BlockKey, CompiledBlock>,
    pub stats: JitStats,
}

impl Default for JitEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl JitEngine {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            stats: JitStats::default(),
        }
    }

    pub fn invalidate_all(&mut self) {
        self.cache.clear();
    }

    /// Execute up to `max_insns` instructions starting at `cpu.pc` using compiled x86-64 blocks.
    pub fn run<M: GuestMemory + SysRegBus + ?Sized>(
        &mut self,
        cpu: &mut CpuState,
        mem: &mut M,
        mmu: &mut Mmu,
        max_insns: u64,
    ) -> Result<StepOutcome> {
        self.run_until(cpu, mem, mmu, max_insns, None)
    }

    /// Execute compiled x86-64 blocks until `max_insns`, `stop_pc`, or a synchronous stop event.
    pub fn run_until<M: GuestMemory + SysRegBus + ?Sized>(
        &mut self,
        cpu: &mut CpuState,
        mem: &mut M,
        mmu: &mut Mmu,
        max_insns: u64,
        stop_pc: Option<u64>,
    ) -> Result<StepOutcome> {
        let start_insns = cpu.insn_count;
        let (ram_base, ram_ptr, ram_len) = mem
            .ram_slice_mut()
            .unwrap_or((0, std::ptr::null_mut(), 0));

        let mut host_ref = HostStateRef {
            mem,
            mmu,
            last_error: None,
        };

        let mut ctx = JitRuntimeCtx {
            cpu_ptr: cpu as *mut CpuState,
            ram_base,
            ram_ptr,
            ram_len: ram_len as u64,
            mmu_enabled: if cpu.mmu_enabled() { 1 } else { 0 },
            fallback_fn: jit_fallback_helper,
            mem_read_fn: jit_mem_read_helper,
            mem_write_fn: jit_mem_write_helper,
            host_state: (&mut host_ref as *mut HostStateRef<M>) as *mut (),
            fallback_vtable: vtable_fallback::<M>,
            mem_read_vtable: vtable_mem_read::<M>,
            mem_write_vtable: vtable_mem_write::<M>,
            fault_pending: 0,
            fault_addr: 0,
            interp_fallbacks: 0,
            cache_invalidate_pending: 0,
        };

        loop {
            if Some(cpu.pc) == stop_pc {
                self.stats.interp_fallbacks += ctx.interp_fallbacks;
                self.stats.insns_executed = cpu.insn_count;
                return Ok(StepOutcome::Continue);
            }
            let remaining = max_insns.saturating_sub(cpu.insn_count - start_insns);
            if remaining == 0 {
                self.stats.interp_fallbacks += ctx.interp_fallbacks;
                self.stats.insns_executed = cpu.insn_count;
                return Ok(StepOutcome::Continue);
            }
            let block_cap = remaining.min(MAX_BLOCK_INSNS as u64) as usize;

            if ctx.cache_invalidate_pending != 0 {
                self.cache.clear();
                ctx.cache_invalidate_pending = 0;
            }

            let block_pc = cpu.pc;
            let key = BlockKey {
                pc: block_pc,
                mmu_enabled: cpu.mmu_enabled(),
                ttbr0: cpu.ttbr0_el1,
                ttbr1: cpu.ttbr1_el1,
                el: cpu.el,
                max_insns_cap: block_cap as u8,
            };
            let entry_fn = if let Some(b) = self.cache.get(&key) {
                self.stats.jit_hits += 1;
                b.entry
            } else {
                self.stats.jit_misses += 1;
                let block = compile_block(cpu, &mut host_ref, block_pc, block_cap)?;
                let f = block.entry;
                self.cache.insert(key, block);
                self.stats.blocks_compiled += 1;
                f
            };

            ctx.mmu_enabled = if cpu.mmu_enabled() { 1 } else { 0 };
            ctx.fault_pending = 0;

            let raw_ret = unsafe { entry_fn(cpu as *mut CpuState, &mut ctx as *mut JitRuntimeCtx) };

            if ctx.fault_pending != 0 {
                self.stats.interp_fallbacks += ctx.interp_fallbacks;
                self.stats.insns_executed = cpu.insn_count;
                if let Some(err) = host_ref.last_error.take() {
                    return Err(err);
                }
                return Err(ArmtError::MemoryFault {
                    addr: ctx.fault_addr,
                    size: 8,
                    write: false,
                    reason: "JIT memory helper fault".into(),
                });
            }

            let tag = raw_ret & 0xFF;
            let payload = (raw_ret >> 8) as u32;
            match tag {
                EXIT_CONTINUE | 10 => continue,
                EXIT_IRQ_YIELD => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Continue);
                }
                EXIT_HLT => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Hlt(payload as u16));
                }
                EXIT_SVC => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Svc(payload as u16));
                }
                EXIT_HVC => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Hvc(payload as u16));
                }
                EXIT_SMC => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Smc(payload as u16));
                }
                EXIT_BRK => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Brk(payload as u16));
                }
                EXIT_WFI => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Wfi);
                }
                EXIT_WFE => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Wfe);
                }
                EXIT_ERET => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Eret);
                }
                EXIT_UNKNOWN => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    return Ok(StepOutcome::Unknown(payload));
                }
                _ => {
                    self.stats.interp_fallbacks += ctx.interp_fallbacks;
                    self.stats.insns_executed = cpu.insn_count;
                    if let Some(err) = host_ref.last_error.take() {
                        return Err(err);
                    }
                    return Err(ArmtError::Jit(format!("JIT fault exit code {raw_ret}")));
                }
            }
        }
    }
}

fn compile_block<M: GuestMemory + SysRegBus + ?Sized>(
    cpu: &CpuState,
    host: &mut HostStateRef<M>,
    start_pc: u64,
    max_block_insns: usize,
) -> Result<CompiledBlock> {
    let mut e = X64Emitter::new();
    emit_prologue(&mut e);

    let mut exit_patches = Vec::new();
    let mut fault_patches: Vec<(usize, u64, usize)> = Vec::new();
    let mut cur_pc = start_pc;
    let mut compiled_insns = 0usize;
    let block_limit = max_block_insns.clamp(1, MAX_BLOCK_INSNS);

    for idx in 0..block_limit {
        let pa = match host.mmu.translate_exec(cpu, host.mem, cur_pc) {
            Ok(p) => p,
            Err(err) => {
                if idx == 0 {
                    return Err(err);
                }
                emit_add_insn_count(&mut e, compiled_insns);
                e.mov64_imm(Gpr::Rax, cur_pc);
                e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
                e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
                exit_patches.push(e.jmp_rel32_placeholder());
                break;
            }
        };
        let word = host.mem.read_u32(pa)?;
        let inst = decode(word);
        compiled_insns = idx + 1;

        let is_term = emit_instruction(
            &mut e,
            cur_pc,
            word,
            &inst,
            compiled_insns,
            &mut exit_patches,
            &mut fault_patches,
        );

        cur_pc = cur_pc.wrapping_add(4);
        if is_term {
            break;
        }
        if idx + 1 == block_limit {
            // End of block budget without branch: advance PC & insn_count and exit cleanly
            emit_add_insn_count(&mut e, compiled_insns);
            e.mov64_imm(Gpr::Rax, cur_pc);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
        }
    }

    for (p, fpc, done_insns) in fault_patches {
        let fpos = e.pos();
        e.patch_rel32(p, fpos);
        if done_insns > 1 {
            emit_add_insn_count(&mut e, done_insns - 1);
        }
        e.mov64_imm(Gpr::Rax, fpc);
        e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
        e.mov32_imm(Gpr::Rax, EXIT_FAULT as u32);
        exit_patches.push(e.jmp_rel32_placeholder());
    }

    let epilogue_pos = e.pos();
    for p in exit_patches {
        e.patch_rel32(p, epilogue_pos);
    }
    emit_epilogue(&mut e);

    let buf = CodeBuffer::from_code(&e.code)?;
    let entry: JitBlockFn = unsafe { std::mem::transmute(buf.as_ptr()) };
    Ok(CompiledBlock {
        _buf: buf,
        entry,
        _guest_insns: compiled_insns,
    })
}

fn emit_prologue(e: &mut X64Emitter) {
    if cfg!(windows) {
        for r in [
            Gpr::Rbx,
            Gpr::Rbp,
            Gpr::Rdi,
            Gpr::Rsi,
            Gpr::R12,
            Gpr::R13,
            Gpr::R14,
            Gpr::R15,
        ] {
            e.push(r);
        }
        // 8 pushes (64B) + ret (8B) = 72B; sub 40 -> 112B (16B-aligned + 32B shadow space)
        e.alu64_ri(5, Gpr::Rsp, 40);
        e.mov64_rr(Gpr::R15, Gpr::Rcx); // R15 = CpuState*
        e.mov64_rr(Gpr::R14, Gpr::Rdx); // R14 = JitRuntimeCtx*
    } else {
        for r in [Gpr::Rbx, Gpr::Rbp, Gpr::R12, Gpr::R13, Gpr::R14, Gpr::R15] {
            e.push(r);
        }
        e.alu64_ri(5, Gpr::Rsp, 8);
        e.mov64_rr(Gpr::R15, Gpr::Rdi);
        e.mov64_rr(Gpr::R14, Gpr::Rsi);
    }
}

fn emit_epilogue(e: &mut X64Emitter) {
    if cfg!(windows) {
        e.alu64_ri(0, Gpr::Rsp, 40);
        for r in [
            Gpr::R15,
            Gpr::R14,
            Gpr::R13,
            Gpr::R12,
            Gpr::Rsi,
            Gpr::Rdi,
            Gpr::Rbp,
            Gpr::Rbx,
        ] {
            e.pop(r);
        }
    } else {
        e.alu64_ri(0, Gpr::Rsp, 8);
        for r in [
            Gpr::R15,
            Gpr::R14,
            Gpr::R13,
            Gpr::R12,
            Gpr::Rbp,
            Gpr::Rbx,
        ] {
            e.pop(r);
        }
    }
    e.ret();
}

#[inline]
fn emit_add_insn_count(e: &mut X64Emitter, count: usize) {
    e.add64_mr_imm(Gpr::R15, off_insn_count(), count as i32);
}

#[inline]
fn emit_load_gpr(e: &mut X64Emitter, dst: Gpr, reg: Reg, w: Width, sp_allowed: bool) {
    if reg.is_zr_or_sp() {
        if sp_allowed {
            match w {
                Width::W32 => e.mov32_rm(dst, Gpr::R15, off_sp()),
                Width::X64 => e.mov64_rm(dst, Gpr::R15, off_sp()),
            }
        } else {
            e.mov32_imm(dst, 0);
        }
    } else {
        match w {
            Width::W32 => e.mov32_rm(dst, Gpr::R15, off_x(reg)),
            Width::X64 => e.mov64_rm(dst, Gpr::R15, off_x(reg)),
        }
    }
}

#[inline]
fn emit_store_gpr(e: &mut X64Emitter, reg: Reg, src: Gpr, w: Width, sp_allowed: bool) {
    if reg.is_zr_or_sp() {
        if sp_allowed {
            if w == Width::W32 {
                e.mov32_rr(src, src); // zero-extend to 64
            }
            e.mov64_mr(Gpr::R15, off_sp(), src);
        }
    } else if w == Width::W32 {
        e.mov32_rr(src, src); // zero-extend low 32 into 64
        e.mov64_mr(Gpr::R15, off_x(reg), src);
    } else {
        e.mov64_mr(Gpr::R15, off_x(reg), src);
    }
}

/// Capture x86 RFLAGS immediately after `ADD` or `SUB/CMP` into `CpuState.nzcv`.
/// CRITICAL: ARM64 `SUBS`/`CMP` sets `C = !borrow` (x86 `AE`/NC), whereas `ADDS` sets `C = carry` (x86 `B`/C).
fn emit_store_nzcv_from_flags(e: &mut X64Emitter, is_sub: bool, clear_cv: bool) {
    e.setcc(X64Cc::S, Gpr::R8); // N
    e.setcc(X64Cc::E, Gpr::R9); // Z
    if clear_cv {
        e.mov32_imm(Gpr::R10, 0);
        e.mov32_imm(Gpr::R11, 0);
    } else {
        e.setcc(if is_sub { X64Cc::Ae } else { X64Cc::B }, Gpr::R10); // C
        e.setcc(X64Cc::O, Gpr::R11); // V
    }
    e.movzx_r32_r8(Gpr::R8, Gpr::R8);
    e.movzx_r32_r8(Gpr::R9, Gpr::R9);
    e.shift32_imm(4, Gpr::R8, 31);
    e.shift32_imm(4, Gpr::R9, 30);
    e.alu32_rr(0x09, Gpr::R8, Gpr::R9); // OR
    if !clear_cv {
        e.movzx_r32_r8(Gpr::R10, Gpr::R10);
        e.movzx_r32_r8(Gpr::R11, Gpr::R11);
        e.shift32_imm(4, Gpr::R10, 29);
        e.shift32_imm(4, Gpr::R11, 28);
        e.alu32_rr(0x09, Gpr::R8, Gpr::R10);
        e.alu32_rr(0x09, Gpr::R8, Gpr::R11);
    }
    e.mov64_mr(Gpr::R15, off_nzcv(), Gpr::R8);
}

fn emit_shift_reg_in_rcx(e: &mut X64Emitter, reg: Gpr, shift: ShiftType, amount: u8, w: Width) {
    let bits = w.bits() as u8;
    let amt = amount & (bits - 1);
    if amt == 0 {
        return;
    }
    let subop = match shift {
        ShiftType::Lsl => 4,
        ShiftType::Lsr => 5,
        ShiftType::Asr => 7,
        ShiftType::Ror => 1,
    };
    match w {
        Width::W32 => e.shift32_imm(subop, reg, amt),
        Width::X64 => e.shift64_imm(subop, reg, amt),
    }
}

/// Emit code testing AArch64 `Condition` against `CpuState.nzcv`; returns a jump patch
/// that jumps to `taken` if the condition HOLDS.
fn emit_jcc_arm_cond(e: &mut X64Emitter, cond: Condition) -> usize {
    // Load nzcv >> 28 into RAX (bits 3..0 = N(8), Z(4), C(2), V(1))
    e.mov64_rm(Gpr::Rax, Gpr::R15, off_nzcv());
    e.shift64_imm(5, Gpr::Rax, 28);
    match cond {
        Condition::Eq => {
            e.alu32_ri(4, Gpr::Rax, 4); // test Z (bit 2)
            e.jcc_rel32_placeholder(X64Cc::Ne)
        }
        Condition::Ne => {
            e.alu32_ri(4, Gpr::Rax, 4);
            e.jcc_rel32_placeholder(X64Cc::E)
        }
        Condition::Cs => {
            e.alu32_ri(4, Gpr::Rax, 2); // test C (bit 1)
            e.jcc_rel32_placeholder(X64Cc::Ne)
        }
        Condition::Cc => {
            e.alu32_ri(4, Gpr::Rax, 2);
            e.jcc_rel32_placeholder(X64Cc::E)
        }
        Condition::Mi => {
            e.alu32_ri(4, Gpr::Rax, 8); // test N (bit 3)
            e.jcc_rel32_placeholder(X64Cc::Ne)
        }
        Condition::Pl => {
            e.alu32_ri(4, Gpr::Rax, 8);
            e.jcc_rel32_placeholder(X64Cc::E)
        }
        Condition::Vs => {
            e.alu32_ri(4, Gpr::Rax, 1); // test V (bit 0)
            e.jcc_rel32_placeholder(X64Cc::Ne)
        }
        Condition::Vc => {
            e.alu32_ri(4, Gpr::Rax, 1);
            e.jcc_rel32_placeholder(X64Cc::E)
        }
        Condition::Hi => {
            // C == 1 && Z == 0 -> (flags & 0b0110) == 0b0010
            e.alu32_ri(4, Gpr::Rax, 6);
            e.alu32_ri(7, Gpr::Rax, 2); // cmp eax, 2
            e.jcc_rel32_placeholder(X64Cc::E)
        }
        Condition::Ls => {
            // C == 0 || Z == 1 -> (flags & 0b0110) != 0b0010
            e.alu32_ri(4, Gpr::Rax, 6);
            e.alu32_ri(7, Gpr::Rax, 2);
            e.jcc_rel32_placeholder(X64Cc::Ne)
        }
        Condition::Ge | Condition::Lt => {
            // N == V -> ((flags >> 3) ^ flags) & 1 == 0
            e.mov32_rr(Gpr::Rcx, Gpr::Rax);
            e.shift32_imm(5, Gpr::Rcx, 3);
            e.alu32_rr(0x31, Gpr::Rax, Gpr::Rcx); // xor
            e.alu32_ri(4, Gpr::Rax, 1);
            e.jcc_rel32_placeholder(if cond == Condition::Ge {
                X64Cc::E
            } else {
                X64Cc::Ne
            })
        }
        Condition::Gt | Condition::Le => {
            // GT: Z == 0 && N == V; LE: Z == 1 || N != V
            e.mov32_rr(Gpr::Rcx, Gpr::Rax);
            e.shift32_imm(5, Gpr::Rcx, 3);
            e.alu32_rr(0x31, Gpr::Rcx, Gpr::Rax);
            e.alu32_ri(4, Gpr::Rcx, 1); // bit0 = N^V
            e.alu32_ri(4, Gpr::Rax, 4); // bit2 = Z
            e.alu32_rr(0x09, Gpr::Rax, Gpr::Rcx); // nonzero iff Z==1 || N!=V
            e.jcc_rel32_placeholder(if cond == Condition::Gt {
                X64Cc::E
            } else {
                X64Cc::Ne
            })
        }
        Condition::Al | Condition::Nv => e.jmp_rel32_placeholder(),
    }
}

fn emit_call_mem_read(e: &mut X64Emitter, addr_reg: Gpr, size: u8, out_reg: Gpr) {
    // Load helper fn ptr first into R11
    let fn_off = offset_of!(JitRuntimeCtx, mem_read_fn) as i32;
    e.mov64_rm(Gpr::R11, Gpr::R14, fn_off);
    if cfg!(windows) {
        // RCX=ctx (R14), RDX=addr, R8=size
        e.mov64_rr(Gpr::Rdx, addr_reg);
        e.mov64_rr(Gpr::Rcx, Gpr::R14);
        e.mov32_imm(Gpr::R8, size as u32);
    } else {
        // RDI=ctx (R14), RSI=addr, RDX=size
        e.mov64_rr(Gpr::Rsi, addr_reg);
        e.mov64_rr(Gpr::Rdi, Gpr::R14);
        e.mov32_imm(Gpr::Rdx, size as u32);
    }
    e.call_r(Gpr::R11);
    if out_reg != Gpr::Rax {
        e.mov64_rr(out_reg, Gpr::Rax);
    }
}

fn emit_call_mem_write(e: &mut X64Emitter, addr_reg: Gpr, size: u8, val_reg: Gpr) {
    let fn_off = offset_of!(JitRuntimeCtx, mem_write_fn) as i32;
    e.mov64_rm(Gpr::R11, Gpr::R14, fn_off);
    if cfg!(windows) {
        // RCX=ctx, RDX=addr, R8=size, R9=val
        e.mov64_rr(Gpr::R9, val_reg);
        e.mov64_rr(Gpr::Rdx, addr_reg);
        e.mov64_rr(Gpr::Rcx, Gpr::R14);
        e.mov32_imm(Gpr::R8, size as u32);
    } else {
        // RDI=ctx, RSI=addr, RDX=size, RCX=val
        e.mov64_rr(Gpr::Rcx, val_reg);
        e.mov64_rr(Gpr::Rsi, addr_reg);
        e.mov64_rr(Gpr::Rdi, Gpr::R14);
        e.mov32_imm(Gpr::Rdx, size as u32);
    }
    e.call_r(Gpr::R11);
}

fn emit_fast_or_slow_read(
    e: &mut X64Emitter,
    addr_reg: Gpr, // must be R12 or Rsi (preserved across helper checks; we use R12!)
    size_bytes: u8,
    out_reg: Gpr,
    pc: u64,
    insns_so_far: usize,
    fault_patches: &mut Vec<(usize, u64, usize)>,
) {
    // Check mmu_enabled == 0
    let mmu_off = offset_of!(JitRuntimeCtx, mmu_enabled) as i32;
    e.mov64_rm(Gpr::Rax, Gpr::R14, mmu_off);
    e.alu64_rr(0x85, Gpr::Rax, Gpr::Rax);
    let slow1 = e.jcc_rel32_placeholder(X64Cc::Ne);

    // Compute offset = addr - ram_base; check offset <= ram_len - size_bytes
    let rbase_off = offset_of!(JitRuntimeCtx, ram_base) as i32;
    let rlen_off = offset_of!(JitRuntimeCtx, ram_len) as i32;
    let rptr_off = offset_of!(JitRuntimeCtx, ram_ptr) as i32;

    e.mov64_rr(Gpr::Rcx, addr_reg);
    e.mov64_rm(Gpr::Rdx, Gpr::R14, rbase_off);
    e.alu64_rr(0x29, Gpr::Rcx, Gpr::Rdx); // rcx = addr - ram_base
    let slow2 = e.jcc_rel32_placeholder(X64Cc::B); // addr < ram_base

    e.mov64_rm(Gpr::Rdx, Gpr::R14, rlen_off);
    e.alu64_ri(5, Gpr::Rdx, size_bytes as i32); // ram_len - size
    let slow_tiny = e.jcc_rel32_placeholder(X64Cc::B);
    e.alu64_rr(0x39, Gpr::Rcx, Gpr::Rdx); // cmp offset, ram_len - size
    let slow3 = e.jcc_rel32_placeholder(X64Cc::A);

    // Fast path: load directly from [ram_ptr + offset]
    e.mov64_rm(Gpr::Rdx, Gpr::R14, rptr_off);
    if size_bytes == 4 {
        e.mov32_sib_rm(out_reg, Gpr::Rdx, Gpr::Rcx, 1, 0);
    } else {
        e.mov64_sib_rm(out_reg, Gpr::Rdx, Gpr::Rcx, 1, 0);
    }
    let done = e.jmp_rel32_placeholder();

    let slow_pos = e.pos();
    for p in [slow1, slow2, slow_tiny, slow3] {
        e.patch_rel32(p, slow_pos);
    }
    emit_call_mem_read(e, addr_reg, size_bytes, out_reg);
    let f_off = offset_of!(JitRuntimeCtx, fault_pending) as i32;
    e.mov64_rm(Gpr::Rcx, Gpr::R14, f_off);
    e.alu64_rr(0x85, Gpr::Rcx, Gpr::Rcx);
    fault_patches.push((e.jcc_rel32_placeholder(X64Cc::Ne), pc, insns_so_far));

    let done_pos = e.pos();
    e.patch_rel32(done, done_pos);
}

fn emit_fast_or_slow_write(
    e: &mut X64Emitter,
    addr_reg: Gpr, // R12
    val_reg: Gpr,  // R13
    size_bytes: u8,
    pc: u64,
    insns_so_far: usize,
    fault_patches: &mut Vec<(usize, u64, usize)>,
) {
    let mmu_off = offset_of!(JitRuntimeCtx, mmu_enabled) as i32;
    e.mov64_rm(Gpr::Rax, Gpr::R14, mmu_off);
    e.alu64_rr(0x85, Gpr::Rax, Gpr::Rax);
    let slow1 = e.jcc_rel32_placeholder(X64Cc::Ne);

    let rbase_off = offset_of!(JitRuntimeCtx, ram_base) as i32;
    let rlen_off = offset_of!(JitRuntimeCtx, ram_len) as i32;
    let rptr_off = offset_of!(JitRuntimeCtx, ram_ptr) as i32;

    e.mov64_rr(Gpr::Rcx, addr_reg);
    e.mov64_rm(Gpr::Rdx, Gpr::R14, rbase_off);
    e.alu64_rr(0x29, Gpr::Rcx, Gpr::Rdx);
    let slow2 = e.jcc_rel32_placeholder(X64Cc::B);

    e.mov64_rm(Gpr::Rdx, Gpr::R14, rlen_off);
    e.alu64_ri(5, Gpr::Rdx, size_bytes as i32);
    let slow_tiny = e.jcc_rel32_placeholder(X64Cc::B);
    e.alu64_rr(0x39, Gpr::Rcx, Gpr::Rdx);
    let slow3 = e.jcc_rel32_placeholder(X64Cc::A);

    e.mov64_rm(Gpr::Rdx, Gpr::R14, rptr_off);
    if size_bytes == 4 {
        e.mov32_sib_mr(Gpr::Rdx, Gpr::Rcx, 1, 0, val_reg);
    } else {
        e.mov64_sib_mr(Gpr::Rdx, Gpr::Rcx, 1, 0, val_reg);
    }
    let done = e.jmp_rel32_placeholder();

    let slow_pos = e.pos();
    for p in [slow1, slow2, slow_tiny, slow3] {
        e.patch_rel32(p, slow_pos);
    }
    emit_call_mem_write(e, addr_reg, size_bytes, val_reg);
    let f_off = offset_of!(JitRuntimeCtx, fault_pending) as i32;
    e.mov64_rm(Gpr::Rcx, Gpr::R14, f_off);
    e.alu64_rr(0x85, Gpr::Rcx, Gpr::Rcx);
    fault_patches.push((e.jcc_rel32_placeholder(X64Cc::Ne), pc, insns_so_far));

    let done_pos = e.pos();
    e.patch_rel32(done, done_pos);
}

/// Emit native x86-64 code for `inst`. Returns `true` if `inst` terminates the basic block.
fn emit_instruction(
    e: &mut X64Emitter,
    pc: u64,
    word: u32,
    inst: &Instruction,
    insns_so_far: usize,
    exit_patches: &mut Vec<usize>,
    fault_patches: &mut Vec<(usize, u64, usize)>,
) -> bool {
    match *inst {
        Instruction::Nop
        | Instruction::Barrier { .. }
        | Instruction::Prfm
        | Instruction::PacHint {
            op: PacOp::Paciasp | PacOp::Autiasp | PacOp::Pacia | PacOp::Pacib | PacOp::Pacda | PacOp::Pacdb,
            ..
        } => false,

        Instruction::Movz { w, rd, imm16, shift } => {
            let val = ((imm16 as u64) << shift) & w.mask();
            e.mov64_imm(Gpr::Rax, val);
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::Movn { w, rd, imm16, shift } => {
            let val = (!((imm16 as u64) << shift)) & w.mask();
            e.mov64_imm(Gpr::Rax, val);
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::Movk { w, rd, imm16, shift } => {
            emit_load_gpr(e, Gpr::Rax, rd, Width::X64, false);
            let clear_mask = !(0xFFFFu64 << shift);
            e.mov64_imm(Gpr::Rcx, clear_mask);
            e.alu64_rr(0x21, Gpr::Rax, Gpr::Rcx); // AND
            let ins = (imm16 as u64) << shift;
            if ins != 0 {
                e.mov64_imm(Gpr::Rcx, ins);
                e.alu64_rr(0x09, Gpr::Rax, Gpr::Rcx); // OR
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::Adr { rd, offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            e.mov64_imm(Gpr::Rax, target);
            emit_store_gpr(e, rd, Gpr::Rax, Width::X64, false);
            false
        }
        Instruction::Adrp { rd, offset } => {
            let base = (pc & !0xFFF) as i64;
            let target = base.wrapping_add(offset) as u64;
            e.mov64_imm(Gpr::Rax, target);
            emit_store_gpr(e, rd, Gpr::Rax, Width::X64, false);
            false
        }

        Instruction::AddImm {
            w,
            rd,
            rn,
            imm,
            set_flags,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, true);
            e.mov64_imm(Gpr::Rcx, imm & w.mask());
            match w {
                Width::W32 => e.alu32_rr(0x01, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x01, Gpr::Rax, Gpr::Rcx),
            }
            if set_flags {
                emit_store_nzcv_from_flags(e, false, false);
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, !set_flags);
            false
        }
        Instruction::SubImm {
            w,
            rd,
            rn,
            imm,
            set_flags,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, true);
            e.mov64_imm(Gpr::Rcx, imm & w.mask());
            match w {
                Width::W32 => e.alu32_rr(0x29, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x29, Gpr::Rax, Gpr::Rcx),
            }
            if set_flags {
                emit_store_nzcv_from_flags(e, true, false);
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, !set_flags);
            false
        }
        Instruction::AddReg {
            w,
            rd,
            rn,
            rm,
            shift,
            amount,
            set_flags,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            emit_shift_reg_in_rcx(e, Gpr::Rcx, shift, amount, w);
            match w {
                Width::W32 => e.alu32_rr(0x01, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x01, Gpr::Rax, Gpr::Rcx),
            }
            if set_flags {
                emit_store_nzcv_from_flags(e, false, false);
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::SubReg {
            w,
            rd,
            rn,
            rm,
            shift,
            amount,
            set_flags,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            emit_shift_reg_in_rcx(e, Gpr::Rcx, shift, amount, w);
            match w {
                Width::W32 => e.alu32_rr(0x29, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x29, Gpr::Rax, Gpr::Rcx),
            }
            if set_flags {
                emit_store_nzcv_from_flags(e, true, false);
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }

        Instruction::AndImm {
            w,
            rd,
            rn,
            imm,
            set_flags,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            e.mov64_imm(Gpr::Rcx, imm & w.mask());
            match w {
                Width::W32 => e.alu32_rr(0x21, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x21, Gpr::Rax, Gpr::Rcx),
            }
            if set_flags {
                emit_store_nzcv_from_flags(e, false, true);
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, !set_flags);
            false
        }
        Instruction::OrrImm { w, rd, rn, imm } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            e.mov64_imm(Gpr::Rcx, imm & w.mask());
            match w {
                Width::W32 => e.alu32_rr(0x09, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x09, Gpr::Rax, Gpr::Rcx),
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, true);
            false
        }
        Instruction::EorImm { w, rd, rn, imm } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            e.mov64_imm(Gpr::Rcx, imm & w.mask());
            match w {
                Width::W32 => e.alu32_rr(0x31, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x31, Gpr::Rax, Gpr::Rcx),
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, true);
            false
        }
        Instruction::AndReg {
            w,
            rd,
            rn,
            rm,
            shift,
            amount,
            set_flags,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            emit_shift_reg_in_rcx(e, Gpr::Rcx, shift, amount, w);
            match w {
                Width::W32 => e.alu32_rr(0x21, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x21, Gpr::Rax, Gpr::Rcx),
            }
            if set_flags {
                emit_store_nzcv_from_flags(e, false, true);
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::OrrReg {
            w,
            rd,
            rn,
            rm,
            shift,
            amount,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            emit_shift_reg_in_rcx(e, Gpr::Rcx, shift, amount, w);
            match w {
                Width::W32 => e.alu32_rr(0x09, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x09, Gpr::Rax, Gpr::Rcx),
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::EorReg {
            w,
            rd,
            rn,
            rm,
            shift,
            amount,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            emit_shift_reg_in_rcx(e, Gpr::Rcx, shift, amount, w);
            match w {
                Width::W32 => e.alu32_rr(0x31, Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x31, Gpr::Rax, Gpr::Rcx),
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::BicReg {
            w,
            rd,
            rn,
            rm,
            shift,
            amount,
            set_flags,
        } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            emit_shift_reg_in_rcx(e, Gpr::Rcx, shift, amount, w);
            match w {
                Width::W32 => {
                    e.not32(Gpr::Rcx);
                    e.alu32_rr(0x21, Gpr::Rax, Gpr::Rcx);
                }
                Width::X64 => {
                    e.not64(Gpr::Rcx);
                    e.alu64_rr(0x21, Gpr::Rax, Gpr::Rcx);
                }
            }
            if set_flags {
                emit_store_nzcv_from_flags(e, false, true);
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }

        Instruction::Lslv { w, rd, rn, rm }
        | Instruction::Lsrv { w, rd, rn, rm }
        | Instruction::Asrv { w, rd, rn, rm }
        | Instruction::Rorv { w, rd, rn, rm } => {
            let subop = match *inst {
                Instruction::Lslv { .. } => 4,
                Instruction::Lsrv { .. } => 5,
                Instruction::Asrv { .. } => 7,
                _ => 1,
            };
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            match w {
                Width::W32 => e.shift32_cl(subop, Gpr::Rax),
                Width::X64 => e.shift64_cl(subop, Gpr::Rax),
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }

        Instruction::Madd { w, rd, rn, rm, ra } => {
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            match w {
                Width::W32 => e.imul32_rr(Gpr::Rax, Gpr::Rcx),
                Width::X64 => e.imul64_rr(Gpr::Rax, Gpr::Rcx),
            }
            if !ra.is_zr_or_sp() {
                emit_load_gpr(e, Gpr::Rcx, ra, w, false);
                match w {
                    Width::W32 => e.alu32_rr(0x01, Gpr::Rax, Gpr::Rcx),
                    Width::X64 => e.alu64_rr(0x01, Gpr::Rax, Gpr::Rcx),
                }
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }
        Instruction::Msub { w, rd, rn, rm, ra } => {
            emit_load_gpr(e, Gpr::Rdx, rn, w, false);
            emit_load_gpr(e, Gpr::Rcx, rm, w, false);
            match w {
                Width::W32 => e.imul32_rr(Gpr::Rdx, Gpr::Rcx),
                Width::X64 => e.imul64_rr(Gpr::Rdx, Gpr::Rcx),
            }
            emit_load_gpr(e, Gpr::Rax, ra, w, false);
            match w {
                Width::W32 => e.alu32_rr(0x29, Gpr::Rax, Gpr::Rdx),
                Width::X64 => e.alu64_rr(0x29, Gpr::Rax, Gpr::Rdx),
            }
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }

        Instruction::Csel { w, rd, rn, rm, cond }
        | Instruction::Csinc { w, rd, rn, rm, cond }
        | Instruction::Csinv { w, rd, rn, rm, cond }
        | Instruction::Csneg { w, rd, rn, rm, cond } => {
            let j_taken = emit_jcc_arm_cond(e, cond);
            emit_load_gpr(e, Gpr::Rax, rm, w, false);
            match *inst {
                Instruction::Csinc { .. } => {
                    if w == Width::W32 {
                        e.alu32_ri(0, Gpr::Rax, 1);
                    } else {
                        e.alu64_ri(0, Gpr::Rax, 1);
                    }
                }
                Instruction::Csinv { .. } => {
                    if w == Width::W32 {
                        e.not32(Gpr::Rax);
                    } else {
                        e.not64(Gpr::Rax);
                    }
                }
                Instruction::Csneg { .. } => {
                    if w == Width::W32 {
                        e.neg32(Gpr::Rax);
                    } else {
                        e.neg64(Gpr::Rax);
                    }
                }
                _ => {}
            }
            let j_done = e.jmp_rel32_placeholder();
            let taken_pos = e.pos();
            e.patch_rel32(j_taken, taken_pos);
            emit_load_gpr(e, Gpr::Rax, rn, w, false);
            let done_pos = e.pos();
            e.patch_rel32(j_done, done_pos);
            emit_store_gpr(e, rd, Gpr::Rax, w, false);
            false
        }

        // Fast-path Load/Store UnsignedOffset (W32 / X64)
        Instruction::Ldr {
            w,
            rt,
            rn,
            mode: AddrMode::UnsignedOffset(off),
        } => {
            emit_load_gpr(e, Gpr::R12, rn, Width::X64, true);
            if off != 0 {
                e.mov64_imm(Gpr::Rax, off as u64);
                e.alu64_rr(0x01, Gpr::R12, Gpr::Rax);
            }
            let sz = if w == Width::W32 { 4 } else { 8 };
            emit_fast_or_slow_read(e, Gpr::R12, sz, Gpr::Rax, pc, insns_so_far, fault_patches);
            emit_store_gpr(e, rt, Gpr::Rax, w, false);
            false
        }
        Instruction::Str {
            w,
            rt,
            rn,
            mode: AddrMode::UnsignedOffset(off),
        } => {
            emit_load_gpr(e, Gpr::R12, rn, Width::X64, true);
            if off != 0 {
                e.mov64_imm(Gpr::Rax, off as u64);
                e.alu64_rr(0x01, Gpr::R12, Gpr::Rax);
            }
            emit_load_gpr(e, Gpr::R13, rt, w, false);
            let sz = if w == Width::W32 { 4 } else { 8 };
            emit_fast_or_slow_write(e, Gpr::R12, Gpr::R13, sz, pc, insns_so_far, fault_patches);
            false
        }

        // --- Branch Terminators ---
        Instruction::B { offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            emit_add_insn_count(e, insns_so_far);
            e.mov64_imm(Gpr::Rax, target);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
            true
        }
        Instruction::Bl { offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            let lr = pc.wrapping_add(4);
            e.mov64_imm(Gpr::Rax, lr);
            emit_store_gpr(e, Reg::LR, Gpr::Rax, Width::X64, false);
            emit_add_insn_count(e, insns_so_far);
            e.mov64_imm(Gpr::Rax, target);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
            true
        }
        Instruction::BCond { cond, offset } => {
            let taken_pc = (pc as i64).wrapping_add(offset) as u64;
            let fall_pc = pc.wrapping_add(4);
            let j_taken = emit_jcc_arm_cond(e, cond);
            e.mov64_imm(Gpr::Rax, fall_pc);
            let j_end = e.jmp_rel32_placeholder();
            let tpos = e.pos();
            e.patch_rel32(j_taken, tpos);
            e.mov64_imm(Gpr::Rax, taken_pc);
            let epos = e.pos();
            e.patch_rel32(j_end, epos);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
            emit_add_insn_count(e, insns_so_far);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
            true
        }
        Instruction::Cbz { w, rt, offset } | Instruction::Cbnz { w, rt, offset } => {
            let is_nz = matches!(inst, Instruction::Cbnz { .. });
            let taken_pc = (pc as i64).wrapping_add(offset) as u64;
            let fall_pc = pc.wrapping_add(4);
            emit_load_gpr(e, Gpr::Rcx, rt, w, false);
            match w {
                Width::W32 => e.alu32_rr(0x85, Gpr::Rcx, Gpr::Rcx),
                Width::X64 => e.alu64_rr(0x85, Gpr::Rcx, Gpr::Rcx),
            }
            let j_taken = e.jcc_rel32_placeholder(if is_nz { X64Cc::Ne } else { X64Cc::E });
            e.mov64_imm(Gpr::Rax, fall_pc);
            let j_end = e.jmp_rel32_placeholder();
            let tpos = e.pos();
            e.patch_rel32(j_taken, tpos);
            e.mov64_imm(Gpr::Rax, taken_pc);
            let epos = e.pos();
            e.patch_rel32(j_end, epos);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
            emit_add_insn_count(e, insns_so_far);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
            true
        }
        Instruction::Tbz { rt, bit, offset } | Instruction::Tbnz { rt, bit, offset } => {
            let is_nz = matches!(inst, Instruction::Tbnz { .. });
            let taken_pc = (pc as i64).wrapping_add(offset) as u64;
            let fall_pc = pc.wrapping_add(4);
            emit_load_gpr(e, Gpr::Rcx, rt, Width::X64, false);
            if bit > 0 {
                e.shift64_imm(5, Gpr::Rcx, bit);
            }
            e.alu32_ri(4, Gpr::Rcx, 1);
            let j_taken = e.jcc_rel32_placeholder(if is_nz { X64Cc::Ne } else { X64Cc::E });
            e.mov64_imm(Gpr::Rax, fall_pc);
            let j_end = e.jmp_rel32_placeholder();
            let tpos = e.pos();
            e.patch_rel32(j_taken, tpos);
            e.mov64_imm(Gpr::Rax, taken_pc);
            let epos = e.pos();
            e.patch_rel32(j_end, epos);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
            emit_add_insn_count(e, insns_so_far);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
            true
        }
        Instruction::Br { rn } | Instruction::Ret { rn } => {
            emit_load_gpr(e, Gpr::Rax, rn, Width::X64, false);
            // Strip PAC bits cleanly inline: keep bits 0..47, sign-extend bit 55 if set
            e.mov64_imm(Gpr::Rcx, strip_pac(0x0000_FFFF_FFFF_FFFF));
            e.mov64_rr(Gpr::Rdx, Gpr::Rax);
            e.shift64_imm(5, Gpr::Rdx, 55);
            e.alu32_ri(4, Gpr::Rdx, 1);
            let is_kern = e.jcc_rel32_placeholder(X64Cc::Ne);
            e.alu64_rr(0x21, Gpr::Rax, Gpr::Rcx);
            let done_pac = e.jmp_rel32_placeholder();
            let kpos = e.pos();
            e.patch_rel32(is_kern, kpos);
            e.mov64_imm(Gpr::Rcx, 0xFFFF_0000_0000_0000);
            e.alu64_rr(0x09, Gpr::Rax, Gpr::Rcx);
            let dpos = e.pos();
            e.patch_rel32(done_pac, dpos);

            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);
            emit_add_insn_count(e, insns_so_far);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
            true
        }
        Instruction::Blr { rn } => {
            emit_load_gpr(e, Gpr::R12, rn, Width::X64, false);
            // Strip PAC bits identically to Br/Ret
            e.mov64_imm(Gpr::Rcx, 0x0000_FFFF_FFFF_FFFF);
            e.mov64_rr(Gpr::Rdx, Gpr::R12);
            e.shift64_imm(5, Gpr::Rdx, 55);
            e.alu32_ri(4, Gpr::Rdx, 1);
            let is_kern = e.jcc_rel32_placeholder(X64Cc::Ne);
            e.alu64_rr(0x21, Gpr::R12, Gpr::Rcx);
            let done_pac = e.jmp_rel32_placeholder();
            let kpos = e.pos();
            e.patch_rel32(is_kern, kpos);
            e.mov64_imm(Gpr::Rcx, 0xFFFF_0000_0000_0000);
            e.alu64_rr(0x09, Gpr::R12, Gpr::Rcx);
            let dpos = e.pos();
            e.patch_rel32(done_pac, dpos);

            e.mov64_imm(Gpr::Rax, pc.wrapping_add(4));
            emit_store_gpr(e, Reg::LR, Gpr::Rax, Width::X64, false);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::R12);
            emit_add_insn_count(e, insns_so_far);
            e.mov32_imm(Gpr::Rax, EXIT_CONTINUE as u32);
            exit_patches.push(e.jmp_rel32_placeholder());
            true
        }

        // Everything else (Sbfm/Ubfm/Bfm, Ldrb/Strb/Ldp/Stp, Mrs/Msr, Svc/Hlt/Brk, Atomics, FP/SIMD)
        // delegates to the reference interpreter helper.
        _ => {
            // Keep cpu.pc exact before fallback so SVC/HLT/BRK/WFI/Fault report exact PC
            e.mov64_imm(Gpr::Rax, pc);
            e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rax);

            let fn_off = offset_of!(JitRuntimeCtx, fallback_fn) as i32;
            e.mov64_rm(Gpr::R11, Gpr::R14, fn_off);
            if cfg!(windows) {
                e.mov64_rr(Gpr::Rcx, Gpr::R15);
                e.mov64_rr(Gpr::Rdx, Gpr::R14);
                e.mov64_imm(Gpr::R8, pc);
                e.mov32_imm(Gpr::R9, word);
            } else {
                e.mov64_rr(Gpr::Rdi, Gpr::R15);
                e.mov64_rr(Gpr::Rsi, Gpr::R14);
                e.mov64_imm(Gpr::Rdx, pc);
                e.mov32_imm(Gpr::Rcx, word);
            }
            e.call_r(Gpr::R11);

            let terminates = matches!(
                inst,
                Instruction::Svc { .. }
                    | Instruction::Hvc { .. }
                    | Instruction::Smc { .. }
                    | Instruction::Brk { .. }
                    | Instruction::Hlt { .. }
                    | Instruction::Eret
                    | Instruction::Msr {
                        sysreg: armt_core::isa::SysReg::SCTLR_EL1
                            | armt_core::isa::SysReg::TTBR0_EL1
                            | armt_core::isa::SysReg::TTBR1_EL1
                            | armt_core::isa::SysReg::TCR_EL1,
                        ..
                    }
                    | Instruction::Tlbi { .. }
                    | Instruction::Ic { .. }
                    | Instruction::MsrImm { .. }
                    | Instruction::Hint { imm: 2 }
                    | Instruction::PacHint {
                        op: PacOp::Retaa | PacOp::Retab | PacOp::Braa | PacOp::Brab | PacOp::Blraa | PacOp::Blrab,
                        ..
                    }
                    | Instruction::Unknown(_)
            );

            // Check if fallback returned nonzero (SVC/HLT/Branch/Fault)
            e.alu64_rr(0x85, Gpr::Rax, Gpr::Rax);
            let cont_patch = e.jcc_rel32_placeholder(X64Cc::E);
            emit_add_insn_count(e, insns_so_far);
            exit_patches.push(e.jmp_rel32_placeholder());

            let cont_pos = e.pos();
            e.patch_rel32(cont_patch, cont_pos);

            if terminates {
                // Fallback returned 0 unexpectedly or finished — update PC & insn_count
                emit_add_insn_count(e, insns_so_far);
                e.mov64_imm(Gpr::Rcx, pc.wrapping_add(4));
                e.mov64_mr(Gpr::R15, off_pc(), Gpr::Rcx);
                exit_patches.push(e.jmp_rel32_placeholder());
            }
            terminates
        }
    }
}
