use crate::hypercall::{handle_hvc, HypercallAction, HypercallState};
use crate::loader::{init_cpu_from_image, LoadedImage};
use armt_core::{ArmtError, CpuState, GuestMemory, Result};
use armt_decoder::{decode, disasm};
use armt_hw::{Framebuffer, Vma2Bus, Vma2MemMap, VirtioBlk, VirtioInputShared};
use armt_jit::{Interpreter, JitEngine, Mmu, StepOutcome};
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    Jit,
    Interpreter,
}

#[derive(Debug, Clone)]
pub struct VmConfig {
    pub mode: ExecMode,
    pub ram_mb: u64,
    pub echo_uart: bool,
    pub max_steps: u64,
    pub trace: bool,
}

impl Default for VmConfig {
    fn default() -> Self {
        Self {
            mode: ExecMode::Jit,
            ram_mb: 256,
            echo_uart: true,
            max_steps: 5_000_000,
            trace: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VmStats {
    pub insns_executed: u64,
    pub elapsed_secs: f64,
    pub mips: f64,
    pub blocks_compiled: u64,
    pub jit_hits: u64,
    pub jit_misses: u64,
    pub interp_fallbacks: u64,
    pub uart_bytes: usize,
    pub fb_non_zero_pixels: usize,
    pub exit_reason: String,
    pub final_pc: u64,
    pub x0: u64,
    pub unknown_sysregs: Vec<(u16, u64)>,
    pub unknown_mmio_pages: Vec<(u64, u64)>,
    pub hvc_calls: Vec<(u64, u64)>,
}

/// Live control handles for driving `ArmVm` asynchronously from the RTX GUI thread.
#[derive(Clone)]
pub struct VmControl {
    pub stop: Arc<AtomicBool>,
    pub paused: Arc<AtomicBool>,
    pub unlimited_steps: bool,
    pub frame_pace_ms: u64,
}

impl Default for VmControl {
    fn default() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            unlimited_steps: true,
            frame_pace_ms: 12,
        }
    }
}

/// Live telemetry snapshot updated by the VM worker thread for the GUI inspector.
#[derive(Debug, Clone, Default)]
pub struct VmLiveTelemetry {
    pub running: bool,
    pub insns_executed: u64,
    pub mips: f64,
    pub blocks_compiled: u64,
    pub pc: u64,
    pub el: u8,
    pub mmu_enabled: bool,
    pub guest_frames: u64,
    pub hvc_total: u64,
    pub unknown_sysreg_count: usize,
    pub status_line: String,
    pub exit_reason: Option<String>,
}

/// Complete sandboxed ARM64 / Apple VirtualMac2,1 emulator orchestrator.
pub struct ArmVm {
    pub cpu: CpuState,
    pub bus: Vma2Bus,
    pub mmu: Mmu,
    pub jit: JitEngine,
    pub config: VmConfig,
    pub is_apple_kernel: bool,
    pub hypercall_state: HypercallState,
    pub trace_tail: usize,
}

impl ArmVm {
    pub fn new(config: VmConfig) -> Self {
        let memmap = Vma2MemMap::with_ram_mb(config.ram_mb);
        let bus = Vma2Bus::new(memmap, config.echo_uart);
        let cpu = CpuState::default_vma2_el1(bus.memmap.ram_base, bus.memmap.ram_base + 0x100_0000, 0);
        Self {
            cpu,
            bus,
            mmu: Mmu::new(),
            jit: JitEngine::new(),
            config,
            is_apple_kernel: false,
            hypercall_state: HypercallState::default(),
            trace_tail: 0,
        }
    }

    pub fn apply_image(&mut self, img: &LoadedImage) {
        self.cpu = init_cpu_from_image(img);
        self.is_apple_kernel = img.is_apple_kernel;
        self.hypercall_state = HypercallState::default();
        self.mmu = Mmu::new();
        self.jit.invalidate_all();
    }

    /// Attach a host disk image (`.iso`, `.img`, `.raw`, `.dmg`) to the PCIe VirtIO-Blk controller.
    pub fn attach_disk_image(&mut self, path: &Path, read_only: bool) -> Result<()> {
        self.bus.virtio_blk = VirtioBlk::open_disk_image(path, read_only)?;
        Ok(())
    }

    pub fn framebuffer(&self) -> Framebuffer {
        self.bus.framebuffer.clone()
    }

    pub fn shared_input(&self) -> Arc<Mutex<VirtioInputShared>> {
        self.bus.virtio_input.shared()
    }

    pub fn shared_uart_tx(&self) -> Arc<Mutex<Vec<u8>>> {
        self.bus.uart0.shared_tx()
    }

    pub fn shared_uart_rx(&self) -> Arc<Mutex<VecDeque<u8>>> {
        self.bus.uart0.shared_rx()
    }

    pub fn uart_output(&self) -> String {
        self.bus.uart0.tx_string()
    }

    /// Execute synchronously up to `config.max_steps` or halt.
    pub fn run(&mut self) -> Result<VmStats> {
        let control = VmControl {
            stop: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            unlimited_steps: false,
            frame_pace_ms: 0,
        };
        let telemetry = Arc::new(Mutex::new(VmLiveTelemetry::default()));
        self.run_live(&control, &telemetry)
    }

    /// Execute until `control.stop` is signaled, `config.max_steps` (if bounded), or synchronous halt.
    pub fn run_live(
        &mut self,
        control: &VmControl,
        telemetry: &Arc<Mutex<VmLiveTelemetry>>,
    ) -> Result<VmStats> {
        let start = Instant::now();
        let max_steps = self.config.max_steps;
        let mut exit_reason = String::new();
        let mut wfi_count = 0u64;

        {
            let mut t = telemetry.lock().unwrap();
            t.running = true;
            t.exit_reason = None;
            t.status_line = if self.is_apple_kernel {
                "Running Apple VirtualMac2,1 XNU kernel...".into()
            } else {
                "Running ARM64 image...".into()
            };
        }

        match self.config.mode {
            ExecMode::Jit if !self.config.trace => {
                let mut remaining = if control.unlimited_steps {
                    u64::MAX
                } else {
                    max_steps
                };
                let mut last_fault: Option<(u64, u64, u32)> = None;
                let mut repeat_faults = 0u32;
                let mut fiq_count = 0u64;
                let mut irq_count = 0u64;
                let mut svc_count = 0u64;
                let mut el0_logged = false;
                let mut next_hb = 500_000_000u64;
                while remaining > 0 {
                    if control.stop.load(Ordering::Acquire) {
                        exit_reason = "Stopped by user".into();
                        break;
                    }
                    if control.paused.load(Ordering::Acquire) {
                        std::thread::sleep(Duration::from_millis(16));
                        continue;
                    }
                    if self.cpu.el == 0 && !el0_logged {
                        el0_logged = true;
                        if self.trace_tail > 0 {
                            eprintln!(
                                "[EL0_ENTER] insn={} pc=0x{:016x} sp_el0=0x{:016x} ttbr0=0x{:016x}",
                                self.cpu.insn_count, self.cpu.pc, self.cpu.sp_el0, self.cpu.ttbr0_el1
                            );
                        }
                    }
                    if self.trace_tail > 0 && self.cpu.insn_count >= next_hb {
                        next_hb += 500_000_000;
                        eprintln!(
                            "[HB] insn={} el={} pc=0x{:016x} lr=0x{:016x} fiqs={fiq_count} irqs={irq_count} svcs={svc_count}",
                            self.cpu.insn_count, self.cpu.el, self.cpu.pc, self.cpu.x[30]
                        );
                    }

                    // Deliver pending GIC Group-0 FIQ or Group-1 IRQ into EL1 if unmasked
                    if self.cpu.vbar_el1 != 0 {
                        if (self.cpu.daif & 1) == 0 && self.bus.gic.pending_group(0).is_some() {
                            fiq_count += 1;
                            Mmu::take_fiq(&mut self.cpu);
                        } else if (self.cpu.daif & 2) == 0 && self.bus.gic.pending_group(1).is_some()
                        {
                            irq_count += 1;
                            Mmu::take_exception(&mut self.cpu, 0, 0, 0, true);
                        }
                    }

                    let chunk = remaining.min(100_000);
                    let before = self.cpu.insn_count;
                    match self.jit.run(&mut self.cpu, &mut self.bus, &mut self.mmu, chunk) {
                        Ok(StepOutcome::Continue) => {
                            let ran = (self.cpu.insn_count - before).max(1);
                            self.bus.tick(ran);
                            if !control.unlimited_steps {
                                remaining = remaining.saturating_sub(ran);
                                if remaining == 0 {
                                    exit_reason = format!("Reached step budget ({max_steps})");
                                }
                            }
                            wfi_count = 0;
                        }
                        Ok(StepOutcome::Hlt(imm)) => {
                            exit_reason = format!("HLT #{imm}");
                            break;
                        }
                        Ok(StepOutcome::Brk(imm)) => {
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 && self.cpu.el == 0 {
                                Mmu::take_exception(&mut self.cpu, 0x3C, imm as u32, 0, false);
                                self.bus.tick(1);
                                let ran = (self.cpu.insn_count - before).max(1);
                                if !control.unlimited_steps {
                                    remaining = remaining.saturating_sub(ran);
                                    if remaining == 0 {
                                        exit_reason = format!("Reached step budget ({max_steps})");
                                    }
                                }
                            } else {
                                exit_reason = format!("BRK #0x{imm:x}");
                                break;
                            }
                        }
                        Ok(StepOutcome::Svc(imm)) => {
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 {
                                svc_count += 1;
                                if self.trace_tail > 0 && svc_count <= 20 {
                                    eprintln!(
                                        "[SVC #{svc_count}] insn={} el={} imm=0x{imm:x} pc=0x{:016x} x16={} x0=0x{:x} x1=0x{:x}",
                                        self.cpu.insn_count,
                                        self.cpu.el,
                                        self.cpu.pc,
                                        self.cpu.x[16] as i64,
                                        self.cpu.x[0],
                                        self.cpu.x[1]
                                    );
                                }
                                Mmu::take_svc(&mut self.cpu, imm);
                                self.bus.tick(1);
                                let ran = (self.cpu.insn_count - before).max(1);
                                if !control.unlimited_steps {
                                    remaining = remaining.saturating_sub(ran);
                                    if remaining == 0 {
                                        exit_reason = format!("Reached step budget ({max_steps})");
                                    }
                                }
                            } else {
                                exit_reason = format!("SVC #0x{imm:x}");
                                break;
                            }
                        }
                        Ok(StepOutcome::Hvc(imm)) | Ok(StepOutcome::Smc(imm)) => {
                            match handle_hvc(&mut self.cpu, &mut self.hypercall_state, imm) {
                                HypercallAction::Continue => {
                                    self.cpu.pc = self.cpu.pc.wrapping_add(4);
                                    self.bus.tick(1);
                                    let ran = (self.cpu.insn_count - before).max(1);
                                    if !control.unlimited_steps {
                                        remaining = remaining.saturating_sub(ran);
                                        if remaining == 0 {
                                            exit_reason =
                                                format!("Reached step budget ({max_steps})");
                                        }
                                    }
                                }
                                HypercallAction::Shutdown(msg) => {
                                    exit_reason = msg.to_string();
                                    break;
                                }
                            }
                        }
                        Ok(StepOutcome::Wfi) => {
                            wfi_count += 1;
                            self.bus.tick(10_000);
                            self.cpu.pc = self.cpu.pc.wrapping_add(4);
                            let ran = (self.cpu.insn_count - before).max(1);
                            if !control.unlimited_steps {
                                remaining = remaining.saturating_sub(ran);
                                if remaining == 0 {
                                    exit_reason = format!("Reached step budget ({max_steps})");
                                }
                            }
                            if wfi_count > 1024 || !self.is_apple_kernel {
                                exit_reason = "WFI idle".into();
                                break;
                            }
                        }
                        Ok(StepOutcome::Wfe) => {
                            self.bus.tick(5_000);
                            self.cpu.pc = self.cpu.pc.wrapping_add(4);
                            let ran = (self.cpu.insn_count - before).max(1);
                            if !control.unlimited_steps {
                                remaining = remaining.saturating_sub(ran);
                                if remaining == 0 {
                                    exit_reason = format!("Reached step budget ({max_steps})");
                                }
                            }
                        }
                        Ok(StepOutcome::Eret) if self.is_apple_kernel => {
                            let ran = (self.cpu.insn_count - before).max(1);
                            self.bus.tick(ran);
                            if !control.unlimited_steps {
                                remaining = remaining.saturating_sub(ran);
                                if remaining == 0 {
                                    exit_reason = format!("Reached step budget ({max_steps})");
                                }
                            }
                            wfi_count = 0;
                        }
                        Ok(StepOutcome::Unknown(word)) => {
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 && (self.cpu.el == 0 || (word & 0xFFFF_FFFE) == 0xE7FF_DEFE) {
                                if self.trace_tail > 0 {
                                    let s1 = read_guest_cstr(&mut self.mmu, &self.cpu, &mut self.bus, self.cpu.x[1]);
                                    let s2 = read_guest_cstr(&mut self.mmu, &self.cpu, &mut self.bus, self.cpu.x[2]);
                                    eprintln!(
                                        "[TRAP] insn={} pc=0x{:016x} word=0x{word:08x} x0=0x{:x} x1=0x{:x} ({:?}) x2=0x{:x} ({:?}) lr=0x{:016x}",
                                        self.cpu.insn_count,
                                        self.cpu.pc,
                                        self.cpu.x[0],
                                        self.cpu.x[1],
                                        s1,
                                        self.cpu.x[2],
                                        s2,
                                        self.cpu.x[30],
                                    );
                                }
                                let key = (self.cpu.pc, 0, word);
                                if last_fault == Some(key) {
                                    repeat_faults += 1;
                                } else {
                                    last_fault = Some(key);
                                    repeat_faults = 1;
                                }
                                if repeat_faults <= 2 {
                                    Mmu::take_exception(&mut self.cpu, 0x00, 0, 0, false);
                                    continue;
                                }
                            }
                            exit_reason = if self.is_apple_kernel {
                                format!(
                                    "Unimplemented ARM64 instruction 0x{word:08x} at PC=0x{:016x}",
                                    self.cpu.pc
                                )
                            } else {
                                format!("Unknown instruction 0x{word:08x}")
                            };
                            break;
                        }
                        Ok(other) => {
                            exit_reason = format!("{other:?}");
                            break;
                        }
                        Err(ArmtError::MmuFault { va, iss, reason })
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 =>
                        {
                            if self.trace_tail > 0 {
                                eprintln!(
                                    "[MMU_FAULT] insn={} pc=0x{:016x} va=0x{va:016x} iss=0x{iss:08x} lr=0x{:016x} ({reason})",
                                    self.cpu.insn_count,
                                    self.cpu.pc,
                                    self.cpu.x[30],
                                );
                            }
                            let key = (self.cpu.pc, va, iss);
                            if last_fault == Some(key) {
                                repeat_faults += 1;
                            } else {
                                last_fault = Some(key);
                                repeat_faults = 1;
                            }
                            if repeat_faults > 8 {
                                exit_reason = format!(
                                    "Guest boundary at PC=0x{:016x}: repeated MMU fault at VA=0x{va:016x} (ISS=0x{iss:08x}): {reason}",
                                    self.cpu.pc
                                );
                                break;
                            }
                            let is_iabt = (iss & 0x40) == 0 && va == self.cpu.pc;
                            let ec = match (is_iabt, self.cpu.el) {
                                (true, 0) => 0x20,
                                (true, _) => 0x21,
                                (false, 0) => 0x24,
                                (false, _) => 0x25,
                            };
                            Mmu::take_exception(&mut self.cpu, ec, iss, va, false);
                        }
                        Err(e) => {
                            if self.is_apple_kernel {
                                exit_reason =
                                    format!("Guest boundary at PC=0x{:016x}: {e}", self.cpu.pc);
                                break;
                            }
                            telemetry.lock().unwrap().running = false;
                            return Err(e);
                        }
                    }

                    update_telemetry(
                        telemetry,
                        &self.cpu,
                        &self.jit,
                        &self.bus,
                        &self.hypercall_state.hvc_counts,
                        start,
                    );
                }
                if exit_reason.is_empty() {
                    exit_reason = format!("Completed step budget ({max_steps})");
                }
                if self.trace_tail > 0 {
                    println!("=== Post-JIT trace ({} steps from PC=0x{:016x}) ===", self.trace_tail, self.cpu.pc);
                    for _ in 0..self.trace_tail {
                        let Ok(pa) = self.mmu.translate(&self.cpu, &mut self.bus, self.cpu.pc, false) else {
                            break;
                        };
                        let Ok(w) = self.bus.read_u32(pa) else { break };
                        let inst = decode(w);
                        println!(
                            "#{:<8} [0x{:016x} -> pa 0x{:08x}] {:08x}  {:<38} x0={:016x} x1={:016x} x2={:016x} x8={:016x} lr={:016x} sp={:016x}",
                            self.cpu.insn_count,
                            self.cpu.pc,
                            pa,
                            w,
                            disasm(&inst, self.cpu.pc),
                            self.cpu.x[0],
                            self.cpu.x[1],
                            self.cpu.x[2],
                            self.cpu.x[8],
                            self.cpu.x[30],
                            self.cpu.sp,
                        );
                        match Interpreter::step(&mut self.cpu, &mut self.bus, &mut self.mmu) {
                            Ok(StepOutcome::Continue)
                            | Ok(StepOutcome::Branch(_))
                            | Ok(StepOutcome::Eret) => self.bus.tick(1),
                            Ok(StepOutcome::Wfe) => {
                                self.cpu.pc = self.cpu.pc.wrapping_add(4);
                                self.bus.tick(1);
                            }
                            Err(ArmtError::MmuFault { va, iss, reason })
                                if self.is_apple_kernel && self.cpu.vbar_el1 != 0 =>
                            {
                                println!(
                                    "  [trace MMU_FAULT] pc=0x{:016x} va=0x{va:016x} iss=0x{iss:08x}: {reason}",
                                    self.cpu.pc
                                );
                                let is_iabt = (iss & 0x40) == 0 && va == self.cpu.pc;
                                let ec = match (is_iabt, self.cpu.el) {
                                    (true, 0) => 0x20,
                                    (true, _) => 0x21,
                                    (false, 0) => 0x24,
                                    (false, _) => 0x25,
                                };
                                Mmu::take_exception(&mut self.cpu, ec, iss, va, false);
                            }
                            Ok(other) => {
                                println!("  [trace stop] {other:?}");
                                break;
                            }
                            Err(e) => {
                                println!("  [trace err] {e}");
                                break;
                            }
                        }
                    }
                    println!(
                        "=== CPU regs: el={} daif=0x{:x} sctlr=0x{:016x} tcr=0x{:016x} ttbr0=0x{:016x} ttbr1=0x{:016x} vbar=0x{:016x} tpidr_el1=0x{:016x} esr=0x{:016x} far=0x{:016x} ===",
                        self.cpu.el,
                        self.cpu.daif,
                        self.cpu.sctlr_el1,
                        self.cpu.tcr_el1,
                        self.cpu.ttbr0_el1,
                        self.cpu.ttbr1_el1,
                        self.cpu.vbar_el1,
                        self.cpu.tpidr_el1,
                        self.cpu.esr_el1,
                        self.cpu.far_el1,
                    );
                    println!(
                        "=== IRQ/Timer: fiqs={fiq_count} irqs={irq_count} svcs={svc_count} cnt={} cntv_ctl=0x{:x} cntv_cval={} cntp_ctl=0x{:x} cntp_cval={} | {} ===",
                        self.bus.timer.counter,
                        self.bus.timer.cntv_ctl,
                        self.bus.timer.cntv_cval,
                        self.bus.timer.cntp_ctl,
                        self.bus.timer.cntp_cval,
                        self.bus.gic.debug_summary(),
                    );
                    println!(
                        "=== PCIe: {} | AES1_writes={} last=(0x{:x},0x{:x}) AES2_writes={} last=(0x{:x},0x{:x}) ===",
                        self.bus.pcie.debug_summary(),
                        self.bus.aes1.write_count,
                        self.bus.aes1.last_write.0,
                        self.bus.aes1.last_write.1,
                        self.bus.aes2.write_count,
                        self.bus.aes2.last_write.0,
                        self.bus.aes2.last_write.1,
                    );
                    // Walk frame-pointer chain (FP = X29) to print guest call stack
                    let mut fp = self.cpu.x[29];
                    let mut frames = Vec::new();
                    for _ in 0..16 {
                        if fp < 0xffff_fe00_0000_0000 || (fp & 7) != 0 {
                            break;
                        }
                        let Ok(pa_fp) = self.mmu.translate(&self.cpu, &mut self.bus, fp, false) else {
                            break;
                        };
                        let Ok(pa_lr) = self.mmu.translate(&self.cpu, &mut self.bus, fp.wrapping_add(8), false) else {
                            break;
                        };
                        let Ok(next_fp) = self.bus.read_u64(pa_fp) else { break };
                        let Ok(lr_val) = self.bus.read_u64(pa_lr) else { break };
                        frames.push(format!("0x{:016x}", lr_val));
                        if next_fp <= fp {
                            break;
                        }
                        fp = next_fp;
                    }
                    println!("=== Guest FP backtrace: [{}] ===", frames.join(", "));
                    // Dump any printable ASCII strings near SP (e.g. XNU panic buffer on stack)
                    let sp_base = self.cpu.sp & !0xF;
                    let mut ascii = String::new();
                    for off in 0..1024u64 {
                        if let Ok(pa) = self.mmu.translate(&self.cpu, &mut self.bus, sp_base.wrapping_add(off), false) {
                            if let Ok(b) = self.bus.read_u8(pa) {
                                if (0x20..0x7F).contains(&b) || b == b'\n' {
                                    ascii.push(b as char);
                                } else if !ascii.ends_with(' ') {
                                    ascii.push(' ');
                                }
                            }
                        }
                    }
                    let trimmed: String = ascii.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !trimmed.is_empty() {
                        println!("=== Stack ASCII near SP=0x{:016x} ===\n{trimmed}", sp_base);
                    }
                }
            }
            _ => {
                // Reference Interpreter (used for `--mode interp`, `--trace`, or `--trace-tail`)
                let limit = if control.unlimited_steps {
                    u64::MAX
                } else {
                    max_steps
                };
                let mut ring: VecDeque<String> = VecDeque::with_capacity(self.trace_tail);
                let mut last_fault: Option<(u64, u64, u32)> = None;
                let mut repeat_faults = 0u32;
                for _ in 0..limit {
                    if control.stop.load(Ordering::Acquire) {
                        exit_reason = "Stopped by user".into();
                        break;
                    }
                    if control.paused.load(Ordering::Acquire) {
                        std::thread::sleep(Duration::from_millis(16));
                        continue;
                    }

                    if self.config.trace || self.trace_tail > 0 {
                        if let Ok(pa) =
                            self.mmu
                                .translate(&self.cpu, &mut self.bus, self.cpu.pc, false)
                        {
                            if let Ok(w) = self.bus.read_u32(pa) {
                                let inst = decode(w);
                                let line = format!(
                                    "#{:<7} [0x{:016x} -> pa 0x{:08x}] {:08x}  {:<38} x0={:016x} x1={:016x} x8={:016x} x16={:016x} lr={:016x} sp={:016x}",
                                    self.cpu.insn_count,
                                    self.cpu.pc,
                                    pa,
                                    w,
                                    disasm(&inst, self.cpu.pc),
                                    self.cpu.x[0],
                                    self.cpu.x[1],
                                    self.cpu.x[8],
                                    self.cpu.x[16],
                                    self.cpu.x[30],
                                    self.cpu.sp,
                                );
                                if self.config.trace {
                                    println!("{line}");
                                }
                                if self.trace_tail > 0 {
                                    if ring.len() == self.trace_tail {
                                        ring.pop_front();
                                    }
                                    ring.push_back(line);
                                }
                            }
                        }
                    }
                    match Interpreter::step(&mut self.cpu, &mut self.bus, &mut self.mmu) {
                        Ok(StepOutcome::Continue)
                        | Ok(StepOutcome::Branch(_))
                        | Ok(StepOutcome::Eret) => {
                            self.bus.tick(1);
                        }
                        Ok(StepOutcome::Hlt(imm)) => {
                            exit_reason = format!("HLT #{imm}");
                            break;
                        }
                        Ok(StepOutcome::Brk(imm)) => {
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 && self.cpu.el == 0 {
                                Mmu::take_exception(&mut self.cpu, 0x3C, imm as u32, 0, false);
                                self.bus.tick(1);
                            } else {
                                exit_reason = format!("BRK #0x{imm:x}");
                                break;
                            }
                        }
                        Ok(StepOutcome::Svc(imm)) => {
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 {
                                Mmu::take_svc(&mut self.cpu, imm);
                                self.bus.tick(1);
                            } else {
                                exit_reason = format!("SVC #0x{imm:x}");
                                break;
                            }
                        }
                        Ok(StepOutcome::Hvc(imm)) | Ok(StepOutcome::Smc(imm)) => {
                            match handle_hvc(&mut self.cpu, &mut self.hypercall_state, imm) {
                                HypercallAction::Continue => {
                                    self.cpu.pc = self.cpu.pc.wrapping_add(4);
                                    self.bus.tick(1);
                                }
                                HypercallAction::Shutdown(msg) => {
                                    exit_reason = msg.to_string();
                                    break;
                                }
                            }
                        }
                        Ok(StepOutcome::Wfi) => {
                            exit_reason = "WFI idle".into();
                            break;
                        }
                        Ok(StepOutcome::Unknown(w)) => {
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 && (self.cpu.el == 0 || (w & 0xFFFF_FFFE) == 0xE7FF_DEFE) {
                                let key = (self.cpu.pc, 0, w);
                                if last_fault == Some(key) {
                                    repeat_faults += 1;
                                } else {
                                    last_fault = Some(key);
                                    repeat_faults = 1;
                                }
                                if repeat_faults <= 2 {
                                    Mmu::take_exception(&mut self.cpu, 0x00, 0, 0, false);
                                    continue;
                                }
                            }
                            exit_reason = format!(
                                "Unimplemented ARM64 instruction 0x{w:08x} at PC=0x{:016x}",
                                self.cpu.pc
                            );
                            break;
                        }
                        Ok(other) => {
                            exit_reason = format!("{other:?}");
                            break;
                        }
                        Err(ArmtError::MmuFault { va, iss, reason })
                            if self.is_apple_kernel && self.cpu.vbar_el1 != 0 =>
                        {
                            let key = (self.cpu.pc, va, iss);
                            if last_fault == Some(key) {
                                repeat_faults += 1;
                            } else {
                                last_fault = Some(key);
                                repeat_faults = 1;
                            }
                            if repeat_faults > 8 {
                                exit_reason = format!(
                                    "Guest boundary at PC=0x{:016x}: repeated MMU fault at VA=0x{va:016x} (ISS=0x{iss:08x}): {reason}",
                                    self.cpu.pc
                                );
                                break;
                            }
                            let is_iabt = (iss & 0x40) == 0 && va == self.cpu.pc;
                            let ec = match (is_iabt, self.cpu.el) {
                                (true, 0) => 0x20,
                                (true, _) => 0x21,
                                (false, 0) => 0x24,
                                (false, _) => 0x25,
                            };
                            Mmu::take_exception(&mut self.cpu, ec, iss, va, false);
                        }
                        Err(e) => {
                            if self.is_apple_kernel {
                                exit_reason =
                                    format!("Guest boundary at PC=0x{:016x}: {e}", self.cpu.pc);
                                break;
                            }
                            telemetry.lock().unwrap().running = false;
                            return Err(e);
                        }
                    }

                    if (self.cpu.insn_count & 0x3FFF) == 0 {
                        update_telemetry(
                            telemetry,
                            &self.cpu,
                            &self.jit,
                            &self.bus,
                            &self.hypercall_state.hvc_counts,
                            start,
                        );
                    }
                }
                if exit_reason.is_empty() {
                    exit_reason = format!("Completed step budget ({max_steps})");
                }
                if !ring.is_empty() {
                    println!("=== Last {} executed instructions ===", ring.len());
                    for l in &ring {
                        println!("{l}");
                    }
                    println!(
                        "=== CPU regs at halt: sctlr=0x{:016x} tcr=0x{:016x} ttbr0=0x{:016x} ttbr1=0x{:016x} vbar=0x{:016x} ===",
                        self.cpu.sctlr_el1,
                        self.cpu.tcr_el1,
                        self.cpu.ttbr0_el1,
                        self.cpu.ttbr1_el1,
                        self.cpu.vbar_el1,
                    );
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64().max(1e-9);
        let insns = self.cpu.insn_count;
        let mips = (insns as f64 / 1_000_000.0) / elapsed;

        update_telemetry(
            telemetry,
            &self.cpu,
            &self.jit,
            &self.bus,
            &self.hypercall_state.hvc_counts,
            start,
        );
        {
            let mut t = telemetry.lock().unwrap();
            t.running = false;
            t.exit_reason = Some(exit_reason.clone());
            t.status_line = exit_reason.clone();
        }

        let mut mmio_pages: Vec<(u64, u64)> = self
            .bus
            .unknown_mmio_reads
            .iter()
            .chain(self.bus.unknown_mmio_writes.iter())
            .map(|(&k, &v)| (k, v))
            .collect();
        mmio_pages.sort_by_key(|&(k, _)| k);

        let sysregs: Vec<(u16, u64)> = self
            .bus
            .unknown_sysregs
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();

        let hvc_calls: Vec<(u64, u64)> = self
            .hypercall_state
            .hvc_counts
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();

        Ok(VmStats {
            insns_executed: insns,
            elapsed_secs: elapsed,
            mips,
            blocks_compiled: self.jit.stats.blocks_compiled,
            jit_hits: self.jit.stats.jit_hits,
            jit_misses: self.jit.stats.jit_misses,
            interp_fallbacks: self.jit.stats.interp_fallbacks,
            uart_bytes: self.bus.uart0.tx_bytes().len(),
            fb_non_zero_pixels: self.bus.framebuffer.non_zero_pixels(),
            exit_reason,
            final_pc: self.cpu.pc,
            x0: self.cpu.x[0],
            unknown_sysregs: sysregs,
            unknown_mmio_pages: mmio_pages,
            hvc_calls,
        })
    }
}

fn read_guest_cstr(mmu: &mut Mmu, cpu: &CpuState, bus: &mut Vma2Bus, va: u64) -> String {
    if va == 0 {
        return String::new();
    }
    let mut out = String::new();
    for i in 0..160u64 {
        let Ok(pa) = mmu.translate(cpu, bus, va.wrapping_add(i), false) else {
            break;
        };
        let Ok(b) = bus.read_u8(pa) else { break };
        if b == 0 {
            break;
        }
        if (0x20..0x7F).contains(&b) {
            out.push(b as char);
        } else if b == b'\n' {
            out.push_str("\\n");
        } else {
            break;
        }
    }
    out
}

fn update_telemetry(
    telemetry: &Arc<Mutex<VmLiveTelemetry>>,
    cpu: &CpuState,
    jit: &JitEngine,
    bus: &Vma2Bus,
    hvc_counts: &BTreeMap<u64, u64>,
    start: Instant,
) {
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let mut t = telemetry.lock().unwrap();
    t.insns_executed = cpu.insn_count;
    t.mips = (cpu.insn_count as f64 / 1_000_000.0) / elapsed;
    t.blocks_compiled = jit.stats.blocks_compiled;
    t.pc = cpu.pc;
    t.el = cpu.el;
    t.mmu_enabled = cpu.mmu_enabled();
    t.guest_frames = bus.apple_gfx.frame_count + bus.virtio_gpu.flushes;
    t.hvc_total = hvc_counts.values().sum();
    t.unknown_sysreg_count = bus.unknown_sysregs.len();
}
