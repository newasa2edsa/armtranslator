use crate::loader::{init_cpu_from_image, LoadedImage};
use armt_core::{CpuState, GuestMemory, Result};
use armt_decoder::{decode, disasm};
use armt_hw::{Framebuffer, Vma2Bus, Vma2MemMap};
use armt_jit::{Interpreter, JitEngine, Mmu, StepOutcome};
use std::time::Instant;

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
}

/// Complete sandboxed ARM64 / Apple VirtualMac2,1 emulator orchestrator.
pub struct ArmVm {
    pub cpu: CpuState,
    pub bus: Vma2Bus,
    pub mmu: Mmu,
    pub jit: JitEngine,
    pub config: VmConfig,
    pub is_apple_kernel: bool,
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
        }
    }

    pub fn apply_image(&mut self, img: &LoadedImage) {
        self.cpu = init_cpu_from_image(img);
        self.is_apple_kernel = img.is_apple_kernel;
    }

    pub fn framebuffer(&self) -> Framebuffer {
        self.bus.framebuffer.clone()
    }

    pub fn uart_output(&self) -> String {
        self.bus.uart0.tx_string()
    }

    /// Execute the loaded ARM64 image up to `config.max_steps` or synchronous halt.
    pub fn run(&mut self) -> Result<VmStats> {
        let start = Instant::now();
        let max_steps = self.config.max_steps;
        let mut exit_reason = String::new();

        match self.config.mode {
            ExecMode::Jit if !self.config.trace => {
                let mut remaining = max_steps;
                let mut wfi_count = 0u64;
                while remaining > 0 {
                    let chunk = remaining.min(100_000);
                    let before = self.cpu.insn_count;
                    match self.jit.run(&mut self.cpu, &mut self.bus, &mut self.mmu, chunk) {
                        Ok(StepOutcome::Continue) => {
                            let ran = (self.cpu.insn_count - before).max(1);
                            self.bus.tick(ran);
                            remaining = remaining.saturating_sub(ran);
                            if remaining == 0 {
                                exit_reason = format!("Reached step budget ({max_steps})");
                            }
                        }
                        Ok(StepOutcome::Hlt(imm)) => {
                            exit_reason = format!("HLT #{imm}");
                            break;
                        }
                        Ok(StepOutcome::Brk(imm)) => {
                            exit_reason = format!("BRK #0x{imm:x}");
                            break;
                        }
                        Ok(StepOutcome::Svc(imm)) => {
                            exit_reason = format!("SVC #0x{imm:x}");
                            break;
                        }
                        Ok(StepOutcome::Wfi) => {
                            wfi_count += 1;
                            self.bus.tick(1000);
                            self.cpu.pc = self.cpu.pc.wrapping_add(4);
                            remaining = remaining.saturating_sub(1);
                            if wfi_count > 256 || !self.is_apple_kernel {
                                exit_reason = "WFI idle".into();
                                break;
                            }
                        }
                        Ok(StepOutcome::Unknown(word)) => {
                            if self.is_apple_kernel {
                                exit_reason = format!(
                                    "Unimplemented ARM64 instruction 0x{word:08x} at PC=0x{:016x}",
                                    self.cpu.pc
                                );
                                break;
                            }
                            exit_reason = format!("Unknown instruction 0x{word:08x}");
                            break;
                        }
                        Ok(other) => {
                            exit_reason = format!("{other:?}");
                            break;
                        }
                        Err(e) => {
                            if self.is_apple_kernel {
                                exit_reason = format!("Guest boundary at PC=0x{:016x}: {e}", self.cpu.pc);
                                break;
                            }
                            return Err(e);
                        }
                    }
                }
                if exit_reason.is_empty() {
                    exit_reason = format!("Completed step budget ({max_steps})");
                }
            }
            _ => {
                // Reference Interpreter (used for `--mode interp` or `--trace`)
                for _ in 0..max_steps {
                    if self.config.trace {
                        if let Ok(pa) = self.mmu.translate(&self.cpu, &mut self.bus, self.cpu.pc, false) {
                            if let Ok(w) = self.bus.read_u32(pa) {
                                let inst = decode(w);
                                println!(
                                    "[0x{:016x}] {:08x}  {}",
                                    self.cpu.pc,
                                    w,
                                    disasm(&inst, self.cpu.pc)
                                );
                            }
                        }
                    }
                    match Interpreter::step(&mut self.cpu, &mut self.bus, &mut self.mmu) {
                        Ok(StepOutcome::Continue) | Ok(StepOutcome::Branch(_)) => {
                            self.bus.tick(1);
                        }
                        Ok(StepOutcome::Hlt(imm)) => {
                            exit_reason = format!("HLT #{imm}");
                            break;
                        }
                        Ok(StepOutcome::Brk(imm)) => {
                            exit_reason = format!("BRK #0x{imm:x}");
                            break;
                        }
                        Ok(StepOutcome::Svc(imm)) => {
                            exit_reason = format!("SVC #0x{imm:x}");
                            break;
                        }
                        Ok(StepOutcome::Wfi) => {
                            exit_reason = "WFI idle".into();
                            break;
                        }
                        Ok(StepOutcome::Unknown(w)) => {
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
                        Err(e) => {
                            if self.is_apple_kernel {
                                exit_reason = format!("Guest boundary at PC=0x{:016x}: {e}", self.cpu.pc);
                                break;
                            }
                            return Err(e);
                        }
                    }
                }
                if exit_reason.is_empty() {
                    exit_reason = format!("Completed step budget ({max_steps})");
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64().max(1e-9);
        let insns = self.cpu.insn_count;
        let mips = (insns as f64 / 1_000_000.0) / elapsed;

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
        })
    }
}
