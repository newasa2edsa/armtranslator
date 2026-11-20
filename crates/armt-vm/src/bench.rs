use crate::machine::{ArmVm, ExecMode, VmConfig};
use armt_core::isa::{Condition, Reg, ShiftType, Width};
use armt_core::{GuestMemory, Result};
use armt_decoder::asm;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BenchReport {
    pub iterations: u64,
    pub jit_insns: u64,
    pub jit_ms: f64,
    pub jit_mips: f64,
    pub interp_insns: u64,
    pub interp_ms: f64,
    pub interp_mips: f64,
    pub speedup: f64,
    pub differential_pass: bool,
    pub jit_blocks_compiled: u64,
}

fn build_bench_program(iterations: u64) -> Vec<u8> {
    // Tight multi-block loop summing values + storing/loading scratch RAM
    // X0 = sum, X1 = iter counter, X2 = ram scratch ptr, X3 = step const
    let mut words = Vec::new();
    words.extend(asm::load_u64(Reg(1), iterations));
    words.extend(asm::load_u64(Reg(2), 0x8004_0000));
    words.push(asm::movz(Width::X64, Reg(0), 0, 0));
    words.push(asm::movz(Width::X64, Reg(3), 7, 0));
    // loop (4 instructions before loop -> loop starts here):
    words.push(asm::add_reg(
        Width::X64,
        Reg(0),
        Reg(0),
        Reg(1),
        ShiftType::Lsl,
        0,
        false,
    ));
    words.push(asm::eor_reg(Width::X64, Reg(0), Reg(0), Reg(3)));
    words.push(asm::str_imm(Width::X64, Reg(0), Reg(2), 0));
    words.push(asm::ldr_imm(Width::X64, Reg(4), Reg(2), 0));
    words.push(asm::add_imm(Width::X64, Reg(0), Reg(4), 1, false));
    words.push(asm::sub_imm(Width::X64, Reg(1), Reg(1), 1, true));
    words.push(asm::b_cond(Condition::Ne, -24));
    words.push(asm::hlt(0));
    asm::to_bytes(&words)
}

/// Run a differential equivalence check & speed benchmark comparing the x86-64 Dynarec JIT
/// against the Reference Interpreter.
pub fn run_benchmark(iterations: u64) -> Result<BenchReport> {
    let prog = build_bench_program(iterations);
    let max_steps = iterations.saturating_mul(10).saturating_add(1000);

    // 1. Run on Reference Interpreter
    let mut vm_i = ArmVm::new(VmConfig {
        mode: ExecMode::Interpreter,
        ram_mb: 32,
        echo_uart: false,
        max_steps,
        trace: false,
    });
    let entry = vm_i.bus.memmap.ram_base + 0x10_0000;
    vm_i.bus.write_bytes(entry, &prog)?;
    vm_i.cpu.pc = entry;
    let ti = Instant::now();
    let stats_i = vm_i.run()?;
    let interp_ms = ti.elapsed().as_secs_f64() * 1000.0;

    // 2. Run on Native x86-64 Dynarec JIT
    let mut vm_j = ArmVm::new(VmConfig {
        mode: ExecMode::Jit,
        ram_mb: 32,
        echo_uart: false,
        max_steps,
        trace: false,
    });
    vm_j.bus.write_bytes(entry, &prog)?;
    vm_j.cpu.pc = entry;
    let tj = Instant::now();
    let stats_j = vm_j.run()?;
    let jit_ms = tj.elapsed().as_secs_f64() * 1000.0;

    let diff_pass = vm_i.cpu.x == vm_j.cpu.x
        && vm_i.cpu.nzcv == vm_j.cpu.nzcv
        && vm_i.cpu.pc == vm_j.cpu.pc
        && vm_i.bus.read_u64(0x8004_0000)? == vm_j.bus.read_u64(0x8004_0000)?;

    let jit_secs = (jit_ms / 1000.0).max(1e-9);
    let interp_secs = (interp_ms / 1000.0).max(1e-9);
    let jit_mips = (stats_j.insns_executed as f64 / 1_000_000.0) / jit_secs;
    let interp_mips = (stats_i.insns_executed as f64 / 1_000_000.0) / interp_secs;
    let speedup = interp_secs / jit_secs;

    Ok(BenchReport {
        iterations,
        jit_insns: stats_j.insns_executed,
        jit_ms,
        jit_mips,
        interp_insns: stats_i.insns_executed,
        interp_ms,
        interp_mips,
        speedup,
        differential_pass: diff_pass,
        jit_blocks_compiled: stats_j.blocks_compiled,
    })
}
