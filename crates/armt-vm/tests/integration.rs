use armt_core::isa::{Reg, ShiftType, Width};
use armt_core::{GuestMemory, HostCallArgs, HostCallResult};
use armt_decoder::asm;
use armt_jit::{invoke_arm64_fn, HostServiceTable, JitEngine, Mmu};
use armt_vm::{load_demo_kernel, run_benchmark, ArmVm, ExecMode, VmConfig};

#[test]
fn test_demo_kernel_boot_jit_and_interp() {
    let mut vm_jit = ArmVm::new(VmConfig {
        mode: ExecMode::Jit,
        ram_mb: 64,
        echo_uart: false,
        max_steps: 10_000_000,
        trace: false,
    });
    let img = load_demo_kernel(&mut vm_jit.bus).unwrap();
    vm_jit.apply_image(&img);
    let stats_jit = vm_jit.run().unwrap();

    assert_eq!(stats_jit.exit_reason, "HLT #0");
    assert_eq!(stats_jit.x0, 6765);
    assert!(stats_jit.fb_non_zero_pixels > 500_000);
    assert!(vm_jit.uart_output().contains("Fibonacci(20) == 6765"));
    assert!(vm_jit.uart_output().contains("24000000 Hz: PASS"));

    let mut vm_int = ArmVm::new(VmConfig {
        mode: ExecMode::Interpreter,
        ram_mb: 64,
        echo_uart: false,
        max_steps: 10_000_000,
        trace: false,
    });
    let img2 = load_demo_kernel(&mut vm_int.bus).unwrap();
    vm_int.apply_image(&img2);
    let stats_int = vm_int.run().unwrap();

    assert_eq!(stats_jit.insns_executed, stats_int.insns_executed);
    assert_eq!(vm_jit.uart_output(), vm_int.uart_output());
    assert_eq!(stats_jit.fb_non_zero_pixels, stats_int.fb_non_zero_pixels);
}

#[test]
fn test_benchmark_differential_equivalence() {
    let rep = run_benchmark(20_000).unwrap();
    assert!(rep.differential_pass, "JIT and Interpreter diverged!");
    assert!(rep.jit_insns > 100_000);
    assert!(rep.jit_blocks_compiled > 0);
}

#[test]
fn test_vm_abi_trampoline() {
    let mut vm = ArmVm::new(VmConfig {
        mode: ExecMode::Jit,
        ram_mb: 16,
        echo_uart: false,
        max_steps: 10_000,
        trace: false,
    });
    let fn_addr = vm.bus.memmap.ram_base + 0x2000;
    let code = asm::to_bytes(&[
        asm::add_reg(Width::X64, Reg(0), Reg(0), Reg(1), ShiftType::Lsl, 0, false),
        asm::movz(Width::X64, Reg(16), 7, 0),
        asm::svc(0x80),
        asm::ret(Reg::LR),
    ]);
    vm.bus.write_bytes(fn_addr, &code).unwrap();

    let mut jit = JitEngine::new();
    let mut mmu = Mmu::new();
    let mut table = HostServiceTable::new();
    table.register(7, |args, _| HostCallResult {
        x0: args[0] + 100,
        x1: 1,
    });

    let res = invoke_arm64_fn(
        &mut jit,
        &mut vm.cpu,
        &mut vm.bus,
        &mut mmu,
        &mut table,
        fn_addr,
        HostCallArgs::from_slice(&[20, 22]),
        5_000,
    )
    .unwrap();
    assert_eq!(res.x0, 142);
    assert_eq!(res.x1, 1);
}
