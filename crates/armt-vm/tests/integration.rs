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

#[test]
fn test_simd_movi_and_uminv_in_vm() {
    let mut vm = ArmVm::new(VmConfig {
        mode: ExecMode::Interpreter,
        ram_mb: 16,
        echo_uart: false,
        max_steps: 100,
        trace: false,
    });
    let entry = vm.bus.memmap.ram_base + 0x2000;
    let data = vm.bus.memmap.ram_base + 0x3000;
    vm.bus.write_bytes(data, b"Hello, VirtualMac").unwrap();
    vm.cpu.x[0] = data;
    vm.cpu.v[0] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];

    let code = asm::to_bytes(&[
        0x6F00_E400, // movi v0.2d, #0
        0x3DC0_0002, // ldr q2, [x0] -> "Hello, VirtualMa" (min byte = ' ' = 0x20)
        0x6E31_A841, // uminv b1, v2.16b
        0x0E01_3C20, // umov w0, v1.b[0]
        asm::hlt(0),
    ]);
    vm.bus.write_bytes(entry, &code).unwrap();
    vm.cpu.pc = entry;

    let stats = vm.run().unwrap();
    assert_eq!(stats.exit_reason, "HLT #0");
    assert_eq!(vm.cpu.v[0], [0, 0]);
    assert_eq!(vm.cpu.x[0], b' ' as u64);
}

#[test]
fn test_virtio_blk_file_disk_cow_overlay() {
    use armt_core::GuestMemory;

    let tmp_dir = std::env::temp_dir().join("armt_disk_test");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let img_path = tmp_dir.join("test_disk.img");
    let mut backing = vec![0u8; 4096];
    backing[0..8].copy_from_slice(b"ARMTDISK");
    std::fs::write(&img_path, &backing).unwrap();

    let mut vm = ArmVm::new(VmConfig {
        mode: ExecMode::Interpreter,
        ram_mb: 16,
        echo_uart: false,
        max_steps: 100,
        trace: false,
    });
    vm.attach_disk_image(&img_path, true).unwrap();

    let mut sec0 = [0u8; 512];
    vm.bus.virtio_blk.read_sector(0, &mut sec0).unwrap();
    assert_eq!(&sec0[0..8], b"ARMTDISK");

    // Write via CoW overlay without touching backing file on disk
    let dirty = [0x5Au8; 512];
    vm.bus.virtio_blk.write_sector(0, &dirty).unwrap();
    let mut verify = [0u8; 512];
    vm.bus.virtio_blk.read_sector(0, &mut verify).unwrap();
    assert_eq!(verify[0], 0x5A);

    let on_disk = std::fs::read(&img_path).unwrap();
    assert_eq!(&on_disk[0..8], b"ARMTDISK");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    let _ = vm.bus.read_u32(vm.bus.memmap.ram_base);
}

#[test]
fn test_jit_and_interp_irq_fiq_and_daif_transitions() {
    use armt_core::isa::SysReg;
    use armt_core::IrqController;

    const MSR_DAIFSET_1: u32 = 0xD503_40DF | (1 << 8);
    const MSR_DAIFSET_2: u32 = 0xD503_40DF | (2 << 8);
    const MSR_DAIFSET_3: u32 = 0xD503_40DF | (3 << 8);
    const MSR_DAIFCLR_1: u32 = 0xD503_40FF | (1 << 8);
    const MSR_DAIFCLR_2: u32 = 0xD503_40FF | (2 << 8);
    const ERET: u32 = 0xD69F_03E0;

    fn run_daifclr_case(mode: ExecMode) -> [u64; 8] {
        let mut vm = ArmVm::new(VmConfig {
            mode,
            ram_mb: 16,
            echo_uart: false,
            max_steps: 50_000,
            trace: false,
        });
        let vbar = vm.bus.memmap.ram_base + 0x10_0000;
        let entry = vm.bus.memmap.ram_base + 0x2000;

        // EL1h IRQ vector (+0x280): count IRQs in x10, store INTID in x12, EOI & ERET
        let irq_vec = asm::to_bytes(&[
            asm::mrs(Reg(9), SysReg::ICC_IAR1_EL1),
            asm::add_imm(Width::X64, Reg(10), Reg(10), 1, false),
            asm::mov_reg(Width::X64, Reg(12), Reg(9)),
            asm::msr(SysReg::ICC_EOIR1_EL1, Reg(9)),
            ERET,
        ]);
        vm.bus.write_bytes(vbar + 0x280, &irq_vec).unwrap();

        // EL1h FIQ vector (+0x300): ack Group-0 timer FIQ 27, disarm CNTV_CTL_EL0, count in x11, EOI & ERET
        let fiq_vec = asm::to_bytes(&[
            asm::mrs(Reg(9), SysReg::ICC_IAR0_EL1),
            asm::msr(SysReg::CNTV_CTL_EL0, Reg::XZR),
            asm::add_imm(Width::X64, Reg(11), Reg(11), 1, false),
            asm::mov_reg(Width::X64, Reg(13), Reg(9)),
            asm::msr(SysReg::ICC_EOIR0_EL1, Reg(9)),
            ERET,
        ]);
        vm.bus.write_bytes(vbar + 0x300, &fiq_vec).unwrap();

        let main_code = asm::to_bytes(&[
            // Enable GIC Group 0 & Group 1 + priority mask while masked (DAIF = 0xF)
            asm::movz(Width::X64, Reg(0), 0xFF, 0),
            asm::msr(SysReg::ICC_PMR_EL1, Reg(0)),
            asm::movz(Width::X64, Reg(0), 1, 0),
            asm::msr(SysReg::ICC_IGRPEN0_EL1, Reg(0)),
            asm::msr(SysReg::ICC_IGRPEN1_EL1, Reg(0)),
            // Snapshot counts while both FIQ and IRQ are still masked
            asm::add_reg(Width::X64, Reg(14), Reg(10), Reg(11), ShiftType::Lsl, 0, false),
            // Unmask FIQ for a single instruction before re-masking (must still fire immediately)
            MSR_DAIFCLR_1,
            MSR_DAIFSET_1,
            // Capture (x10 << 32) | x11 before unmasking IRQ: FIQ must be 1, IRQ still 0
            asm::lsl_imm(Width::X64, Reg(15), Reg(10), 32),
            asm::orr_reg(Width::X64, Reg(15), Reg(15), Reg(11)),
            // Unmask IRQ for a single instruction before re-masking (must still fire immediately)
            MSR_DAIFCLR_2,
            MSR_DAIFSET_2,
            asm::mrs(Reg(16), SysReg::DAIF),
            asm::hlt(0),
        ]);
        vm.bus.write_bytes(entry, &main_code).unwrap();

        vm.cpu.vbar_el1 = vbar;
        vm.cpu.daif = 0xF;
        vm.cpu.pc = entry;
        vm.bus.timer.cntv_cval = 0;
        vm.bus.timer.cntv_ctl = 1;
        vm.bus.gic.set_apple_timer_fiq_routing();
        vm.bus.gic.set_enabled(27, true);
        vm.bus.gic.set_enabled(33, true);
        vm.bus.tick(1);
        vm.bus.gic.raise_irq(33);

        let stats = vm.run().unwrap();
        assert_eq!(stats.exit_reason, "HLT #0");
        [
            vm.cpu.x[10],
            vm.cpu.x[11],
            vm.cpu.x[12],
            vm.cpu.x[13],
            vm.cpu.x[14],
            vm.cpu.x[15],
            vm.cpu.x[16],
            vm.cpu.daif as u64,
        ]
    }

    fn run_msr_daif_case(mode: ExecMode) -> [u64; 4] {
        let mut vm = ArmVm::new(VmConfig {
            mode,
            ram_mb: 16,
            echo_uart: false,
            max_steps: 50_000,
            trace: false,
        });
        let vbar = vm.bus.memmap.ram_base + 0x10_0000;
        let entry = vm.bus.memmap.ram_base + 0x2000;

        let irq_vec = asm::to_bytes(&[
            asm::mrs(Reg(9), SysReg::ICC_IAR1_EL1),
            asm::add_imm(Width::X64, Reg(10), Reg(10), 1, false),
            asm::mov_reg(Width::X64, Reg(12), Reg(9)),
            asm::msr(SysReg::ICC_EOIR1_EL1, Reg(9)),
            ERET,
        ]);
        vm.bus.write_bytes(vbar + 0x280, &irq_vec).unwrap();

        let main_code = asm::to_bytes(&[
            asm::movz(Width::X64, Reg(0), 0xFF, 0),
            asm::msr(SysReg::ICC_PMR_EL1, Reg(0)),
            asm::movz(Width::X64, Reg(0), 1, 0),
            asm::msr(SysReg::ICC_IGRPEN1_EL1, Reg(0)),
            // Keep IRQ masked via MSR DAIF, X0 (D=1,A=1,I=1,F=0 -> 0xE << 6 = 0x380)
            asm::movz(Width::X64, Reg(0), 0x380, 0),
            asm::msr(SysReg::DAIF, Reg(0)),
            asm::mov_reg(Width::X64, Reg(14), Reg(10)), // must still be 0
            // Clear IRQ mask via MSR DAIF, X0 (0x0) followed immediately by DAIFSet #3
            asm::msr(SysReg::DAIF, Reg(31)),
            MSR_DAIFSET_3,
            asm::mrs(Reg(15), SysReg::DAIF),
            asm::hlt(0),
        ]);
        vm.bus.write_bytes(entry, &main_code).unwrap();

        vm.cpu.vbar_el1 = vbar;
        vm.cpu.daif = 0xF;
        vm.cpu.pc = entry;
        vm.bus.gic.set_enabled(48, true);
        vm.bus.gic.raise_irq(48);

        let stats = vm.run().unwrap();
        assert_eq!(stats.exit_reason, "HLT #0");
        [vm.cpu.x[10], vm.cpu.x[12], vm.cpu.x[14], vm.cpu.x[15]]
    }

    let jit1 = run_daifclr_case(ExecMode::Jit);
    let interp1 = run_daifclr_case(ExecMode::Interpreter);
    assert_eq!(jit1, [1, 1, 33, 27, 0, 1, 0xF << 6, 0xF]);
    assert_eq!(jit1, interp1);

    let jit2 = run_msr_daif_case(ExecMode::Jit);
    let interp2 = run_msr_daif_case(ExecMode::Interpreter);
    assert_eq!(jit2, [1, 48, 0, 0x3 << 6]);
    assert_eq!(jit2, interp2);
}
