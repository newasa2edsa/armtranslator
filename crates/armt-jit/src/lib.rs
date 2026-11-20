pub mod abi_bridge;
pub mod dynarec;
pub mod interp;
pub mod mmu;
pub mod wx_mem;
pub mod x64;

pub use abi_bridge::{invoke_arm64_fn, HostServiceTable};
pub use dynarec::{JitEngine, JitStats};
pub use interp::{Interpreter, StepOutcome};
pub use mmu::Mmu;

#[cfg(test)]
mod tests {
    use super::*;
    use armt_core::isa::{Condition, Reg, ShiftType, SysReg, Width};
    use armt_core::{CpuState, GuestMemory, HostCallArgs, HostCallResult, Result, SysRegBus};
    use armt_decoder::asm;

    struct TestMem {
        base: u64,
        ram: Vec<u8>,
        uart_out: Vec<u8>,
    }

    impl TestMem {
        fn new(base: u64, size: usize) -> Self {
            Self {
                base,
                ram: vec![0u8; size],
                uart_out: Vec::new(),
            }
        }
        fn load_words(&mut self, addr: u64, words: &[u32]) {
            let bytes = asm::to_bytes(words);
            self.write_bytes(addr, &bytes).unwrap();
        }
    }

    impl GuestMemory for TestMem {
        fn read_u8(&mut self, a: u64) -> Result<u8> {
            Ok(self.ram[(a - self.base) as usize])
        }
        fn read_u16(&mut self, a: u64) -> Result<u16> {
            let i = (a - self.base) as usize;
            Ok(u16::from_le_bytes(self.ram[i..i + 2].try_into().unwrap()))
        }
        fn read_u32(&mut self, a: u64) -> Result<u32> {
            let i = (a - self.base) as usize;
            Ok(u32::from_le_bytes(self.ram[i..i + 4].try_into().unwrap()))
        }
        fn read_u64(&mut self, a: u64) -> Result<u64> {
            let i = (a - self.base) as usize;
            Ok(u64::from_le_bytes(self.ram[i..i + 8].try_into().unwrap()))
        }
        fn write_u8(&mut self, a: u64, v: u8) -> Result<()> {
            if a == 0x2001_0000 {
                self.uart_out.push(v);
                return Ok(());
            }
            self.ram[(a - self.base) as usize] = v;
            Ok(())
        }
        fn write_u16(&mut self, a: u64, v: u16) -> Result<()> {
            let i = (a - self.base) as usize;
            self.ram[i..i + 2].copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
        fn write_u32(&mut self, a: u64, v: u32) -> Result<()> {
            if a == 0x2001_0000 {
                self.uart_out.push(v as u8);
                return Ok(());
            }
            let i = (a - self.base) as usize;
            self.ram[i..i + 4].copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
        fn write_u64(&mut self, a: u64, v: u64) -> Result<()> {
            let i = (a - self.base) as usize;
            self.ram[i..i + 8].copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
        fn ram_slice_mut(&mut self) -> Option<(u64, *mut u8, usize)> {
            Some((self.base, self.ram.as_mut_ptr(), self.ram.len()))
        }
    }

    impl SysRegBus for TestMem {
        fn read_sysreg(&mut self, reg: SysReg) -> Option<u64> {
            if reg == SysReg::CNTFRQ_EL0 {
                Some(24_000_000)
            } else {
                None
            }
        }
        fn write_sysreg(&mut self, _reg: SysReg, _val: u64) -> bool {
            false
        }
    }

    fn run_diff_check(words: &[u32], init: impl Fn(&mut CpuState)) -> (CpuState, CpuState) {
        let base = 0x8000_0000u64;
        let mut mem_i = TestMem::new(base, 1024 * 1024);
        let mut mem_j = TestMem::new(base, 1024 * 1024);
        mem_i.load_words(base, words);
        mem_j.load_words(base, words);

        let mut cpu_i = CpuState::default_vma2_el1(base, base + 0x8_0000, 0);
        let mut cpu_j = CpuState::default_vma2_el1(base, base + 0x8_0000, 0);
        init(&mut cpu_i);
        init(&mut cpu_j);

        let mut mmu_i = Mmu::new();
        for _ in 0..100_000 {
            match Interpreter::step(&mut cpu_i, &mut mem_i, &mut mmu_i).unwrap() {
                StepOutcome::Hlt(_) => break,
                StepOutcome::Continue | StepOutcome::Branch(_) => continue,
                other => panic!("unexpected interp outcome: {other:?}"),
            }
        }

        let mut mmu_j = Mmu::new();
        let mut jit = JitEngine::new();
        let out = jit.run(&mut cpu_j, &mut mem_j, &mut mmu_j, 100_000).unwrap();
        assert!(matches!(out, StepOutcome::Hlt(_)));

        assert_eq!(cpu_i.x, cpu_j.x, "GPR mismatch");
        assert_eq!(cpu_i.sp, cpu_j.sp, "SP mismatch");
        assert_eq!(cpu_i.pc, cpu_j.pc, "PC mismatch");
        assert_eq!(cpu_i.nzcv, cpu_j.nzcv, "NZCV mismatch");
        assert_eq!(cpu_i.insn_count, cpu_j.insn_count, "insn_count mismatch");
        assert_eq!(mem_i.ram, mem_j.ram, "RAM mismatch");
        assert_eq!(mem_i.uart_out, mem_j.uart_out, "MMIO UART mismatch");
        (cpu_i, cpu_j)
    }

    #[test]
    fn test_diff_alu_and_carry_polarity() {
        // Test SUBS / CMP carry & overflow edge cases:
        // Case 1: 5 - 3 -> C=1 (no borrow), Z=0, N=0, V=0 -> B.CS taken
        // Case 2: 3 - 5 -> C=0 (borrow), Z=0, N=1, V=0 -> B.CC taken
        // Case 3: i64::MIN - 1 -> V=1 (signed overflow) -> B.VS taken
        let mut prog = vec![
            asm::movz(Width::X64, Reg(0), 5, 0),
            asm::movz(Width::X64, Reg(1), 3, 0),
            asm::sub_reg(Width::X64, Reg(2), Reg(0), Reg(1), ShiftType::Lsl, 0, true), // 5 - 3
            asm::b_cond(Condition::Cs, 8),
            asm::movz(Width::X64, Reg(10), 0xDEAD, 0), // skipped if C==1
            asm::movz(Width::X64, Reg(10), 1, 0),
            asm::sub_reg(Width::X64, Reg(3), Reg(1), Reg(0), ShiftType::Lsl, 0, true), // 3 - 5
            asm::b_cond(Condition::Cc, 8),
            asm::movz(Width::X64, Reg(11), 0xDEAD, 0), // skipped if C==0
            asm::movz(Width::X64, Reg(11), 2, 0),
            asm::movz(Width::X64, Reg(4), 0x8000, 48), // i64::MIN
            asm::sub_imm(Width::X64, Reg(5), Reg(4), 1, true), // overflow -> V=1
            asm::b_cond(Condition::Vs, 8),
            asm::movz(Width::X64, Reg(12), 0xDEAD, 0),
            asm::movz(Width::X64, Reg(12), 3, 0),
            asm::hlt(0),
        ];
        let (ci, _) = run_diff_check(&prog, |_| {});
        assert_eq!(ci.x[10], 1);
        assert_eq!(ci.x[11], 2);
        assert_eq!(ci.x[12], 3);
        prog.clear();
    }

    #[test]
    fn test_diff_fibonacci_and_memory_mmio() {
        // Compute fib(20) = 6765 in loop + store/load stack + write MMIO UART
        let words = vec![
            asm::movz(Width::X64, Reg(0), 20, 0),  // N = 20
            asm::movz(Width::X64, Reg(1), 0, 0),   // a = 0
            asm::movz(Width::X64, Reg(2), 1, 0),   // b = 1
            asm::movz(Width::X64, Reg(3), 0, 0),   // i = 0
            // loop (pc + 16):
            asm::sub_reg(Width::X64, Reg::XZR, Reg(3), Reg(0), ShiftType::Lsl, 0, true), // cmp i, N
            asm::b_cond(Condition::Ge, 24),        // -> done
            asm::add_reg(Width::X64, Reg(4), Reg(1), Reg(2), ShiftType::Lsl, 0, false),
            asm::mov_reg(Width::X64, Reg(1), Reg(2)),
            asm::mov_reg(Width::X64, Reg(2), Reg(4)),
            asm::add_imm(Width::X64, Reg(3), Reg(3), 1, false),
            asm::b(-24),
            // done: store to [sp + 16], load back into x5, emit 'O','K' to UART 0x2001_0000
            asm::str_imm(Width::X64, Reg(1), Reg::SP, 16),
            asm::ldr_imm(Width::X64, Reg(5), Reg::SP, 16),
            asm::movz(Width::X64, Reg(6), 0x0000, 0),
            asm::movk(Width::X64, Reg(6), 0x2001, 16),
            asm::movz(Width::X64, Reg(7), b'O' as u16, 0),
            asm::str_imm(Width::W32, Reg(7), Reg(6), 0),
            asm::movz(Width::X64, Reg(7), b'K' as u16, 0),
            asm::str_imm(Width::W32, Reg(7), Reg(6), 0),
            asm::hlt(0),
        ];
        let (ci, _) = run_diff_check(&words, |_| {});
        assert_eq!(ci.x[5], 6765);
    }

    #[test]
    fn test_diff_bitfield_pair_mul_shift_fp_and_cnd() {
        let words = vec![
            // Bitwise + Shifts + Mul + Sub + CSEL/CSINC/STP/LDP/LDRB/STRB + FP math
            asm::movz(Width::X64, Reg(0), 0x1234, 0),
            asm::movk(Width::X64, Reg(0), 0x0001, 16),
            asm::movz(Width::X64, Reg(1), 4, 0),
            asm::lslv(Width::X64, Reg(2), Reg(0), Reg(1)),
            asm::lsrv(Width::X64, Reg(3), Reg(2), Reg(1)),
            asm::mul(Width::X64, Reg(4), Reg(3), Reg(1)),
            // Save pair onto stack and restore into different regs
            asm::stp_pre(Width::X64, Reg(2), Reg(4), Reg::SP, -16),
            asm::ldp_post(Width::X64, Reg(8), Reg(9), Reg::SP, 16),
            // Store byte / load byte
            asm::movz(Width::X64, Reg(10), 0xAB, 0),
            asm::strb_imm(Reg(10), Reg::SP, 8),
            asm::ldrb_imm(Reg(11), Reg::SP, 8),
            // MRS CNTFRQ_EL0
            asm::mrs(Reg(12), SysReg::CNTFRQ_EL0),
            // SCVTF D0, X4; FMOV D1, #1.0 (0x70); FADD D2, D0, D1; FCVTZS X13, D2
            0x9E62_0080, // scvtf d0, x4
            0x1E6E_1001, // fmov d1, #1.0 (wait: in 64-bit fmov imm8: 0x1E60_1000 | (0x70<<13) | 1 = 0x1E6E_1001)
            0x1E61_2802, // fadd d2, d0, d1
            0x9E78_004D, // fcvtzs x13, d2
            asm::hlt(0),
        ];
        let (ci, _) = run_diff_check(&words, |_| {});
        assert_eq!(ci.x[8], ci.x[2]);
        assert_eq!(ci.x[9], ci.x[4]);
        assert_eq!(ci.x[11], 0xAB);
        assert_eq!(ci.x[12], 24_000_000);
        assert_eq!(ci.x[13], ci.x[4] + 1);
    }

    #[test]
    fn test_abi_bridge_host_to_arm64_and_svc_callback() {
        let base = 0x8000_0000u64;
        let mut mem = TestMem::new(base, 1024 * 1024);
        // Guest function: X0 = arg0, X1 = arg1; computes arg0 * 10 + arg1, calls SVC #0x80 (num 42)
        // which host multiplies by 2 and returns in X0, then RET!
        let fn_words = vec![
            asm::movz(Width::X64, Reg(2), 10, 0),
            asm::mul(Width::X64, Reg(0), Reg(0), Reg(2)),
            asm::add_reg(Width::X64, Reg(0), Reg(0), Reg(1), ShiftType::Lsl, 0, false),
            asm::movz(Width::X64, Reg(16), 42, 0),
            asm::svc(0x80),
            asm::ret(Reg::LR),
        ];
        mem.load_words(base, &fn_words);

        let mut cpu = CpuState::default_vma2_el1(base, base + 0x8_0000, 0);
        let mut mmu = Mmu::new();
        let mut jit = JitEngine::new();
        let mut bridge = HostServiceTable::new();
        bridge.register(42, |args, _| HostCallResult {
            x0: args[0] * 2,
            x1: 99,
        });

        let res = invoke_arm64_fn(
            &mut jit,
            &mut cpu,
            &mut mem,
            &mut mmu,
            &mut bridge,
            base,
            HostCallArgs::from_slice(&[7, 5]),
            10_000,
        )
        .unwrap();
        // (7 * 10 + 5) * 2 = 150
        assert_eq!(res.x0, 150);
        assert_eq!(res.x1, 99);
    }
}
