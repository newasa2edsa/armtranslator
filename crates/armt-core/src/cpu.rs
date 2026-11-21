use crate::isa::{Condition, Reg};

/// Architectural state of an AArch64 vCPU.
///
/// Field layout is `#[repr(C)]` so the x86-64 Dynarec JIT can reference fields
/// directly via `std::mem::offset_of!`.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuState {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    /// Bits 31..28 hold N, Z, C, V respectively.
    pub nzcv: u64,
    /// 128-bit NEON / FP registers V0..V31 (little-endian `[low64, high64]`).
    pub v: [[u64; 2]; 32],
    pub fpcr: u32,
    pub fpsr: u32,
    pub el: u8,
    pub sp_sel: u8,
    pub daif: u8,
    pub pan: u8,
    pub sp_el0: u64,
    pub sp_el1: u64,
    pub sctlr_el1: u64,
    pub tcr_el1: u64,
    pub ttbr0_el1: u64,
    pub ttbr1_el1: u64,
    pub mair_el1: u64,
    pub vbar_el1: u64,
    pub elr_el1: u64,
    pub spsr_el1: u64,
    pub esr_el1: u64,
    pub far_el1: u64,
    pub tpidr_el0: u64,
    pub tpidr_el1: u64,
    pub tpidrro_el0: u64,
    pub cpacr_el1: u64,
    pub exclusive_addr: u64,
    pub exclusive_valid: u8,
    pub insn_count: u64,
    pub par_el1: u64,
    pub csselr_el1: u64,
    pub contextidr_el1: u64,
    pub amair_el1: u64,
    pub mdscr_el1: u64,
    pub pac_keys: [u64; 8],
    pub pac_el0_div: u64,
}

impl Default for CpuState {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuState {
    pub fn new() -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            nzcv: 0,
            v: [[0; 2]; 32],
            fpcr: 0,
            fpsr: 0,
            el: 1,
            sp_sel: 1,
            daif: 0x0F,
            pan: 0,
            sp_el0: 0,
            sp_el1: 0,
            sctlr_el1: 0x00C5_0838,
            tcr_el1: 0,
            ttbr0_el1: 0,
            ttbr1_el1: 0,
            mair_el1: 0,
            vbar_el1: 0,
            elr_el1: 0,
            spsr_el1: 0,
            esr_el1: 0,
            far_el1: 0,
            tpidr_el0: 0,
            tpidr_el1: 0,
            tpidrro_el0: 0,
            cpacr_el1: 0x0030_0000, // Enable FP/SIMD at EL1
            exclusive_addr: 0,
            exclusive_valid: 0,
            insn_count: 0,
            par_el1: 0,
            csselr_el1: 0,
            contextidr_el1: 0,
            amair_el1: 0,
            mdscr_el1: 0,
            pac_keys: [0; 8],
            pac_el0_div: 0,
        }
    }

    /// Initialize default `VirtualMac2,1` EL1h state (FP enabled, MMU off at handoff).
    pub fn default_vma2_el1(pc: u64, sp: u64, boot_args_phys: u64) -> Self {
        let mut cpu = Self::new();
        cpu.pc = pc;
        cpu.sp = sp;
        cpu.sp_el1 = sp;
        cpu.x[0] = boot_args_phys;
        cpu.el = 1;
        cpu.sp_sel = 1;
        cpu.daif = 0x0F;
        cpu.cpacr_el1 = 0x0030_0000;
        cpu.sctlr_el1 = 0x00C5_0838; // MMU initially off (kernel sets TTBR/SCTLR.M)
        cpu
    }

    #[inline]
    pub const fn n(&self) -> bool {
        (self.nzcv & (1 << 31)) != 0
    }

    #[inline]
    pub const fn z(&self) -> bool {
        (self.nzcv & (1 << 30)) != 0
    }

    #[inline]
    pub const fn c(&self) -> bool {
        (self.nzcv & (1 << 29)) != 0
    }

    #[inline]
    pub const fn v(&self) -> bool {
        (self.nzcv & (1 << 28)) != 0
    }

    #[inline]
    pub fn set_nzcv(&mut self, n: bool, z: bool, c: bool, v: bool) {
        self.nzcv = ((n as u64) << 31)
            | ((z as u64) << 30)
            | ((c as u64) << 29)
            | ((v as u64) << 28);
    }

    #[inline]
    pub const fn cond_holds(&self, cond: Condition) -> bool {
        let n = self.n();
        let z = self.z();
        let c = self.c();
        let v = self.v();
        match cond {
            Condition::Eq => z,
            Condition::Ne => !z,
            Condition::Cs => c,
            Condition::Cc => !c,
            Condition::Mi => n,
            Condition::Pl => !n,
            Condition::Vs => v,
            Condition::Vc => !v,
            Condition::Hi => c && !z,
            Condition::Ls => !c || z,
            Condition::Ge => n == v,
            Condition::Lt => n != v,
            Condition::Gt => !z && (n == v),
            Condition::Le => z || (n != v),
            Condition::Al | Condition::Nv => true,
        }
    }

    /// Read X0..X30 or XZR (31 -> 0).
    #[inline]
    pub const fn read_x(&self, r: Reg) -> u64 {
        let idx = r.index();
        if idx == 31 {
            0
        } else {
            self.x[idx]
        }
    }

    /// Write X0..X30 or discard if XZR (31).
    #[inline]
    pub fn write_x(&mut self, r: Reg, val: u64) {
        let idx = r.index();
        if idx < 31 {
            self.x[idx] = val;
        }
    }

    #[inline]
    pub const fn read_w(&self, r: Reg) -> u32 {
        self.read_x(r) as u32
    }

    #[inline]
    pub fn write_w(&mut self, r: Reg, val: u32) {
        self.write_x(r, val as u64);
    }

    /// Read X0..X30 or active SP (31 -> `self.sp`).
    #[inline]
    pub const fn read_sp_or_x(&self, r: Reg) -> u64 {
        let idx = r.index();
        if idx == 31 {
            self.sp
        } else {
            self.x[idx]
        }
    }

    /// Write X0..X30 or active SP (31 -> `self.sp`).
    #[inline]
    pub fn write_sp_or_x(&mut self, r: Reg, val: u64) {
        let idx = r.index();
        if idx == 31 {
            self.sp = val;
        } else {
            self.x[idx] = val;
        }
    }

    /// Switch stack pointer bank when SPSEL changes.
    pub fn set_sp_sel(&mut self, sel: u8) {
        let new_sel = sel & 1;
        if self.sp_sel != new_sel {
            if self.sp_sel == 0 {
                self.sp_el0 = self.sp;
            } else {
                self.sp_el1 = self.sp;
            }
            self.sp_sel = new_sel;
            self.sp = if new_sel == 0 {
                self.sp_el0
            } else {
                self.sp_el1
            };
        }
    }

    #[inline]
    pub fn mmu_enabled(&self) -> bool {
        (self.sctlr_el1 & 1) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nzcv_conditions() {
        let mut cpu = CpuState::new();
        cpu.set_nzcv(false, true, true, false); // Z=1, C=1
        assert!(cpu.cond_holds(Condition::Eq));
        assert!(!cpu.cond_holds(Condition::Ne));
        assert!(cpu.cond_holds(Condition::Cs));
        assert!(!cpu.cond_holds(Condition::Hi)); // C && !Z
        assert!(cpu.cond_holds(Condition::Ls)); // !C || Z

        cpu.set_nzcv(true, false, false, false); // N=1, V=0 -> LT
        assert!(cpu.cond_holds(Condition::Lt));
        assert!(!cpu.cond_holds(Condition::Ge));
        assert!(cpu.cond_holds(Condition::Le));
        assert!(!cpu.cond_holds(Condition::Gt));
    }

    #[test]
    fn test_register_xzr_and_sp() {
        let mut cpu = CpuState::new();
        cpu.write_x(Reg::XZR, 0xDEAD_BEEF);
        assert_eq!(cpu.read_x(Reg::XZR), 0);
        cpu.write_sp_or_x(Reg::SP, 0x9000_0000);
        assert_eq!(cpu.read_sp_or_x(Reg::SP), 0x9000_0000);
        cpu.write_w(Reg(5), 0xFFFF_FFFF);
        assert_eq!(cpu.read_x(Reg(5)), 0x0000_0000_FFFF_FFFF);
    }
}
