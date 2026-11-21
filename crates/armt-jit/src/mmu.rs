use armt_core::{ArmtError, CpuState, GuestMemory, Result};

const TLB_SIZE: usize = 256;

#[derive(Clone, Copy, Default)]
struct TlbEntry {
    tag: u64,
    pa_base: u64,
    page_mask: u64,
    ap_el0: bool,
    ap_ro: bool,
    level: u8,
    valid: bool,
}

/// ARMv8 Stage-1 MMU supporting 4 KB and 16 KB (Apple Silicon macOS) translation granules.
pub struct Mmu {
    tlb: [TlbEntry; TLB_SIZE],
}

impl Default for Mmu {
    fn default() -> Self {
        Self::new()
    }
}

impl Mmu {
    pub fn new() -> Self {
        Self {
            tlb: [TlbEntry::default(); TLB_SIZE],
        }
    }

    pub fn tlbi_all(&mut self) {
        for entry in &mut self.tlb {
            entry.valid = false;
        }
    }

    pub fn tlbi_va(&mut self, va: u64) {
        let idx4 = ((va >> 12) as usize) & (TLB_SIZE - 1);
        let idx16 = ((va >> 14) as usize) & (TLB_SIZE - 1);
        self.tlb[idx4].valid = false;
        self.tlb[idx16].valid = false;
    }

    /// Translate guest VA -> PA using `TTBR0_EL1`/`TTBR1_EL1` and `TCR_EL1`.
    pub fn translate<M: GuestMemory + ?Sized>(
        &mut self,
        cpu: &CpuState,
        mem: &mut M,
        va: u64,
        is_write: bool,
    ) -> Result<u64> {
        if !cpu.mmu_enabled() {
            return Ok(va);
        }

        let upper = ((va >> 55) & 1) != 0;
        let (ttbr, tsz, granule_shift) = if upper {
            let t1sz = ((cpu.tcr_el1 >> 16) & 0x3F) as u32;
            let tg1 = (cpu.tcr_el1 >> 30) & 3;
            let shift = match tg1 {
                1 => 14u32, // 16 KB (Apple Silicon default)
                2 => 12u32, // 4 KB
                3 => 16u32, // 64 KB
                _ => 14u32,
            };
            (cpu.ttbr1_el1, t1sz, shift)
        } else {
            let t0sz = (cpu.tcr_el1 & 0x3F) as u32;
            let tg0 = (cpu.tcr_el1 >> 14) & 3;
            let shift = match tg0 {
                0 => 12u32, // 4 KB
                2 => 14u32, // 16 KB
                1 => 16u32, // 64 KB
                _ => 12u32,
            };
            (cpu.ttbr0_el1, t0sz, shift)
        };

        let page_size = 1u64 << granule_shift;
        let page_mask = page_size - 1;
        let va_page = va & !page_mask;
        let tlb_idx = ((va >> granule_shift) as usize) & (TLB_SIZE - 1);
        let tag = va_page | if upper { 1 } else { 0 };

        let cached = self.tlb[tlb_idx];
        if cached.valid && cached.tag == tag && cached.page_mask == page_mask {
            if cpu.el == 0 && !cached.ap_el0 {
                return Err(ArmtError::MmuFault {
                    va,
                    iss: (if is_write { 0x4C } else { 0x0C }) + (cached.level as u32),
                    reason: "Permission fault (EL0 access denied)".into(),
                });
            }
            if is_write && cached.ap_ro {
                return Err(ArmtError::MmuFault {
                    va,
                    iss: 0x4C + (cached.level as u32),
                    reason: "Permission fault (write to read-only page)".into(),
                });
            }
            return Ok(cached.pa_base | (va & page_mask));
        }

        let stride = granule_shift - 3; // 9 bits for 4KB, 11 bits for 16KB
        let va_bits = 64u32.saturating_sub(tsz);
        if va_bits < granule_shift || ttbr == 0 {
            return Err(ArmtError::MmuFault {
                va,
                iss: if is_write { 0x44 } else { 0x04 },
                reason: "Invalid TCR or TTBR=0 during MMU walk".into(),
            });
        }

        // Compute starting lookup level from VA width and granule stride
        let resolved_bits = va_bits - granule_shift;
        let num_levels = resolved_bits.div_ceil(stride).clamp(1, 4);
        let start_level = 4 - num_levels;

        let index_mask = (1u64 << stride) - 1;
        let start_lsb = granule_shift + (3 - start_level) * stride;
        let start_top_bits = va_bits.saturating_sub(start_lsb).min(stride);
        let root_table_bytes = (1u64 << start_top_bits) * 8;
        let root_align_mask = !(root_table_bytes.max(64) - 1);
        let mut table_addr = ttbr & 0x0000_FFFF_FFFF_FFC0 & root_align_mask;
        let oa_mask = 0x0000_FFFF_FFFF_FFFFu64 & !page_mask;
        let mut table_ap_ro = false;
        let mut table_ap_deny_el0 = false;

        for level in start_level..=3 {
            let lsb = granule_shift + (3 - level) * stride;
            let level_mask = if level == start_level {
                let top_bits = va_bits.saturating_sub(lsb).min(stride);
                if top_bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << top_bits) - 1
                }
            } else {
                index_mask
            };
            let index = (va >> lsb) & level_mask;
            let desc_addr = table_addr + index * 8;
            let desc = mem.read_u64(desc_addr).map_err(|_| ArmtError::MmuFault {
                va,
                iss: (if is_write { 0x44 } else { 0x04 }) + (level as u32),
                reason: format!("Page table read fault at PA=0x{desc_addr:x}"),
            })?;

            let kind = desc & 0x3;
            if kind == 0 || (kind == 2) {
                return Err(ArmtError::MmuFault {
                    va,
                    iss: (if is_write { 0x44 } else { 0x04 }) + (level as u32),
                    reason: format!("Translation fault at level {level} (desc=0x{desc:x})"),
                });
            }

            // Block descriptor (valid at level < 3 when kind == 1) or L3 Page descriptor (kind == 3)
            if (level < 3 && kind == 1) || (level == 3 && kind == 3) {
                let ap_ro = table_ap_ro || ((desc >> 7) & 1) != 0;
                let ap_el0 = !table_ap_deny_el0 && ((desc >> 6) & 1) != 0;
                let block_mask = (1u64 << lsb) - 1;
                let oa_base = desc & oa_mask & !block_mask;
                let pa = oa_base | (va & block_mask);
                let pa_page = pa & !page_mask;
                self.tlb[tlb_idx] = TlbEntry {
                    tag,
                    pa_base: pa_page,
                    page_mask,
                    ap_el0,
                    ap_ro,
                    level: level as u8,
                    valid: true,
                };
                if cpu.el == 0 && !ap_el0 {
                    return Err(ArmtError::MmuFault {
                        va,
                        iss: (if is_write { 0x4C } else { 0x0C }) + (level as u32),
                        reason: "Permission fault (EL0 access denied)".into(),
                    });
                }
                if is_write && ap_ro {
                    return Err(ArmtError::MmuFault {
                        va,
                        iss: 0x4C + (level as u32),
                        reason: "Permission fault (write to read-only page)".into(),
                    });
                }
                return Ok(pa);
            }

            // Table descriptor -> propagate APTable restrictions & advance
            if ((desc >> 62) & 1) != 0 {
                table_ap_ro = true;
            }
            if ((desc >> 61) & 1) != 0 {
                table_ap_deny_el0 = true;
            }
            table_addr = desc & oa_mask;
        }

        Err(ArmtError::MmuFault {
            va,
            iss: 0x04,
            reason: "MMU walk exhausted levels".into(),
        })
    }

    /// Deliver a synchronous exception or IRQ into EL1 via `VBAR_EL1`.
    pub fn take_exception(cpu: &mut CpuState, ec: u8, iss: u32, far: u64, is_irq: bool) {
        let vector_offset = if cpu.el == 1 {
            let base = if cpu.sp_sel == 0 { 0x000u64 } else { 0x200u64 };
            if is_irq {
                base + 0x080
            } else {
                base
            }
        } else if is_irq {
            0x480
        } else {
            0x400
        };

        Self::enter_el1_vector(cpu, ec, iss, far, vector_offset, is_irq);
    }

    /// Deliver an AArch64 `SVC #imm` system call into EL1 (`EC=0x15`, `ELR_EL1=PC+4`).
    pub fn take_svc(cpu: &mut CpuState, imm: u16) {
        let ret_pc = cpu.pc.wrapping_add(4);
        Self::take_exception(cpu, 0x15, imm as u32, 0, false);
        cpu.elr_el1 = ret_pc;
    }

    /// Deliver a GIC Group-0 FIQ into EL1 via `VBAR_EL1` (`+0x100` / `+0x300` / `+0x500`).
    pub fn take_fiq(cpu: &mut CpuState) {
        let vector_offset = if cpu.el == 1 {
            let base = if cpu.sp_sel == 0 { 0x000u64 } else { 0x200u64 };
            base + 0x100
        } else {
            0x500
        };
        Self::enter_el1_vector(cpu, 0, 0, 0, vector_offset, true);
    }

    fn enter_el1_vector(
        cpu: &mut CpuState,
        ec: u8,
        iss: u32,
        far: u64,
        vector_offset: u64,
        is_async: bool,
    ) {
        cpu.elr_el1 = cpu.pc;
        let daif = (cpu.daif as u64) << 6;
        let el_mode = ((cpu.el as u64) << 2) | (cpu.sp_sel as u64);
        cpu.spsr_el1 = cpu.nzcv | daif | el_mode | ((cpu.pan as u64) << 22);
        let il = if is_async { 0 } else { 1u64 << 25 };
        cpu.esr_el1 = ((ec as u64) << 26) | il | (iss as u64);
        cpu.far_el1 = far;

        cpu.el = 1;
        cpu.set_sp_sel(1);
        cpu.daif = 0x0F;
        if (cpu.sctlr_el1 & (1 << 23)) == 0 {
            cpu.pan = 1;
        }
        cpu.pc = cpu.vbar_el1.wrapping_add(vector_offset);
    }

    /// Return from an EL1 exception via `ERET`, restoring PSTATE from `SPSR_EL1` and `PC` from `ELR_EL1`.
    pub fn eret(cpu: &mut CpuState) {
        let spsr = cpu.spsr_el1;
        cpu.nzcv = spsr & 0xF000_0000;
        cpu.daif = ((spsr >> 6) & 0xF) as u8;
        cpu.pan = ((spsr >> 22) & 1) as u8;
        cpu.el = ((spsr >> 2) & 0x3) as u8;
        cpu.set_sp_sel((spsr & 0x1) as u8);
        cpu.pc = cpu.elr_el1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FlatRam(Vec<u8>);
    impl GuestMemory for FlatRam {
        fn read_u8(&mut self, a: u64) -> Result<u8> {
            Ok(self.0[a as usize])
        }
        fn read_u16(&mut self, a: u64) -> Result<u16> {
            Ok(u16::from_le_bytes(self.0[a as usize..a as usize + 2].try_into().unwrap()))
        }
        fn read_u32(&mut self, a: u64) -> Result<u32> {
            Ok(u32::from_le_bytes(self.0[a as usize..a as usize + 4].try_into().unwrap()))
        }
        fn read_u64(&mut self, a: u64) -> Result<u64> {
            Ok(u64::from_le_bytes(self.0[a as usize..a as usize + 8].try_into().unwrap()))
        }
        fn write_u8(&mut self, a: u64, v: u8) -> Result<()> {
            self.0[a as usize] = v;
            Ok(())
        }
        fn write_u16(&mut self, a: u64, v: u16) -> Result<()> {
            self.0[a as usize..a as usize + 2].copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
        fn write_u32(&mut self, a: u64, v: u32) -> Result<()> {
            self.0[a as usize..a as usize + 4].copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
        fn write_u64(&mut self, a: u64, v: u64) -> Result<()> {
            self.0[a as usize..a as usize + 8].copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
    }

    #[test]
    fn test_mmu_16kb_granule_translation() {
        let mut ram = FlatRam(vec![0u8; 256 * 1024]);
        // Apple 16KB granule on TTBR1 (T1SZ = 16 -> 48-bit VA, 3 levels starting at L1?
        // With 16KB granule: L3=11 bits (14..25), L2=11 bits (25..36), L1=11 bits (36..47), L0=1 bit (47).
        // Let's use T1SZ = 28 -> 36-bit VA -> starts at L2! L2 table at 0x10000, L3 table at 0x14000.
        let mut cpu = CpuState::new();
        cpu.sctlr_el1 |= 1;
        // T1SZ = 28, TG1 = 1 (16KB) -> (1 << 30) | (28 << 16)
        cpu.tcr_el1 = (1u64 << 30) | (28u64 << 16);
        cpu.ttbr1_el1 = 0x10000;

        let va = 0xFFFF_FFF0_0000_4123u64;
        let l2_idx = (va >> 25) & 0x7FF;
        let l3_idx = (va >> 14) & 0x7FF;
        // L2 entry -> L3 table at 0x14000 (type 3)
        ram.write_u64(0x10000 + l2_idx * 8, 0x14000 | 0x3).unwrap();
        // L3 entry -> PA 0x20000 (type 3 + AF bit 10)
        ram.write_u64(0x14000 + l3_idx * 8, 0x20000 | 0x403).unwrap();

        let mut mmu = Mmu::new();
        let pa = mmu.translate(&cpu, &mut ram, va, false).unwrap();
        assert_eq!(pa, 0x20123);
    }

    #[test]
    fn test_mmu_16kb_48bit_kernel_va() {
        let mut ram = FlatRam(vec![0u8; 512 * 1024]);
        let mut cpu = CpuState::new();
        cpu.sctlr_el1 |= 1;
        // T1SZ = 16 (48-bit VA), TG1 = 1 (16KB): L0 has 1 bit [47], L1 [46:36], L2 [35:25], L3 [24:14]
        cpu.tcr_el1 = (1u64 << 30) | (16u64 << 16);
        cpu.ttbr1_el1 = 0x10000;

        let va = 0xFFFF_FE00_0700_4123u64;
        let l0_idx = (va >> 47) & 0x1;
        let l1_idx = (va >> 36) & 0x7FF;
        let l2_idx = (va >> 25) & 0x7FF;
        let l3_idx = (va >> 14) & 0x7FF;

        ram.write_u64(0x10000 + l0_idx * 8, 0x14000 | 0x3).unwrap();
        ram.write_u64(0x14000 + l1_idx * 8, 0x18000 | 0x3).unwrap();
        ram.write_u64(0x18000 + l2_idx * 8, 0x1C000 | 0x3).unwrap();
        ram.write_u64(0x1C000 + l3_idx * 8, 0x30000 | 0x403).unwrap();

        let mut mmu = Mmu::new();
        let pa = mmu.translate(&cpu, &mut ram, va, false).unwrap();
        assert_eq!(pa, 0x30123);
    }

    #[test]
    fn test_el0_svc_eret_and_ap_permissions() {
        let mut ram = FlatRam(vec![0u8; 256 * 1024]);
        let mut cpu = CpuState::new();
        cpu.sctlr_el1 |= 1;
        cpu.tcr_el1 = 28; // T0SZ=28 (36-bit VA, 4KB granule, starts at L1)
        cpu.ttbr0_el1 = 0x10000;
        cpu.vbar_el1 = 0xFFFF_FE00_0000_0000;
        cpu.el = 0;
        cpu.set_sp_sel(0);
        cpu.sp = 0x7000_1000;
        cpu.pc = 0x1_0000_4000;

        let va_ro_user = 0x0000_0000_0000_4123u64;
        let l1_idx = (va_ro_user >> 30) & 0x1FF;
        let l2_idx = (va_ro_user >> 21) & 0x1FF;
        let l3_idx = (va_ro_user >> 12) & 0x1FF;
        ram.write_u64(0x10000 + l1_idx * 8, 0x11000 | 0x3).unwrap();
        ram.write_u64(0x11000 + l2_idx * 8, 0x12000 | 0x3).unwrap();
        // L3: AP[2:1] = 0b11 (RO + EL0 accessible: bits 7 and 6 set)
        ram.write_u64(0x12000 + l3_idx * 8, 0x20000 | 0x4C3).unwrap();

        let mut mmu = Mmu::new();
        assert_eq!(mmu.translate(&cpu, &mut ram, va_ro_user, false).unwrap(), 0x20123);
        assert!(mmu.translate(&cpu, &mut ram, va_ro_user, true).is_err());

        Mmu::take_svc(&mut cpu, 0x80);
        assert_eq!(cpu.el, 1);
        assert_eq!(cpu.pc, 0xFFFF_FE00_0000_0400);
        assert_eq!(cpu.elr_el1, 0x1_0000_4004);
        assert_eq!((cpu.esr_el1 >> 26) & 0x3F, 0x15);
        assert_eq!(cpu.esr_el1 & 0xFFFF, 0x80);

        Mmu::eret(&mut cpu);
        assert_eq!(cpu.el, 0);
        assert_eq!(cpu.pc, 0x1_0000_4004);
        assert_eq!(cpu.sp, 0x7000_1000);
    }
}
