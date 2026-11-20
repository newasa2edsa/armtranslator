use armt_core::{ArmtError, CpuState, GuestMemory, Result};

const TLB_SIZE: usize = 256;

#[derive(Clone, Copy, Default)]
struct TlbEntry {
    tag: u64,
    pa_base: u64,
    page_mask: u64,
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

        let mut table_addr = ttbr & 0x0000_FFFF_FFFF_F000;
        let index_mask = (1u64 << stride) - 1;

        for level in start_level..=3 {
            let lsb = granule_shift + (3 - level) * stride;
            let index = (va >> lsb) & index_mask;
            let desc_addr = table_addr + index * 8;
            let desc = mem.read_u64(desc_addr).map_err(|_| ArmtError::MmuFault {
                va,
                iss: if is_write { 0x44 } else { 0x04 },
                reason: format!("Page table read fault at PA=0x{desc_addr:x}"),
            })?;

            let kind = desc & 0x3;
            if kind == 0 || (kind == 2) {
                return Err(ArmtError::MmuFault {
                    va,
                    iss: if is_write { 0x44 } else { 0x04 },
                    reason: format!("Translation fault at level {level} (desc=0x{desc:x})"),
                });
            }

            // Block descriptor (valid at level < 3 when kind == 1) or L3 Page descriptor (kind == 3)
            if (level < 3 && kind == 1) || (level == 3 && kind == 3) {
                let block_mask = (1u64 << lsb) - 1;
                let oa_base = desc & 0x0000_FFFF_FFFF_F000 & !block_mask;
                let pa = oa_base | (va & block_mask);
                let pa_page = (oa_base | (va & block_mask)) & !page_mask;
                self.tlb[tlb_idx] = TlbEntry {
                    tag,
                    pa_base: pa_page,
                    page_mask,
                    valid: true,
                };
                return Ok(pa);
            }

            // Table descriptor -> next level
            table_addr = desc & 0x0000_FFFF_FFFF_F000;
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

        cpu.elr_el1 = cpu.pc;
        let daif = (cpu.daif as u64) << 6;
        let el_mode = ((cpu.el as u64) << 2) | (cpu.sp_sel as u64);
        cpu.spsr_el1 = cpu.nzcv | daif | el_mode;
        cpu.esr_el1 = ((ec as u64) << 26) | (iss as u64);
        cpu.far_el1 = far;

        cpu.el = 1;
        cpu.set_sp_sel(1);
        cpu.daif = 0x0F;
        cpu.pc = cpu.vbar_el1.wrapping_add(vector_offset);
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
}
