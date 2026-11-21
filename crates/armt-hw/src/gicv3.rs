use armt_core::{IrqController, Result, SysReg, SysRegBus};

const NUM_IRQS: usize = 256;

/// ARM GICv3 Distributor + Redistributor + ICC system register interface.
pub struct GicV3 {
    ctlr: u32,
    ctlr_el1: u64,
    igrpen0: u64,
    igrpen1: u64,
    pmr: u64,
    sre: u64,
    bpr0: u64,
    bpr1: u64,
    group1: [bool; NUM_IRQS],
    enabled: [bool; NUM_IRQS],
    pending: [bool; NUM_IRQS],
    active: [bool; NUM_IRQS],
    priority: [u8; NUM_IRQS],
}

impl Default for GicV3 {
    fn default() -> Self {
        Self::new()
    }
}

impl GicV3 {
    pub fn new() -> Self {
        Self {
            ctlr: 0x37,
            ctlr_el1: 0,
            igrpen0: 1,
            igrpen1: 1,
            pmr: 0xFF,
            sre: 0x7,
            bpr0: 0,
            bpr1: 0,
            group1: [true; NUM_IRQS],
            enabled: [false; NUM_IRQS],
            pending: [false; NUM_IRQS],
            active: [false; NUM_IRQS],
            priority: [0x80; NUM_IRQS],
        }
    }

    /// Configure ARM generic timer PPIs (`27` virtual, `30` physical) as Group-0 FIQs
    /// matching Apple `vmapple` / Apple Silicon GICv3 routing.
    pub fn set_apple_timer_fiq_routing(&mut self) {
        self.group1[27] = false;
        self.group1[30] = false;
    }

    pub fn set_enabled(&mut self, irq: u32, on: bool) {
        if (irq as usize) < NUM_IRQS {
            self.enabled[irq as usize] = on;
        }
    }

    pub fn read_gicd(&self, offset: u64) -> Result<u64> {
        let val = match offset {
            0x0000 => self.ctlr as u64,
            0x0004 => 0x0000_0007, // TYPER: 256 IRQs
            0x0008 => 0x0400_043B, // IIDR (ARM)
            0x0080..=0x009C => {
                let word_idx = ((offset - 0x0080) / 4) as usize;
                self.read_bitmap(&self.group1, word_idx) as u64
            }
            0xFFE8 => 0x30,        // PIDR2: GICv3
            0x0100..=0x011C => {
                let word_idx = ((offset - 0x0100) / 4) as usize;
                self.read_bitmap(&self.enabled, word_idx) as u64
            }
            0x0200..=0x021C => {
                let word_idx = ((offset - 0x0200) / 4) as usize;
                self.read_bitmap(&self.pending, word_idx) as u64
            }
            0x0400..=0x04FC => {
                let base = (offset - 0x0400) as usize;
                let mut w = 0u32;
                for i in 0..4 {
                    if base + i < NUM_IRQS {
                        w |= (self.priority[base + i] as u32) << (i * 8);
                    }
                }
                w as u64
            }
            _ => 0,
        };
        Ok(val)
    }

    pub fn write_gicd(&mut self, offset: u64, val: u64) -> Result<()> {
        match offset {
            0x0000 => self.ctlr = val as u32,
            0x0080..=0x009C => {
                let word_idx = ((offset - 0x0080) / 4) as usize;
                let base = word_idx * 32;
                let bits = val as u32;
                for i in 0..32 {
                    if base + i < NUM_IRQS {
                        self.group1[base + i] = ((bits >> i) & 1) != 0;
                    }
                }
            }
            0x0100..=0x011C => {
                let word_idx = ((offset - 0x0100) / 4) as usize;
                Self::write_bitmap_set(&mut self.enabled, word_idx, val as u32, true);
            }
            0x0180..=0x019C => {
                let word_idx = ((offset - 0x0180) / 4) as usize;
                Self::write_bitmap_set(&mut self.enabled, word_idx, val as u32, false);
            }
            0x0200..=0x021C => {
                let word_idx = ((offset - 0x0200) / 4) as usize;
                Self::write_bitmap_set(&mut self.pending, word_idx, val as u32, true);
            }
            0x0280..=0x029C => {
                let word_idx = ((offset - 0x0280) / 4) as usize;
                Self::write_bitmap_set(&mut self.pending, word_idx, val as u32, false);
            }
            0x0400..=0x04FC => {
                let base = (offset - 0x0400) as usize;
                for i in 0..4 {
                    if base + i < NUM_IRQS {
                        self.priority[base + i] = ((val >> (i * 8)) & 0xFF) as u8;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Read GICv2m / Apple vmapple MSI frame register (`MSI_TYPER` at `+0x008`).
    pub fn read_msi(&self, offset: u64) -> Result<u64> {
        let val = match offset {
            // Bit 31: valid/enabled, bits [28:16]: vectorCount=64, bits [12:0]: vectorBase=128
            0x0008 => 0x8040_0080,
            0x0FCC => 0x0400_043B,
            0x0FE8 => 0x30,
            _ => 0,
        };
        Ok(val)
    }

    /// Write GICv2m / Apple vmapple MSI frame register (`MSI_SETSPI_NS` at `+0x040`).
    pub fn write_msi(&mut self, offset: u64, val: u64) -> Result<()> {
        if offset == 0x0040 {
            let irq = (val & 0x3FF) as u32;
            if (irq as usize) < NUM_IRQS {
                self.pending[irq as usize] = true;
            }
        }
        Ok(())
    }

    pub fn read_gicr(&self, offset: u64) -> Result<u64> {
        let cpu_idx = offset / 0x2_0000;
        let frame_off = offset % 0x2_0000;
        let val = match frame_off {
            0x0000 => 0x0100_0000, // GICR_CTLR
            0x0004 => 0x0400_043B, // GICR_IIDR
            0x0008 => {
                // GICR_TYPER (low 32 + affinity high 32): bit 4 = Last redistributor
                let last = if cpu_idx == 0 { 0x10 } else { 0 };
                (cpu_idx << 32) | (cpu_idx << 8) | last
            }
            0x000C => cpu_idx,     // High 32 bits of GICR_TYPER (affinity)
            0x0014 => 0,           // WAKER: ProcessorSleep=0, ChildrenAsleep=0
            0xFFE8 => 0x30,
            0x10080 if cpu_idx == 0 => self.read_bitmap(&self.group1, 0) as u64,
            0x10100 if cpu_idx == 0 => self.read_bitmap(&self.enabled, 0) as u64,
            0x10200 if cpu_idx == 0 => self.read_bitmap(&self.pending, 0) as u64,
            0x10300 if cpu_idx == 0 => self.read_bitmap(&self.active, 0) as u64,
            0x10400..=0x1041C if cpu_idx == 0 => {
                let base = (frame_off - 0x10400) as usize;
                let mut w = 0u32;
                for i in 0..4 {
                    if base + i < 32 {
                        w |= (self.priority[base + i] as u32) << (i * 8);
                    }
                }
                w as u64
            }
            _ => 0,
        };
        Ok(val)
    }

    pub fn write_gicr(&mut self, offset: u64, val: u64) -> Result<()> {
        let cpu_idx = offset / 0x2_0000;
        let frame_off = offset % 0x2_0000;
        if cpu_idx == 0 {
            match frame_off {
                0x10080 => {
                    let bits = val as u32;
                    for i in 0..32.min(NUM_IRQS) {
                        self.group1[i] = ((bits >> i) & 1) != 0;
                    }
                }
                0x10100 => Self::write_bitmap_set(&mut self.enabled, 0, val as u32, true),
                0x10180 => Self::write_bitmap_set(&mut self.enabled, 0, val as u32, false),
                0x10200 => Self::write_bitmap_set(&mut self.pending, 0, val as u32, true),
                0x10280 => Self::write_bitmap_set(&mut self.pending, 0, val as u32, false),
                0x10300 => Self::write_bitmap_set(&mut self.active, 0, val as u32, true),
                0x10380 => Self::write_bitmap_set(&mut self.active, 0, val as u32, false),
                0x10400..=0x1041C => {
                    let base = (frame_off - 0x10400) as usize;
                    for i in 0..4 {
                        if base + i < 32 {
                            self.priority[base + i] = ((val >> (i * 8)) & 0xFF) as u8;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn read_bitmap(&self, arr: &[bool; NUM_IRQS], word_idx: usize) -> u32 {
        let base = word_idx * 32;
        let mut v = 0u32;
        for b in 0..32 {
            if base + b < NUM_IRQS && arr[base + b] {
                v |= 1 << b;
            }
        }
        v
    }

    fn write_bitmap_set(arr: &mut [bool; NUM_IRQS], word_idx: usize, bits: u32, set: bool) {
        let base = word_idx * 32;
        for b in 0..32 {
            if (bits & (1 << b)) != 0 && base + b < NUM_IRQS {
                arr[base + b] = set;
            }
        }
    }
    pub fn debug_summary(&self) -> String {
        let mut en = Vec::new();
        let mut pend = Vec::new();
        let mut act = Vec::new();
        let mut g0 = Vec::new();
        for i in 0..NUM_IRQS {
            if self.enabled[i] {
                en.push(i);
            }
            if self.pending[i] {
                pend.push(i);
            }
            if self.active[i] {
                act.push(i);
            }
            if !self.group1[i] {
                g0.push(i);
            }
        }
        format!(
            "ctlr=0x{:x} igrpen0={} igrpen1={} pmr=0x{:x} enabled={:?} pending={:?} active={:?} grp0={:?}",
            self.ctlr, self.igrpen0, self.igrpen1, self.pmr, en, pend, act, g0
        )
    }

    /// Returns highest-priority pending interrupt for GIC Group `group` (`0` = FIQ, `1` = IRQ).
    pub fn pending_group(&self, group: u8) -> Option<u32> {
        if (group == 0 && self.igrpen0 == 0) || (group != 0 && self.igrpen1 == 0) {
            return None;
        }
        let want_g1 = group != 0;
        let mut best: Option<(u8, u32)> = None;
        for irq in 0..NUM_IRQS {
            if self.group1[irq] == want_g1
                && self.enabled[irq]
                && self.pending[irq]
                && !self.active[irq]
                && (self.priority[irq] as u64) < self.pmr
            {
                let prio = self.priority[irq];
                if best.map_or(true, |(bp, _)| prio < bp) {
                    best = Some((prio, irq as u32));
                }
            }
        }
        best.map(|(_, irq)| irq)
    }

    fn send_sgi(&mut self, val: u64) {
        let intid = ((val >> 24) & 0xF) as usize;
        let irm = ((val >> 40) & 1) != 0;
        let target_list = (val & 0xFFFF) as u16;
        let aff1 = (val >> 16) & 0xFF;
        let aff2 = (val >> 32) & 0xFF;
        let aff3 = (val >> 48) & 0xFF;
        if !irm && aff3 == 0 && aff2 == 0 && aff1 == 0 && (target_list & 1) != 0 && intid < 16 {
            self.pending[intid] = true;
        }
    }

    fn ack_group(&mut self, group: u8) -> u32 {
        if let Some(irq) = self.pending_group(group) {
            self.pending[irq as usize] = false;
            self.active[irq as usize] = true;
            irq
        } else {
            1023 // Spurious
        }
    }
}

impl IrqController for GicV3 {
    fn raise_irq(&mut self, irq: u32) {
        if (irq as usize) < NUM_IRQS {
            self.pending[irq as usize] = true;
        }
    }

    fn clear_irq(&mut self, irq: u32) {
        if (irq as usize) < NUM_IRQS {
            self.pending[irq as usize] = false;
        }
    }

    fn pending_irq(&self) -> Option<u32> {
        self.pending_group(1)
    }

    fn ack_irq(&mut self) -> u32 {
        self.ack_group(1)
    }

    fn eoi_irq(&mut self, irq: u32) {
        if (irq as usize) < NUM_IRQS {
            self.active[irq as usize] = false;
        }
    }
}

impl SysRegBus for GicV3 {
    fn read_sysreg(&mut self, reg: SysReg) -> Option<u64> {
        match reg {
            SysReg::ISR_EL1 => {
                let mut v = 0u64;
                if self.pending_group(0).is_some() {
                    v |= 1 << 6; // F bit
                }
                if self.pending_group(1).is_some() {
                    v |= 1 << 7; // I bit
                }
                Some(v)
            }
            SysReg::ICC_PMR_EL1 => Some(self.pmr),
            SysReg::ICC_IAR0_EL1 => Some(self.ack_group(0) as u64),
            SysReg::ICC_HPPIR0_EL1 => Some(self.pending_group(0).unwrap_or(1023) as u64),
            SysReg::ICC_BPR0_EL1 => Some(self.bpr0),
            SysReg::ICC_RPR_EL1 => Some(0xFF),
            SysReg::ICC_IAR1_EL1 => Some(self.ack_group(1) as u64),
            SysReg::ICC_HPPIR1_EL1 => Some(self.pending_group(1).unwrap_or(1023) as u64),
            SysReg::ICC_BPR1_EL1 => Some(self.bpr1),
            SysReg::ICC_CTLR_EL1 => Some(self.ctlr_el1),
            SysReg::ICC_SRE_EL1 => Some(self.sre),
            SysReg::ICC_IGRPEN0_EL1 => Some(self.igrpen0),
            SysReg::ICC_IGRPEN1_EL1 => Some(self.igrpen1),
            _ => None,
        }
    }

    fn write_sysreg(&mut self, reg: SysReg, val: u64) -> bool {
        match reg {
            SysReg::ICC_PMR_EL1 => self.pmr = val & 0xFF,
            SysReg::ICC_EOIR0_EL1 | SysReg::ICC_EOIR1_EL1 => {
                if (self.ctlr_el1 & 2) == 0 {
                    self.eoi_irq(val as u32);
                }
            }
            SysReg::ICC_DIR_EL1 => self.eoi_irq(val as u32),
            SysReg::ICC_SGI1R_EL1 | SysReg::ICC_ASGI1R_EL1 | SysReg::ICC_SGI0R_EL1 => {
                self.send_sgi(val);
            }
            SysReg::ICC_BPR0_EL1 => self.bpr0 = val & 0x7,
            SysReg::ICC_BPR1_EL1 => self.bpr1 = val & 0x7,
            SysReg::ICC_CTLR_EL1 => self.ctlr_el1 = val & 0x2,
            SysReg::ICC_SRE_EL1 => self.sre = val | 1,
            SysReg::ICC_IGRPEN0_EL1 => self.igrpen0 = val & 1,
            SysReg::ICC_IGRPEN1_EL1 => self.igrpen1 = val & 1,
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgi_and_gicv2m_msi() {
        let mut gic = GicV3::new();
        // Enable SGI 1 (Group 1) and SPI 128 (Group 1)
        gic.write_gicr(0x10100, 1 << 1).unwrap();
        gic.write_gicd(0x0110, 1 << 0).unwrap(); // IRQ 128 = word 4 (0x0100 + 16 = 0x0110), bit 0

        // Send SGI 1 to PE 0 via ICC_SGI1R_EL1 (INTID=1 at bits 27:24, TargetList=1)
        assert!(gic.write_sysreg(SysReg::ICC_SGI1R_EL1, (1 << 24) | 1));
        assert_eq!(gic.read_sysreg(SysReg::ISR_EL1), Some(1 << 7));
        assert_eq!(gic.pending_group(1), Some(1));
        assert_eq!(gic.read_sysreg(SysReg::ICC_IAR1_EL1), Some(1));
        assert_eq!(gic.read_sysreg(SysReg::ISR_EL1), Some(0));
        assert!(gic.write_sysreg(SysReg::ICC_EOIR1_EL1, 1));
        assert_eq!(gic.pending_group(1), None);

        // Read GICv2m MSI_TYPER and ring MSI_SETSPI_NS for SPI 128
        let typer = gic.read_msi(0x0008).unwrap();
        assert_ne!(typer & 0x8000_0000, 0);
        assert_eq!(typer & 0x1FFF, 128);
        assert_eq!((typer >> 16) & 0x1FFF, 64);
        gic.write_msi(0x0040, 128).unwrap();
        assert_eq!(gic.pending_group(1), Some(128));
    }
}
