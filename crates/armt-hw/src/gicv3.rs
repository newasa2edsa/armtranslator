use armt_core::{IrqController, Result, SysReg, SysRegBus};

const NUM_IRQS: usize = 128;

/// ARM GICv3 Distributor + Redistributor + ICC system register interface.
pub struct GicV3 {
    ctlr: u32,
    igrpen1: u64,
    pmr: u64,
    sre: u64,
    bpr1: u64,
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
            igrpen1: 1,
            pmr: 0xFF,
            sre: 0x7,
            bpr1: 0,
            enabled: [false; NUM_IRQS],
            pending: [false; NUM_IRQS],
            active: [false; NUM_IRQS],
            priority: [0x80; NUM_IRQS],
        }
    }

    pub fn set_enabled(&mut self, irq: u32, on: bool) {
        if (irq as usize) < NUM_IRQS {
            self.enabled[irq as usize] = on;
        }
    }

    pub fn read_gicd(&self, offset: u64) -> Result<u64> {
        let val = match offset {
            0x0000 => self.ctlr as u64,
            0x0004 => 0x0000_0003, // TYPER: 128 IRQs
            0x0008 => 0x0400_043B, // IIDR (ARM)
            0xFFE8 => 0x30,        // PIDR2: GICv3
            0x0100..=0x011C => {
                let word_idx = ((offset - 0x0100) / 4) as usize;
                self.read_bitmap(&self.enabled, word_idx) as u64
            }
            0x0200..=0x021C => {
                let word_idx = ((offset - 0x0200) / 4) as usize;
                self.read_bitmap(&self.pending, word_idx) as u64
            }
            0x0400..=0x047C => {
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
            0x0400..=0x047C => {
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

    pub fn read_gicr(&self, offset: u64) -> Result<u64> {
        let val = match offset {
            0x0004 => 0x0400_043B,
            0x0008 => 0x0000_0010, // Last redistributor bit
            0x0014 => 0,           // WAKER: ProcessorSleep=0, ChildrenAsleep=0
            0xFFE8 => 0x30,
            0x10100 => self.read_bitmap(&self.enabled, 0) as u64,
            0x10200 => self.read_bitmap(&self.pending, 0) as u64,
            _ => 0,
        };
        Ok(val)
    }

    pub fn write_gicr(&mut self, offset: u64, val: u64) -> Result<()> {
        match offset {
            0x10100 => Self::write_bitmap_set(&mut self.enabled, 0, val as u32, true),
            0x10180 => Self::write_bitmap_set(&mut self.enabled, 0, val as u32, false),
            0x10200 => Self::write_bitmap_set(&mut self.pending, 0, val as u32, true),
            0x10280 => Self::write_bitmap_set(&mut self.pending, 0, val as u32, false),
            _ => {}
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
        if self.igrpen1 == 0 {
            return None;
        }
        let mut best: Option<(u8, u32)> = None;
        for irq in 0..NUM_IRQS {
            if self.enabled[irq]
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

    fn ack_irq(&mut self) -> u32 {
        if let Some(irq) = self.pending_irq() {
            self.pending[irq as usize] = false;
            self.active[irq as usize] = true;
            irq
        } else {
            1023 // Spurious
        }
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
            SysReg::ICC_PMR_EL1 => Some(self.pmr),
            SysReg::ICC_IAR1_EL1 => Some(self.ack_irq() as u64),
            SysReg::ICC_BPR1_EL1 => Some(self.bpr1),
            SysReg::ICC_SRE_EL1 => Some(self.sre),
            SysReg::ICC_IGRPEN1_EL1 => Some(self.igrpen1),
            _ => None,
        }
    }

    fn write_sysreg(&mut self, reg: SysReg, val: u64) -> bool {
        match reg {
            SysReg::ICC_PMR_EL1 => self.pmr = val & 0xFF,
            SysReg::ICC_EOIR1_EL1 => self.eoi_irq(val as u32),
            SysReg::ICC_BPR1_EL1 => self.bpr1 = val & 0x7,
            SysReg::ICC_SRE_EL1 => self.sre = val | 1,
            SysReg::ICC_IGRPEN1_EL1 => self.igrpen1 = val & 1,
            _ => return false,
        }
        true
    }
}
