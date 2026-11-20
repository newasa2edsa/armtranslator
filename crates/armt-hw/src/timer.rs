use armt_core::{SysReg, SysRegBus};

/// ARMv8 Generic Timer (24 MHz Apple Silicon timebase).
pub struct GenericTimer {
    pub cntfrq: u64,
    pub counter: u64,
    pub cntp_ctl: u64,
    pub cntp_cval: u64,
    pub cntv_ctl: u64,
    pub cntv_cval: u64,
}

impl Default for GenericTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl GenericTimer {
    pub fn new() -> Self {
        Self {
            cntfrq: 24_000_000,
            counter: 0,
            cntp_ctl: 0,
            cntp_cval: 0,
            cntv_ctl: 0,
            cntv_cval: 0,
        }
    }

    pub fn tick(&mut self, delta: u64) {
        self.counter = self.counter.wrapping_add(delta);
        if (self.cntv_ctl & 1) != 0 && self.counter >= self.cntv_cval {
            self.cntv_ctl |= 0b100; // ISTATUS
        } else {
            self.cntv_ctl &= !0b100;
        }
        if (self.cntp_ctl & 1) != 0 && self.counter >= self.cntp_cval {
            self.cntp_ctl |= 0b100;
        } else {
            self.cntp_ctl &= !0b100;
        }
    }

    /// Returns any asserted PPI IRQ (`27` for CNTV, `30` for CNTP).
    pub fn active_irq(&self) -> Option<u32> {
        if (self.cntv_ctl & 0b111) == 0b101 {
            Some(27)
        } else if (self.cntp_ctl & 0b111) == 0b101 {
            Some(30)
        } else {
            None
        }
    }
}

impl SysRegBus for GenericTimer {
    fn read_sysreg(&mut self, reg: SysReg) -> Option<u64> {
        match reg {
            SysReg::CNTFRQ_EL0 => Some(self.cntfrq),
            SysReg::CNTPCT_EL0 | SysReg::CNTVCT_EL0 => {
                self.counter = self.counter.wrapping_add(1);
                Some(self.counter)
            }
            SysReg::CNTP_CTL_EL0 => Some(self.cntp_ctl),
            SysReg::CNTP_CVAL_EL0 => Some(self.cntp_cval),
            SysReg::CNTP_TVAL_EL0 => Some(self.cntp_cval.wrapping_sub(self.counter) as i32 as i64 as u64),
            SysReg::CNTV_CTL_EL0 => Some(self.cntv_ctl),
            SysReg::CNTV_CVAL_EL0 => Some(self.cntv_cval),
            SysReg::CNTV_TVAL_EL0 => Some(self.cntv_cval.wrapping_sub(self.counter) as i32 as i64 as u64),
            _ => None,
        }
    }

    fn write_sysreg(&mut self, reg: SysReg, val: u64) -> bool {
        match reg {
            SysReg::CNTFRQ_EL0 => self.cntfrq = val,
            SysReg::CNTP_CTL_EL0 => {
                self.cntp_ctl = (self.cntp_ctl & 0b100) | (val & 0b011);
                self.tick(0);
            }
            SysReg::CNTP_CVAL_EL0 => {
                self.cntp_cval = val;
                self.tick(0);
            }
            SysReg::CNTP_TVAL_EL0 => {
                let signed = (val as i32) as i64;
                self.cntp_cval = (self.counter as i64).wrapping_add(signed) as u64;
                self.tick(0);
            }
            SysReg::CNTV_CTL_EL0 => {
                self.cntv_ctl = (self.cntv_ctl & 0b100) | (val & 0b011);
                self.tick(0);
            }
            SysReg::CNTV_CVAL_EL0 => {
                self.cntv_cval = val;
                self.tick(0);
            }
            SysReg::CNTV_TVAL_EL0 => {
                let signed = (val as i32) as i64;
                self.cntv_cval = (self.counter as i64).wrapping_add(signed) as u64;
                self.tick(0);
            }
            _ => return false,
        }
        true
    }
}
