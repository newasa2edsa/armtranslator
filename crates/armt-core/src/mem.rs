use crate::error::Result;
use crate::isa::SysReg;

/// Physical/guest memory interface implemented by `Vma2Bus` and test RAM adapters.
pub trait GuestMemory {
    fn read_u8(&mut self, addr: u64) -> Result<u8>;
    fn read_u16(&mut self, addr: u64) -> Result<u16>;
    fn read_u32(&mut self, addr: u64) -> Result<u32>;
    fn read_u64(&mut self, addr: u64) -> Result<u64>;

    fn write_u8(&mut self, addr: u64, val: u8) -> Result<()>;
    fn write_u16(&mut self, addr: u64, val: u16) -> Result<()>;
    fn write_u32(&mut self, addr: u64, val: u32) -> Result<()>;
    fn write_u64(&mut self, addr: u64, val: u64) -> Result<()>;

    fn read_bytes(&mut self, addr: u64, dst: &mut [u8]) -> Result<()> {
        for (i, b) in dst.iter_mut().enumerate() {
            *b = self.read_u8(addr + i as u64)?;
        }
        Ok(())
    }

    fn write_bytes(&mut self, addr: u64, src: &[u8]) -> Result<()> {
        for (i, &b) in src.iter().enumerate() {
            self.write_u8(addr + i as u64, b)?;
        }
        Ok(())
    }

    /// Optional direct pointer to contiguous guest RAM `(phys_base, host_ptr, len_bytes)`
    /// enabling fast-path bounds-checked inline load/store codegen in the x86-64 JIT.
    fn ram_slice_mut(&mut self) -> Option<(u64, *mut u8, usize)> {
        None
    }
}

/// Memory-mapped peripheral device trait.
pub trait MmioDevice: Send {
    fn name(&self) -> &str;
    fn read(&mut self, offset: u64, size: u8) -> Result<u64>;
    fn write(&mut self, offset: u64, size: u8, val: u64) -> Result<()>;
    fn tick(&mut self, _cycles: u64) {}
}

/// Interrupt controller (GICv3) trait.
pub trait IrqController {
    fn raise_irq(&mut self, irq: u32);
    fn clear_irq(&mut self, irq: u32);
    fn pending_irq(&self) -> Option<u32>;
    fn ack_irq(&mut self) -> u32;
    fn eoi_irq(&mut self, irq: u32);
}

/// Bus for hardware-backed system registers (Generic Timer, GICv3 ICC_*).
pub trait SysRegBus {
    fn read_sysreg(&mut self, reg: SysReg) -> Option<u64>;
    fn write_sysreg(&mut self, reg: SysReg, val: u64) -> bool;
}
