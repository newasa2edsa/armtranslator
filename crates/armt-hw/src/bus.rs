use crate::gicv3::GicV3;
use crate::memmap::Vma2MemMap;
use crate::timer::GenericTimer;
use crate::uart::Pl011Uart;
use crate::virtio::{Framebuffer, VirtioBlk, VirtioInput};
use crate::vmapple::{PvPanicDevice, VmaAesDevice, VmaBdifDevice, VmaCfgDevice};
use armt_core::{ArmtError, GuestMemory, IrqController, MmioDevice, Result, SysReg, SysRegBus};
use std::collections::BTreeMap;

/// Unified system bus for the sandboxed `VirtualMac2,1` platform.
pub struct Vma2Bus {
    pub memmap: Vma2MemMap,
    pub ram: Vec<u8>,
    pub uart0: Pl011Uart,
    pub gic: GicV3,
    pub timer: GenericTimer,
    pub cfg: VmaCfgDevice,
    pub bdif: VmaBdifDevice,
    pub aes1: VmaAesDevice,
    pub aes2: VmaAesDevice,
    pub pvpanic: PvPanicDevice,
    pub virtio_blk: VirtioBlk,
    pub virtio_input: VirtioInput,
    pub framebuffer: Framebuffer,
    pub unknown_mmio_reads: BTreeMap<u64, u64>,
    pub unknown_mmio_writes: BTreeMap<u64, u64>,
    pub unknown_sysregs: BTreeMap<u16, u64>,
}

impl Vma2Bus {
    pub fn new(memmap: Vma2MemMap, echo_uart: bool) -> Self {
        let ram = vec![0u8; memmap.ram_size as usize];
        let fb = Framebuffer::new(memmap.framebuffer_width, memmap.framebuffer_height);
        let ram_mb = memmap.ram_size / (1024 * 1024);
        Self {
            memmap,
            ram,
            uart0: Pl011Uart::new(echo_uart),
            gic: GicV3::new(),
            timer: GenericTimer::new(),
            cfg: VmaCfgDevice::new(ram_mb),
            bdif: VmaBdifDevice::new(64 * 1024),
            aes1: VmaAesDevice::new(),
            aes2: VmaAesDevice::new(),
            pvpanic: PvPanicDevice::new(),
            virtio_blk: VirtioBlk::new_memory_disk(1024 * 1024),
            virtio_input: VirtioInput::new(),
            framebuffer: fb,
            unknown_mmio_reads: BTreeMap::new(),
            unknown_mmio_writes: BTreeMap::new(),
            unknown_sysregs: BTreeMap::new(),
        }
    }

    pub fn tick(&mut self, cycles: u64) {
        self.timer.tick(cycles);
        if let Some(irq) = self.timer.active_irq() {
            self.gic.raise_irq(irq);
        }
        if self.uart0.irq_pending() {
            self.gic.raise_irq(33);
        }
    }

    #[inline]
    fn ram_offset(&self, addr: u64, size: usize) -> Option<usize> {
        if addr >= self.memmap.ram_base {
            let off = (addr - self.memmap.ram_base) as usize;
            if off.checked_add(size)? <= self.ram.len() {
                return Some(off);
            }
        }
        None
    }

    fn read_mmio(&mut self, addr: u64, size: u8) -> Result<u64> {
        let m = &self.memmap;
        if addr >= m.framebuffer_base && addr < m.framebuffer_base + m.framebuffer_size_bytes() {
            return Ok(self.framebuffer.read_bytes(addr - m.framebuffer_base, size));
        }
        if addr >= m.uart0_base && addr < m.uart0_base + m.uart0_size {
            return self.uart0.read(addr - m.uart0_base, size);
        }
        if addr >= m.gicd_base && addr < m.gicd_base + m.gicd_size {
            return self.gic.read_gicd(addr - m.gicd_base);
        }
        if addr >= m.gicr_base && addr < m.gicr_base + m.gicr_size {
            return self.gic.read_gicr(addr - m.gicr_base);
        }
        if addr >= m.pvpanic_base && addr < m.pvpanic_base + m.pvpanic_size {
            return self.pvpanic.read(addr - m.pvpanic_base, size);
        }
        if addr >= m.bdif_base && addr < m.bdif_base + m.bdif_size {
            return self.bdif.read(addr - m.bdif_base, size);
        }
        if addr >= m.aes1_base && addr < m.aes1_base + m.aes1_size {
            return self.aes1.read(addr - m.aes1_base, size);
        }
        if addr >= m.aes2_base && addr < m.aes2_base + m.aes2_size {
            return self.aes2.read(addr - m.aes2_base, size);
        }
        if addr >= m.cfg_base && addr < m.cfg_base + m.cfg_size {
            return self.cfg.read(addr - m.cfg_base, size);
        }
        if addr >= m.virtio_blk_base && addr < m.virtio_blk_base + m.virtio_blk_size {
            return self.virtio_blk.read(addr - m.virtio_blk_base, size);
        }
        if addr >= m.virtio_input_base && addr < m.virtio_input_base + m.virtio_input_size {
            return self.virtio_input.read(addr - m.virtio_input_base, size);
        }

        // Graceful probe handling for Apple-specific MMIO probes during early boot.
        if addr < 0x8000_0000 {
            *self.unknown_mmio_reads.entry(addr & !0xFFF).or_insert(0) += 1;
            return Ok(0);
        }

        Err(ArmtError::MemoryFault {
            addr,
            size,
            write: false,
            reason: "Unmapped physical read".into(),
        })
    }

    fn write_mmio(&mut self, addr: u64, size: u8, val: u64) -> Result<()> {
        let m = &self.memmap;
        if addr >= m.framebuffer_base && addr < m.framebuffer_base + m.framebuffer_size_bytes() {
            self.framebuffer.write_bytes(addr - m.framebuffer_base, size, val);
            return Ok(());
        }
        if addr >= m.uart0_base && addr < m.uart0_base + m.uart0_size {
            return self.uart0.write(addr - m.uart0_base, size, val);
        }
        if addr >= m.gicd_base && addr < m.gicd_base + m.gicd_size {
            return self.gic.write_gicd(addr - m.gicd_base, val);
        }
        if addr >= m.gicr_base && addr < m.gicr_base + m.gicr_size {
            return self.gic.write_gicr(addr - m.gicr_base, val);
        }
        if addr >= m.pvpanic_base && addr < m.pvpanic_base + m.pvpanic_size {
            return self.pvpanic.write(addr - m.pvpanic_base, size, val);
        }
        if addr >= m.bdif_base && addr < m.bdif_base + m.bdif_size {
            return self.bdif.write(addr - m.bdif_base, size, val);
        }
        if addr >= m.aes1_base && addr < m.aes1_base + m.aes1_size {
            return self.aes1.write(addr - m.aes1_base, size, val);
        }
        if addr >= m.aes2_base && addr < m.aes2_base + m.aes2_size {
            return self.aes2.write(addr - m.aes2_base, size, val);
        }
        if addr >= m.cfg_base && addr < m.cfg_base + m.cfg_size {
            return self.cfg.write(addr - m.cfg_base, size, val);
        }
        if addr >= m.virtio_blk_base && addr < m.virtio_blk_base + m.virtio_blk_size {
            return self.virtio_blk.write(addr - m.virtio_blk_base, size, val);
        }
        if addr >= m.virtio_input_base && addr < m.virtio_input_base + m.virtio_input_size {
            return self.virtio_input.write(addr - m.virtio_input_base, size, val);
        }

        if addr < 0x8000_0000 {
            *self.unknown_mmio_writes.entry(addr & !0xFFF).or_insert(0) += 1;
            return Ok(());
        }

        Err(ArmtError::MemoryFault {
            addr,
            size,
            write: true,
            reason: "Unmapped physical write".into(),
        })
    }
}

impl GuestMemory for Vma2Bus {
    fn read_u8(&mut self, addr: u64) -> Result<u8> {
        if let Some(off) = self.ram_offset(addr, 1) {
            return Ok(self.ram[off]);
        }
        self.read_mmio(addr, 1).map(|v| v as u8)
    }

    fn read_u16(&mut self, addr: u64) -> Result<u16> {
        if let Some(off) = self.ram_offset(addr, 2) {
            return Ok(u16::from_le_bytes([self.ram[off], self.ram[off + 1]]));
        }
        self.read_mmio(addr, 2).map(|v| v as u16)
    }

    fn read_u32(&mut self, addr: u64) -> Result<u32> {
        if let Some(off) = self.ram_offset(addr, 4) {
            return Ok(u32::from_le_bytes(
                self.ram[off..off + 4].try_into().unwrap(),
            ));
        }
        self.read_mmio(addr, 4).map(|v| v as u32)
    }

    fn read_u64(&mut self, addr: u64) -> Result<u64> {
        if let Some(off) = self.ram_offset(addr, 8) {
            return Ok(u64::from_le_bytes(
                self.ram[off..off + 8].try_into().unwrap(),
            ));
        }
        self.read_mmio(addr, 8)
    }

    fn write_u8(&mut self, addr: u64, val: u8) -> Result<()> {
        if let Some(off) = self.ram_offset(addr, 1) {
            self.ram[off] = val;
            return Ok(());
        }
        self.write_mmio(addr, 1, val as u64)
    }

    fn write_u16(&mut self, addr: u64, val: u16) -> Result<()> {
        if let Some(off) = self.ram_offset(addr, 2) {
            self.ram[off..off + 2].copy_from_slice(&val.to_le_bytes());
            return Ok(());
        }
        self.write_mmio(addr, 2, val as u64)
    }

    fn write_u32(&mut self, addr: u64, val: u32) -> Result<()> {
        if let Some(off) = self.ram_offset(addr, 4) {
            self.ram[off..off + 4].copy_from_slice(&val.to_le_bytes());
            return Ok(());
        }
        self.write_mmio(addr, 4, val as u64)
    }

    fn write_u64(&mut self, addr: u64, val: u64) -> Result<()> {
        if let Some(off) = self.ram_offset(addr, 8) {
            self.ram[off..off + 8].copy_from_slice(&val.to_le_bytes());
            return Ok(());
        }
        self.write_mmio(addr, 8, val)
    }

    fn write_bytes(&mut self, addr: u64, src: &[u8]) -> Result<()> {
        if let Some(off) = self.ram_offset(addr, src.len()) {
            self.ram[off..off + src.len()].copy_from_slice(src);
            return Ok(());
        }
        for (i, &b) in src.iter().enumerate() {
            self.write_u8(addr + i as u64, b)?;
        }
        Ok(())
    }

    fn ram_slice_mut(&mut self) -> Option<(u64, *mut u8, usize)> {
        Some((self.memmap.ram_base, self.ram.as_mut_ptr(), self.ram.len()))
    }
}

impl IrqController for Vma2Bus {
    fn raise_irq(&mut self, irq: u32) {
        self.gic.raise_irq(irq);
    }

    fn clear_irq(&mut self, irq: u32) {
        self.gic.clear_irq(irq);
    }

    fn pending_irq(&self) -> Option<u32> {
        self.gic.pending_irq()
    }

    fn ack_irq(&mut self) -> u32 {
        self.gic.ack_irq()
    }

    fn eoi_irq(&mut self, irq: u32) {
        self.gic.eoi_irq(irq);
    }
}

impl SysRegBus for Vma2Bus {
    fn read_sysreg(&mut self, reg: SysReg) -> Option<u64> {
        if let Some(v) = self.timer.read_sysreg(reg) {
            return Some(v);
        }
        if let Some(v) = self.gic.read_sysreg(reg) {
            return Some(v);
        }
        *self.unknown_sysregs.entry(reg.0).or_insert(0) += 1;
        None
    }

    fn write_sysreg(&mut self, reg: SysReg, val: u64) -> bool {
        if self.timer.write_sysreg(reg, val) {
            return true;
        }
        if self.gic.write_sysreg(reg, val) {
            return true;
        }
        *self.unknown_sysregs.entry(reg.0).or_insert(0) += 1;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_ram_uart_timer_gic_aes() {
        let mut bus = Vma2Bus::new(Vma2MemMap::with_ram_mb(16), false);

        // RAM roundtrip
        bus.write_u64(0x8000_1000, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(bus.read_u64(0x8000_1000).unwrap(), 0x1122_3344_5566_7788);

        // PL011 UART TX
        for &b in b"Hello Mac\n" {
            bus.write_u32(0x2001_0000, b as u32).unwrap();
        }
        assert_eq!(bus.uart0.tx_string(), "Hello Mac\n");

        // Generic timer countdown -> GICv3 IRQ 27
        bus.gic.set_enabled(27, true);
        assert!(bus.write_sysreg(SysReg::CNTV_TVAL_EL0, 10));
        assert!(bus.write_sysreg(SysReg::CNTV_CTL_EL0, 1));
        bus.tick(15);
        assert_eq!(bus.pending_irq(), Some(27));
        assert_eq!(bus.ack_irq(), 27);
        bus.eoi_irq(27);

        // NIST FIPS-197 AES-128 test vector:
        // Key: 2b7e151628aed2a6abf7158809cf4f3c
        // PT : 3243f6a8885a308d313198a2e0370734
        // CT : 3925841d02dc09fbdc118597196a0b32
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let mut block: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ];
        let expected: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a,
            0x0b, 0x32,
        ];
        VmaAesDevice::encrypt_block(&key, &mut block);
        assert_eq!(block, expected);

        // VirtIO-blk sector roundtrip
        let mut wbuf = [0u8; 512];
        wbuf[..5].copy_from_slice(b"APFS!");
        bus.virtio_blk.write_sector(1, &wbuf).unwrap();
        let mut rbuf = [0u8; 512];
        bus.virtio_blk.read_sector(1, &mut rbuf).unwrap();
        assert_eq!(&rbuf[..5], b"APFS!");

        // Framebuffer write
        bus.write_u32(0x6000_0000, 0x00FF_8800).unwrap();
        assert_eq!(bus.framebuffer.non_zero_pixels(), 1);
    }
}
