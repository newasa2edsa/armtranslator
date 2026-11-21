use crate::gicv3::GicV3;
use crate::gpu::{AppleParavirtGpu, VirtioGpu};
use crate::memmap::Vma2MemMap;
use crate::pcie::PcieRootComplex;
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
    pub virtio_gpu: VirtioGpu,
    pub virtio_input: VirtioInput,
    pub apple_gfx: AppleParavirtGpu,
    pub pcie: PcieRootComplex,
    pub framebuffer: Framebuffer,
    pub apple_sysregs: BTreeMap<u16, u64>,
    pub unknown_mmio_reads: BTreeMap<u64, u64>,
    pub unknown_mmio_writes: BTreeMap<u64, u64>,
    pub unknown_sysregs: BTreeMap<u16, u64>,
}

impl Vma2Bus {
    pub fn new(memmap: Vma2MemMap, echo_uart: bool) -> Self {
        let ram = vec![0u8; memmap.ram_size as usize];
        let fb = Framebuffer::new(memmap.framebuffer_width, memmap.framebuffer_height);
        let ram_mb = memmap.ram_size / (1024 * 1024);
        let w = memmap.framebuffer_width;
        let h = memmap.framebuffer_height;
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
            virtio_blk: VirtioBlk::new_memory_disk(16 * 1024 * 1024),
            virtio_gpu: VirtioGpu::new(w, h),
            virtio_input: VirtioInput::new(),
            apple_gfx: AppleParavirtGpu::new(w, h),
            pcie: PcieRootComplex::new(),
            framebuffer: fb,
            apple_sysregs: BTreeMap::new(),
            unknown_mmio_reads: BTreeMap::new(),
            unknown_mmio_writes: BTreeMap::new(),
            unknown_sysregs: BTreeMap::new(),
        }
    }

    pub fn tick(&mut self, cycles: u64) {
        self.timer.tick(cycles);
        let active_timer_irq = self.timer.active_irq();
        if active_timer_irq != Some(27) {
            self.gic.clear_irq(27);
        }
        if active_timer_irq != Some(30) {
            self.gic.clear_irq(30);
        }
        if let Some(irq) = active_timer_irq {
            self.gic.raise_irq(irq);
        }
        if self.uart0.irq_pending() {
            self.gic.raise_irq(33);
        }
        if self.virtio_gpu.irq_pending() {
            self.gic.raise_irq(34);
        }
        if self.virtio_input.irq_pending() {
            self.gic.raise_irq(35);
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
        if addr >= m.gic_msi_base && addr < m.gic_msi_base + m.gic_msi_size {
            return self.gic.read_msi(addr - m.gic_msi_base);
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
        if addr >= m.virtio_gpu_base && addr < m.virtio_gpu_base + m.virtio_gpu_size {
            return self.virtio_gpu.read(addr - m.virtio_gpu_base, size);
        }
        if addr >= m.virtio_input_base && addr < m.virtio_input_base + m.virtio_input_size {
            return self.virtio_input.read(addr - m.virtio_input_base, size);
        }
        if addr >= m.apple_gfx_base && addr < m.apple_gfx_base + m.apple_gfx_size {
            return self.apple_gfx.read(addr - m.apple_gfx_base, size);
        }
        if addr >= m.pcie_ecam_base && addr < m.pcie_ecam_base + m.pcie_ecam_size {
            return self.pcie.read_ecam(addr - m.pcie_ecam_base, size);
        }
        if self.pcie.contains_bar(addr) {
            return self.pcie.read_bar(addr, size, &self.virtio_blk);
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
        if addr >= m.gic_msi_base && addr < m.gic_msi_base + m.gic_msi_size {
            return self.gic.write_msi(addr - m.gic_msi_base, val);
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
        if addr >= m.virtio_gpu_base && addr < m.virtio_gpu_base + m.virtio_gpu_size {
            let off = addr - m.virtio_gpu_base;
            self.virtio_gpu.write(off, size, val)?;
            if self.virtio_gpu.notify_pending.is_some() {
                self.virtio_gpu
                    .process_queue(&mut self.ram, self.memmap.ram_base, &self.framebuffer);
            }
            return Ok(());
        }
        if addr >= m.virtio_input_base && addr < m.virtio_input_base + m.virtio_input_size {
            return self.virtio_input.write(addr - m.virtio_input_base, size, val);
        }
        if addr >= m.apple_gfx_base && addr < m.apple_gfx_base + m.apple_gfx_size {
            let off = addr - m.apple_gfx_base;
            self.apple_gfx.write(off, size, val)?;
            if self.apple_gfx.doorbell_pending {
                self.apple_gfx
                    .execute_cmd_ring(&self.ram, self.memmap.ram_base, &self.framebuffer);
            }
            return Ok(());
        }
        if addr >= m.pcie_ecam_base && addr < m.pcie_ecam_base + m.pcie_ecam_size {
            return self.pcie.write_ecam(addr - m.pcie_ecam_base, size, val);
        }
        if self.pcie.contains_bar(addr) {
            let ram_base = self.memmap.ram_base;
            if let Some((msi_addr, msi_data)) = self.pcie.write_bar(
                addr,
                size,
                val,
                &mut self.virtio_blk,
                &mut self.ram,
                ram_base,
            )? {
                let _ = self.write_u32(msi_addr, msi_data);
            }
            return Ok(());
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
        if let Some(&v) = self.apple_sysregs.get(&reg.0) {
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
        if reg.is_apple_impl_defined() {
            self.apple_sysregs.insert(reg.0, val);
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

    #[test]
    fn test_apple_gfx_virtio_gpu_and_input() {
        use crate::gpu::{
            GPU_CMD_CLEAR, GPU_CMD_DRAW_TEXT, GPU_CMD_PRESENT, GPU_CMD_ROUND_RECT,
            GPU_CMD_SET_CURSOR,
        };
        let mut bus = Vma2Bus::new(Vma2MemMap::with_ram_mb(16), false);

        // Shared thread-safe VirtioInput mouse + char
        let shared_in = bus.virtio_input.shared();
        {
            let mut g = shared_in.lock().unwrap();
            g.push_mouse_move(420, 310);
            g.push_mouse_button(0, true, 420, 310);
            g.push_char(b'A');
        }
        assert_eq!(bus.read_u32(0x200a_2100).unwrap(), 420);
        assert_eq!(bus.read_u32(0x200a_2104).unwrap(), 310);
        assert_eq!(bus.read_u32(0x200a_2108).unwrap(), 1);
        assert_eq!(bus.read_u32(0x200a_2110).unwrap(), 3);
        assert_eq!(bus.read_u32(0x200a_2114).unwrap(), 2); // MOUSE_MOVE
        assert_eq!(bus.read_u32(0x200a_2114).unwrap(), 3); // MOUSE_BTN
        assert_eq!(bus.read_u32(0x200a_2114).unwrap(), 4); // CHAR
        assert_eq!(bus.read_u32(0x200a_2118).unwrap(), b'A' as u32);

        // AppleParavirtGpu cmd ring test
        let text_pa = 0x8001_0000u64;
        bus.write_bytes(text_pa, b"Hello RTX!\0").unwrap();
        let ring_pa = 0x8002_0000u64;
        let cmds: [(u32, u32, i32, i32, u32, u32, u32, u32); 5] = [
            (GPU_CMD_CLEAR, 0, 0, 0, 0, 0, 0xFF10_1828, 0),
            (GPU_CMD_ROUND_RECT, 12, 100, 100, 300, 160, 0xFF20_2C44, 0xFF40_6090),
            (
                GPU_CMD_DRAW_TEXT,
                2 | (32 << 8),
                120,
                140,
                text_pa as u32,
                (text_pa >> 32) as u32,
                0xFFFF_FFFF,
                0,
            ),
            (GPU_CMD_SET_CURSOR, 1, 250, 220, 0, 0, 0, 0),
            (GPU_CMD_PRESENT, 0, 0, 0, 0, 0, 0, 0),
        ];
        for (idx, c) in cmds.iter().enumerate() {
            let p = ring_pa + (idx as u64) * 32;
            bus.write_u32(p, c.0).unwrap();
            bus.write_u32(p + 4, c.1).unwrap();
            bus.write_u32(p + 8, c.2 as u32).unwrap();
            bus.write_u32(p + 12, c.3 as u32).unwrap();
            bus.write_u32(p + 16, c.4).unwrap();
            bus.write_u32(p + 20, c.5).unwrap();
            bus.write_u32(p + 24, c.6).unwrap();
            bus.write_u32(p + 28, c.7).unwrap();
        }
        bus.write_u64(0x200b_0020, ring_pa).unwrap();
        bus.write_u32(0x200b_0024, cmds.len() as u32).unwrap();
        bus.write_u32(0x200b_0028, 1).unwrap();
        assert_eq!(bus.apple_gfx.frame_count, 1);
        assert!(bus.framebuffer.non_zero_pixels() > 100_000);
        assert_eq!(bus.framebuffer.cursor_state(), (250, 220, true));

        // VirtIO-GPU v2 GET_DISPLAY_INFO + RESOURCE_CREATE_2D + ATTACH + SET_SCANOUT + TRANSFER + FLUSH
        assert_eq!(bus.read_u32(0x200a_1008).unwrap(), 16);
        let desc = 0x8003_0000u64;
        let avail = 0x8003_1000u64;
        let used = 0x8003_2000u64;
        let req_pa = 0x8003_3000u64;
        let resp_pa = 0x8003_4000u64;
        let backing_pa = 0x8004_0000u64;
        bus.write_u32(backing_pa, 0x00AA_BBCC).unwrap();

        bus.write_u32(0x200a_1030, 0).unwrap();
        bus.write_u32(0x200a_1038, 16).unwrap();
        bus.write_u32(0x200a_1080, desc as u32).unwrap();
        bus.write_u32(0x200a_1090, avail as u32).unwrap();
        bus.write_u32(0x200a_10a0, used as u32).unwrap();
        bus.write_u32(0x200a_1044, 1).unwrap();

        let push_cmd = |bus: &mut Vma2Bus, slot: u16, req_bytes: &[u8]| {
            bus.write_bytes(req_pa, req_bytes).unwrap();
            // desc 0: req
            bus.write_u64(desc, req_pa).unwrap();
            bus.write_u32(desc + 8, req_bytes.len() as u32).unwrap();
            bus.write_u16(desc + 12, 1).unwrap(); // NEXT
            bus.write_u16(desc + 14, 1).unwrap();
            // desc 1: resp
            bus.write_u64(desc + 16, resp_pa).unwrap();
            bus.write_u32(desc + 24, 512).unwrap();
            bus.write_u16(desc + 28, 2).unwrap(); // WRITE
            bus.write_u16(desc + 30, 0).unwrap();
            bus.write_u16(avail + 4 + (slot as u64 % 16) * 2, 0).unwrap();
            bus.write_u16(avail + 2, slot + 1).unwrap();
            bus.write_u32(0x200a_1050, 0).unwrap();
        };

        // 1) RESOURCE_CREATE_2D (id=1, fmt=1, w=16, h=16)
        let mut r1 = vec![0u8; 40];
        r1[0..4].copy_from_slice(&0x0101u32.to_le_bytes());
        r1[24..28].copy_from_slice(&1u32.to_le_bytes());
        r1[28..32].copy_from_slice(&1u32.to_le_bytes());
        r1[32..36].copy_from_slice(&16u32.to_le_bytes());
        r1[36..40].copy_from_slice(&16u32.to_le_bytes());
        push_cmd(&mut bus, 0, &r1);

        // 2) ATTACH_BACKING (id=1, nr=1, addr=backing_pa, len=1024)
        let mut r2 = vec![0u8; 48];
        r2[0..4].copy_from_slice(&0x0106u32.to_le_bytes());
        r2[24..28].copy_from_slice(&1u32.to_le_bytes());
        r2[28..32].copy_from_slice(&1u32.to_le_bytes());
        r2[32..40].copy_from_slice(&backing_pa.to_le_bytes());
        r2[40..44].copy_from_slice(&1024u32.to_le_bytes());
        push_cmd(&mut bus, 1, &r2);

        // 3) TRANSFER_TO_HOST_2D (w=16,h=16, id=1)
        let mut r3 = vec![0u8; 56];
        r3[0..4].copy_from_slice(&0x0105u32.to_le_bytes());
        r3[32..36].copy_from_slice(&16u32.to_le_bytes());
        r3[36..40].copy_from_slice(&16u32.to_le_bytes());
        r3[48..52].copy_from_slice(&1u32.to_le_bytes());
        push_cmd(&mut bus, 2, &r3);

        // 4) RESOURCE_FLUSH (id=1)
        let mut r4 = vec![0u8; 48];
        r4[0..4].copy_from_slice(&0x0104u32.to_le_bytes());
        r4[32..36].copy_from_slice(&16u32.to_le_bytes());
        r4[36..40].copy_from_slice(&16u32.to_le_bytes());
        r4[40..44].copy_from_slice(&1u32.to_le_bytes());
        push_cmd(&mut bus, 3, &r4);

        assert_eq!(bus.virtio_gpu.flushes, 1);
        assert_eq!(bus.framebuffer.read_bytes(0, 4), 0x00AA_BBCC);
    }
}
