use armt_core::AdtNode;

/// Physical memory map for the sandboxed `VirtualMac2,1` (`vma2`) virtual machine.
#[derive(Debug, Clone)]
pub struct Vma2MemMap {
    pub ram_base: u64,
    pub ram_size: u64,
    pub gicd_base: u64,
    pub gicd_size: u64,
    pub gicr_base: u64,
    pub gicr_size: u64,
    pub gic_msi_base: u64,
    pub gic_msi_size: u64,
    pub pvpanic_base: u64,
    pub pvpanic_size: u64,
    pub uart0_base: u64,
    pub uart0_size: u64,
    pub bdif_base: u64,
    pub bdif_size: u64,
    pub aes1_base: u64,
    pub aes1_size: u64,
    pub aes2_base: u64,
    pub aes2_size: u64,
    pub cfg_base: u64,
    pub cfg_size: u64,
    pub virtio_blk_base: u64,
    pub virtio_blk_size: u64,
    pub virtio_gpu_base: u64,
    pub virtio_gpu_size: u64,
    pub virtio_input_base: u64,
    pub virtio_input_size: u64,
    pub apple_gfx_base: u64,
    pub apple_gfx_size: u64,
    pub pcie_ecam_base: u64,
    pub pcie_ecam_size: u64,
    pub framebuffer_base: u64,
    pub framebuffer_width: u32,
    pub framebuffer_height: u32,
}

impl Vma2MemMap {
    pub fn with_ram_mb(ram_mb: u64) -> Self {
        Self {
            ram_base: 0x8000_0000,
            ram_size: ram_mb.max(16) * 1024 * 1024,
            gicd_base: 0x1000_0000,
            gicd_size: 0x1_0000,
            gicr_base: 0x1001_0000,
            gicr_size: 0x40_0000,
            gic_msi_base: 0x1fff_0000,
            gic_msi_size: 0x1_0000,
            pvpanic_base: 0x2000_0000,
            pvpanic_size: 0x1000,
            uart0_base: 0x2001_0000,
            uart0_size: 0x1000,
            bdif_base: 0x2005_0000,
            bdif_size: 0x1_0000,
            aes1_base: 0x2006_0000,
            aes1_size: 0x4000,
            aes2_base: 0x2007_0000,
            aes2_size: 0x4000,
            cfg_base: 0x2008_0000,
            cfg_size: 0x1000,
            virtio_blk_base: 0x200a_0000,
            virtio_blk_size: 0x1000,
            virtio_gpu_base: 0x200a_1000,
            virtio_gpu_size: 0x1000,
            virtio_input_base: 0x200a_2000,
            virtio_input_size: 0x1000,
            apple_gfx_base: 0x200b_0000,
            apple_gfx_size: 0x1_0000,
            pcie_ecam_base: 0x4000_0000,
            pcie_ecam_size: 0x1000_0000,
            framebuffer_base: 0x6000_0000,
            framebuffer_width: 1280,
            framebuffer_height: 800,
        }
    }

    /// Overlay non-overlapping device MMIO addresses discovered from a real Apple `vma2` ADT.
    pub fn from_adt(adt: &AdtNode, ram_mb: u64) -> Self {
        let mut map = Self::with_ram_mb(ram_mb);
        let translate_arm_io = |addr: u64| -> u64 {
            if let Some(arm_io) = adt.find_path("arm-io") {
                if let Some(ranges) = arm_io.prop("ranges") {
                    for chunk in ranges.chunks_exact(24) {
                        let child = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
                        let parent = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
                        let size = u64::from_le_bytes(chunk[16..24].try_into().unwrap());
                        if addr >= child && addr < child.saturating_add(size) {
                            return parent.wrapping_add(addr - child);
                        }
                    }
                }
            }
            addr
        };
        if let Some(uart) = adt.find_path("arm-io/uart0") {
            if let Some(&(raw_base, size)) = uart.prop_regs().first() {
                let base = translate_arm_io(raw_base);
                if base >= 0x2000_0000 && base < map.ram_base {
                    map.uart0_base = base;
                    if size >= 0x1000 {
                        map.uart0_size = size;
                    }
                }
            }
        }
        if let Some(gic) = adt.find_path("arm-io/gic").or_else(|| adt.find_path("arm-io/aic")) {
            let regs = gic.prop_regs();
            if let Some(&(raw_base, size)) = regs.first() {
                let base = translate_arm_io(raw_base);
                if base >= 0x1000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.gicd_base = base;
                    if size >= 0x1000 {
                        map.gicd_size = size;
                    }
                }
            }
            if let Some(&(raw_base, size)) = regs.get(1) {
                let base = translate_arm_io(raw_base);
                if base >= 0x1000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.gicr_base = base;
                    if size >= 0x1000 {
                        map.gicr_size = size;
                    }
                }
            }
            if let Some(&(raw_base, size)) = regs.get(2) {
                let base = translate_arm_io(raw_base);
                if base >= 0x1000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.gic_msi_base = base;
                    if size >= 0x1000 {
                        map.gic_msi_size = size;
                    }
                }
            }
        }
        if let Some(aes) = adt.find_path("arm-io/aes") {
            let regs = aes.prop_regs();
            if let Some(&(raw_base, size)) = regs.first() {
                let base = translate_arm_io(raw_base);
                if base >= 0x2000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.aes1_base = base;
                    if size >= 0x1000 {
                        map.aes1_size = size;
                    }
                }
            }
            if let Some(&(raw_base, size)) = regs.get(1) {
                let base = translate_arm_io(raw_base);
                if base >= 0x2000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.aes2_base = base;
                    if size >= 0x1000 {
                        map.aes2_size = size;
                    }
                }
            }
        }
        if let Some(pvpanic) = adt.find_path("arm-io/pvpanic") {
            if let Some(&(raw_base, size)) = pvpanic.prop_regs().first() {
                let base = translate_arm_io(raw_base);
                if base >= 0x2000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.pvpanic_base = base;
                    if size >= 0x1000 {
                        map.pvpanic_size = size;
                    }
                }
            }
        }
        if let Some(gfx) = adt.find_path("arm-io/gfx") {
            if let Some(&(raw_base, size)) = gfx.prop_regs().first() {
                let base = translate_arm_io(raw_base);
                if base >= 0x2000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.apple_gfx_base = base;
                    if size >= 0x1000 {
                        map.apple_gfx_size = size;
                    }
                }
            }
        }
        if let Some(pcie) = adt.find_path("arm-io/pcie") {
            if let Some(&(raw_base, size)) = pcie.prop_regs().first() {
                let base = translate_arm_io(raw_base);
                if base >= 0x2000_0000 && base < map.ram_base && base != map.uart0_base {
                    map.pcie_ecam_base = base;
                    if size >= 0x1000 {
                        map.pcie_ecam_size = size;
                    }
                }
            }
        }
        map
    }

    pub fn framebuffer_size_bytes(&self) -> u64 {
        (self.framebuffer_width as u64) * (self.framebuffer_height as u64) * 4
    }
}
