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
            gicr_size: 0x4_0000,
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
            framebuffer_base: 0x6000_0000,
            framebuffer_width: 1024,
            framebuffer_height: 768,
        }
    }

    /// Overlay non-overlapping device MMIO addresses discovered from a real Apple `vma2` ADT.
    pub fn from_adt(adt: &AdtNode, ram_mb: u64) -> Self {
        let mut map = Self::with_ram_mb(ram_mb);
        if let Some(uart) = adt.find_path("arm-io/uart0") {
            if let Some(&(base, size)) = uart.prop_regs().first() {
                if base != 0 && base < map.ram_base {
                    map.uart0_base = base;
                    if size >= 0x1000 {
                        map.uart0_size = size;
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
