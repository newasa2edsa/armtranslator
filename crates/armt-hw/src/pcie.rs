use crate::virtio::VirtioBlk;
use armt_core::Result;

const VIRTIO_STATUS_ACKNOWLEDGE: u8 = 1;
const VIRTIO_STATUS_DRIVER: u8 = 2;
const VIRTIO_STATUS_DRIVER_OK: u8 = 4;
const VIRTIO_STATUS_FEATURES_OK: u8 = 8;

const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;
const VIRTIO_BLK_T_FLUSH: u32 = 4;
const VIRTIO_BLK_T_GET_ID: u32 = 8;

const VIRTIO_BLK_S_OK: u8 = 0;
const VIRTIO_BLK_S_IOERR: u8 = 1;
const VIRTIO_BLK_S_UNSUPP: u8 = 2;

const BAR0_SIZE: u64 = 0x4000;

#[derive(Debug, Clone, Copy, Default)]
struct MsixEntry {
    addr: u64,
    data: u32,
    ctrl: u32,
}

/// Virtual PCIe Root Complex (`pcie,vmapple1`) exposing:
/// - `00:00.0`: Apple/QEMU Generic PCIe Host Bridge (`0x1b36:0x0008`)
/// - `00:01.0`: Modern VirtIO-Blk PCI device (`0x1af4:0x1042`) with MSI-X
pub struct PcieRootComplex {
    host_cmd: u16,
    blk_cmd: u16,
    blk_bar0: u64,
    bar0_sizing_lo: bool,
    bar0_sizing_hi: bool,
    msix_ctrl: u16,
    msix_table: [MsixEntry; 2],
    msix_pba: u64,

    // VirtIO Modern PCI Common Config state
    device_feature_sel: u32,
    driver_feature_sel: u32,
    driver_features: u64,
    config_msix_vector: u16,
    device_status: u8,
    queue_select: u16,
    queue_size: u16,
    queue_msix_vector: u16,
    queue_enable: u16,
    queue_desc: u64,
    queue_avail: u64,
    queue_used: u64,
    last_avail_idx: u16,
    isr_status: u8,
    pub io_requests: u64,
    pub last_req_type: u32,
    pub last_req_sector: u64,
    pub last_req_bytes: u32,
}

impl Default for PcieRootComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl PcieRootComplex {
    pub fn new() -> Self {
        Self {
            host_cmd: 0x0006,
            blk_cmd: 0x0006,
            blk_bar0: 0x5000_0000,
            bar0_sizing_lo: false,
            bar0_sizing_hi: false,
            msix_ctrl: 0x0001, // 2 table entries (N-1 = 1), disabled initially
            msix_table: [
                MsixEntry { addr: 0, data: 0, ctrl: 1 },
                MsixEntry { addr: 0, data: 0, ctrl: 1 },
            ],
            msix_pba: 0,
            device_feature_sel: 0,
            driver_feature_sel: 0,
            driver_features: 0,
            config_msix_vector: 0xFFFF,
            device_status: 0,
            queue_select: 0,
            queue_size: 128,
            queue_msix_vector: 0xFFFF,
            queue_enable: 0,
            queue_desc: 0,
            queue_avail: 0,
            queue_used: 0,
            last_avail_idx: 0,
            isr_status: 0,
            io_requests: 0,
            last_req_type: 0,
            last_req_sector: 0,
            last_req_bytes: 0,
        }
    }

    pub fn debug_summary(&self) -> String {
        format!(
            "bar0=0x{:x} blk_cmd=0x{:x} status=0x{:x} drv_feat=0x{:x} q_en={} q_size={} q_vec={} msix_ctrl=0x{:x} msi_addr=0x{:x} msi_data={} io_reqs={} last=(type={},sec={},bytes={})",
            self.blk_bar0,
            self.blk_cmd,
            self.device_status,
            self.driver_features,
            self.queue_enable,
            self.queue_size,
            self.queue_msix_vector,
            self.msix_ctrl,
            self.msix_table[0].addr,
            self.msix_table[0].data,
            self.io_requests,
            self.last_req_type,
            self.last_req_sector,
            self.last_req_bytes,
        )
    }

    #[inline]
    fn decode_ecam(offset: u64) -> (u8, u8, u8, u16) {
        let bus = ((offset >> 20) & 0xFF) as u8;
        let dev = ((offset >> 15) & 0x1F) as u8;
        let func = ((offset >> 12) & 0x7) as u8;
        let reg = (offset & 0xFFF) as u16;
        (bus, dev, func, reg)
    }

    fn read_cfg_u32(&self, bus: u8, dev: u8, func: u8, reg: u16) -> u32 {
        if bus != 0 || func != 0 {
            return 0xFFFF_FFFF;
        }
        match dev {
            0 => match reg & !3 {
                0x00 => 0x0008_1B36, // Red Hat/QEMU Generic PCIe Host Bridge
                0x04 => self.host_cmd as u32,
                0x08 => 0x0600_0000, // Host Bridge
                0x0C => 0x0000_0000, // Single function type 0
                _ => 0,
            },
            1 => self.read_blk_cfg_u32(reg & !3),
            _ => 0xFFFF_FFFF,
        }
    }

    fn read_blk_cfg_u32(&self, reg: u16) -> u32 {
        match reg {
            0x00 => 0x1042_1AF4, // VirtIO Block PCI Modern (1AF4:1042)
            0x04 => (0x0010u32 << 16) | (self.blk_cmd as u32), // Status: Capabilities List
            0x08 => 0x0180_0001, // Mass storage (other), Rev 1 (VirtIO 1.0)
            0x0C => 0x0000_0000, // Type 0 header
            0x10 => {
                if self.bar0_sizing_lo {
                    (!(BAR0_SIZE as u32 - 1)) | 0x4
                } else {
                    (self.blk_bar0 as u32 & !0xF) | 0x4 // 64-bit MMIO BAR
                }
            }
            0x14 => {
                if self.bar0_sizing_hi {
                    0xFFFF_FFFF
                } else {
                    (self.blk_bar0 >> 32) as u32
                }
            }
            0x2C => 0x0002_1AF4, // Subsys Vendor 1AF4, Subsys Device 0002 (VirtIO-Blk)
            0x34 => 0x0000_0040, // Capabilities pointer
            0x3C => 0x0000_01FF, // Interrupt Pin INTA# (1), Line 0xFF

            // Cap 1 @ 0x40: VIRTIO_PCI_CAP_COMMON_CFG (type=1, bar=0, off=0x0000, len=0x1000)
            0x40 => 0x0110_5009,
            0x44 => 0x0000_0000,
            0x48 => 0x0000_0000,
            0x4C => 0x0000_1000,

            // Cap 2 @ 0x50: VIRTIO_PCI_CAP_NOTIFY_CFG (type=2, bar=0, off=0x1000, len=0x1000, mult=4)
            0x50 => 0x0214_6409,
            0x54 => 0x0000_0000,
            0x58 => 0x0000_1000,
            0x5C => 0x0000_1000,
            0x60 => 0x0000_0004,

            // Cap 3 @ 0x64: VIRTIO_PCI_CAP_ISR_CFG (type=3, bar=0, off=0x2000, len=0x1000)
            0x64 => 0x0310_7409,
            0x68 => 0x0000_0000,
            0x6C => 0x0000_2000,
            0x70 => 0x0000_1000,

            // Cap 4 @ 0x74: VIRTIO_PCI_CAP_DEVICE_CFG (type=4, bar=0, off=0x3000, len=0x1000)
            0x74 => 0x0410_8409,
            0x78 => 0x0000_0000,
            0x7C => 0x0000_3000,
            0x80 => 0x0000_1000,

            // Cap 5 @ 0x84: PCI MSI-X (id=0x11, next=0x00, table=0x3800, pba=0x3C00)
            0x84 => ((self.msix_ctrl as u32) << 16) | 0x0011,
            0x88 => 0x0000_3800,
            0x8C => 0x0000_3C00,
            _ => 0,
        }
    }

    pub fn read_ecam(&self, offset: u64, size: u8) -> Result<u64> {
        let (bus, dev, func, reg) = Self::decode_ecam(offset);
        let aligned = reg & !3;
        let shift = ((reg & 3) * 8) as u32;
        let lo = self.read_cfg_u32(bus, dev, func, aligned);
        let val = if size == 8 && (reg & 7) == 0 {
            let hi = self.read_cfg_u32(bus, dev, func, aligned + 4);
            ((hi as u64) << 32) | (lo as u64)
        } else {
            let mask = match size {
                1 => 0xFFu64,
                2 => 0xFFFFu64,
                4 => 0xFFFF_FFFFu64,
                _ => u64::MAX,
            };
            ((lo as u64) >> shift) & mask
        };
        Ok(val)
    }

    pub fn write_ecam(&mut self, offset: u64, size: u8, val: u64) -> Result<()> {
        let (bus, dev, func, reg) = Self::decode_ecam(offset);
        if bus != 0 || func != 0 {
            return Ok(());
        }
        if dev == 0 {
            if (reg & !3) == 0x04 {
                self.host_cmd = val as u16;
            }
            return Ok(());
        }
        if dev != 1 {
            return Ok(());
        }
        match reg & !3 {
            0x04 => {
                if (reg & 3) == 0 {
                    self.blk_cmd = val as u16;
                }
            }
            0x10 => {
                let v = val as u32;
                if v == 0xFFFF_FFFF {
                    self.bar0_sizing_lo = true;
                } else {
                    self.bar0_sizing_lo = false;
                    let lo = (v as u64) & !(BAR0_SIZE - 1);
                    self.blk_bar0 = (self.blk_bar0 & 0xFFFF_FFFF_0000_0000) | lo;
                }
                if size == 8 {
                    let hi = (val >> 32) as u32;
                    if hi == 0xFFFF_FFFF {
                        self.bar0_sizing_hi = true;
                    } else {
                        self.bar0_sizing_hi = false;
                        self.blk_bar0 =
                            (self.blk_bar0 & 0x0000_0000_FFFF_FFFF) | ((hi as u64) << 32);
                    }
                }
            }
            0x14 => {
                let v = val as u32;
                if v == 0xFFFF_FFFF {
                    self.bar0_sizing_hi = true;
                } else {
                    self.bar0_sizing_hi = false;
                    self.blk_bar0 = (self.blk_bar0 & 0x0000_0000_FFFF_FFFF) | ((v as u64) << 32);
                }
            }
            0x84 => {
                if size >= 2 && (reg & 3) == 2 {
                    self.msix_ctrl = (self.msix_ctrl & 0x07FF) | ((val as u16) & 0xC000);
                } else if size == 4 && (reg & 3) == 0 {
                    self.msix_ctrl = (self.msix_ctrl & 0x07FF) | (((val >> 16) as u16) & 0xC000);
                }
            }
            _ => {}
        }
        Ok(())
    }

    #[inline]
    pub fn contains_bar(&self, addr: u64) -> bool {
        (self.blk_cmd & 2) != 0
            && self.blk_bar0 != 0
            && addr >= self.blk_bar0
            && addr < self.blk_bar0 + BAR0_SIZE
    }

    pub fn read_bar(&mut self, addr: u64, size: u8, blk: &VirtioBlk) -> Result<u64> {
        let off = addr - self.blk_bar0;
        if (0x3000..0x3040).contains(&off) {
            let mut cfg = [0u8; 64];
            cfg[0x00..0x08].copy_from_slice(&blk.capacity_sectors().to_le_bytes());
            cfg[0x08..0x0C].copy_from_slice(&0x0010_0000u32.to_le_bytes()); // size_max = 1 MiB
            cfg[0x0C..0x10].copy_from_slice(&126u32.to_le_bytes()); // seg_max
            cfg[0x14..0x18].copy_from_slice(&512u32.to_le_bytes()); // blk_size
            cfg[0x1A..0x1C].copy_from_slice(&1u16.to_le_bytes()); // min_io_size
            cfg[0x1C..0x20].copy_from_slice(&8u32.to_le_bytes()); // opt_io_size
            cfg[0x20] = 1; // writeback
            let start = (off - 0x3000) as usize;
            let n = (size as usize).min(8).min(cfg.len() - start);
            let mut buf = [0u8; 8];
            buf[..n].copy_from_slice(&cfg[start..start + n]);
            return Ok(u64::from_le_bytes(buf));
        }
        let val = match off {
            // Common config (0x0000..0x003F)
            0x0000 => self.device_feature_sel as u64,
            0x0004 => match self.device_feature_sel {
                // SIZE_MAX (1) | SEG_MAX (2) | BLK_SIZE (6) | FLUSH (9)
                0 => (1 << 1) | (1 << 2) | (1 << 6) | (1 << 9),
                1 => 1, // VIRTIO_F_VERSION_1 (bit 32)
                _ => 0,
            },
            0x0008 => self.driver_feature_sel as u64,
            0x000C => (self.driver_features >> ((self.driver_feature_sel.min(1)) * 32)) & 0xFFFF_FFFF,
            0x0010 => self.config_msix_vector as u64,
            0x0012 => 1, // num_queues = 1
            0x0014 => {
                if size == 1 {
                    self.device_status as u64
                } else {
                    (self.device_status as u64) | ((self.queue_select as u64) << 16)
                }
            }
            0x0015 => 0, // config_generation
            0x0016 => self.queue_select as u64,
            0x0018 => {
                if self.queue_select == 0 {
                    self.queue_size as u64
                } else {
                    0
                }
            }
            0x001A => self.queue_msix_vector as u64,
            0x001C => self.queue_enable as u64,
            0x001E => 0, // queue_notify_off
            0x0020 => self.queue_desc,
            0x0024 => self.queue_desc >> 32,
            0x0028 => self.queue_avail,
            0x002C => self.queue_avail >> 32,
            0x0030 => self.queue_used,
            0x0034 => self.queue_used >> 32,

            // ISR (0x2000) - read clears
            0x2000 => {
                let s = self.isr_status as u64;
                self.isr_status = 0;
                s
            }

            // MSI-X Table (0x3800..0x381F)
            0x3800..=0x381F => {
                let idx = ((off - 0x3800) / 16) as usize;
                let field = (off - 0x3800) % 16;
                let e = &self.msix_table[idx];
                match field {
                    0x0 => if size == 8 { e.addr } else { e.addr & 0xFFFF_FFFF },
                    0x4 => e.addr >> 32,
                    0x8 => e.data as u64,
                    0xC => e.ctrl as u64,
                    _ => 0,
                }
            }
            0x3C00 => self.msix_pba,
            _ => 0,
        };
        Ok(val)
    }

    /// Writes to BAR0; returns `Some((msi_addr, msi_data))` if queue processing completed and MSI-X should fire.
    pub fn write_bar(
        &mut self,
        addr: u64,
        size: u8,
        val: u64,
        blk: &mut VirtioBlk,
        ram: &mut [u8],
        ram_base: u64,
    ) -> Result<Option<(u64, u32)>> {
        let off = addr - self.blk_bar0;
        match off {
            0x0000 => self.device_feature_sel = val as u32,
            0x0008 => self.driver_feature_sel = val as u32,
            0x000C => {
                if self.driver_feature_sel == 0 {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
                } else if self.driver_feature_sel == 1 {
                    self.driver_features =
                        (self.driver_features & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
                }
            }
            0x0010 => self.config_msix_vector = val as u16,
            0x0014 => {
                let st = val as u8;
                if st == 0 {
                    self.reset_virtio();
                } else {
                    self.device_status = st
                        & (VIRTIO_STATUS_ACKNOWLEDGE
                            | VIRTIO_STATUS_DRIVER
                            | VIRTIO_STATUS_DRIVER_OK
                            | VIRTIO_STATUS_FEATURES_OK);
                }
                if size >= 4 {
                    self.queue_select = (val >> 16) as u16;
                }
            }
            0x0016 => self.queue_select = val as u16,
            0x0018 => {
                if self.queue_select == 0 {
                    self.queue_size = (val as u16).min(128);
                }
            }
            0x001A => {
                if self.queue_select == 0 {
                    self.queue_msix_vector = val as u16;
                }
            }
            0x001C => {
                if self.queue_select == 0 {
                    self.queue_enable = val as u16;
                }
            }
            0x0020 => {
                if size == 8 {
                    self.queue_desc = val;
                } else {
                    self.queue_desc = (self.queue_desc & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
                }
            }
            0x0024 => {
                self.queue_desc = (self.queue_desc & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            0x0028 => {
                if size == 8 {
                    self.queue_avail = val;
                } else {
                    self.queue_avail =
                        (self.queue_avail & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
                }
            }
            0x002C => {
                self.queue_avail =
                    (self.queue_avail & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            0x0030 => {
                if size == 8 {
                    self.queue_used = val;
                } else {
                    self.queue_used = (self.queue_used & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
                }
            }
            0x0034 => {
                self.queue_used = (self.queue_used & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }

            // Notify region (0x1000..0x1FFF)
            0x1000..=0x1FFF => {
                return Ok(self.process_blk_queue(blk, ram, ram_base));
            }

            // MSI-X Table (0x3800..0x381F)
            0x3800..=0x381F => {
                let idx = ((off - 0x3800) / 16) as usize;
                let field = (off - 0x3800) % 16;
                let e = &mut self.msix_table[idx];
                match field {
                    0x0 => {
                        if size == 8 {
                            e.addr = val;
                        } else {
                            e.addr = (e.addr & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
                        }
                    }
                    0x4 => {
                        e.addr = (e.addr & 0x0000_0000_FFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
                    }
                    0x8 => e.data = val as u32,
                    0xC => e.ctrl = val as u32,
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn reset_virtio(&mut self) {
        self.device_status = 0;
        self.driver_features = 0;
        self.queue_enable = 0;
        self.last_avail_idx = 0;
        self.isr_status = 0;
    }

    fn process_blk_queue(
        &mut self,
        blk: &mut VirtioBlk,
        ram: &mut [u8],
        ram_base: u64,
    ) -> Option<(u64, u32)> {
        if self.queue_enable == 0
            || self.queue_size == 0
            || self.queue_desc == 0
            || self.queue_avail == 0
            || self.queue_used == 0
        {
            return None;
        }
        let q_num = self.queue_size;
        let avail_idx = ram_read_u16(ram, ram_base, self.queue_avail + 2)?;
        let mut completed = false;

        while self.last_avail_idx != avail_idx {
            let slot = (self.last_avail_idx % q_num) as u64;
            let Some(head) = ram_read_u16(ram, ram_base, self.queue_avail + 4 + slot * 2) else {
                break;
            };
            self.last_avail_idx = self.last_avail_idx.wrapping_add(1);

            let (written, rtype, rsec, rbytes) = self.exec_blk_chain(blk, ram, ram_base, q_num, head);
            self.last_req_type = rtype;
            self.last_req_sector = rsec;
            self.last_req_bytes = rbytes;
            let used_idx = ram_read_u16(ram, ram_base, self.queue_used + 2).unwrap_or(0);
            let used_slot = (used_idx % q_num) as u64;
            let elem = self.queue_used + 4 + used_slot * 8;
            let _ = ram_write_u32(ram, ram_base, elem, head as u32);
            let _ = ram_write_u32(ram, ram_base, elem + 4, written);
            let _ = ram_write_u16(ram, ram_base, self.queue_used + 2, used_idx.wrapping_add(1));
            self.isr_status |= 1;
            self.io_requests += 1;
            completed = true;
        }

        if completed && (self.msix_ctrl & 0x8000) != 0 && (self.msix_ctrl & 0x4000) == 0 {
            let vec_idx = self.queue_msix_vector as usize;
            if vec_idx < self.msix_table.len() {
                let entry = self.msix_table[vec_idx];
                if (entry.ctrl & 1) == 0 && entry.addr != 0 {
                    return Some((entry.addr, entry.data));
                }
            }
        }
        None
    }

    fn exec_blk_chain(
        &self,
        blk: &mut VirtioBlk,
        ram: &mut [u8],
        ram_base: u64,
        q_num: u16,
        head: u16,
    ) -> (u32, u32, u64, u32) {
        let mut idx = head;
        let mut descs = Vec::new();
        for _ in 0..q_num {
            let base = self.queue_desc + (idx as u64) * 16;
            let Some(addr) = ram_read_u64(ram, ram_base, base) else {
                break;
            };
            let Some(len) = ram_read_u32(ram, ram_base, base + 8) else {
                break;
            };
            let Some(flags) = ram_read_u16(ram, ram_base, base + 12) else {
                break;
            };
            let Some(next) = ram_read_u16(ram, ram_base, base + 14) else {
                break;
            };
            descs.push((addr, len, flags));
            if (flags & 1) == 0 {
                break;
            }
            idx = next;
        }

        if descs.len() < 2 {
            return (0, 0xFFFF, 0, 0);
        }
        let (hdr_addr, hdr_len, _) = descs[0];
        if hdr_len < 16 {
            return (0, 0xFFFE, 0, 0);
        }
        let Some(req_type) = ram_read_u32(ram, ram_base, hdr_addr) else {
            return (0, 0xFFFD, 0, 0);
        };
        let Some(sector) = ram_read_u64(ram, ram_base, hdr_addr + 8) else {
            return (0, 0xFFFC, 0, 0);
        };

        let mut total_written = 0u32;
        let mut data_bytes = 0u32;
        let mut status = VIRTIO_BLK_S_OK;
        let data_descs = &descs[1..descs.len() - 1];
        let (status_addr, status_len, _) = descs[descs.len() - 1];

        match req_type {
            VIRTIO_BLK_T_IN => {
                let mut cur_lba = sector;
                for &(addr, len, flags) in data_descs {
                    if (flags & 2) == 0 {
                        continue;
                    }
                    data_bytes = data_bytes.saturating_add(len);
                    let mut buf = vec![0u8; len as usize];
                    if blk.read_sector(cur_lba, &mut buf).is_err() {
                        status = VIRTIO_BLK_S_IOERR;
                        break;
                    }
                    if !ram_write_slice(ram, ram_base, addr, &buf) {
                        status = VIRTIO_BLK_S_IOERR;
                        break;
                    }
                    total_written = total_written.saturating_add(len);
                    cur_lba = cur_lba.saturating_add((len as u64) / 512);
                }
            }
            VIRTIO_BLK_T_OUT => {
                let mut cur_lba = sector;
                for &(addr, len, _) in data_descs {
                    data_bytes = data_bytes.saturating_add(len);
                    let Some(buf) = ram_read_slice(ram, ram_base, addr, len as usize) else {
                        status = VIRTIO_BLK_S_IOERR;
                        break;
                    };
                    if blk.write_sector(cur_lba, buf).is_err() {
                        status = VIRTIO_BLK_S_IOERR;
                        break;
                    }
                    cur_lba = cur_lba.saturating_add((len as u64) / 512);
                }
            }
            VIRTIO_BLK_T_FLUSH => {}
            VIRTIO_BLK_T_GET_ID => {
                if let Some(&(addr, len, _)) = data_descs.first() {
                    let mut id = [0u8; 20];
                    let tag = b"armt-vma2-disk0";
                    id[..tag.len()].copy_from_slice(tag);
                    let n = (len as usize).min(20);
                    let _ = ram_write_slice(ram, ram_base, addr, &id[..n]);
                    total_written = total_written.saturating_add(n as u32);
                    data_bytes = n as u32;
                }
            }
            _ => {
                status = VIRTIO_BLK_S_UNSUPP;
            }
        }

        if status_len >= 1 && ram_write_slice(ram, ram_base, status_addr, &[status]) {
            total_written = total_written.saturating_add(1);
        }
        (total_written, req_type, sector, data_bytes)
    }
}

#[inline]
fn ram_off(ram: &[u8], ram_base: u64, addr: u64, len: usize) -> Option<usize> {
    if addr < ram_base {
        return None;
    }
    let off = (addr - ram_base) as usize;
    if off.checked_add(len)? <= ram.len() {
        Some(off)
    } else {
        None
    }
}

fn ram_read_u16(ram: &[u8], ram_base: u64, addr: u64) -> Option<u16> {
    let off = ram_off(ram, ram_base, addr, 2)?;
    Some(u16::from_le_bytes([ram[off], ram[off + 1]]))
}

fn ram_read_u32(ram: &[u8], ram_base: u64, addr: u64) -> Option<u32> {
    let off = ram_off(ram, ram_base, addr, 4)?;
    Some(u32::from_le_bytes(ram[off..off + 4].try_into().unwrap()))
}

fn ram_read_u64(ram: &[u8], ram_base: u64, addr: u64) -> Option<u64> {
    let off = ram_off(ram, ram_base, addr, 8)?;
    Some(u64::from_le_bytes(ram[off..off + 8].try_into().unwrap()))
}

fn ram_read_slice(ram: &[u8], ram_base: u64, addr: u64, len: usize) -> Option<&[u8]> {
    let off = ram_off(ram, ram_base, addr, len)?;
    Some(&ram[off..off + len])
}

fn ram_write_u16(ram: &mut [u8], ram_base: u64, addr: u64, val: u16) -> bool {
    let Some(off) = ram_off(ram, ram_base, addr, 2) else {
        return false;
    };
    ram[off..off + 2].copy_from_slice(&val.to_le_bytes());
    true
}

fn ram_write_u32(ram: &mut [u8], ram_base: u64, addr: u64, val: u32) -> bool {
    let Some(off) = ram_off(ram, ram_base, addr, 4) else {
        return false;
    };
    ram[off..off + 4].copy_from_slice(&val.to_le_bytes());
    true
}

fn ram_write_slice(ram: &mut [u8], ram_base: u64, addr: u64, src: &[u8]) -> bool {
    let Some(off) = ram_off(ram, ram_base, addr, src.len()) else {
        return false;
    };
    ram[off..off + src.len()].copy_from_slice(src);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcie_ecam_and_virtio_blk_msix() {
        let mut pcie = PcieRootComplex::new();
        let mut blk = VirtioBlk::new_memory_disk(64 * 1024);
        let mut sec0 = [0u8; 512];
        sec0[..8].copy_from_slice(b"EFI PART");
        blk.write_sector(1, &sec0).unwrap();

        assert_eq!(pcie.read_ecam(0x0000, 4).unwrap(), 0x0008_1B36);
        assert_eq!(pcie.read_ecam(0x8000, 4).unwrap(), 0x1042_1AF4);
        assert_eq!(pcie.read_ecam(0x10000, 4).unwrap(), 0xFFFF_FFFF);

        // BAR0 sizing probe
        pcie.write_ecam(0x8010, 4, 0xFFFF_FFFF).unwrap();
        assert_eq!(pcie.read_ecam(0x8010, 4).unwrap(), 0xFFFF_C004);
        pcie.write_ecam(0x8010, 4, 0x5000_0000).unwrap();

        // Configure MSI-X vector 0 -> 0x1fff_0040 data 128, enable MSI-X
        let ram_base = 0x8000_0000u64;
        let mut ram = vec![0u8; 0x10_0000];
        pcie.write_bar(0x5000_3800, 8, 0x1fff_0040, &mut blk, &mut ram, ram_base)
            .unwrap();
        pcie.write_bar(0x5000_3808, 4, 128, &mut blk, &mut ram, ram_base)
            .unwrap();
        pcie.write_bar(0x5000_380C, 4, 0, &mut blk, &mut ram, ram_base)
            .unwrap();
        pcie.write_ecam(0x8086, 2, 0x8000).unwrap();

        // Setup virtqueue 0
        let desc = 0x8001_0000u64;
        let avail = 0x8001_1000u64;
        let used = 0x8001_2000u64;
        let hdr = 0x8001_3000u64;
        let data = 0x8001_4000u64;
        let st = 0x8001_5000u64;
        ram_write_u32(&mut ram, ram_base, hdr, VIRTIO_BLK_T_IN);
        ram_write_u32(&mut ram, ram_base, hdr + 4, 0);
        ram_write_slice(&mut ram, ram_base, hdr + 8, &1u64.to_le_bytes());

        // desc 0 (hdr), desc 1 (data, WRITE), desc 2 (status, WRITE)
        ram_write_slice(&mut ram, ram_base, desc, &hdr.to_le_bytes());
        ram_write_u32(&mut ram, ram_base, desc + 8, 16);
        ram_write_u16(&mut ram, ram_base, desc + 12, 1);
        ram_write_u16(&mut ram, ram_base, desc + 14, 1);

        ram_write_slice(&mut ram, ram_base, desc + 16, &data.to_le_bytes());
        ram_write_u32(&mut ram, ram_base, desc + 24, 512);
        ram_write_u16(&mut ram, ram_base, desc + 28, 3);
        ram_write_u16(&mut ram, ram_base, desc + 30, 2);

        ram_write_slice(&mut ram, ram_base, desc + 32, &st.to_le_bytes());
        ram_write_u32(&mut ram, ram_base, desc + 40, 1);
        ram_write_u16(&mut ram, ram_base, desc + 44, 2);
        ram_write_u16(&mut ram, ram_base, desc + 46, 0);

        ram_write_u16(&mut ram, ram_base, avail + 4, 0);
        ram_write_u16(&mut ram, ram_base, avail + 2, 1);

        pcie.write_bar(0x5000_001A, 2, 0, &mut blk, &mut ram, ram_base).unwrap();
        pcie.write_bar(0x5000_0020, 8, desc, &mut blk, &mut ram, ram_base).unwrap();
        pcie.write_bar(0x5000_0028, 8, avail, &mut blk, &mut ram, ram_base).unwrap();
        pcie.write_bar(0x5000_0030, 8, used, &mut blk, &mut ram, ram_base).unwrap();
        pcie.write_bar(0x5000_001C, 2, 1, &mut blk, &mut ram, ram_base).unwrap();

        let msi = pcie
            .write_bar(0x5000_1000, 2, 0, &mut blk, &mut ram, ram_base)
            .unwrap();
        assert_eq!(msi, Some((0x1fff_0040, 128)));
        assert_eq!(&ram[0x1_4000..0x1_4008], b"EFI PART");
        assert_eq!(ram[0x1_5000], VIRTIO_BLK_S_OK);
    }
}
