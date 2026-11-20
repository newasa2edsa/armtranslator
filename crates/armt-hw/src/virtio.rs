use armt_core::{ArmtError, MmioDevice, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// VirtIO MMIO v2 block device (100% sandboxed in-memory or explicit file disk image).
pub struct VirtioBlk {
    sectors: Vec<u8>,
    status: u32,
    driver_features: u32,
    queue_sel: u32,
    queue_num: u32,
}

impl VirtioBlk {
    pub fn new_memory_disk(size_bytes: usize) -> Self {
        Self {
            sectors: vec![0u8; size_bytes],
            status: 0,
            driver_features: 0,
            queue_sel: 0,
            queue_num: 128,
        }
    }

    pub fn read_sector(&self, lba: u64, dst: &mut [u8]) -> Result<()> {
        let off = (lba as usize).saturating_mul(512);
        if off + dst.len() > self.sectors.len() {
            return Err(ArmtError::Device("VirtioBlk read out of bounds".into()));
        }
        dst.copy_from_slice(&self.sectors[off..off + dst.len()]);
        Ok(())
    }

    pub fn write_sector(&mut self, lba: u64, src: &[u8]) -> Result<()> {
        let off = (lba as usize).saturating_mul(512);
        if off + src.len() > self.sectors.len() {
            return Err(ArmtError::Device("VirtioBlk write out of bounds".into()));
        }
        self.sectors[off..off + src.len()].copy_from_slice(src);
        Ok(())
    }

    pub fn capacity_sectors(&self) -> u64 {
        (self.sectors.len() / 512) as u64
    }
}

impl MmioDevice for VirtioBlk {
    fn name(&self) -> &str {
        "virtio-blk"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        let val = match offset {
            0x000 => 0x7472_6976, // "virt"
            0x004 => 2,           // VirtIO MMIO v2
            0x008 => 2,           // Block device
            0x00C => 0x1AF4,      // Vendor ID
            0x010 => 0,           // Device features
            0x034 => 128,         // QueueNumMax
            0x070 => self.status as u64,
            0x0FC => 1,           // Config generation
            0x100 => self.capacity_sectors() & 0xFFFF_FFFF,
            0x104 => self.capacity_sectors() >> 32,
            _ => 0,
        };
        Ok(val)
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) -> Result<()> {
        match offset {
            0x020 => self.driver_features = val as u32,
            0x030 => self.queue_sel = val as u32,
            0x038 => self.queue_num = val as u32,
            0x070 => {
                if val == 0 {
                    self.status = 0;
                } else {
                    self.status = val as u32;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputEvent {
    pub ev_type: u16,
    pub code: u16,
    pub value: i32,
}

/// VirtIO MMIO v2 keyboard/mouse input queue.
pub struct VirtioInput {
    events: VecDeque<InputEvent>,
    status: u32,
}

impl Default for VirtioInput {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtioInput {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            status: 0,
        }
    }

    pub fn push_key(&mut self, code: u16, pressed: bool) {
        self.events.push_back(InputEvent {
            ev_type: 1,
            code,
            value: if pressed { 1 } else { 0 },
        });
    }

    pub fn pop_event(&mut self) -> Option<InputEvent> {
        self.events.pop_front()
    }
}

impl MmioDevice for VirtioInput {
    fn name(&self) -> &str {
        "virtio-input"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        Ok(match offset {
            0x000 => 0x7472_6976,
            0x004 => 2,
            0x008 => 18, // Input device
            0x00C => 0x1AF4,
            0x070 => self.status as u64,
            _ => 0,
        })
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) -> Result<()> {
        if offset == 0x070 {
            self.status = val as u32;
        }
        Ok(())
    }
}

/// Linear BGRA (`0x00RRGGBB` in `u32` for `minifb`) framebuffer shared with GUI window.
#[derive(Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<Mutex<Vec<u32>>>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: Arc::new(Mutex::new(vec![0u32; (width as usize) * (height as usize)])),
        }
    }

    pub fn read_bytes(&self, offset: u64, size: u8) -> u64 {
        let guard = self.pixels.lock().unwrap();
        let pix_idx = (offset / 4) as usize;
        if pix_idx >= guard.len() {
            return 0;
        }
        let word = guard[pix_idx];
        let shift = ((offset % 4) * 8) as u32;
        let mask = match size {
            1 => 0xFFu64,
            2 => 0xFFFFu64,
            4 => 0xFFFF_FFFFu64,
            _ => {
                let hi = guard.get(pix_idx + 1).copied().unwrap_or(0) as u64;
                return (word as u64) | (hi << 32);
            }
        };
        ((word as u64) >> shift) & mask
    }

    pub fn write_bytes(&self, offset: u64, size: u8, val: u64) {
        let mut guard = self.pixels.lock().unwrap();
        let pix_idx = (offset / 4) as usize;
        if pix_idx >= guard.len() {
            return;
        }
        if size == 8 && (offset % 4) == 0 {
            guard[pix_idx] = val as u32;
            if pix_idx + 1 < guard.len() {
                guard[pix_idx + 1] = (val >> 32) as u32;
            }
            return;
        }
        if size == 4 && (offset % 4) == 0 {
            guard[pix_idx] = val as u32;
            return;
        }
        let shift = ((offset % 4) * 8) as u32;
        let mask = if size == 1 { 0xFFu32 } else { 0xFFFFu32 } << shift;
        let mut cur = guard[pix_idx];
        cur = (cur & !mask) | (((val as u32) << shift) & mask);
        guard[pix_idx] = cur;
    }

    pub fn clear(&self, rgb: u32) {
        let mut guard = self.pixels.lock().unwrap();
        guard.fill(rgb);
    }

    pub fn fill_rect(&self, x: u32, y: u32, w: u32, h: u32, rgb: u32) {
        let mut guard = self.pixels.lock().unwrap();
        let max_x = (x + w).min(self.width);
        let max_y = (y + h).min(self.height);
        for yy in y..max_y {
            let row = (yy as usize) * (self.width as usize);
            for xx in x..max_x {
                guard[row + xx as usize] = rgb;
            }
        }
    }

    pub fn non_zero_pixels(&self) -> usize {
        self.pixels
            .lock()
            .unwrap()
            .iter()
            .filter(|&&p| p != 0)
            .count()
    }
}
