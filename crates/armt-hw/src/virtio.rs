use crate::dmg::UdifDmg;
use armt_core::{ArmtError, MmioDevice, Result};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

enum DiskBackend {
    Memory(Vec<u8>),
    File {
        file: File,
        total_sectors: u64,
        read_only: bool,
        cow: HashMap<u64, [u8; 512]>,
    },
    UdifDmg {
        dmg: UdifDmg,
        cow: HashMap<u64, [u8; 512]>,
    },
}

/// VirtIO MMIO/PCI block device supporting in-memory disks, streaming `.iso`/`.img`/`.raw` files,
/// and UDIF (`koly`) `.dmg` images with an in-memory copy-on-write overlay for read-only media.
pub struct VirtioBlk {
    backend: DiskBackend,
    status: u32,
    driver_features: u32,
    queue_sel: u32,
    queue_num: u32,
}

impl VirtioBlk {
    pub fn new_memory_disk(size_bytes: usize) -> Self {
        Self {
            backend: DiskBackend::Memory(vec![0u8; size_bytes]),
            status: 0,
            driver_features: 0,
            queue_sel: 0,
            queue_num: 128,
        }
    }

    /// Open a real host disk image (`.iso`, `.img`, `.raw`, `.cdr`, or UDIF `.dmg`).
    /// `.iso`/`.cdr`/`.dmg` or `read_only=true` images use an in-memory CoW sector overlay
    /// so guest writes never mutate the source image.
    pub fn open_disk_image(path: &Path, read_only: bool) -> Result<Self> {
        let ext_ro = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_ascii_lowercase().as_str(), "iso" | "cdr" | "dmg"))
            .unwrap_or(false);
        let effective_ro = read_only || ext_ro;
        let file = OpenOptions::new()
            .read(true)
            .write(!effective_ro)
            .open(path)?;
        let file_len = file.metadata()?.len();

        if let Some(dmg) = UdifDmg::try_open(file.try_clone()?, file_len)? {
            return Ok(Self {
                backend: DiskBackend::UdifDmg {
                    dmg,
                    cow: HashMap::new(),
                },
                status: 0,
                driver_features: 0,
                queue_sel: 0,
                queue_num: 128,
            });
        }

        let total_sectors = file_len / 512;
        if total_sectors == 0 {
            return Err(ArmtError::Device(format!(
                "Disk image '{}' is smaller than 512 bytes",
                path.display()
            )));
        }
        Ok(Self {
            backend: DiskBackend::File {
                file,
                total_sectors,
                read_only: effective_ro,
                cow: HashMap::new(),
            },
            status: 0,
            driver_features: 0,
            queue_sel: 0,
            queue_num: 128,
        })
    }

    pub fn read_sector(&mut self, lba: u64, dst: &mut [u8]) -> Result<()> {
        if dst.len() % 512 != 0 {
            return Err(ArmtError::Device("VirtioBlk read must be 512-byte aligned".into()));
        }
        let n_secs = dst.len() / 512;
        match &mut self.backend {
            DiskBackend::Memory(sectors) => {
                let off = (lba as usize).saturating_mul(512);
                if off + dst.len() > sectors.len() {
                    return Err(ArmtError::Device("VirtioBlk read out of bounds".into()));
                }
                dst.copy_from_slice(&sectors[off..off + dst.len()]);
                Ok(())
            }
            DiskBackend::File {
                file,
                total_sectors,
                cow,
                ..
            } => {
                if lba.saturating_add(n_secs as u64) > *total_sectors {
                    return Err(ArmtError::Device("VirtioBlk file read out of bounds".into()));
                }
                for i in 0..n_secs {
                    let cur = lba + (i as u64);
                    let slice = &mut dst[i * 512..(i + 1) * 512];
                    if let Some(ov) = cow.get(&cur) {
                        slice.copy_from_slice(ov);
                    } else {
                        file.seek(SeekFrom::Start(cur * 512))?;
                        file.read_exact(slice)?;
                    }
                }
                Ok(())
            }
            DiskBackend::UdifDmg { dmg, cow } => {
                if lba.saturating_add(n_secs as u64) > dmg.capacity_sectors() {
                    return Err(ArmtError::Device("VirtioBlk DMG read out of bounds".into()));
                }
                for i in 0..n_secs {
                    let cur = lba + (i as u64);
                    let slice = &mut dst[i * 512..(i + 1) * 512];
                    if let Some(ov) = cow.get(&cur) {
                        slice.copy_from_slice(ov);
                    } else {
                        dmg.read_sector(cur, slice)?;
                    }
                }
                Ok(())
            }
        }
    }

    pub fn write_sector(&mut self, lba: u64, src: &[u8]) -> Result<()> {
        if src.len() % 512 != 0 {
            return Err(ArmtError::Device("VirtioBlk write must be 512-byte aligned".into()));
        }
        let n_secs = src.len() / 512;
        match &mut self.backend {
            DiskBackend::Memory(sectors) => {
                let off = (lba as usize).saturating_mul(512);
                if off + src.len() > sectors.len() {
                    return Err(ArmtError::Device("VirtioBlk write out of bounds".into()));
                }
                sectors[off..off + src.len()].copy_from_slice(src);
                Ok(())
            }
            DiskBackend::File {
                file,
                total_sectors,
                read_only,
                cow,
            } => {
                if lba.saturating_add(n_secs as u64) > *total_sectors {
                    return Err(ArmtError::Device("VirtioBlk file write out of bounds".into()));
                }
                if *read_only {
                    for i in 0..n_secs {
                        let mut sec = [0u8; 512];
                        sec.copy_from_slice(&src[i * 512..(i + 1) * 512]);
                        cow.insert(lba + (i as u64), sec);
                    }
                } else {
                    file.seek(SeekFrom::Start(lba * 512))?;
                    file.write_all(src)?;
                }
                Ok(())
            }
            DiskBackend::UdifDmg { dmg, cow } => {
                if lba.saturating_add(n_secs as u64) > dmg.capacity_sectors() {
                    return Err(ArmtError::Device("VirtioBlk DMG write out of bounds".into()));
                }
                for i in 0..n_secs {
                    let mut sec = [0u8; 512];
                    sec.copy_from_slice(&src[i * 512..(i + 1) * 512]);
                    cow.insert(lba + (i as u64), sec);
                }
                Ok(())
            }
        }
    }

    pub fn capacity_sectors(&self) -> u64 {
        match &self.backend {
            DiskBackend::Memory(sectors) => (sectors.len() / 512) as u64,
            DiskBackend::File { total_sectors, .. } => *total_sectors,
            DiskBackend::UdifDmg { dmg, .. } => dmg.capacity_sectors(),
        }
    }

    /// Format the in-memory disk with a protective MBR + valid CRC32 GPT header
    /// containing a single `Apple_HFS` partition so XNU's `IOGUIDPartitionScheme` probes it.
    pub fn format_gpt_stub(&mut self) {
        if !matches!(self.backend, DiskBackend::Memory(_)) {
            return;
        }
        let total_sectors = self.capacity_sectors();
        if total_sectors < 128 {
            return;
        }
        // Protective MBR at LBA 0
        let mut mbr = [0u8; 512];
        mbr[446 + 4] = 0xEE; // GPT protective partition type
        mbr[446 + 8..446 + 12].copy_from_slice(&1u32.to_le_bytes());
        let mbr_sz = (total_sectors - 1).min(0xFFFF_FFFF) as u32;
        mbr[446 + 12..446 + 16].copy_from_slice(&mbr_sz.to_le_bytes());
        mbr[510] = 0x55;
        mbr[511] = 0xAA;
        let _ = self.write_sector(0, &mbr);

        // Single GPT partition entry at LBA 2 (Apple_HFS GUID: 48465300-0000-11AA-AA11-00306543ECAC)
        let mut entries = [0u8; 16384]; // 128 entries * 128 bytes = 32 sectors (LBA 2..33)
        let hfs_type_guid: [u8; 16] = [
            0x00, 0x53, 0x46, 0x48, 0x00, 0x00, 0xAA, 0x11, 0xAA, 0x11, 0x00, 0x30, 0x65, 0x43,
            0xEC, 0xAC,
        ];
        let part_guid: [u8; 16] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10,
        ];
        entries[0..16].copy_from_slice(&hfs_type_guid);
        entries[16..32].copy_from_slice(&part_guid);
        entries[32..40].copy_from_slice(&40u64.to_le_bytes()); // first_lba
        entries[40..48].copy_from_slice(&(total_sectors - 34).to_le_bytes()); // last_lba
        for (i, ch) in "ArmTranslator".encode_utf16().enumerate() {
            entries[56 + i * 2..58 + i * 2].copy_from_slice(&ch.to_le_bytes());
        }
        let entries_crc = crc32_ieee(&entries);
        for s in 0..32u64 {
            let off = (s as usize) * 512;
            let _ = self.write_sector(2 + s, &entries[off..off + 512]);
        }

        // Primary GPT Header at LBA 1
        let mut hdr = [0u8; 512];
        hdr[0..8].copy_from_slice(b"EFI PART");
        hdr[8..12].copy_from_slice(&0x0001_0000u32.to_le_bytes()); // rev 1.0
        hdr[12..16].copy_from_slice(&92u32.to_le_bytes()); // header size
        hdr[24..32].copy_from_slice(&1u64.to_le_bytes()); // my_lba
        hdr[32..40].copy_from_slice(&(total_sectors - 1).to_le_bytes()); // alt_lba
        hdr[40..48].copy_from_slice(&34u64.to_le_bytes()); // first_usable_lba
        hdr[48..56].copy_from_slice(&(total_sectors - 34).to_le_bytes()); // last_usable_lba
        hdr[56..72].copy_from_slice(&part_guid);
        hdr[72..80].copy_from_slice(&2u64.to_le_bytes()); // partition_entry_lba
        hdr[80..84].copy_from_slice(&128u32.to_le_bytes()); // num_partition_entries
        hdr[84..88].copy_from_slice(&128u32.to_le_bytes()); // sizeof_partition_entry
        hdr[88..92].copy_from_slice(&entries_crc.to_le_bytes());
        let hdr_crc = crc32_ieee(&hdr[0..92]);
        hdr[16..20].copy_from_slice(&hdr_crc.to_le_bytes());
        let _ = self.write_sector(1, &hdr);
    }
}

fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
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

pub const INPUT_EV_KEY: u16 = 1;
pub const INPUT_EV_MOUSE_MOVE: u16 = 2;
pub const INPUT_EV_MOUSE_BTN: u16 = 3;
pub const INPUT_EV_CHAR: u16 = 4;

/// Thread-safe input state shared between the RTX GUI event loop and the guest VM.
#[derive(Debug, Default)]
pub struct VirtioInputShared {
    pub events: VecDeque<InputEvent>,
    pub mouse_x: i32,
    pub mouse_y: i32,
    pub mouse_buttons: u32,
    pub scroll_delta: i32,
}

impl VirtioInputShared {
    pub fn push_mouse_move(&mut self, x: i32, y: i32) {
        self.mouse_x = x;
        self.mouse_y = y;
        if self.events.len() < 256 {
            self.events.push_back(InputEvent {
                ev_type: INPUT_EV_MOUSE_MOVE,
                code: x as u16,
                value: y,
            });
        }
    }

    pub fn push_mouse_button(&mut self, button: u8, pressed: bool, x: i32, y: i32) {
        self.mouse_x = x;
        self.mouse_y = y;
        let bit = 1u32 << (button & 7);
        if pressed {
            self.mouse_buttons |= bit;
        } else {
            self.mouse_buttons &= !bit;
        }
        self.events.push_back(InputEvent {
            ev_type: INPUT_EV_MOUSE_BTN,
            code: button as u16,
            value: if pressed { 1 } else { 0 },
        });
    }

    pub fn push_scroll(&mut self, delta: i32) {
        self.scroll_delta = self.scroll_delta.saturating_add(delta);
    }

    pub fn push_key(&mut self, code: u16, pressed: bool) {
        self.events.push_back(InputEvent {
            ev_type: INPUT_EV_KEY,
            code,
            value: if pressed { 1 } else { 0 },
        });
    }

    pub fn push_char(&mut self, ch: u8) {
        self.events.push_back(InputEvent {
            ev_type: INPUT_EV_CHAR,
            code: ch as u16,
            value: 1,
        });
    }
}

/// VirtIO MMIO v2 keyboard/mouse input queue with live shared event bridge.
pub struct VirtioInput {
    shared: Arc<Mutex<VirtioInputShared>>,
    status: u32,
    latched_code: u32,
    latched_value: u32,
}

impl Default for VirtioInput {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtioInput {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(VirtioInputShared::default())),
            status: 0,
            latched_code: 0,
            latched_value: 0,
        }
    }

    pub fn shared(&self) -> Arc<Mutex<VirtioInputShared>> {
        self.shared.clone()
    }

    pub fn push_key(&mut self, code: u16, pressed: bool) {
        self.shared.lock().unwrap().push_key(code, pressed);
    }

    pub fn pop_event(&mut self) -> Option<InputEvent> {
        self.shared.lock().unwrap().events.pop_front()
    }

    pub fn irq_pending(&self) -> bool {
        !self.shared.lock().unwrap().events.is_empty()
    }
}

impl MmioDevice for VirtioInput {
    fn name(&self) -> &str {
        "virtio-input"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        let mut s = self.shared.lock().unwrap();
        Ok(match offset {
            0x000 => 0x7472_6976,
            0x004 => 2,
            0x008 => 18, // Input device
            0x00C => 0x1AF4,
            0x070 => self.status as u64,
            0x100 => s.mouse_x as u32 as u64,
            0x104 => s.mouse_y as u32 as u64,
            0x108 => s.mouse_buttons as u64,
            0x10C => {
                let d = s.scroll_delta;
                s.scroll_delta = 0;
                d as u32 as u64
            }
            0x110 => s.events.len() as u64,
            0x114 => {
                if let Some(ev) = s.events.pop_front() {
                    self.latched_code = ev.code as u32;
                    self.latched_value = ev.value as u32;
                    ev.ev_type as u64
                } else {
                    self.latched_code = 0;
                    self.latched_value = 0;
                    0
                }
            }
            0x118 => self.latched_code as u64,
            0x11C => self.latched_value as u64,
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

/// Linear BGRA (`0x00RRGGBB` in `u32`, `[B, G, R, A]` in little-endian memory) scanout framebuffer
/// shared between guest devices and the NVIDIA RTX `wgpu` renderer.
#[derive(Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<Mutex<Vec<u32>>>,
    pub generation: Arc<AtomicU64>,
    pub cursor_x: Arc<AtomicI32>,
    pub cursor_y: Arc<AtomicI32>,
    pub cursor_visible: Arc<AtomicBool>,
    pub cursor_rgba: Arc<Mutex<Vec<u32>>>,
}

fn default_mac_cursor_32x32() -> Vec<u32> {
    // Crisp 18x22 classic macOS arrow pointer (black with white outline) in 32x32 RGBA8
    const SHAPE: [&str; 20] = [
        "W...............",
        "WW..............",
        "WBW.............",
        "WBBW............",
        "WBBBW...........",
        "WBBBBW..........",
        "WBBBBBW.........",
        "WBBBBBBW........",
        "WBBBBBBBW.......",
        "WBBBBBBBBW......",
        "WBBBBBBBBBW.....",
        "WBBBBBBBBBBW....",
        "WBBBBBBWWWWWW...",
        "WBBBWWBW........",
        "WBBW.WBBW.......",
        "WBW..WBBW.......",
        "WW....WBBW......",
        "W.....WBBW......",
        ".......WW.......",
        "................",
    ];
    let mut out = vec![0u32; 32 * 32];
    for (y, row) in SHAPE.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            let rgba = match ch {
                'W' => 0xFFFF_FFFF, // white outline (ABGR/RGBA little-endian opaque white)
                'B' => 0xFF14_1418, // crisp dark fill
                _ => 0x0000_0000,   // transparent
            };
            out[y * 32 + x] = rgba;
        }
    }
    out
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: Arc::new(Mutex::new(vec![0u32; (width as usize) * (height as usize)])),
            generation: Arc::new(AtomicU64::new(1)),
            cursor_x: Arc::new(AtomicI32::new(width as i32 / 2)),
            cursor_y: Arc::new(AtomicI32::new(height as i32 / 2)),
            cursor_visible: Arc::new(AtomicBool::new(false)),
            cursor_rgba: Arc::new(Mutex::new(default_mac_cursor_32x32())),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn mark_dirty(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    pub fn set_cursor_pos(&self, x: i32, y: i32, visible: bool) {
        self.cursor_x.store(x, Ordering::Release);
        self.cursor_y.store(y, Ordering::Release);
        self.cursor_visible.store(visible, Ordering::Release);
    }

    pub fn cursor_state(&self) -> (i32, i32, bool) {
        (
            self.cursor_x.load(Ordering::Acquire),
            self.cursor_y.load(Ordering::Acquire),
            self.cursor_visible.load(Ordering::Acquire),
        )
    }

    pub fn copy_bgra_bytes(&self, dst: &mut Vec<u8>) {
        let guard = self.pixels.lock().unwrap();
        let needed = guard.len() * 4;
        dst.resize(needed, 0);
        for (i, &px) in guard.iter().enumerate() {
            let opaque = px | 0xFF00_0000;
            dst[i * 4..i * 4 + 4].copy_from_slice(&opaque.to_le_bytes());
        }
    }

    pub fn present_from(&self, backbuffer: &[u32]) {
        let mut guard = self.pixels.lock().unwrap();
        let n = guard.len().min(backbuffer.len());
        guard[..n].copy_from_slice(&backbuffer[..n]);
        drop(guard);
        self.mark_dirty();
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
            drop(guard);
            self.mark_dirty();
            return;
        }
        if size == 4 && (offset % 4) == 0 {
            guard[pix_idx] = val as u32;
            drop(guard);
            self.mark_dirty();
            return;
        }
        let shift = ((offset % 4) * 8) as u32;
        let mask = if size == 1 { 0xFFu32 } else { 0xFFFFu32 } << shift;
        let mut cur = guard[pix_idx];
        cur = (cur & !mask) | (((val as u32) << shift) & mask);
        guard[pix_idx] = cur;
        drop(guard);
        self.mark_dirty();
    }

    pub fn clear(&self, rgb: u32) {
        let mut guard = self.pixels.lock().unwrap();
        guard.fill(rgb);
        drop(guard);
        self.mark_dirty();
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
        drop(guard);
        self.mark_dirty();
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
