use crate::virtio::Framebuffer;
use armt_core::{MmioDevice, Result};
use std::collections::HashMap;

const VIRTIO_GPU_CMD_GET_DISPLAY_INFO: u32 = 0x0100;
const VIRTIO_GPU_CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
const VIRTIO_GPU_CMD_RESOURCE_UNREF: u32 = 0x0102;
const VIRTIO_GPU_CMD_SET_SCANOUT: u32 = 0x0103;
const VIRTIO_GPU_CMD_RESOURCE_FLUSH: u32 = 0x0104;
const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
const VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
const VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;
const VIRTIO_GPU_CMD_UPDATE_CURSOR: u32 = 0x0300;
const VIRTIO_GPU_CMD_MOVE_CURSOR: u32 = 0x0301;

const VIRTIO_GPU_RESP_OK_NODATA: u32 = 0x1100;
const VIRTIO_GPU_RESP_OK_DISPLAY_INFO: u32 = 0x1101;
const VIRTIO_GPU_RESP_ERR_UNSPEC: u32 = 0x1200;

#[derive(Debug, Clone)]
struct GpuResource2D {
    width: u32,
    height: u32,
    _format: u32,
    pixels: Vec<u32>,
    backing_ranges: Vec<(u64, u32)>,
}

#[derive(Debug, Clone, Copy, Default)]
struct VirtQueueState {
    num: u32,
    ready: u32,
    desc_addr: u64,
    avail_addr: u64,
    used_addr: u64,
    last_avail_idx: u16,
}

/// VirtIO-GPU v2 MMIO device (DeviceID 16) supporting 2D scanouts, host transfers & hardware cursor.
pub struct VirtioGpu {
    pub width: u32,
    pub height: u32,
    pub status: u32,
    pub driver_features: u64,
    pub driver_features_sel: u32,
    pub device_features_sel: u32,
    pub interrupt_status: u32,
    pub queue_sel: u32,
    pub notify_pending: Option<u32>,
    pub commands_processed: u64,
    pub flushes: u64,
    pub cursor_updates: u64,
    pub active_scanout_res: u32,
    queues: [VirtQueueState; 2],
    resources: HashMap<u32, GpuResource2D>,
}

impl VirtioGpu {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            status: 0,
            driver_features: 0,
            driver_features_sel: 0,
            device_features_sel: 0,
            interrupt_status: 0,
            queue_sel: 0,
            notify_pending: None,
            commands_processed: 0,
            flushes: 0,
            cursor_updates: 0,
            active_scanout_res: 0,
            queues: [VirtQueueState::default(); 2],
            resources: HashMap::new(),
        }
    }

    pub fn irq_pending(&self) -> bool {
        self.interrupt_status != 0
    }

    /// Process any newly posted VirtIO-GPU controlq / cursorq descriptor chains in guest RAM.
    pub fn process_queue(&mut self, ram: &mut [u8], ram_base: u64, fb: &Framebuffer) {
        let q_idx = self.notify_pending.take().unwrap_or(0) as usize;
        if q_idx >= self.queues.len() {
            return;
        }
        let (q_num, desc_addr, avail_addr, used_addr, mut last_avail_idx) = {
            let q = &self.queues[q_idx];
            if q.ready == 0 || q.num == 0 || q.desc_addr == 0 || q.avail_addr == 0 || q.used_addr == 0 {
                return;
            }
            (q.num as u16, q.desc_addr, q.avail_addr, q.used_addr, q.last_avail_idx)
        };
        let Some(avail_idx) = ram_read_u16(ram, ram_base, avail_addr + 2) else {
            return;
        };

        while last_avail_idx != avail_idx {
            let ring_slot = (last_avail_idx % q_num) as u64;
            let Some(head_desc) = ram_read_u16(ram, ram_base, avail_addr + 4 + ring_slot * 2)
            else {
                break;
            };
            last_avail_idx = last_avail_idx.wrapping_add(1);

            let (req_bytes, resp_addr, resp_cap) =
                collect_chain(ram, ram_base, desc_addr, q_num, head_desc);
            let resp_bytes = if req_bytes.len() >= 24 {
                self.handle_control_cmd(&req_bytes, ram, ram_base, fb)
            } else {
                encode_resp_hdr(VIRTIO_GPU_RESP_ERR_UNSPEC)
            };

            let written = if resp_addr != 0 && resp_cap > 0 {
                let n = resp_bytes.len().min(resp_cap as usize);
                let _ = ram_write_slice(ram, ram_base, resp_addr, &resp_bytes[..n]);
                n as u32
            } else {
                0
            };

            // Push into Used ring
            let used_idx = ram_read_u16(ram, ram_base, used_addr + 2).unwrap_or(0);
            let used_slot = (used_idx % q_num) as u64;
            let elem_addr = used_addr + 4 + used_slot * 8;
            let _ = ram_write_u32(ram, ram_base, elem_addr, head_desc as u32);
            let _ = ram_write_u32(ram, ram_base, elem_addr + 4, written);
            let _ = ram_write_u16(ram, ram_base, used_addr + 2, used_idx.wrapping_add(1));
            self.interrupt_status |= 1;
        }
        self.queues[q_idx].last_avail_idx = last_avail_idx;
    }

    fn handle_control_cmd(
        &mut self,
        req: &[u8],
        ram: &[u8],
        ram_base: u64,
        fb: &Framebuffer,
    ) -> Vec<u8> {
        let cmd_type = u32::from_le_bytes(req[0..4].try_into().unwrap());
        self.commands_processed += 1;

        match cmd_type {
            VIRTIO_GPU_CMD_GET_DISPLAY_INFO => {
                let mut out = vec![0u8; 24 + 16 * 24];
                out[0..4].copy_from_slice(&VIRTIO_GPU_RESP_OK_DISPLAY_INFO.to_le_bytes());
                // pmodes[0]: rect {x=0,y=0,w,h}, enabled=1
                out[24 + 8..24 + 12].copy_from_slice(&self.width.to_le_bytes());
                out[24 + 12..24 + 16].copy_from_slice(&self.height.to_le_bytes());
                out[24 + 16..24 + 20].copy_from_slice(&1u32.to_le_bytes());
                out
            }
            VIRTIO_GPU_CMD_RESOURCE_CREATE_2D if req.len() >= 40 => {
                let res_id = u32::from_le_bytes(req[24..28].try_into().unwrap());
                let format = u32::from_le_bytes(req[28..32].try_into().unwrap());
                let w = u32::from_le_bytes(req[32..36].try_into().unwrap()).min(4096);
                let h = u32::from_le_bytes(req[36..40].try_into().unwrap()).min(4096);
                let count = (w as usize).saturating_mul(h as usize).min(16_777_216);
                self.resources.insert(
                    res_id,
                    GpuResource2D {
                        width: w,
                        height: h,
                        _format: format,
                        pixels: vec![0u32; count],
                        backing_ranges: Vec::new(),
                    },
                );
                encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
            }
            VIRTIO_GPU_CMD_RESOURCE_UNREF if req.len() >= 28 => {
                let res_id = u32::from_le_bytes(req[24..28].try_into().unwrap());
                self.resources.remove(&res_id);
                if self.active_scanout_res == res_id {
                    self.active_scanout_res = 0;
                }
                encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
            }
            VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING if req.len() >= 32 => {
                let res_id = u32::from_le_bytes(req[24..28].try_into().unwrap());
                let nr_entries = u32::from_le_bytes(req[28..32].try_into().unwrap()) as usize;
                if let Some(res) = self.resources.get_mut(&res_id) {
                    res.backing_ranges.clear();
                    for i in 0..nr_entries.min(256) {
                        let off = 32 + i * 16;
                        if off + 16 > req.len() {
                            break;
                        }
                        let addr = u64::from_le_bytes(req[off..off + 8].try_into().unwrap());
                        let len = u32::from_le_bytes(req[off + 8..off + 12].try_into().unwrap());
                        res.backing_ranges.push((addr, len));
                    }
                    encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
                } else {
                    encode_resp_hdr(VIRTIO_GPU_RESP_ERR_UNSPEC)
                }
            }
            VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING if req.len() >= 28 => {
                let res_id = u32::from_le_bytes(req[24..28].try_into().unwrap());
                if let Some(res) = self.resources.get_mut(&res_id) {
                    res.backing_ranges.clear();
                }
                encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
            }
            VIRTIO_GPU_CMD_SET_SCANOUT if req.len() >= 48 => {
                let res_id = u32::from_le_bytes(req[44..48].try_into().unwrap());
                self.active_scanout_res = res_id;
                encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
            }
            VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D if req.len() >= 56 => {
                let rx = u32::from_le_bytes(req[24..28].try_into().unwrap());
                let ry = u32::from_le_bytes(req[28..32].try_into().unwrap());
                let rw = u32::from_le_bytes(req[32..36].try_into().unwrap());
                let rh = u32::from_le_bytes(req[36..40].try_into().unwrap());
                let offset = u64::from_le_bytes(req[40..48].try_into().unwrap());
                let res_id = u32::from_le_bytes(req[48..52].try_into().unwrap());
                if let Some(res) = self.resources.get_mut(&res_id) {
                    copy_backing_to_resource(res, ram, ram_base, rx, ry, rw, rh, offset);
                    encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
                } else {
                    encode_resp_hdr(VIRTIO_GPU_RESP_ERR_UNSPEC)
                }
            }
            VIRTIO_GPU_CMD_RESOURCE_FLUSH if req.len() >= 48 => {
                let res_id = u32::from_le_bytes(req[40..44].try_into().unwrap());
                if let Some(res) = self.resources.get(&res_id) {
                    let mut guard = fb.pixels.lock().unwrap();
                    let copy_w = res.width.min(fb.width) as usize;
                    let copy_h = res.height.min(fb.height) as usize;
                    for y in 0..copy_h {
                        let dst_row = y * fb.width as usize;
                        let src_row = y * res.width as usize;
                        guard[dst_row..dst_row + copy_w]
                            .copy_from_slice(&res.pixels[src_row..src_row + copy_w]);
                    }
                    drop(guard);
                    fb.mark_dirty();
                    self.flushes += 1;
                    encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
                } else {
                    encode_resp_hdr(VIRTIO_GPU_RESP_ERR_UNSPEC)
                }
            }
            VIRTIO_GPU_CMD_UPDATE_CURSOR | VIRTIO_GPU_CMD_MOVE_CURSOR if req.len() >= 40 => {
                let x = u32::from_le_bytes(req[28..32].try_into().unwrap()) as i32;
                let y = u32::from_le_bytes(req[32..36].try_into().unwrap()) as i32;
                let res_id = u32::from_le_bytes(req[40..44].try_into().unwrap_or([1, 0, 0, 0]));
                fb.set_cursor_pos(x, y, res_id != 0);
                self.cursor_updates += 1;
                encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA)
            }
            _ => encode_resp_hdr(VIRTIO_GPU_RESP_OK_NODATA),
        }
    }
}

impl MmioDevice for VirtioGpu {
    fn name(&self) -> &str {
        "virtio-gpu"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        let q = &self.queues[(self.queue_sel as usize).min(1)];
        Ok(match offset {
            0x000 => 0x7472_6976, // "virt"
            0x004 => 2,           // VirtIO MMIO v2
            0x008 => 16,          // VirtIO-GPU
            0x00C => 0x1AF4,      // Vendor ID
            0x010 => {
                if self.device_features_sel == 1 {
                    1 // VIRTIO_F_VERSION_1
                } else {
                    0
                }
            }
            0x034 => 256, // QueueNumMax
            0x044 => q.ready as u64,
            0x060 => self.interrupt_status as u64,
            0x070 => self.status as u64,
            0x0FC => 1,
            0x100 => 0, // events_read
            0x104 => 0, // events_clear
            0x108 => 1, // num_scanouts
            0x10C => 0, // num_capsets
            _ => 0,
        })
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) -> Result<()> {
        let idx = (self.queue_sel as usize).min(1);
        match offset {
            0x014 => self.device_features_sel = val as u32,
            0x020 => {
                if self.driver_features_sel == 0 {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
                } else {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
                }
            }
            0x024 => self.driver_features_sel = val as u32,
            0x030 => self.queue_sel = val as u32,
            0x038 => self.queues[idx].num = (val as u32).min(256),
            0x044 => self.queues[idx].ready = (val & 1) as u32,
            0x050 => self.notify_pending = Some(val as u32),
            0x064 => self.interrupt_status &= !(val as u32),
            0x070 => {
                if val == 0 {
                    self.status = 0;
                    self.queues = [VirtQueueState::default(); 2];
                } else {
                    self.status = val as u32;
                }
            }
            0x080 => {
                self.queues[idx].desc_addr =
                    (self.queues[idx].desc_addr & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
            }
            0x084 => {
                self.queues[idx].desc_addr =
                    (self.queues[idx].desc_addr & 0xFFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            0x090 => {
                self.queues[idx].avail_addr =
                    (self.queues[idx].avail_addr & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
            }
            0x094 => {
                self.queues[idx].avail_addr =
                    (self.queues[idx].avail_addr & 0xFFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            0x0A0 => {
                self.queues[idx].used_addr =
                    (self.queues[idx].used_addr & 0xFFFF_FFFF_0000_0000) | (val & 0xFFFF_FFFF);
            }
            0x0A4 => {
                self.queues[idx].used_addr =
                    (self.queues[idx].used_addr & 0xFFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            _ => {}
        }
        Ok(())
    }
}

fn encode_resp_hdr(resp_type: u32) -> Vec<u8> {
    let mut out = vec![0u8; 24];
    out[0..4].copy_from_slice(&resp_type.to_le_bytes());
    out
}

fn ram_slice_at(ram: &[u8], ram_base: u64, addr: u64, len: usize) -> Option<&[u8]> {
    if addr < ram_base {
        return None;
    }
    let off = (addr - ram_base) as usize;
    ram.get(off..off.checked_add(len)?)
}

fn ram_read_u16(ram: &[u8], ram_base: u64, addr: u64) -> Option<u16> {
    let s = ram_slice_at(ram, ram_base, addr, 2)?;
    Some(u16::from_le_bytes([s[0], s[1]]))
}

fn ram_read_u32(ram: &[u8], ram_base: u64, addr: u64) -> Option<u32> {
    let s = ram_slice_at(ram, ram_base, addr, 4)?;
    Some(u32::from_le_bytes(s.try_into().unwrap()))
}

fn ram_read_u64(ram: &[u8], ram_base: u64, addr: u64) -> Option<u64> {
    let s = ram_slice_at(ram, ram_base, addr, 8)?;
    Some(u64::from_le_bytes(s.try_into().unwrap()))
}

fn ram_write_slice(ram: &mut [u8], ram_base: u64, addr: u64, data: &[u8]) -> Option<()> {
    if addr < ram_base {
        return None;
    }
    let off = (addr - ram_base) as usize;
    let dst = ram.get_mut(off..off.checked_add(data.len())?)?;
    dst.copy_from_slice(data);
    Some(())
}

fn ram_write_u16(ram: &mut [u8], ram_base: u64, addr: u64, val: u16) -> Option<()> {
    ram_write_slice(ram, ram_base, addr, &val.to_le_bytes())
}

fn ram_write_u32(ram: &mut [u8], ram_base: u64, addr: u64, val: u32) -> Option<()> {
    ram_write_slice(ram, ram_base, addr, &val.to_le_bytes())
}

fn collect_chain(
    ram: &[u8],
    ram_base: u64,
    desc_table: u64,
    q_num: u16,
    head: u16,
) -> (Vec<u8>, u64, u32) {
    let mut req = Vec::new();
    let mut resp_addr = 0u64;
    let mut resp_len = 0u32;
    let mut idx = head;
    for _ in 0..16 {
        if idx >= q_num {
            break;
        }
        let d_addr = desc_table + (idx as u64) * 16;
        let Some(addr) = ram_read_u64(ram, ram_base, d_addr) else {
            break;
        };
        let Some(len) = ram_read_u32(ram, ram_base, d_addr + 8) else {
            break;
        };
        let Some(flags) = ram_read_u16(ram, ram_base, d_addr + 12) else {
            break;
        };
        let next = ram_read_u16(ram, ram_base, d_addr + 14).unwrap_or(0);
        if (flags & 2) != 0 {
            if resp_addr == 0 {
                resp_addr = addr;
                resp_len = len;
            }
        } else if let Some(chunk) = ram_slice_at(ram, ram_base, addr, (len as usize).min(65536)) {
            req.extend_from_slice(chunk);
        }
        if (flags & 1) == 0 {
            break;
        }
        idx = next;
    }
    (req, resp_addr, resp_len)
}

fn copy_backing_to_resource(
    res: &mut GpuResource2D,
    ram: &[u8],
    ram_base: u64,
    rx: u32,
    ry: u32,
    rw: u32,
    rh: u32,
    offset: u64,
) {
    let Some(&(backing_pa, backing_len)) = res.backing_ranges.first() else {
        return;
    };
    let Some(backing) = ram_slice_at(ram, ram_base, backing_pa, backing_len as usize) else {
        return;
    };
    let stride = res.width as usize * 4;
    let max_x = rx.saturating_add(rw).min(res.width);
    let max_y = ry.saturating_add(rh).min(res.height);
    for y in ry..max_y {
        for x in rx..max_x {
            let src_off = (offset as usize)
                .saturating_add((y as usize) * stride)
                .saturating_add((x as usize) * 4);
            if src_off + 4 <= backing.len() {
                let px = u32::from_le_bytes(backing[src_off..src_off + 4].try_into().unwrap());
                res.pixels[(y as usize) * (res.width as usize) + (x as usize)] = px;
            }
        }
    }
}

// ============================================================================
// AppleParavirtGpu (`apple-gfx` 2D Compositor & Hardware Cursor Bridge)
// ============================================================================

/// Command structure written by ARM64 guest drivers into RAM for `AppleParavirtGpu`.
/// Size = 32 bytes (`#[repr(C)]`), little-endian.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuCmd {
    pub kind: u32,
    pub flags: u32,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub color: u32,
    pub aux: u32, // radius/border/text_len/end_color
}

pub const GPU_CMD_CLEAR: u32 = 1;
pub const GPU_CMD_FILL_RECT: u32 = 2;
pub const GPU_CMD_ROUND_RECT: u32 = 3;
pub const GPU_CMD_GRADIENT_RECT: u32 = 4;
pub const GPU_CMD_DRAW_TEXT: u32 = 5;
pub const GPU_CMD_SET_CURSOR: u32 = 6;
pub const GPU_CMD_PRESENT: u32 = 7;

/// `AppleParavirtGpu` (`apple-gfx`) MMIO device at `0x200b_0000`.
/// Accelerates 2D compositing + rounded macOS surfaces + crisp text + atomic backbuffer swap
/// directly into the shared `Framebuffer` for upload to the host NVIDIA RTX GPU.
pub struct AppleParavirtGpu {
    pub width: u32,
    pub height: u32,
    pub cmd_ring_phys: u64,
    pub cmd_count: u32,
    pub frame_count: u64,
    pub doorbell_pending: bool,
    backbuffer: Vec<u32>,
}

impl AppleParavirtGpu {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            cmd_ring_phys: 0,
            cmd_count: 0,
            frame_count: 0,
            doorbell_pending: false,
            backbuffer: vec![0xFF10_141E; (width as usize) * (height as usize)],
        }
    }

    pub fn execute_cmd_ring(&mut self, ram: &[u8], ram_base: u64, fb: &Framebuffer) {
        self.doorbell_pending = false;
        let count = self.cmd_count.min(4096) as usize;
        if self.cmd_ring_phys < ram_base || count == 0 {
            return;
        }
        let base_off = (self.cmd_ring_phys - ram_base) as usize;
        let needed = count * 32;
        let Some(cmd_bytes) = ram.get(base_off..base_off.saturating_add(needed)) else {
            return;
        };

        let mut presented = false;
        for chunk in cmd_bytes.chunks_exact(32) {
            let kind = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let flags = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let x = i32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let y = i32::from_le_bytes(chunk[12..16].try_into().unwrap());
            let w = u32::from_le_bytes(chunk[16..20].try_into().unwrap());
            let h = u32::from_le_bytes(chunk[20..24].try_into().unwrap());
            let color = u32::from_le_bytes(chunk[24..28].try_into().unwrap());
            let aux = u32::from_le_bytes(chunk[28..32].try_into().unwrap());

            match kind {
                GPU_CMD_CLEAR => {
                    self.backbuffer.fill(color | 0xFF00_0000);
                }
                GPU_CMD_FILL_RECT => {
                    self.fill_rect_alpha(x, y, w as i32, h as i32, color);
                }
                GPU_CMD_ROUND_RECT => {
                    let radius = (flags & 0xFF) as i32;
                    let shadow = (flags & 0x100) != 0;
                    if shadow {
                        self.round_rect_alpha(x + 2, y + 6, w as i32, h as i32, radius, 0x5500_0000);
                    }
                    self.round_rect_alpha(x, y, w as i32, h as i32, radius, color);
                    if aux != 0 {
                        self.round_rect_border(x, y, w as i32, h as i32, radius, aux);
                    }
                }
                GPU_CMD_GRADIENT_RECT => {
                    self.gradient_rect(x, y, w as i32, h as i32, color, aux);
                }
                GPU_CMD_DRAW_TEXT => {
                    // w = low32(str_pa), h = high32(str_pa), flags = scale (1 or 2) | (max_len << 8)
                    let str_pa = (w as u64) | ((h as u64) << 32);
                    let scale = (flags & 0xF).max(1).min(3);
                    let max_len = ((flags >> 8) & 0xFFF) as usize;
                    if str_pa >= ram_base {
                        let off = (str_pa - ram_base) as usize;
                        if off < ram.len() {
                            let slice = &ram[off..(off + max_len.max(256)).min(ram.len())];
                            let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
                            self.draw_text(x, y, &slice[..end], color, scale);
                        }
                    }
                }
                GPU_CMD_SET_CURSOR => {
                    fb.set_cursor_pos(x, y, flags != 0);
                }
                GPU_CMD_PRESENT => {
                    fb.present_from(&self.backbuffer);
                    self.frame_count += 1;
                    presented = true;
                }
                _ => {}
            }
        }
        if !presented {
            fb.present_from(&self.backbuffer);
            self.frame_count += 1;
        }
    }

    #[inline]
    fn blend_px(&mut self, px: i32, py: i32, argb: u32) {
        if px < 0 || py < 0 || px >= self.width as i32 || py >= self.height as i32 {
            return;
        }
        let a = (argb >> 24) & 0xFF;
        let idx = (py as usize) * (self.width as usize) + (px as usize);
        if a == 255 || a == 0 {
            if a == 255 {
                self.backbuffer[idx] = argb;
            }
            return;
        }
        let dst = self.backbuffer[idx];
        let inv = 255 - a;
        let r = (((argb >> 16) & 0xFF) * a + ((dst >> 16) & 0xFF) * inv) >> 8;
        let g = (((argb >> 8) & 0xFF) * a + ((dst >> 8) & 0xFF) * inv) >> 8;
        let b = ((argb & 0xFF) * a + (dst & 0xFF) * inv) >> 8;
        self.backbuffer[idx] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
    }

    fn fill_rect_alpha(&mut self, x: i32, y: i32, w: i32, h: i32, argb: u32) {
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = x.saturating_add(w).min(self.width as i32);
        let y1 = y.saturating_add(h).min(self.height as i32);
        let alpha = (argb >> 24) & 0xFF;
        if alpha == 0 {
            return;
        }
        if alpha == 255 {
            for yy in y0..y1 {
                let row = (yy as usize) * (self.width as usize);
                self.backbuffer[row + x0 as usize..row + x1 as usize].fill(argb);
            }
            return;
        }
        for yy in y0..y1 {
            for xx in x0..x1 {
                self.blend_px(xx, yy, argb);
            }
        }
    }

    fn gradient_rect(&mut self, x: i32, y: i32, w: i32, h: i32, top_c: u32, bot_c: u32) {
        if w <= 0 || h <= 0 {
            return;
        }
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = x.saturating_add(w).min(self.width as i32);
        let y1 = y.saturating_add(h).min(self.height as i32);
        let tr = (top_c >> 16) & 0xFF;
        let tg = (top_c >> 8) & 0xFF;
        let tb = top_c & 0xFF;
        let br = (bot_c >> 16) & 0xFF;
        let bg = (bot_c >> 8) & 0xFF;
        let bb = bot_c & 0xFF;
        let denom = (h - 1).max(1) as u32;
        for yy in y0..y1 {
            let t = (yy - y) as u32;
            let inv = denom - t;
            let r = (tr * inv + br * t) / denom;
            let g = (tg * inv + bg * t) / denom;
            let b = (tb * inv + bb * t) / denom;
            let c = 0xFF00_0000 | (r << 16) | (g << 8) | b;
            let row = (yy as usize) * (self.width as usize);
            self.backbuffer[row + x0 as usize..row + x1 as usize].fill(c);
        }
    }

    fn round_rect_alpha(&mut self, x: i32, y: i32, w: i32, h: i32, r: i32, argb: u32) {
        if w <= 0 || h <= 0 {
            return;
        }
        let r = r.min(w / 2).min(h / 2).max(0);
        let r2 = r * r;
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = x.saturating_add(w).min(self.width as i32);
        let y1 = y.saturating_add(h).min(self.height as i32);
        for yy in y0..y1 {
            let dy = if yy < y + r {
                y + r - yy
            } else if yy >= y + h - r {
                yy - (y + h - r - 1)
            } else {
                0
            };
            for xx in x0..x1 {
                let dx = if xx < x + r {
                    x + r - xx
                } else if xx >= x + w - r {
                    xx - (x + w - r - 1)
                } else {
                    0
                };
                if dx == 0 || dy == 0 || dx * dx + dy * dy <= r2 {
                    self.blend_px(xx, yy, argb);
                }
            }
        }
    }

    fn round_rect_border(&mut self, x: i32, y: i32, w: i32, h: i32, r: i32, argb: u32) {
        if w <= 2 || h <= 2 {
            return;
        }
        let r = r.min(w / 2).min(h / 2).max(0);
        let r2 = r * r;
        let inner_r2 = (r - 1).max(0) * (r - 1).max(0);
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = x.saturating_add(w).min(self.width as i32);
        let y1 = y.saturating_add(h).min(self.height as i32);
        for yy in y0..y1 {
            let dy = if yy < y + r {
                y + r - yy
            } else if yy >= y + h - r {
                yy - (y + h - r - 1)
            } else {
                0
            };
            for xx in x0..x1 {
                let dx = if xx < x + r {
                    x + r - xx
                } else if xx >= x + w - r {
                    xx - (x + w - r - 1)
                } else {
                    0
                };
                if dx == 0 && dy == 0 {
                    if xx == x || xx == x + w - 1 || yy == y || yy == y + h - 1 {
                        self.blend_px(xx, yy, argb);
                    }
                } else {
                    let d2 = dx * dx + dy * dy;
                    if d2 <= r2 && d2 >= inner_r2 {
                        self.blend_px(xx, yy, argb);
                    }
                }
            }
        }
    }

    fn draw_text(&mut self, x: i32, y: i32, text: &[u8], color: u32, scale: u32) {
        let s = scale as i32;
        let color = if (color >> 24) == 0 {
            color | 0xFF00_0000
        } else {
            color
        };
        let mut cx = x;
        for &ch in text {
            if ch == b'\n' {
                continue;
            }
            let glyph = glyph_6x8(ch);
            for (row_idx, &bits) in glyph.iter().enumerate() {
                for col in 0..6i32 {
                    if (bits & (1 << (5 - col))) != 0 {
                        for sy in 0..s {
                            for sx in 0..s {
                                self.blend_px(cx + col * s + sx, y + (row_idx as i32) * s + sy, color);
                            }
                        }
                    }
                }
            }
            cx += 7 * s;
        }
    }
}

impl MmioDevice for AppleParavirtGpu {
    fn name(&self) -> &str {
        "apple-gfx"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        Ok(match offset {
            0x00 => 0x4150_4755, // "APGU"
            0x04 => 1,           // Version 1
            0x08 => self.width as u64,
            0x0C => self.height as u64,
            0x10 => 0x10DE_0001, // RTX paravirt acceleration capability flag
            0x14 => self.frame_count,
            0x20 => self.cmd_ring_phys & 0xFFFF_FFFF,
            0x24 => self.cmd_count as u64,
            0x2C => self.cmd_ring_phys >> 32,
            _ => 0,
        })
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) -> Result<()> {
        match offset {
            0x20 => {
                // Support both 64-bit write to 0x20 and split 32-bit writes to 0x20/0x2C
                if val > 0xFFFF_FFFF {
                    self.cmd_ring_phys = val;
                } else {
                    self.cmd_ring_phys = (self.cmd_ring_phys & !0xFFFF_FFFF) | (val & 0xFFFF_FFFF);
                }
            }
            0x24 => self.cmd_count = val as u32,
            0x28 => self.doorbell_pending = true,
            0x2C => {
                self.cmd_ring_phys = (self.cmd_ring_phys & 0xFFFF_FFFF) | ((val & 0xFFFF_FFFF) << 32);
            }
            _ => {}
        }
        Ok(())
    }
}

/// Compact 6x8 bitmap font covering ASCII 0x20..=0x7E (6 bits wide, 8 rows tall).
fn glyph_6x8(ch: u8) -> [u8; 8] {
    match ch {
        b' ' => [0x00; 8],
        b'!' => [0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x08, 0x00],
        b'"' => [0x14, 0x14, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00],
        b'#' => [0x14, 0x3E, 0x14, 0x14, 0x3E, 0x14, 0x00, 0x00],
        b'$' => [0x08, 0x1E, 0x28, 0x1C, 0x0A, 0x3C, 0x08, 0x00],
        b'%' => [0x32, 0x34, 0x08, 0x08, 0x16, 0x26, 0x00, 0x00],
        b'&' => [0x18, 0x24, 0x18, 0x2A, 0x24, 0x1A, 0x00, 0x00],
        b'\'' => [0x08, 0x08, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00],
        b'(' => [0x04, 0x08, 0x10, 0x10, 0x10, 0x08, 0x04, 0x00],
        b')' => [0x10, 0x08, 0x04, 0x04, 0x04, 0x08, 0x10, 0x00],
        b'*' => [0x00, 0x22, 0x14, 0x3E, 0x14, 0x22, 0x00, 0x00],
        b'+' => [0x00, 0x08, 0x08, 0x3E, 0x08, 0x08, 0x00, 0x00],
        b',' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x08, 0x10],
        b'-' => [0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00],
        b'.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],
        b'/' => [0x02, 0x04, 0x08, 0x08, 0x10, 0x20, 0x00, 0x00],
        b'0' => [0x1C, 0x22, 0x26, 0x2A, 0x32, 0x22, 0x1C, 0x00],
        b'1' => [0x08, 0x18, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00],
        b'2' => [0x1C, 0x22, 0x02, 0x0C, 0x10, 0x20, 0x3E, 0x00],
        b'3' => [0x3E, 0x02, 0x04, 0x0C, 0x02, 0x22, 0x1C, 0x00],
        b'4' => [0x04, 0x0C, 0x14, 0x24, 0x3E, 0x04, 0x04, 0x00],
        b'5' => [0x3E, 0x20, 0x3C, 0x02, 0x02, 0x22, 0x1C, 0x00],
        b'6' => [0x0E, 0x10, 0x20, 0x3C, 0x22, 0x22, 0x1C, 0x00],
        b'7' => [0x3E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x10, 0x00],
        b'8' => [0x1C, 0x22, 0x22, 0x1C, 0x22, 0x22, 0x1C, 0x00],
        b'9' => [0x1C, 0x22, 0x22, 0x1E, 0x02, 0x04, 0x18, 0x00],
        b':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00, 0x00],
        b';' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x10, 0x00],
        b'<' => [0x04, 0x08, 0x10, 0x20, 0x10, 0x08, 0x04, 0x00],
        b'=' => [0x00, 0x00, 0x3E, 0x00, 0x3E, 0x00, 0x00, 0x00],
        b'>' => [0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10, 0x00],
        b'?' => [0x1C, 0x22, 0x04, 0x08, 0x08, 0x00, 0x08, 0x00],
        b'@' => [0x1C, 0x22, 0x2E, 0x2A, 0x2E, 0x20, 0x1C, 0x00],
        b'A' => [0x08, 0x14, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x00],
        b'B' => [0x3C, 0x22, 0x22, 0x3C, 0x22, 0x22, 0x3C, 0x00],
        b'C' => [0x1C, 0x22, 0x20, 0x20, 0x20, 0x22, 0x1C, 0x00],
        b'D' => [0x38, 0x24, 0x22, 0x22, 0x22, 0x24, 0x38, 0x00],
        b'E' => [0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x3E, 0x00],
        b'F' => [0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x20, 0x00],
        b'G' => [0x1C, 0x22, 0x20, 0x2E, 0x22, 0x22, 0x1E, 0x00],
        b'H' => [0x22, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x22, 0x00],
        b'I' => [0x1C, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00],
        b'J' => [0x0E, 0x04, 0x04, 0x04, 0x24, 0x24, 0x18, 0x00],
        b'K' => [0x22, 0x24, 0x28, 0x30, 0x28, 0x24, 0x22, 0x00],
        b'L' => [0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x3E, 0x00],
        b'M' => [0x22, 0x36, 0x2A, 0x2A, 0x22, 0x22, 0x22, 0x00],
        b'N' => [0x22, 0x32, 0x2A, 0x26, 0x22, 0x22, 0x22, 0x00],
        b'O' => [0x1C, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C, 0x00],
        b'P' => [0x3C, 0x22, 0x22, 0x3C, 0x20, 0x20, 0x20, 0x00],
        b'Q' => [0x1C, 0x22, 0x22, 0x22, 0x2A, 0x24, 0x1A, 0x00],
        b'R' => [0x3C, 0x22, 0x22, 0x3C, 0x28, 0x24, 0x22, 0x00],
        b'S' => [0x1C, 0x22, 0x20, 0x1C, 0x02, 0x22, 0x1C, 0x00],
        b'T' => [0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00],
        b'U' => [0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C, 0x00],
        b'V' => [0x22, 0x22, 0x22, 0x22, 0x14, 0x14, 0x08, 0x00],
        b'W' => [0x22, 0x22, 0x22, 0x2A, 0x2A, 0x36, 0x22, 0x00],
        b'X' => [0x22, 0x22, 0x14, 0x08, 0x14, 0x22, 0x22, 0x00],
        b'Y' => [0x22, 0x22, 0x14, 0x08, 0x08, 0x08, 0x08, 0x00],
        b'Z' => [0x3E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3E, 0x00],
        b'[' => [0x1C, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1C, 0x00],
        b'\\' => [0x20, 0x10, 0x08, 0x08, 0x04, 0x02, 0x00, 0x00],
        b']' => [0x1C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1C, 0x00],
        b'^' => [0x08, 0x14, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00],
        b'_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3E, 0x00],
        b'`' => [0x10, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00],
        b'a' => [0x00, 0x00, 0x1C, 0x02, 0x1E, 0x22, 0x1E, 0x00],
        b'b' => [0x20, 0x20, 0x3C, 0x22, 0x22, 0x22, 0x3C, 0x00],
        b'c' => [0x00, 0x00, 0x1C, 0x22, 0x20, 0x22, 0x1C, 0x00],
        b'd' => [0x02, 0x02, 0x1E, 0x22, 0x22, 0x22, 0x1E, 0x00],
        b'e' => [0x00, 0x00, 0x1C, 0x22, 0x3E, 0x20, 0x1E, 0x00],
        b'f' => [0x0C, 0x12, 0x10, 0x38, 0x10, 0x10, 0x10, 0x00],
        b'g' => [0x00, 0x00, 0x1E, 0x22, 0x1E, 0x02, 0x1C, 0x00],
        b'h' => [0x20, 0x20, 0x3C, 0x22, 0x22, 0x22, 0x22, 0x00],
        b'i' => [0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x1C, 0x00],
        b'j' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x24, 0x18, 0x00],
        b'k' => [0x20, 0x20, 0x24, 0x28, 0x30, 0x28, 0x24, 0x00],
        b'l' => [0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00],
        b'm' => [0x00, 0x00, 0x34, 0x2A, 0x2A, 0x22, 0x22, 0x00],
        b'n' => [0x00, 0x00, 0x3C, 0x22, 0x22, 0x22, 0x22, 0x00],
        b'o' => [0x00, 0x00, 0x1C, 0x22, 0x22, 0x22, 0x1C, 0x00],
        b'p' => [0x00, 0x00, 0x3C, 0x22, 0x3C, 0x20, 0x20, 0x00],
        b'q' => [0x00, 0x00, 0x1E, 0x22, 0x1E, 0x02, 0x02, 0x00],
        b'r' => [0x00, 0x00, 0x2C, 0x32, 0x20, 0x20, 0x20, 0x00],
        b's' => [0x00, 0x00, 0x1E, 0x20, 0x1C, 0x02, 0x3C, 0x00],
        b't' => [0x10, 0x10, 0x38, 0x10, 0x10, 0x12, 0x0C, 0x00],
        b'u' => [0x00, 0x00, 0x22, 0x22, 0x22, 0x26, 0x1A, 0x00],
        b'v' => [0x00, 0x00, 0x22, 0x22, 0x22, 0x14, 0x08, 0x00],
        b'w' => [0x00, 0x00, 0x22, 0x22, 0x2A, 0x2A, 0x14, 0x00],
        b'x' => [0x00, 0x00, 0x22, 0x14, 0x08, 0x14, 0x22, 0x00],
        b'y' => [0x00, 0x00, 0x22, 0x22, 0x1E, 0x02, 0x1C, 0x00],
        b'z' => [0x00, 0x00, 0x3E, 0x04, 0x08, 0x10, 0x3E, 0x00],
        b'{' => [0x0C, 0x10, 0x10, 0x20, 0x10, 0x10, 0x0C, 0x00],
        b'|' => [0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00],
        b'}' => [0x18, 0x04, 0x04, 0x02, 0x04, 0x04, 0x18, 0x00],
        b'~' => [0x00, 0x14, 0x2A, 0x00, 0x00, 0x00, 0x00, 0x00],
        _ => [0x1C, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C, 0x00],
    }
}
