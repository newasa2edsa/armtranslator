use aes::cipher::{generic_array::GenericArray, BlockDecrypt, BlockEncrypt, KeyInit};
use aes::{Aes128, Aes192, Aes256};
use armt_core::{MmioDevice, Result};

/// `vmapple-cfg` platform descriptor device exposing `VirtualMac2,1` topology.
pub struct VmaCfgDevice {
    pub version: u32,
    pub vcpus: u32,
    pub ram_mb: u64,
    pub serial: [u8; 32],
    pub ecid: u64,
}

impl VmaCfgDevice {
    pub fn new(ram_mb: u64) -> Self {
        let mut serial = [0u8; 32];
        let s = b"ARMT-VMA2-0001";
        serial[..s.len()].copy_from_slice(s);
        Self {
            version: 1,
            vcpus: 1,
            ram_mb,
            serial,
            ecid: 0x4152_4D54_0000_0001,
        }
    }
}

impl MmioDevice for VmaCfgDevice {
    fn name(&self) -> &str {
        "vmapple-cfg"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        Ok(match offset {
            0x00 => self.version as u64,
            0x04 => self.vcpus as u64,
            0x08 => self.ram_mb,
            0x10 => self.ecid,
            _ => 0,
        })
    }

    fn write(&mut self, _offset: u64, _size: u8, _val: u64) -> Result<()> {
        Ok(())
    }
}

/// `vmapple-bdif` backdoor boot-dict & sandboxed auxiliary storage device.
pub struct VmaBdifDevice {
    aux: Vec<u8>,
    selected_offset: u64,
}

impl VmaBdifDevice {
    pub fn new(aux_size: usize) -> Self {
        Self {
            aux: vec![0u8; aux_size],
            selected_offset: 0,
        }
    }

    pub fn aux_len(&self) -> usize {
        self.aux.len()
    }
}

impl MmioDevice for VmaBdifDevice {
    fn name(&self) -> &str {
        "vmapple-bdif"
    }

    fn read(&mut self, offset: u64, size: u8) -> Result<u64> {
        if offset == 0x00 {
            return Ok(0x4244_4946); // "BDIF"
        }
        if offset == 0x08 {
            return Ok(self.aux.len() as u64);
        }
        let idx = (self.selected_offset as usize).saturating_add(offset as usize);
        if idx + (size as usize) <= self.aux.len() {
            let mut buf = [0u8; 8];
            buf[..size as usize].copy_from_slice(&self.aux[idx..idx + (size as usize)]);
            return Ok(u64::from_le_bytes(buf));
        }
        Ok(0)
    }

    fn write(&mut self, offset: u64, size: u8, val: u64) -> Result<()> {
        if offset == 0x04 {
            self.selected_offset = val;
            return Ok(());
        }
        let idx = (self.selected_offset as usize).saturating_add(offset as usize);
        if idx + (size as usize) <= self.aux.len() {
            let bytes = val.to_le_bytes();
            self.aux[idx..idx + (size as usize)].copy_from_slice(&bytes[..size as usize]);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AesBlockMode {
    Ecb,
    Cbc,
    Ctr,
}

#[derive(Debug, Clone)]
struct AesChannelState {
    key: [u8; 32],
    key_len: usize,
    iv: [u8; 16],
    mode: AesBlockMode,
    encrypt: bool,
}

impl Default for AesChannelState {
    fn default() -> Self {
        Self {
            key: [0u8; 32],
            key_len: 16,
            iv: [0u8; 16],
            mode: AesBlockMode::Cbc,
            encrypt: false,
        }
    }
}

/// `vmapple-aes` (`aes,s8000`) AES-128/192/256 hardware engine with FIFO/DMA + legacy registers.
pub struct VmaAesDevice {
    pub key: [u8; 16],
    pub data: [u8; 16],
    pub status: u32,
    pub write_count: u64,
    pub last_write: (u64, u64),
    ctrl: u32,
    int_status: u32,
    int_enable: u32,
    watermark: u32,
    flag_tag: u32,
    hw_iv: [u8; 16],
    fifo: Vec<u32>,
    fifo_expected: usize,
    active: AesChannelState,
}

impl Default for VmaAesDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl VmaAesDevice {
    pub fn new() -> Self {
        Self {
            key: [0u8; 16],
            data: [0u8; 16],
            status: 1,
            write_count: 0,
            last_write: (0, 0),
            ctrl: 0,
            int_status: 0,
            int_enable: 0,
            watermark: 0,
            flag_tag: 0,
            hw_iv: [0u8; 16],
            fifo: Vec::with_capacity(20),
            fifo_expected: 0,
            active: AesChannelState::default(),
        }
    }

    pub fn encrypt_block(key: &[u8; 16], block: &mut [u8; 16]) {
        let cipher = Aes128::new(GenericArray::from_slice(key));
        cipher.encrypt_block(GenericArray::from_mut_slice(block));
    }

    pub fn decrypt_block(key: &[u8; 16], block: &mut [u8; 16]) {
        let cipher = Aes128::new(GenericArray::from_slice(key));
        cipher.decrypt_block(GenericArray::from_mut_slice(block));
    }

    #[inline]
    pub fn irq_pending(&self) -> bool {
        (self.int_status & self.int_enable) != 0
    }

    fn aes_transform_block(&self, enc: bool, block: &mut [u8; 16]) {
        let arr = GenericArray::from_mut_slice(block);
        match self.active.key_len {
            24 => {
                let c = Aes192::new(GenericArray::from_slice(&self.active.key[..24]));
                if enc {
                    c.encrypt_block(arr);
                } else {
                    c.decrypt_block(arr);
                }
            }
            32 => {
                let c = Aes256::new(GenericArray::from_slice(&self.active.key[..32]));
                if enc {
                    c.encrypt_block(arr);
                } else {
                    c.decrypt_block(arr);
                }
            }
            _ => {
                let c = Aes128::new(GenericArray::from_slice(&self.active.key[..16]));
                if enc {
                    c.encrypt_block(arr);
                } else {
                    c.decrypt_block(arr);
                }
            }
        }
    }

    fn execute_dma(&mut self, len: usize, src_pa: u64, dst_pa: u64, ram: &mut [u8], ram_base: u64) {
        if len == 0 || ram.is_empty() || src_pa < ram_base || dst_pa < ram_base {
            return;
        }
        let src_off = (src_pa - ram_base) as usize;
        let dst_off = (dst_pa - ram_base) as usize;
        if src_off.saturating_add(len) > ram.len() || dst_off.saturating_add(len) > ram.len() {
            return;
        }
        let mut buf = ram[src_off..src_off + len].to_vec();
        match self.active.mode {
            AesBlockMode::Ecb => {
                for chunk in buf.chunks_exact_mut(16) {
                    let mut blk: [u8; 16] = chunk.try_into().unwrap();
                    self.aes_transform_block(self.active.encrypt, &mut blk);
                    chunk.copy_from_slice(&blk);
                }
            }
            AesBlockMode::Cbc => {
                if self.active.encrypt {
                    for chunk in buf.chunks_exact_mut(16) {
                        let mut blk: [u8; 16] = chunk.try_into().unwrap();
                        for (b, ivb) in blk.iter_mut().zip(self.active.iv.iter()) {
                            *b ^= *ivb;
                        }
                        self.aes_transform_block(true, &mut blk);
                        self.active.iv = blk;
                        chunk.copy_from_slice(&blk);
                    }
                } else {
                    for chunk in buf.chunks_exact_mut(16) {
                        let ct: [u8; 16] = chunk.try_into().unwrap();
                        let mut blk = ct;
                        self.aes_transform_block(false, &mut blk);
                        for (b, ivb) in blk.iter_mut().zip(self.active.iv.iter()) {
                            *b ^= *ivb;
                        }
                        self.active.iv = ct;
                        chunk.copy_from_slice(&blk);
                    }
                }
            }
            AesBlockMode::Ctr => {
                for chunk in buf.chunks_mut(16) {
                    let mut stream = self.active.iv;
                    self.aes_transform_block(true, &mut stream);
                    for (dst, ks) in chunk.iter_mut().zip(stream.iter()) {
                        *dst ^= *ks;
                    }
                    // Big-endian 128-bit counter increment
                    for byte in self.active.iv.iter_mut().rev() {
                        *byte = byte.wrapping_add(1);
                        if *byte != 0 {
                            break;
                        }
                    }
                }
            }
        }
        ram[dst_off..dst_off + len].copy_from_slice(&buf);
    }

    fn expected_fifo_words(w0: u32) -> usize {
        match w0 >> 28 {
            0x1 => {
                // AppleS8000AES key command:
                // - HW/key-slot when bits [26:24] != 0 -> 1 word
                // - Unwrap/context key command when bit 21 is set -> 17 words (0x44 bytes)
                // - Software key: 128/192/256 bits -> 5 / 7 / 9 words
                if (w0 & 0x0700_0000) != 0 {
                    1
                } else if (w0 & 0x0020_0000) != 0 {
                    17
                } else {
                    match (w0 >> 22) & 3 {
                        0 => 5,
                        1 => 7,
                        _ => 9,
                    }
                }
            }
            0x4 => 2, // Key-context / handle word
            0x5 => 4, // Data DMA command
            0x6 => 2, // IV DMA pointer
            0x8 => 1, // Flag / sync command
            0xD => 9, // Tweak / inline 256-bit key
            _ => 1,
        }
    }

    fn push_fifo_word(&mut self, w: u32, ram: &mut [u8], ram_base: u64) {
        if self.fifo.is_empty() {
            self.fifo_expected = Self::expected_fifo_words(w);
        }
        self.fifo.push(w);
        if self.fifo.len() < self.fifo_expected {
            return;
        }
        let words = std::mem::take(&mut self.fifo);
        self.fifo_expected = 0;
        let w0 = words[0];
        match w0 >> 28 {
            0x1 => {
                let enc = ((w0 >> 27) & 1) != 0;
                let key_len = match (w0 >> 22) & 3 {
                    0 => 16,
                    1 => 24,
                    _ => 32,
                };
                let mode = match (w0 >> 16) & 3 {
                    0 => AesBlockMode::Ecb,
                    2 => AesBlockMode::Ctr,
                    _ => AesBlockMode::Cbc,
                };
                self.active.encrypt = enc;
                self.active.key_len = key_len;
                self.active.mode = mode;
                if (w0 & 0x0700_0000) != 0 {
                    // Deterministic synthetic UID/GID/hardware device key
                    let tag = ((w0 >> 24) & 0xF) as u8 ^ (w0 & 0xFF) as u8;
                    for (i, b) in self.active.key.iter_mut().enumerate() {
                        *b = 0xA5u8.wrapping_add(tag.wrapping_mul(17)).wrapping_add(i as u8);
                    }
                } else if words.len() >= 17 {
                    // 0x44-byte command: key starts at word offset 9 (sp+64)
                    for (idx, word) in words[9..17].iter().enumerate() {
                        self.active.key[idx * 4..idx * 4 + 4]
                            .copy_from_slice(&word.to_le_bytes());
                    }
                } else {
                    for (idx, word) in words[1..].iter().take(8).enumerate() {
                        self.active.key[idx * 4..idx * 4 + 4]
                            .copy_from_slice(&word.to_le_bytes());
                    }
                }
            }
            0x6 => {
                if words.len() >= 2 {
                    let hi = (w0 & 0x3FF) as u64;
                    let lo = words[1] as u64;
                    let iv_pa = (hi << 32) | lo;
                    if iv_pa >= ram_base && !ram.is_empty() {
                        let off = (iv_pa - ram_base) as usize;
                        if off.saturating_add(16) <= ram.len() {
                            self.active.iv.copy_from_slice(&ram[off..off + 16]);
                        }
                    } else {
                        self.active.iv = self.hw_iv;
                    }
                }
            }
            0x5 => {
                if words.len() >= 4 {
                    let len = (w0 & 0x00FF_FFFF) as usize;
                    let dst_hi = (words[1] & 0x3FF) as u64;
                    let src_hi = ((words[1] >> 16) & 0x3FF) as u64;
                    let src_pa = (src_hi << 32) | (words[2] as u64);
                    let dst_pa = (dst_hi << 32) | (words[3] as u64);
                    self.execute_dma(len, src_pa, dst_pa, ram, ram_base);
                }
            }
            0x8 => {
                self.flag_tag = w0 & 0xFF;
                if (w0 & 0x0800_0000) != 0 {
                    self.int_status |= 0x20;
                }
            }
            0xD => {
                for (idx, word) in words[1..].iter().take(8).enumerate() {
                    self.active.key[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            _ => {}
        }
    }

    pub fn write_with_dma(
        &mut self,
        offset: u64,
        size: u8,
        val: u64,
        ram: &mut [u8],
        ram_base: u64,
    ) -> Result<()> {
        self.write_count += 1;
        self.last_write = (offset, val);
        match offset {
            0x04 => {
                // Legacy command: 1 = encrypt, 2 = decrypt
                let mut block = self.data;
                if val == 1 {
                    Self::encrypt_block(&self.key, &mut block);
                } else if val == 2 {
                    Self::decrypt_block(&self.key, &mut block);
                }
                self.data = block;
                self.status = 1;
            }
            0x08 => self.ctrl = val as u32,
            0x10 => self.key[0..8].copy_from_slice(&val.to_le_bytes()),
            0x18 if size <= 4 => self.int_status &= !(val as u32),
            0x18 => self.key[8..16].copy_from_slice(&val.to_le_bytes()),
            0x1C => self.int_enable = val as u32,
            0x20 if size <= 4 => self.watermark = val as u32,
            0x20 => self.data[0..8].copy_from_slice(&val.to_le_bytes()),
            0x28 => self.data[8..16].copy_from_slice(&val.to_le_bytes()),
            0x50 | 0x54 | 0x58 | 0x5C => {
                let idx = (offset - 0x50) as usize;
                self.hw_iv[idx..idx + 4].copy_from_slice(&(val as u32).to_le_bytes());
                self.active.iv = self.hw_iv;
            }
            0x100 | 0x200 => {
                self.push_fifo_word(val as u32, ram, ram_base);
                if size == 8 {
                    self.push_fifo_word((val >> 32) as u32, ram, ram_base);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl MmioDevice for VmaAesDevice {
    fn name(&self) -> &str {
        "vmapple-aes"
    }

    fn read(&mut self, offset: u64, size: u8) -> Result<u64> {
        let val = match offset {
            0x00 => self.status as u64,
            0x08 => self.ctrl as u64,
            // AppleS8000AES v2/v3 status: DPA entropy seeded & hardware ready bits [13:7]
            0x0C => 0x3F80,
            0x18 => self.int_status as u64,
            0x1C => self.int_enable as u64,
            0x20 if size <= 4 => self.watermark as u64,
            0x20 => u64::from_le_bytes(self.data[0..8].try_into().unwrap()),
            0x24 => 0, // QUEUE_STATUS: FIFO ready, bits [15:8] == 0
            0x28 => u64::from_le_bytes(self.data[8..16].try_into().unwrap()),
            0x30 => self.flag_tag as u64, // FLAG_COMMAND: bits [15:8] == 0 (not busy), low byte = completed tag
            _ => 0,
        };
        Ok(val)
    }

    fn write(&mut self, offset: u64, size: u8, val: u64) -> Result<()> {
        self.write_with_dma(offset, size, val, &mut [], 0)
    }
}

#[cfg(test)]
mod aes_s8000_tests {
    use super::*;

    #[test]
    fn test_s8000_aes_fifo_cbc_dma_and_irq() {
        let mut dev = VmaAesDevice::new();
        let mut ram = vec![0u8; 4096];
        let ram_base = 0x8000_0000u64;

        // Place 16-byte key, 16-byte IV at ram+0x100, plaintext at ram+0x200
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let iv: [u8; 16] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f];
        let pt: [u8; 32] = *b"AppleS8000AES_VirtualMac2_1_Test";
        ram[0x100..0x110].copy_from_slice(&iv);
        ram[0x200..0x220].copy_from_slice(&pt);

        dev.write_with_dma(0x1C, 4, 0x3EFFE, &mut ram, ram_base).unwrap();
        assert_eq!(dev.read(0x0C, 4).unwrap(), 0x3F80);

        // Encrypt AES-128 CBC: op=1, enc=1 (bit 27), len=128 (0), mode=CBC (bit 16)
        let key_cmd_enc = 0x1801_0000u64;
        dev.write_with_dma(0x200, 4, key_cmd_enc, &mut ram, ram_base).unwrap();
        for chunk in key.chunks_exact(4) {
            dev.write_with_dma(0x200, 4, u32::from_le_bytes(chunk.try_into().unwrap()) as u64, &mut ram, ram_base).unwrap();
        }
        // IV DMA pointer
        dev.write_with_dma(0x200, 4, 0x6000_0000, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, ram_base + 0x100, &mut ram, ram_base).unwrap();
        // DATA DMA: ram+0x200 -> ram+0x300, 32 bytes
        dev.write_with_dma(0x200, 4, 0x5000_0020, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, 0, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, ram_base + 0x200, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, ram_base + 0x300, &mut ram, ram_base).unwrap();
        // FLAG 1 with IRQ
        dev.write_with_dma(0x200, 4, 0x8800_0001, &mut ram, ram_base).unwrap();
        assert!(dev.irq_pending());
        assert_eq!(dev.read(0x30, 4).unwrap(), 1);
        assert_ne!(&ram[0x300..0x320], &pt);

        // W1C INT_STATUS
        let ist = dev.read(0x18, 4).unwrap();
        assert_ne!(ist & 0x20, 0);
        dev.write_with_dma(0x18, 4, ist, &mut ram, ram_base).unwrap();
        assert!(!dev.irq_pending());

        // Decrypt back to ram+0x400
        let key_cmd_dec = 0x1001_0000u64;
        dev.write_with_dma(0x200, 4, key_cmd_dec, &mut ram, ram_base).unwrap();
        for chunk in key.chunks_exact(4) {
            dev.write_with_dma(0x200, 4, u32::from_le_bytes(chunk.try_into().unwrap()) as u64, &mut ram, ram_base).unwrap();
        }
        dev.write_with_dma(0x200, 4, 0x6000_0000, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, ram_base + 0x100, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, 0x5000_0020, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, 0, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, ram_base + 0x300, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, ram_base + 0x400, &mut ram, ram_base).unwrap();
        dev.write_with_dma(0x200, 4, 0x8800_0002, &mut ram, ram_base).unwrap();
        assert_eq!(dev.read(0x30, 4).unwrap(), 2);
        assert_eq!(&ram[0x400..0x420], &pt);
    }
}

/// `pvpanic-mmio` guest crash / panic notifier.
pub struct PvPanicDevice {
    pub panicked: bool,
    pub last_event: u64,
}

impl Default for PvPanicDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl PvPanicDevice {
    pub fn new() -> Self {
        Self {
            panicked: false,
            last_event: 0,
        }
    }
}

impl MmioDevice for PvPanicDevice {
    fn name(&self) -> &str {
        "pvpanic-mmio"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        Ok(if offset == 0 { 0x3 } else { 0 })
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) -> Result<()> {
        if offset == 0 && val != 0 {
            self.panicked = true;
            self.last_event = val;
        }
        Ok(())
    }
}
