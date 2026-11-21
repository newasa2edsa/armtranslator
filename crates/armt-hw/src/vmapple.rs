use aes::cipher::{generic_array::GenericArray, BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes128;
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

/// `vmapple-aes` AES-128 hardware acceleration engine backed by pure-Rust `aes`.
pub struct VmaAesDevice {
    pub key: [u8; 16],
    pub data: [u8; 16],
    pub status: u32,
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
}

impl MmioDevice for VmaAesDevice {
    fn name(&self) -> &str {
        "vmapple-aes"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        let val = match offset {
            0x00 => self.status as u64,
            // AppleS8000AES v2/v3 status: DPA entropy seeded & hardware ready bits [13:7]
            0x0C => 0x3F80,
            0x20 => u64::from_le_bytes(self.data[0..8].try_into().unwrap()),
            0x28 => u64::from_le_bytes(self.data[8..16].try_into().unwrap()),
            _ => 0,
        };
        Ok(val)
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) -> Result<()> {
        match offset {
            0x04 => {
                // Command: 1 = encrypt, 2 = decrypt
                let mut block = self.data;
                if val == 1 {
                    Self::encrypt_block(&self.key, &mut block);
                } else if val == 2 {
                    Self::decrypt_block(&self.key, &mut block);
                }
                self.data = block;
                self.status = 1;
            }
            0x10 => self.key[0..8].copy_from_slice(&val.to_le_bytes()),
            0x18 => self.key[8..16].copy_from_slice(&val.to_le_bytes()),
            0x20 => self.data[0..8].copy_from_slice(&val.to_le_bytes()),
            0x28 => self.data[8..16].copy_from_slice(&val.to_le_bytes()),
            _ => {}
        }
        Ok(())
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
