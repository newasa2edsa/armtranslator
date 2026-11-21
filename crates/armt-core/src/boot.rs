use crate::isa::SysReg;

/// XNU `Boot_Video` structure (`pexpert/arm64/boot.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct XnuVideo {
    pub v_base_addr: u64,
    pub v_display: u64,
    pub v_row_bytes: u64,
    pub v_width: u64,
    pub v_height: u64,
    pub v_depth: u64,
}

/// Maximum boot command line length in Apple arm64 macOS `boot_args` (`BOOT_LINE_LENGTH = 1024`).
pub const XNU_BOOT_LINE_LENGTH: usize = 1024;

/// XNU ARM64 `boot_args` structure (`revision = 2`, `version = 2`).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct XnuBootArgs {
    pub revision: u16,
    pub version: u16,
    pub virt_base: u64,
    pub phys_base: u64,
    pub mem_size: u64,
    pub top_of_kernel_data: u64,
    pub video: XnuVideo,
    pub machine_type: u32,
    pub device_tree_p: u64,
    pub device_tree_length: u32,
    pub command_line: [u8; XNU_BOOT_LINE_LENGTH],
    pub boot_flags: u64,
    pub mem_size_actual: u64,
}

impl Default for XnuBootArgs {
    fn default() -> Self {
        Self {
            revision: 2,
            version: 2,
            virt_base: 0xFFFF_FE00_0700_0000,
            phys_base: 0x8000_0000,
            mem_size: 256 * 1024 * 1024,
            top_of_kernel_data: 0x8400_0000,
            video: XnuVideo::default(),
            machine_type: 0,
            device_tree_p: 0,
            device_tree_length: 0,
            command_line: [0u8; XNU_BOOT_LINE_LENGTH],
            boot_flags: 0,
            mem_size_actual: 256 * 1024 * 1024,
        }
    }
}

impl XnuBootArgs {
    pub fn set_command_line(&mut self, cmdline: &str) {
        self.command_line.fill(0);
        let bytes = cmdline.as_bytes();
        let len = bytes.len().min(XNU_BOOT_LINE_LENGTH - 1);
        self.command_line[..len].copy_from_slice(&bytes[..len]);
    }

    /// Serialize into standard LP64 C `struct boot_args` wire layout (1152 bytes, macOS ARM64 `BOOT_LINE_LENGTH=1024`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1152);
        out.extend_from_slice(&self.revision.to_le_bytes()); // +0
        out.extend_from_slice(&self.version.to_le_bytes()); // +2
        out.extend_from_slice(&0u32.to_le_bytes()); // +4 pad
        out.extend_from_slice(&self.virt_base.to_le_bytes()); // +8
        out.extend_from_slice(&self.phys_base.to_le_bytes()); // +16
        out.extend_from_slice(&self.mem_size.to_le_bytes()); // +24
        out.extend_from_slice(&self.top_of_kernel_data.to_le_bytes()); // +32
        out.extend_from_slice(&self.video.v_base_addr.to_le_bytes()); // +40
        out.extend_from_slice(&self.video.v_display.to_le_bytes()); // +48
        out.extend_from_slice(&self.video.v_row_bytes.to_le_bytes()); // +56
        out.extend_from_slice(&self.video.v_width.to_le_bytes()); // +64
        out.extend_from_slice(&self.video.v_height.to_le_bytes()); // +72
        out.extend_from_slice(&self.video.v_depth.to_le_bytes()); // +80
        out.extend_from_slice(&self.machine_type.to_le_bytes()); // +88
        out.extend_from_slice(&0u32.to_le_bytes()); // +92 pad
        out.extend_from_slice(&self.device_tree_p.to_le_bytes()); // +96
        out.extend_from_slice(&self.device_tree_length.to_le_bytes()); // +104
        out.extend_from_slice(&self.command_line); // +108 (1024 bytes -> 1132)
        out.extend_from_slice(&0u32.to_le_bytes()); // +1132 pad to 8B align
        out.extend_from_slice(&self.boot_flags.to_le_bytes()); // +1136
        out.extend_from_slice(&self.mem_size_actual.to_le_bytes()); // +1144
        out
    }
}

/// Parsed 64-bit Mach-O section summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachOSection {
    pub sectname: String,
    pub segname: String,
    pub addr: u64,
    pub size: u64,
    pub offset: u32,
}

/// Parsed 64-bit Mach-O segment (`LC_SEGMENT_64`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachOSegment {
    pub name: String,
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
    pub initprot: u32,
    pub maxprot: u32,
    pub sections: Vec<MachOSection>,
}

/// Parsed Mach-O image (standalone or `MH_FILESET` kernelcache).
#[derive(Debug, Clone)]
pub struct MachOImage {
    pub filetype: u32,
    pub entry_pc: Option<u64>,
    pub entry_offset: Option<u64>,
    pub segments: Vec<MachOSegment>,
    pub fileset_entries: Vec<String>,
    pub raw: Vec<u8>,
}

impl MachOImage {
    /// Lowest virtual address across all non-zero `LC_SEGMENT_64` segments (excluding `__PAGEZERO`).
    pub fn min_vmaddr(&self) -> Option<u64> {
        self.segments
            .iter()
            .filter(|s| s.vmsize > 0 && s.name != "__PAGEZERO")
            .map(|s| s.vmaddr)
            .min()
    }
}

/// Single property inside an Apple Device Tree (ADT) node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdtProperty {
    pub name: String,
    pub value: Vec<u8>,
}

/// Node in an Apple Device Tree (ADT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdtNode {
    pub name: String,
    pub properties: Vec<AdtProperty>,
    pub children: Vec<AdtNode>,
}

impl AdtNode {
    pub fn prop(&self, name: &str) -> Option<&[u8]> {
        self.properties
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.value.as_slice())
    }

    pub fn prop_str(&self, name: &str) -> Option<String> {
        let bytes = self.prop(name)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }

    pub fn prop_u32(&self, name: &str) -> Option<u32> {
        let b = self.prop(name)?;
        if b.len() < 4 {
            return None;
        }
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn prop_u64(&self, name: &str) -> Option<u64> {
        let b = self.prop(name)?;
        if b.len() < 8 {
            return None;
        }
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Decode `(base, size)` pairs from an ADT `reg` property (64-bit addr, 64-bit size).
    pub fn prop_regs(&self) -> Vec<(u64, u64)> {
        let Some(b) = self.prop("reg") else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for chunk in b.chunks_exact(16) {
            let addr = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let size = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
            out.push((addr, size));
        }
        out
    }

    /// Lookup a node by slash-separated path relative to this node (e.g. `"arm-io/uart0"`).
    pub fn find_path(&self, path: &str) -> Option<&AdtNode> {
        let trimmed = path.trim_matches('/');
        if trimmed.is_empty() {
            return Some(self);
        }
        let mut cur = self;
        for part in trimmed.split('/') {
            cur = cur
                .children
                .iter()
                .find(|c| c.name == part || c.name_prop() == Some(part))?;
        }
        Some(cur)
    }

    /// Lookup a mutable node by slash-separated path relative to this node.
    pub fn find_path_mut(&mut self, path: &str) -> Option<&mut AdtNode> {
        let trimmed = path.trim_matches('/');
        if trimmed.is_empty() {
            return Some(self);
        }
        let mut cur = self;
        for part in trimmed.split('/') {
            let idx = cur
                .children
                .iter()
                .position(|c| c.name == part || c.name_prop() == Some(part))?;
            cur = &mut cur.children[idx];
        }
        Some(cur)
    }

    /// Set an existing property's value or append a new property.
    pub fn set_prop(&mut self, name: &str, value: Vec<u8>) {
        if let Some(p) = self.properties.iter_mut().find(|p| p.name == name) {
            p.value = value;
        } else {
            self.properties.push(AdtProperty {
                name: name.to_string(),
                value,
            });
        }
    }

    pub fn name_prop(&self) -> Option<&str> {
        let bytes = self.prop("name")?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        std::str::from_utf8(&bytes[..end]).ok()
    }
}

/// macOS IPSW metadata entry fetched from Apple's official catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppleBuildInfo {
    pub model: String,
    pub product_version: String,
    pub build_version: String,
    pub firmware_url: String,
    pub sha1: Option<String>,
}

/// Ensure SysReg is referenced cleanly if needed by boot tooling.
pub type BootSysReg = SysReg;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_args_wire_layout() {
        let mut ba = XnuBootArgs::default();
        ba.set_command_line("-v serial=3");
        let bytes = ba.to_bytes();
        assert_eq!(bytes.len(), 1152);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 2);
        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            0x8000_0000
        );
        assert_eq!(&bytes[108..119], b"-v serial=3");
        assert_eq!(
            u64::from_le_bytes(bytes[1144..1152].try_into().unwrap()),
            256 * 1024 * 1024
        );
    }

    #[test]
    fn test_adt_find_path_and_regs() {
        let mut reg_bytes = Vec::new();
        reg_bytes.extend_from_slice(&0x2001_0000u64.to_le_bytes());
        reg_bytes.extend_from_slice(&0x1000u64.to_le_bytes());

        let uart = AdtNode {
            name: "uart0".into(),
            properties: vec![
                AdtProperty {
                    name: "name".into(),
                    value: b"uart0\0".to_vec(),
                },
                AdtProperty {
                    name: "reg".into(),
                    value: reg_bytes,
                },
            ],
            children: vec![],
        };
        let arm_io = AdtNode {
            name: "arm-io".into(),
            properties: vec![],
            children: vec![uart],
        };
        let root = AdtNode {
            name: "device-tree".into(),
            properties: vec![],
            children: vec![arm_io],
        };

        let found = root.find_path("arm-io/uart0").expect("should find uart0");
        assert_eq!(found.prop_regs(), vec![(0x2001_0000, 0x1000)]);
    }

    #[test]
    fn test_sysreg_encoding() {
        let sr = SysReg::SCTLR_EL1;
        assert_eq!(sr.fields(), (3, 0, 1, 0, 0));
        assert_eq!(sr.name(), "sctlr_el1");
        let apple = SysReg::from_fields(3, 5, 15, 0, 0);
        assert!(apple.is_apple_impl_defined());
    }
}
