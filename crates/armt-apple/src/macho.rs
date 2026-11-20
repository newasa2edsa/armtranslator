use armt_core::{ArmtError, MachOImage, MachOSection, MachOSegment, Result};

const MH_MAGIC_64: u32 = 0xFEED_FACF;
const FAT_MAGIC: u32 = 0xCAFE_BABE;
const FAT_CIGAM: u32 = 0xBEBA_FECA;
const CPU_TYPE_ARM64: u32 = 0x0100_000C;

const LC_UNIXTHREAD: u32 = 0x5;
const LC_SEGMENT_64: u32 = 0x19;
const LC_MAIN: u32 = 0x8000_0028;
const LC_FILESET_ENTRY: u32 = 0x8000_0035;

/// Parse a 64-bit ARM64 Mach-O or `MH_FILESET` kernelcache (including FAT universal wrappers).
pub fn parse_macho(data: &[u8]) -> Result<MachOImage> {
    if data.len() < 32 {
        return Err(ArmtError::MachO("Input too small for Mach-O header".into()));
    }

    let magic_be = u32::from_be_bytes(data[0..4].try_into().unwrap());
    let slice = if magic_be == FAT_MAGIC || magic_be == FAT_CIGAM {
        extract_arm64_fat_slice(data)?
    } else {
        data
    };

    if slice.len() < 32 {
        return Err(ArmtError::MachO("Truncated 64-bit Mach-O header".into()));
    }
    let magic = u32::from_le_bytes(slice[0..4].try_into().unwrap());
    if magic != MH_MAGIC_64 {
        return Err(ArmtError::MachO(format!(
            "Unsupported Mach-O magic 0x{magic:08x} (expected 0x{MH_MAGIC_64:08x})"
        )));
    }

    let cputype = u32::from_le_bytes(slice[4..8].try_into().unwrap());
    if cputype != CPU_TYPE_ARM64 {
        return Err(ArmtError::MachO(format!(
            "Expected ARM64 cputype 0x{CPU_TYPE_ARM64:08x}, got 0x{cputype:08x}"
        )));
    }

    let filetype = u32::from_le_bytes(slice[12..16].try_into().unwrap());
    let ncmds = u32::from_le_bytes(slice[16..20].try_into().unwrap()) as usize;
    let sizeofcmds = u32::from_le_bytes(slice[20..24].try_into().unwrap()) as usize;

    let mut pos = 32usize;
    let cmds_end = (pos + sizeofcmds).min(slice.len());
    let mut segments = Vec::new();
    let mut fileset_entries = Vec::new();
    let mut entry_pc = None;
    let mut entry_offset = None;

    for _ in 0..ncmds {
        if pos + 8 > cmds_end {
            break;
        }
        let cmd = u32::from_le_bytes(slice[pos..pos + 4].try_into().unwrap());
        let cmdsize = u32::from_le_bytes(slice[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if cmdsize < 8 || pos + cmdsize > slice.len() {
            return Err(ArmtError::MachO("Invalid Mach-O load command size".into()));
        }
        let cmd_bytes = &slice[pos..pos + cmdsize];

        match cmd {
            LC_SEGMENT_64 if cmd_bytes.len() >= 72 => {
                let name = read_fixed_str(&cmd_bytes[8..24]);
                let vmaddr = u64::from_le_bytes(cmd_bytes[24..32].try_into().unwrap());
                let vmsize = u64::from_le_bytes(cmd_bytes[32..40].try_into().unwrap());
                let fileoff = u64::from_le_bytes(cmd_bytes[40..48].try_into().unwrap());
                let filesize = u64::from_le_bytes(cmd_bytes[48..56].try_into().unwrap());
                let maxprot = u32::from_le_bytes(cmd_bytes[56..60].try_into().unwrap());
                let initprot = u32::from_le_bytes(cmd_bytes[60..64].try_into().unwrap());
                let nsects = u32::from_le_bytes(cmd_bytes[64..68].try_into().unwrap()) as usize;

                let mut sections = Vec::with_capacity(nsects.min(256));
                let mut sp = 72usize;
                for _ in 0..nsects {
                    if sp + 80 > cmd_bytes.len() {
                        break;
                    }
                    let sectname = read_fixed_str(&cmd_bytes[sp..sp + 16]);
                    let segname = read_fixed_str(&cmd_bytes[sp + 16..sp + 32]);
                    let addr = u64::from_le_bytes(cmd_bytes[sp + 32..sp + 40].try_into().unwrap());
                    let size = u64::from_le_bytes(cmd_bytes[sp + 40..sp + 48].try_into().unwrap());
                    let offset = u32::from_le_bytes(cmd_bytes[sp + 48..sp + 52].try_into().unwrap());
                    sections.push(MachOSection {
                        sectname,
                        segname,
                        addr,
                        size,
                        offset,
                    });
                    sp += 80;
                }

                segments.push(MachOSegment {
                    name,
                    vmaddr,
                    vmsize,
                    fileoff,
                    filesize,
                    initprot,
                    maxprot,
                    sections,
                });
            }
            LC_UNIXTHREAD if cmd_bytes.len() >= 16 + 272 => {
                let flavor = u32::from_le_bytes(cmd_bytes[8..12].try_into().unwrap());
                if flavor == 6 {
                    // ARM_THREAD_STATE64: pc is at index 32 (byte offset 256 in state, +16 cmd hdr = 272)
                    let pc = u64::from_le_bytes(cmd_bytes[272..280].try_into().unwrap());
                    entry_pc = Some(pc);
                }
            }
            LC_MAIN if cmd_bytes.len() >= 16 => {
                let off = u64::from_le_bytes(cmd_bytes[8..16].try_into().unwrap());
                entry_offset = Some(off);
            }
            LC_FILESET_ENTRY if cmd_bytes.len() >= 32 => {
                let str_off = u32::from_le_bytes(cmd_bytes[24..28].try_into().unwrap()) as usize;
                if str_off < cmd_bytes.len() {
                    let end = cmd_bytes[str_off..]
                        .iter()
                        .position(|&b| b == 0)
                        .map(|p| str_off + p)
                        .unwrap_or(cmd_bytes.len());
                    fileset_entries
                        .push(String::from_utf8_lossy(&cmd_bytes[str_off..end]).into_owned());
                }
            }
            _ => {}
        }

        pos += cmdsize;
    }

    Ok(MachOImage {
        filetype,
        entry_pc,
        entry_offset,
        segments,
        fileset_entries,
        raw: slice.to_vec(),
    })
}

fn extract_arm64_fat_slice(data: &[u8]) -> Result<&[u8]> {
    if data.len() < 8 {
        return Err(ArmtError::MachO("Truncated FAT header".into()));
    }
    let nfat = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
    let mut off = 8usize;
    for _ in 0..nfat.min(16) {
        if off + 20 > data.len() {
            break;
        }
        let cputype = u32::from_be_bytes(data[off..off + 4].try_into().unwrap());
        let slice_off = u32::from_be_bytes(data[off + 8..off + 12].try_into().unwrap()) as usize;
        let slice_sz = u32::from_be_bytes(data[off + 12..off + 16].try_into().unwrap()) as usize;
        if cputype == CPU_TYPE_ARM64 && slice_off + slice_sz <= data.len() {
            return Ok(&data[slice_off..slice_off + slice_sz]);
        }
        off += 20;
    }
    Err(ArmtError::MachO(
        "No ARM64 slice found inside FAT binary".into(),
    ))
}

fn read_fixed_str(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_macho64() {
        let mut buf = vec![0u8; 4096];
        buf[0..4].copy_from_slice(&MH_MAGIC_64.to_le_bytes());
        buf[4..8].copy_from_slice(&CPU_TYPE_ARM64.to_le_bytes());
        buf[12..16].copy_from_slice(&2u32.to_le_bytes()); // MH_EXECUTE
        buf[16..20].copy_from_slice(&2u32.to_le_bytes()); // ncmds = 2
        buf[20..24].copy_from_slice(&(72 + 288u32).to_le_bytes());

        // LC_SEGMENT_64 __TEXT at offset 32 (72 bytes)
        buf[32..36].copy_from_slice(&LC_SEGMENT_64.to_le_bytes());
        buf[36..40].copy_from_slice(&72u32.to_le_bytes());
        buf[40..46].copy_from_slice(b"__TEXT");
        buf[56..64].copy_from_slice(&0xFFFF_FE00_0700_0000u64.to_le_bytes());
        buf[64..72].copy_from_slice(&0x4000u64.to_le_bytes());
        buf[72..80].copy_from_slice(&0u64.to_le_bytes());
        buf[80..88].copy_from_slice(&0x1000u64.to_le_bytes());

        // LC_UNIXTHREAD at offset 104 (288 bytes)
        buf[104..108].copy_from_slice(&LC_UNIXTHREAD.to_le_bytes());
        buf[108..112].copy_from_slice(&288u32.to_le_bytes());
        buf[112..116].copy_from_slice(&6u32.to_le_bytes()); // ARM_THREAD_STATE64
        buf[116..120].copy_from_slice(&68u32.to_le_bytes());
        buf[104 + 272..104 + 280].copy_from_slice(&0xFFFF_FE00_0700_0800u64.to_le_bytes());

        let img = parse_macho(&buf).unwrap();
        assert_eq!(img.segments.len(), 1);
        assert_eq!(img.segments[0].name, "__TEXT");
        assert_eq!(img.entry_pc, Some(0xFFFF_FE00_0700_0800));
    }
}
