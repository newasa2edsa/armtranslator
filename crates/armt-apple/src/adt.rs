use armt_core::{AdtNode, AdtProperty, ArmtError, Result};

/// Parse an uncompressed Apple Device Tree (ADT) binary blob into an `AdtNode` tree.
pub fn parse_adt(data: &[u8]) -> Result<AdtNode> {
    let mut cursor = 0usize;
    parse_node(data, &mut cursor)
}

/// Finalize a raw IPSW Apple Device Tree binary blob in-place for XNU boot.
///
/// Raw `DeviceTree.*.im4p` templates mark bootloader-filled placeholder properties
/// with bit 31 (`0x8000_0000`) set in `DeviceTreeNodeProperty.length`.
/// `AVPBooter` / `iBoot` clears bit 31 before passing the ADT to XNU; otherwise
/// XNU's `device_tree.c` panics with "Device tree pointer outside of device tree region".
/// Returns the number of placeholder property flags cleared.
pub fn finalize_adt_for_xnu(data: &mut [u8]) -> Result<usize> {
    let mut cursor = 0usize;
    let mut cleared = 0usize;
    finalize_node(data, &mut cursor, &mut cleared)?;
    Ok(cleared)
}

fn finalize_node(data: &mut [u8], cursor: &mut usize, cleared: &mut usize) -> Result<()> {
    let n_props = read_u32_le(data, cursor)? as usize;
    let n_children = read_u32_le(data, cursor)? as usize;
    if n_props > 4096 || n_children > 4096 {
        return Err(ArmtError::Adt(format!(
            "Unreasonable ADT node counts: props={n_props}, children={n_children}"
        )));
    }
    for _ in 0..n_props {
        if *cursor + 36 > data.len() {
            return Err(ArmtError::Adt("Truncated ADT property header".into()));
        }
        *cursor += 32; // name[32]
        let len_off = *cursor;
        let raw_len = read_u32_le(data, cursor)?;
        if (raw_len & 0x8000_0000) != 0 {
            let clean = raw_len & 0x7FFF_FFFF;
            data[len_off..len_off + 4].copy_from_slice(&clean.to_le_bytes());
            *cleared += 1;
        }
        let size = (raw_len & 0x7FFF_FFFF) as usize;
        let padded = (size + 3) & !3;
        if *cursor + padded > data.len() {
            return Err(ArmtError::Adt(format!(
                "Truncated ADT property value (size={size})"
            )));
        }
        *cursor += padded;
    }
    for _ in 0..n_children {
        finalize_node(data, cursor, cleared)?;
    }
    Ok(())
}

/// Serialize an `AdtNode` tree back into Apple's binary Device Tree (ADT) wire format
/// with clean 32-bit property lengths (bit 31 cleared) and 4-byte value padding.
pub fn serialize_adt(node: &AdtNode) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 * 1024);
    write_node(node, &mut out);
    out
}

fn write_node(node: &AdtNode, out: &mut Vec<u8>) {
    out.extend_from_slice(&(node.properties.len() as u32).to_le_bytes());
    out.extend_from_slice(&(node.children.len() as u32).to_le_bytes());
    for prop in &node.properties {
        let mut name_buf = [0u8; 32];
        let nb = prop.name.as_bytes();
        let nlen = nb.len().min(31);
        name_buf[..nlen].copy_from_slice(&nb[..nlen]);
        out.extend_from_slice(&name_buf);
        let len = (prop.value.len() as u32) & 0x7FFF_FFFF;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&prop.value);
        let pad = (4 - (prop.value.len() & 3)) & 3;
        for _ in 0..pad {
            out.push(0);
        }
    }
    for child in &node.children {
        write_node(child, out);
    }
}

fn read_u32_le(data: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > data.len() {
        return Err(ArmtError::Adt("Unexpected EOF reading ADT u32".into()));
    }
    let v = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn parse_node(data: &[u8], cursor: &mut usize) -> Result<AdtNode> {
    let n_props = read_u32_le(data, cursor)? as usize;
    let n_children = read_u32_le(data, cursor)? as usize;
    if n_props > 4096 || n_children > 4096 {
        return Err(ArmtError::Adt(format!(
            "Unreasonable ADT node counts: props={n_props}, children={n_children}"
        )));
    }

    let mut properties = Vec::with_capacity(n_props);
    let mut node_name = None;

    for _ in 0..n_props {
        if *cursor + 36 > data.len() {
            return Err(ArmtError::Adt("Truncated ADT property header".into()));
        }
        let name_bytes = &data[*cursor..*cursor + 32];
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(32);
        let prop_name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();
        *cursor += 32;

        let raw_len = read_u32_le(data, cursor)?;
        let size = (raw_len & 0x7FFF_FFFF) as usize;
        let padded = (size + 3) & !3;
        if *cursor + padded > data.len() {
            return Err(ArmtError::Adt(format!(
                "Truncated ADT property '{prop_name}' value (size={size})"
            )));
        }
        let value = data[*cursor..*cursor + size].to_vec();
        *cursor += padded;

        if prop_name == "name" {
            let end = value.iter().position(|&b| b == 0).unwrap_or(value.len());
            node_name = Some(String::from_utf8_lossy(&value[..end]).into_owned());
        }
        properties.push(AdtProperty {
            name: prop_name,
            value,
        });
    }

    let mut children = Vec::with_capacity(n_children);
    for _ in 0..n_children {
        children.push(parse_node(data, cursor)?);
    }

    Ok(AdtNode {
        name: node_name.unwrap_or_else(|| "unknown".into()),
        properties,
        children,
    })
}

/// Format an ADT tree cleanly for CLI inspection.
pub fn dump_adt(node: &AdtNode, max_depth: usize) -> String {
    fn fmt_rec(node: &AdtNode, depth: usize, max_depth: usize, out: &mut String) {
        let indent = "  ".repeat(depth);
        let compat = node
            .prop_str("compatible")
            .map(|c| format!(" compatible=\"{c}\""))
            .unwrap_or_default();
        let regs = node.prop_regs();
        let reg_str = if !regs.is_empty() {
            let parts: Vec<String> = regs.iter().map(|(b, s)| format!("0x{b:x}+0x{s:x}")).collect();
            format!(" reg=[{}]", parts.join(", "))
        } else {
            String::new()
        };
        let ranges_str = if let Some(prop) = node.properties.iter().find(|p| p.name == "ranges") {
            let words: Vec<String> = prop
                .value
                .chunks_exact(4)
                .map(|c| format!("0x{:08x}", u32::from_le_bytes(c.try_into().unwrap())))
                .collect();
            format!(" ranges=[{}]", words.join(" "))
        } else {
            String::new()
        };
        out.push_str(&format!(
            "{indent}+ {}{compat}{reg_str}{ranges_str}\n",
            node.name
        ));
        if matches!(
            node.name.as_str(),
            "chosen"
                | "memory-map"
                | "iBoot"
                | "options"
                | "defaults"
                | "vram"
                | "pram"
                | "gic"
                | "gfx"
                | "iosurface-mapper"
                | "aes"
                | "uart0"
                | "avp-rtc"
                | "avp-ctrr"
                | "pcie"
                | "pl061-buttons"
        ) {
            for prop in &node.properties {
                if prop.name == "name" || prop.name == "compatible" {
                    continue;
                }
                let preview = if prop.value.len() <= 32
                    && prop.value.ends_with(&[0])
                    && prop.value[..prop.value.len().saturating_sub(1)]
                        .iter()
                        .all(|&b| (0x20..0x7f).contains(&b))
                {
                    format!(
                        "\"{}\"",
                        String::from_utf8_lossy(&prop.value[..prop.value.len().saturating_sub(1)])
                    )
                } else if prop.value.len() == 4 {
                    format!("0x{:08x}", u32::from_le_bytes(prop.value.clone().try_into().unwrap()))
                } else if prop.value.len() == 8 {
                    format!("0x{:016x}", u64::from_le_bytes(prop.value.clone().try_into().unwrap()))
                } else {
                    format!("<{} bytes>", prop.value.len())
                };
                out.push_str(&format!("{indent}    .{} = {}\n", prop.name, preview));
            }
        }
        if depth < max_depth {
            for child in &node.children {
                fmt_rec(child, depth + 1, max_depth, out);
            }
        } else if !node.children.is_empty() {
            out.push_str(&format!(
                "{indent}  ... ({} children)\n",
                node.children.len()
            ));
        }
    }
    let mut out = String::new();
    fmt_rec(node, 0, max_depth, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_adt_binary() {
        let mut blob = Vec::new();
        // Root node: 2 props, 0 children
        blob.extend_from_slice(&2u32.to_le_bytes());
        blob.extend_from_slice(&0u32.to_le_bytes());
        // Prop 1: name = "device-tree\0"
        let mut pname = [0u8; 32];
        pname[..4].copy_from_slice(b"name");
        blob.extend_from_slice(&pname);
        let val = b"device-tree\0";
        blob.extend_from_slice(&(val.len() as u32).to_le_bytes());
        blob.extend_from_slice(val);
        // Prop 2: model = "VirtualMac2,1\0" (14 bytes -> pad to 16)
        let mut pmodel = [0u8; 32];
        pmodel[..5].copy_from_slice(b"model");
        blob.extend_from_slice(&pmodel);
        let mval = b"VirtualMac2,1\0";
        blob.extend_from_slice(&(mval.len() as u32).to_le_bytes());
        blob.extend_from_slice(mval);
        blob.extend_from_slice(&[0u8; 2]);

        let root = parse_adt(&blob).unwrap();
        assert_eq!(root.name, "device-tree");
        assert_eq!(root.prop_str("model").as_deref(), Some("VirtualMac2,1"));
    }
}
