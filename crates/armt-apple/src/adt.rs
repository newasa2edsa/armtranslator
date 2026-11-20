use armt_core::{AdtNode, AdtProperty, ArmtError, Result};

/// Parse an uncompressed Apple Device Tree (ADT) binary blob into an `AdtNode` tree.
pub fn parse_adt(data: &[u8]) -> Result<AdtNode> {
    let mut cursor = 0usize;
    parse_node(data, &mut cursor)
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
            let (base, size) = regs[0];
            format!(" reg=0x{base:x}+0x{size:x}")
        } else {
            String::new()
        };
        out.push_str(&format!("{indent}+ {}{compat}{reg_str}\n", node.name));
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
