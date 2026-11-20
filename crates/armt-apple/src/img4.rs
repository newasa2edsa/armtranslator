use armt_core::{ArmtError, Result};

/// Parsed Apple IM4P payload container (optionally unwrapped from an outer IMG4 wrapper).
#[derive(Debug, Clone)]
pub struct Im4pContainer {
    pub tag: String,
    pub description: String,
    pub payload: Vec<u8>,
}

/// Read ASN.1 DER `(tag, content_start, content_len)` starting at `pos`.
fn parse_der_tl(data: &[u8], pos: usize) -> Result<(u8, usize, usize)> {
    if pos >= data.len() {
        return Err(ArmtError::AppleContainer("Unexpected EOF in DER TLV".into()));
    }
    let tag = data[pos];
    if pos + 1 >= data.len() {
        return Err(ArmtError::AppleContainer("Missing DER length byte".into()));
    }
    let len_byte = data[pos + 1];
    if len_byte < 0x80 {
        let len = len_byte as usize;
        let start = pos + 2;
        if start + len > data.len() {
            return Err(ArmtError::AppleContainer("Truncated DER element".into()));
        }
        Ok((tag, start, len))
    } else {
        let count = (len_byte & 0x7F) as usize;
        if count == 0 || count > 8 || pos + 2 + count > data.len() {
            return Err(ArmtError::AppleContainer(
                "Invalid long-form DER length".into(),
            ));
        }
        let mut len = 0usize;
        for i in 0..count {
            len = (len << 8) | (data[pos + 2 + i] as usize);
        }
        let start = pos + 2 + count;
        if start + len > data.len() {
            return Err(ArmtError::AppleContainer(
                "Truncated long-form DER element".into(),
            ));
        }
        Ok((tag, start, len))
    }
}

/// Parse an Apple `IMG4` or `IM4P` ASN.1 DER container into its fourcc tag, description, and raw payload.
pub fn parse_img4_or_im4p(data: &[u8]) -> Result<Im4pContainer> {
    let (outer_tag, outer_start, outer_len) = parse_der_tl(data, 0)?;
    if outer_tag != 0x30 {
        return Err(ArmtError::AppleContainer(format!(
            "Expected ASN.1 SEQUENCE (0x30), got 0x{outer_tag:02x}"
        )));
    }
    let outer_end = outer_start + outer_len;

    let (t0, s0, l0) = parse_der_tl(data, outer_start)?;
    if t0 != 0x16 {
        return Err(ArmtError::AppleContainer(
            "First element of IMG4/IM4P must be IA5String".into(),
        ));
    }
    let magic = std::str::from_utf8(&data[s0..s0 + l0])
        .map_err(|_| ArmtError::AppleContainer("Non-UTF8 IM4P/IMG4 magic".into()))?;

    if magic == "IMG4" {
        let next_pos = s0 + l0;
        if next_pos >= outer_end {
            return Err(ArmtError::AppleContainer(
                "IMG4 container missing inner IM4P element".into(),
            ));
        }
        return parse_img4_or_im4p(&data[next_pos..outer_end]);
    }
    if magic != "IM4P" {
        return Err(ArmtError::AppleContainer(format!(
            "Expected IM4P or IMG4 magic, got '{magic}'"
        )));
    }

    let pos1 = s0 + l0;
    let (t1, s1, l1) = parse_der_tl(data, pos1)?;
    if t1 != 0x16 {
        return Err(ArmtError::AppleContainer("Missing IM4P fourcc tag".into()));
    }
    let tag = String::from_utf8_lossy(&data[s1..s1 + l1]).into_owned();

    let pos2 = s1 + l1;
    let (t2, s2, l2) = parse_der_tl(data, pos2)?;
    if t2 != 0x16 {
        return Err(ArmtError::AppleContainer("Missing IM4P description".into()));
    }
    let description = String::from_utf8_lossy(&data[s2..s2 + l2]).into_owned();

    let pos3 = s2 + l2;
    if pos3 >= outer_end {
        return Err(ArmtError::AppleContainer("Missing IM4P payload".into()));
    }
    let (t3, s3, l3) = parse_der_tl(data, pos3)?;
    if t3 != 0x04 {
        return Err(ArmtError::AppleContainer(format!(
            "Expected IM4P OCTET STRING (0x04) payload, got 0x{t3:02x}"
        )));
    }

    Ok(Im4pContainer {
        tag,
        description,
        payload: data[s3..s3 + l3].to_vec(),
    })
}

/// Build a synthetic IM4P DER blob (used in tests).
pub fn encode_im4p(tag: &str, description: &str, payload: &[u8]) -> Vec<u8> {
    fn der_elem(tag: u8, body: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        let len = body.len();
        if len < 0x80 {
            out.push(len as u8);
        } else if len <= 0xFF {
            out.push(0x81);
            out.push(len as u8);
        } else if len <= 0xFFFF {
            out.push(0x82);
            out.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            out.push(0x84);
            out.extend_from_slice(&(len as u32).to_be_bytes());
        }
        out.extend_from_slice(body);
        out
    }

    let mut inner = Vec::new();
    inner.extend_from_slice(&der_elem(0x16, b"IM4P"));
    inner.extend_from_slice(&der_elem(0x16, tag.as_bytes()));
    inner.extend_from_slice(&der_elem(0x16, description.as_bytes()));
    inner.extend_from_slice(&der_elem(0x04, payload));
    der_elem(0x30, &inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im4p_roundtrip() {
        let blob = encode_im4p("krnl", "KernelManagement_host-514", b"bvx2dummy");
        let parsed = parse_img4_or_im4p(&blob).unwrap();
        assert_eq!(parsed.tag, "krnl");
        assert_eq!(parsed.description, "KernelManagement_host-514");
        assert_eq!(parsed.payload, b"bvx2dummy");
    }
}
