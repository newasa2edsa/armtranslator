use armt_core::{ArmtError, Result};
use lzfse_rust::LzfseRingDecoder;
use std::io::Cursor;

/// Decompress Apple payloads (`bvx2`/`bvxn`/`bvx-` LZFSE/LZVN, `complzss`, or raw uncompressed).
pub fn decompress_payload(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() >= 4
        && (&data[..4] == b"bvx2" || &data[..4] == b"bvxn" || &data[..4] == b"bvx-")
    {
        let mut reader = Cursor::new(data);
        let mut out = Vec::with_capacity(data.len().saturating_mul(3));
        let mut decoder = LzfseRingDecoder::default();
        decoder
            .decode(&mut reader, &mut out)
            .map_err(|e| ArmtError::Decompress(format!("LZFSE decode failed: {e}")))?;
        return Ok(out);
    }

    if data.len() >= 0x180 && &data[..8] == b"complzss" {
        return decompress_lzss(data);
    }

    Ok(data.to_vec())
}

/// Decompress Apple `complzss` container (384-byte header + classic 4096-byte ring LZSS).
pub fn decompress_lzss(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 0x180 || &data[..8] != b"complzss" {
        return Err(ArmtError::Decompress("Invalid complzss header".into()));
    }
    let uncomp_size = u32::from_be_bytes(data[12..16].try_into().unwrap()) as usize;
    let comp_size = u32::from_be_bytes(data[16..20].try_into().unwrap()) as usize;
    let src_end = (0x180 + comp_size).min(data.len());
    let src = &data[0x180..src_end];

    const N: usize = 4096;
    const F: usize = 18;
    const THRESHOLD: usize = 2;

    let mut ring = [0x20u8; N + F - 1];
    let mut r = N - F;
    let mut flags: u32 = 0;
    let mut si = 0usize;
    let mut out = Vec::with_capacity(uncomp_size);

    while out.len() < uncomp_size && si < src.len() {
        flags >>= 1;
        if (flags & 0x100) == 0 {
            if si >= src.len() {
                break;
            }
            flags = (src[si] as u32) | 0xFF00;
            si += 1;
        }
        if (flags & 1) != 0 {
            if si >= src.len() {
                break;
            }
            let c = src[si];
            si += 1;
            out.push(c);
            ring[r] = c;
            r = (r + 1) & (N - 1);
        } else {
            if si + 1 >= src.len() {
                break;
            }
            let i_lo = src[si] as usize;
            let j_hi = src[si + 1] as usize;
            si += 2;
            let pos = i_lo | ((j_hi & 0xF0) << 4);
            let len = (j_hi & 0x0F) + THRESHOLD;
            for k in 0..=len {
                if out.len() >= uncomp_size {
                    break;
                }
                let c = ring[(pos + k) & (N - 1)];
                out.push(c);
                ring[r] = c;
                r = (r + 1) & (N - 1);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lzfse_rust::LzfseRingEncoder;

    #[test]
    fn test_lzfse_roundtrip() {
        let original = b"ArmTranslator Apple LZFSE payload verification string repeated! ".repeat(200);
        let mut compressed = Vec::new();
        let mut enc = LzfseRingEncoder::default();
        enc.encode(&mut Cursor::new(&original), &mut compressed)
            .unwrap();
        assert!(compressed.starts_with(b"bvx2") || compressed.starts_with(b"bvx-"));
        let decoded = decompress_payload(&compressed).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_lzss_literal_run() {
        let mut blob = vec![0u8; 0x180];
        blob[..8].copy_from_slice(b"complzss");
        blob[12..16].copy_from_slice(&4u32.to_be_bytes());
        blob[16..20].copy_from_slice(&5u32.to_be_bytes());
        blob.push(0x0F); // 4 literal bytes
        blob.extend_from_slice(b"ABCD");
        let out = decompress_payload(&blob).unwrap();
        assert_eq!(out, b"ABCD");
    }
}
