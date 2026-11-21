use armt_core::{ArmtError, Result};
use flate2::read::DeflateDecoder;
use sha1::{Digest, Sha1};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;

/// Entry metadata parsed from a ZIP / ZIP64 Central Directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipEntry {
    pub name: String,
    pub compression_method: u16,
    pub crc32: u32,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub local_header_offset: u64,
}

/// Pure-Rust HTTP Range ZIP64 reader for extracting individual files from Apple's ~25 GB IPSWs
/// without downloading the full archive.
pub struct RemoteZip {
    pub url: String,
    pub total_size: u64,
    pub entries: Vec<ZipEntry>,
    agent: ureq::Agent,
}

impl RemoteZip {
    /// Open a remote ZIP/IPSW URL, inspect its EOCD/ZIP64 structures, and download the Central Directory.
    pub fn open(url: &str) -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(300))
            .build();

        let head = agent
            .head(url)
            .call()
            .map_err(|e| ArmtError::Network(format!("HEAD failed for {url}: {e}")))?;

        let total_size: u64 = head
            .header("Content-Length")
            .ok_or_else(|| ArmtError::Network("Missing Content-Length header on IPSW".into()))?
            .parse()
            .map_err(|e| ArmtError::Network(format!("Invalid Content-Length: {e}")))?;

        let tail_len = total_size.min(65536);
        let tail_start = total_size - tail_len;
        let tail = fetch_range(&agent, url, tail_start, total_size - 1)?;

        let (cd_offset, cd_size, z64_loc_offset) = parse_eocd_tail(&tail, tail_start)?;
        let (cd_offset, cd_size) = if let Some(z64_off) = z64_loc_offset {
            let z64_end = (z64_off + 127).min(total_size - 1);
            let z64 = fetch_range(&agent, url, z64_off, z64_end)?;
            parse_zip64_eocd(&z64)?
        } else {
            (cd_offset, cd_size)
        };

        if cd_size == 0 || cd_offset + cd_size > total_size {
            return Err(ArmtError::AppleContainer(format!(
                "Invalid ZIP Central Directory bounds: off={cd_offset}, size={cd_size}, total={total_size}"
            )));
        }

        let cd_bytes = fetch_range(&agent, url, cd_offset, cd_offset + cd_size - 1)?;
        let entries = parse_central_directory(&cd_bytes)?;

        Ok(Self {
            url: url.to_string(),
            total_size,
            entries,
            agent,
        })
    }

    /// Find an entry by exact path or case-insensitive suffix/substring.
    pub fn find_entry(&self, name: &str) -> Option<&ZipEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Extract a single entry by exact path using HTTP Range requests.
    pub fn extract_entry(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self
            .find_entry(name)
            .ok_or_else(|| ArmtError::AppleContainer(format!("ZIP entry not found: {name}")))?;

        let off = entry.local_header_offset;
        // Fetch local header + extra padding (or full compressed payload if small enough) in 1 request when possible.
        let pre_fetch = 512u64 + entry.compressed_size;
        let end = (off + pre_fetch - 1).min(self.total_size - 1);
        let mut buf = fetch_range(&self.agent, &self.url, off, end)?;

        if buf.len() < 30 || &buf[0..4] != b"PK\x03\x04" {
            return Err(ArmtError::AppleContainer(format!(
                "Invalid local file header signature for {name}"
            )));
        }
        let lnlen = u16::from_le_bytes([buf[26], buf[27]]) as usize;
        let lelen = u16::from_le_bytes([buf[28], buf[29]]) as usize;
        let data_start = 30 + lnlen + lelen;
        let needed_total = data_start + entry.compressed_size as usize;

        if buf.len() < needed_total {
            let missing_start = off + buf.len() as u64;
            let missing_end = off + needed_total as u64 - 1;
            let rest = fetch_range(&self.agent, &self.url, missing_start, missing_end)?;
            buf.extend_from_slice(&rest);
        }

        let compressed = &buf[data_start..needed_total];
        decompress_zip_entry(entry, compressed)
    }
}

fn fetch_range(agent: &ureq::Agent, url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    let range_hdr = format!("bytes={start}-{end}");
    let resp = agent
        .get(url)
        .set("Range", &range_hdr)
        .call()
        .map_err(|e| ArmtError::Network(format!("HTTP Range {range_hdr} failed: {e}")))?;

    let expected = (end - start + 1) as usize;
    let mut out = Vec::with_capacity(expected);
    resp.into_reader()
        .read_to_end(&mut out)
        .map_err(|e| ArmtError::Network(format!("Range body read failed: {e}")))?;
    Ok(out)
}

pub fn decompress_zip_entry(entry: &ZipEntry, compressed: &[u8]) -> Result<Vec<u8>> {
    match entry.compression_method {
        0 => Ok(compressed.to_vec()),
        8 => {
            let mut dec = DeflateDecoder::new(compressed);
            let mut out = Vec::with_capacity(entry.uncompressed_size.min(64 * 1024 * 1024) as usize);
            dec.read_to_end(&mut out).map_err(|e| {
                ArmtError::Decompress(format!("ZIP deflate failed for {}: {e}", entry.name))
            })?;
            Ok(out)
        }
        other => Err(ArmtError::AppleContainer(format!(
            "Unsupported ZIP compression method {other} for {}",
            entry.name
        ))),
    }
}

/// Locate standard EOCD in `tail` and detect if ZIP64 EOCD Locator is present.
pub fn parse_eocd_tail(tail: &[u8], _tail_base: u64) -> Result<(u64, u64, Option<u64>)> {
    let pos = tail
        .windows(4)
        .rposition(|w| w == b"PK\x05\x06")
        .ok_or_else(|| ArmtError::AppleContainer("ZIP EOCD (PK\\x05\\x06) not found".into()))?;
    if pos + 22 > tail.len() {
        return Err(ArmtError::AppleContainer("Truncated ZIP EOCD".into()));
    }
    let cd_size = u32::from_le_bytes(tail[pos + 12..pos + 16].try_into().unwrap()) as u64;
    let cd_offset = u32::from_le_bytes(tail[pos + 16..pos + 20].try_into().unwrap()) as u64;

    if cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF {
        let loc_pos = tail[..pos]
            .windows(4)
            .rposition(|w| w == b"PK\x06\x07")
            .ok_or_else(|| {
                ArmtError::AppleContainer("ZIP64 EOCD Locator (PK\\x06\\x07) not found".into())
            })?;
        if loc_pos + 20 > pos {
            return Err(ArmtError::AppleContainer(
                "Truncated ZIP64 EOCD Locator".into(),
            ));
        }
        let z64_off = u64::from_le_bytes(tail[loc_pos + 8..loc_pos + 16].try_into().unwrap());
        Ok((cd_offset, cd_size, Some(z64_off)))
    } else {
        Ok((cd_offset, cd_size, None))
    }
}

pub fn parse_zip64_eocd(data: &[u8]) -> Result<(u64, u64)> {
    if data.len() < 56 || &data[0..4] != b"PK\x06\x06" {
        return Err(ArmtError::AppleContainer(
            "Invalid ZIP64 EOCD record (PK\\x06\\x06)".into(),
        ));
    }
    let cd_size = u64::from_le_bytes(data[40..48].try_into().unwrap());
    let cd_offset = u64::from_le_bytes(data[48..56].try_into().unwrap());
    Ok((cd_offset, cd_size))
}

pub fn parse_central_directory(cd: &[u8]) -> Result<Vec<ZipEntry>> {
    let mut entries = Vec::new();
    let mut i = 0usize;
    while i + 46 <= cd.len() {
        if &cd[i..i + 4] != b"PK\x01\x02" {
            break;
        }
        let method = u16::from_le_bytes([cd[i + 10], cd[i + 11]]);
        let crc32 = u32::from_le_bytes(cd[i + 16..i + 20].try_into().unwrap());
        let csz32 = u32::from_le_bytes(cd[i + 20..i + 24].try_into().unwrap());
        let usz32 = u32::from_le_bytes(cd[i + 24..i + 28].try_into().unwrap());
        let nlen = u16::from_le_bytes([cd[i + 28], cd[i + 29]]) as usize;
        let elen = u16::from_le_bytes([cd[i + 30], cd[i + 31]]) as usize;
        let clen = u16::from_le_bytes([cd[i + 32], cd[i + 33]]) as usize;
        let lhoff32 = u32::from_le_bytes(cd[i + 42..i + 46].try_into().unwrap());

        if i + 46 + nlen + elen + clen > cd.len() {
            return Err(ArmtError::AppleContainer(
                "Truncated ZIP Central Directory entry".into(),
            ));
        }

        let name = String::from_utf8_lossy(&cd[i + 46..i + 46 + nlen]).into_owned();
        let extra = &cd[i + 46 + nlen..i + 46 + nlen + elen];

        let mut usz = usz32 as u64;
        let mut csz = csz32 as u64;
        let mut lhoff = lhoff32 as u64;

        if usz32 == 0xFFFF_FFFF || csz32 == 0xFFFF_FFFF || lhoff32 == 0xFFFF_FFFF {
            let mut j = 0usize;
            while j + 4 <= extra.len() {
                let hid = u16::from_le_bytes([extra[j], extra[j + 1]]);
                let hsz = u16::from_le_bytes([extra[j + 2], extra[j + 3]]) as usize;
                let end = (j + 4 + hsz).min(extra.len());
                let hdata = &extra[j + 4..end];
                if hid == 0x0001 {
                    let mut k = 0usize;
                    if usz32 == 0xFFFF_FFFF && k + 8 <= hdata.len() {
                        usz = u64::from_le_bytes(hdata[k..k + 8].try_into().unwrap());
                        k += 8;
                    }
                    if csz32 == 0xFFFF_FFFF && k + 8 <= hdata.len() {
                        csz = u64::from_le_bytes(hdata[k..k + 8].try_into().unwrap());
                        k += 8;
                    }
                    if lhoff32 == 0xFFFF_FFFF && k + 8 <= hdata.len() {
                        lhoff = u64::from_le_bytes(hdata[k..k + 8].try_into().unwrap());
                    }
                    break;
                }
                j += 4 + hsz;
            }
        }

        entries.push(ZipEntry {
            name,
            compression_method: method,
            crc32,
            compressed_size: csz,
            uncompressed_size: usz,
            local_header_offset: lhoff,
        });
        i += 46 + nlen + elen + clen;
    }
    Ok(entries)
}

/// Download a full IPSW from Apple CDN with HTTP Range resume support and optional SHA-1 check.
pub fn download_full_ipsw<F>(
    url: &str,
    dest_path: &Path,
    expected_sha1: Option<&str>,
    mut progress: F,
) -> Result<()>
where
    F: FnMut(u64, u64),
{
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build();
    let head = agent
        .head(url)
        .call()
        .map_err(|e| ArmtError::Network(format!("HEAD failed: {e}")))?;
    let total_size: u64 = head
        .header("Content-Length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let existing_len = std::fs::metadata(dest_path).map(|m| m.len()).unwrap_or(0);
    let mut downloaded = if existing_len > 0 && existing_len < total_size {
        existing_len
    } else if total_size > 0 && existing_len == total_size {
        progress(total_size, total_size);
        return verify_sha1_if_requested(dest_path, expected_sha1);
    } else {
        0
    };

    let mut req = agent.get(url);
    if downloaded > 0 {
        req = req.set("Range", &format!("bytes={downloaded}-"));
    }
    let resp = req
        .call()
        .map_err(|e| ArmtError::Network(format!("Download GET failed: {e}")))?;

    let mut file = if downloaded > 0 {
        OpenOptions::new().append(true).open(dest_path)?
    } else {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dest_path)?
    };

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        progress(downloaded, total_size);
    }
    file.flush()?;
    verify_sha1_if_requested(dest_path, expected_sha1)
}

fn verify_sha1_if_requested(path: &Path, expected: Option<&str>) -> Result<()> {
    let Some(expected_hex) = expected else {
        return Ok(());
    };
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha1::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let actual_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    if !actual_hex.eq_ignore_ascii_case(expected_hex) {
        return Err(ArmtError::AppleContainer(format!(
            "IPSW SHA-1 mismatch: expected {expected_hex}, got {actual_hex}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zip64_eocd_and_cd_parse() {
        // Build a synthetic ZIP64 Central Directory entry with 64-bit offset > 4GB
        let name = b"kernelcache.release.vma2";
        let big_off = 0x0000_0005_F000_1234u64;
        let big_usz = 0x0000_0001_0000_0040u64;
        let mut extra = Vec::new();
        extra.extend_from_slice(&0x0001u16.to_le_bytes()); // ZIP64 tag
        extra.extend_from_slice(&16u16.to_le_bytes());
        extra.extend_from_slice(&big_usz.to_le_bytes());
        extra.extend_from_slice(&big_off.to_le_bytes());

        let mut cd = Vec::new();
        cd.extend_from_slice(b"PK\x01\x02");
        cd.extend_from_slice(&45u16.to_le_bytes()); // version made by
        cd.extend_from_slice(&45u16.to_le_bytes()); // version needed
        cd.extend_from_slice(&0u16.to_le_bytes()); // flags
        cd.extend_from_slice(&8u16.to_le_bytes()); // method=deflate
        cd.extend_from_slice(&0u32.to_le_bytes()); // time/date
        cd.extend_from_slice(&0x12345678u32.to_le_bytes()); // crc32
        cd.extend_from_slice(&12345u32.to_le_bytes()); // csz32
        cd.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // usz32 sentinel
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&(extra.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes()); // comment len
        cd.extend_from_slice(&0u16.to_le_bytes()); // disk
        cd.extend_from_slice(&0u16.to_le_bytes()); // int attrs
        cd.extend_from_slice(&0u32.to_le_bytes()); // ext attrs
        cd.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // lhoff sentinel
        cd.extend_from_slice(name);
        cd.extend_from_slice(&extra);

        let entries = parse_central_directory(&cd).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "kernelcache.release.vma2");
        assert_eq!(entries[0].uncompressed_size, big_usz);
        assert_eq!(entries[0].local_header_offset, big_off);
        assert_eq!(entries[0].compressed_size, 12345);
    }

    #[test]
    fn test_zip64_eocd_tail_locator() {
        let mut tail = Vec::new();
        // ZIP64 EOCD Locator
        tail.extend_from_slice(b"PK\x06\x07");
        tail.extend_from_slice(&0u32.to_le_bytes());
        tail.extend_from_slice(&0x0000_0005_FEDC_BA00u64.to_le_bytes());
        tail.extend_from_slice(&1u32.to_le_bytes());
        // Standard EOCD with 0xFFFFFFFF sentinels
        tail.extend_from_slice(b"PK\x05\x06");
        tail.extend_from_slice(&[0u8; 8]);
        tail.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        tail.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        tail.extend_from_slice(&0u16.to_le_bytes());

        let (_, _, z64_off) = parse_eocd_tail(&tail, 0).unwrap();
        assert_eq!(z64_off, Some(0x0000_0005_FEDC_BA00));
    }
}
