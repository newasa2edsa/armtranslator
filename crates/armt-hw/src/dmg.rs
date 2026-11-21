use armt_core::{ArmtError, Result};
use flate2::read::ZlibDecoder;
use lzfse_rust::LzfseRingDecoder;
use plist::Value;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};

const KOLY_MAGIC: &[u8; 4] = b"koly";
const BLKX_MAGIC: &[u8; 4] = b"mish";

const CHUNK_ZERO: u32 = 0x0000_0000;
const CHUNK_RAW: u32 = 0x0000_0001;
const CHUNK_IGNORE: u32 = 0x0000_0002;
const CHUNK_ZLIB: u32 = 0x8000_0005;
const CHUNK_LZFSE: u32 = 0x8000_0008;
const CHUNK_TERM: u32 = 0xFFFF_FFFF;

#[derive(Debug, Clone)]
struct BlkxRun {
    chunk_type: u32,
    sector_start: u64,
    sector_count: u64,
    comp_offset: u64,
    comp_length: u64,
}

/// Streaming pure-Rust reader for Apple UDIF (`koly`) `.dmg` disk images
/// supporting `RAW`, `ZERO`/`IGNORE`, `ZLIB`, and `LZFSE` compressed block runs.
pub struct UdifDmg {
    file: File,
    data_fork_offset: u64,
    total_sectors: u64,
    runs: Vec<BlkxRun>,
    chunk_cache: VecDeque<(usize, Vec<u8>)>,
}

impl UdifDmg {
    pub fn try_open(mut file: File, file_len: u64) -> Result<Option<Self>> {
        if file_len < 512 {
            return Ok(None);
        }
        file.seek(SeekFrom::Start(file_len - 512))?;
        let mut trailer = [0u8; 512];
        file.read_exact(&mut trailer)?;
        if &trailer[0..4] != KOLY_MAGIC {
            return Ok(None);
        }

        let data_fork_offset = u64::from_be_bytes(trailer[24..32].try_into().unwrap());
        let xml_offset = u64::from_be_bytes(trailer[216..224].try_into().unwrap());
        let xml_length = u64::from_be_bytes(trailer[224..232].try_into().unwrap());
        let total_sectors = u64::from_be_bytes(trailer[492..500].try_into().unwrap());

        if xml_length == 0 || xml_length > 32 * 1024 * 1024 || xml_offset + xml_length > file_len {
            return Err(ArmtError::Device("Invalid UDIF DMG XML plist bounds".into()));
        }

        file.seek(SeekFrom::Start(xml_offset))?;
        let mut xml_bytes = vec![0u8; xml_length as usize];
        file.read_exact(&mut xml_bytes)?;

        let root = Value::from_reader(Cursor::new(&xml_bytes))
            .map_err(|e| ArmtError::Device(format!("Invalid UDIF DMG plist: {e}")))?;

        let mut runs = Vec::new();
        if let Some(blkx_arr) = root
            .as_dictionary()
            .and_then(|d| d.get("resource-fork"))
            .and_then(|v| v.as_dictionary())
            .and_then(|d| d.get("blkx"))
            .and_then(|v| v.as_array())
        {
            for entry in blkx_arr {
                let Some(data) = entry
                    .as_dictionary()
                    .and_then(|d| d.get("Data"))
                    .and_then(|v| v.as_data())
                else {
                    continue;
                };
                parse_mish_table(data, &mut runs)?;
            }
        }

        runs.sort_by_key(|r| r.sector_start);
        Ok(Some(Self {
            file,
            data_fork_offset,
            total_sectors,
            runs,
            chunk_cache: VecDeque::with_capacity(16),
        }))
    }

    pub fn capacity_sectors(&self) -> u64 {
        self.total_sectors
    }

    pub fn read_sector(&mut self, lba: u64, dst: &mut [u8]) -> Result<()> {
        let sector_bytes = 512usize;
        if dst.len() % sector_bytes != 0 {
            return Err(ArmtError::Device("DMG read must be sector-aligned".into()));
        }
        let n_sectors = dst.len() / sector_bytes;
        for i in 0..n_sectors {
            let cur_lba = lba + (i as u64);
            if cur_lba >= self.total_sectors {
                return Err(ArmtError::Device("DMG read out of bounds".into()));
            }
            let out_slice = &mut dst[i * sector_bytes..(i + 1) * sector_bytes];
            self.read_single_sector(cur_lba, out_slice)?;
        }
        Ok(())
    }

    fn read_single_sector(&mut self, lba: u64, dst: &mut [u8]) -> Result<()> {
        let run_idx = self.runs.partition_point(|r| r.sector_start + r.sector_count <= lba);
        let Some(run) = self.runs.get(run_idx).cloned() else {
            dst.fill(0);
            return Ok(());
        };
        if lba < run.sector_start || lba >= run.sector_start + run.sector_count {
            dst.fill(0);
            return Ok(());
        }

        let sec_in_run = (lba - run.sector_start) as usize;
        match run.chunk_type {
            CHUNK_ZERO | CHUNK_IGNORE => {
                dst.fill(0);
                Ok(())
            }
            CHUNK_RAW => {
                let file_off = self.data_fork_offset + run.comp_offset + (sec_in_run as u64) * 512;
                self.file.seek(SeekFrom::Start(file_off))?;
                self.file.read_exact(dst)?;
                Ok(())
            }
            CHUNK_ZLIB | CHUNK_LZFSE => {
                if let Some((_, cached)) = self.chunk_cache.iter().find(|(id, _)| *id == run_idx) {
                    let off = sec_in_run * 512;
                    if off + 512 <= cached.len() {
                        dst.copy_from_slice(&cached[off..off + 512]);
                        return Ok(());
                    }
                }
                let decompressed = self.decompress_run(&run)?;
                let off = sec_in_run * 512;
                if off + 512 <= decompressed.len() {
                    dst.copy_from_slice(&decompressed[off..off + 512]);
                } else {
                    dst.fill(0);
                }
                if self.chunk_cache.len() == 16 {
                    self.chunk_cache.pop_front();
                }
                self.chunk_cache.push_back((run_idx, decompressed));
                Ok(())
            }
            _ => {
                dst.fill(0);
                Ok(())
            }
        }
    }

    fn decompress_run(&mut self, run: &BlkxRun) -> Result<Vec<u8>> {
        let mut comp = vec![0u8; run.comp_length as usize];
        self.file
            .seek(SeekFrom::Start(self.data_fork_offset + run.comp_offset))?;
        self.file.read_exact(&mut comp)?;

        let expected = (run.sector_count as usize) * 512;
        let mut out = Vec::with_capacity(expected);
        match run.chunk_type {
            CHUNK_ZLIB => {
                let mut dec = ZlibDecoder::new(Cursor::new(&comp));
                dec.read_to_end(&mut out)?;
            }
            CHUNK_LZFSE => {
                let mut dec = LzfseRingDecoder::default();
                dec.decode(&mut Cursor::new(&comp), &mut out)
                    .map_err(|e| ArmtError::Decompress(format!("DMG LZFSE decode failed: {e}")))?;
            }
            _ => out.resize(expected, 0),
        }
        Ok(out)
    }
}

fn parse_mish_table(data: &[u8], runs: &mut Vec<BlkxRun>) -> Result<()> {
    if data.len() < 204 || &data[0..4] != BLKX_MAGIC {
        return Ok(());
    }
    let first_sector = u64::from_be_bytes(data[8..16].try_into().unwrap());
    let num_runs = u32::from_be_bytes(data[200..204].try_into().unwrap()) as usize;
    let mut pos = 204usize;
    for _ in 0..num_runs {
        if pos + 40 > data.len() {
            break;
        }
        let chunk_type = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        let sector_start = first_sector + u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap());
        let sector_count = u64::from_be_bytes(data[pos + 16..pos + 24].try_into().unwrap());
        let comp_offset = u64::from_be_bytes(data[pos + 24..pos + 32].try_into().unwrap());
        let comp_length = u64::from_be_bytes(data[pos + 32..pos + 40].try_into().unwrap());
        pos += 40;
        if chunk_type == CHUNK_TERM {
            break;
        }
        if sector_count > 0 {
            runs.push(BlkxRun {
                chunk_type,
                sector_start,
                sector_count,
                comp_offset,
                comp_length,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_udif_dmg_zlib_and_zero_runs() {
        let mut raw_sector = vec![0u8; 512];
        raw_sector[..8].copy_from_slice(b"EFI PART");
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&raw_sector).unwrap();
        let comp_payload = enc.finish().unwrap();

        let mut mish = vec![0u8; 204 + 2 * 40];
        mish[0..4].copy_from_slice(BLKX_MAGIC);
        mish[8..16].copy_from_slice(&0u64.to_be_bytes());
        mish[200..204].copy_from_slice(&2u32.to_be_bytes());
        // Run 0: 1 zero sector at LBA 0
        mish[204..208].copy_from_slice(&CHUNK_ZERO.to_be_bytes());
        mish[204 + 8..204 + 16].copy_from_slice(&0u64.to_be_bytes());
        mish[204 + 16..204 + 24].copy_from_slice(&1u64.to_be_bytes());
        // Run 1: 1 zlib sector at LBA 1
        let r1 = 204 + 40;
        mish[r1..r1 + 4].copy_from_slice(&CHUNK_ZLIB.to_be_bytes());
        mish[r1 + 8..r1 + 16].copy_from_slice(&1u64.to_be_bytes());
        mish[r1 + 16..r1 + 24].copy_from_slice(&1u64.to_be_bytes());
        mish[r1 + 24..r1 + 32].copy_from_slice(&0u64.to_be_bytes());
        mish[r1 + 32..r1 + 40].copy_from_slice(&(comp_payload.len() as u64).to_be_bytes());

        let mut plist_dict = plist::Dictionary::new();
        let mut rfork = plist::Dictionary::new();
        let mut blkx_entry = plist::Dictionary::new();
        blkx_entry.insert("Data".into(), Value::Data(mish));
        rfork.insert("blkx".into(), Value::Array(vec![Value::Dictionary(blkx_entry)]));
        plist_dict.insert("resource-fork".into(), Value::Dictionary(rfork));
        let mut xml = Vec::new();
        plist::to_writer_xml(&mut xml, &Value::Dictionary(plist_dict)).unwrap();

        let xml_off = comp_payload.len() as u64;
        let mut trailer = [0u8; 512];
        trailer[0..4].copy_from_slice(KOLY_MAGIC);
        trailer[24..32].copy_from_slice(&0u64.to_be_bytes());
        trailer[216..224].copy_from_slice(&xml_off.to_be_bytes());
        trailer[224..232].copy_from_slice(&(xml.len() as u64).to_be_bytes());
        trailer[492..500].copy_from_slice(&2u64.to_be_bytes());

        let tmp = std::env::temp_dir().join("armt_test_udif.dmg");
        {
            let mut f = File::create(&tmp).unwrap();
            f.write_all(&comp_payload).unwrap();
            f.write_all(&xml).unwrap();
            f.write_all(&trailer).unwrap();
        }
        let file = File::open(&tmp).unwrap();
        let len = file.metadata().unwrap().len();
        let mut dmg = UdifDmg::try_open(file, len).unwrap().unwrap();
        assert_eq!(dmg.capacity_sectors(), 2);

        let mut s0 = [1u8; 512];
        dmg.read_sector(0, &mut s0).unwrap();
        assert_eq!(s0, [0u8; 512]);

        let mut s1 = [0u8; 512];
        dmg.read_sector(1, &mut s1).unwrap();
        assert_eq!(&s1[..8], b"EFI PART");
        let _ = std::fs::remove_file(&tmp);
    }
}
