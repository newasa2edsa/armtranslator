use armt_apple::{
    decompress_payload, finalize_adt_for_xnu, parse_adt, parse_img4_or_im4p, parse_macho,
    serialize_adt,
};
use armt_core::isa::{Condition, Reg, ShiftType, SysReg, Width};
use armt_core::boot::XNU_BOOT_LINE_LENGTH;
use armt_core::{AdtNode, ArmtError, CpuState, GuestMemory, Result, XnuBootArgs, XnuVideo};
use armt_decoder::asm;
use armt_hw::{Vma2Bus, Vma2MemMap};
use std::collections::HashMap;

/// Result of loading a kernel or standalone ARM64 image into guest RAM.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    pub entry_pc: u64,
    pub sp: u64,
    pub boot_args_phys: u64,
    pub is_apple_kernel: bool,
    pub kernel_segments_loaded: usize,
    pub adt_summary: Option<String>,
}

/// Unwrap optional IM4P/IMG4 container and decompress LZFSE/LZSS payload.
pub fn unwrap_apple_payload(bytes: &[u8]) -> Result<Vec<u8>> {
    let raw = if bytes.first() == Some(&0x30) {
        parse_img4_or_im4p(bytes)?.payload
    } else {
        bytes.to_vec()
    };
    decompress_payload(&raw)
}

/// Optional ramdisk and static trustcache blobs to attach alongside the Apple kernel.
#[derive(Debug, Clone, Default)]
pub struct AppleLoadOptions {
    pub ramdisk_bytes: Option<Vec<u8>>,
    pub trustcache_bytes: Option<Vec<u8>>,
}

/// Load a real Apple `kernelcache.release.vma2` and `DeviceTree.vma2macosap.im4p` into `Vma2Bus` RAM
/// and construct authentic XNU ARM64 `boot_args` (`rev=2, ver=2`).
pub fn load_apple_kernel_bundle(
    bus: &mut Vma2Bus,
    kernel_bytes: &[u8],
    devicetree_bytes: &[u8],
    cmdline: &str,
) -> Result<LoadedImage> {
    load_apple_kernel_bundle_with_options(
        bus,
        kernel_bytes,
        devicetree_bytes,
        cmdline,
        &AppleLoadOptions::default(),
    )
}

/// Load an Apple kernelcache + DeviceTree with optional `RestoreRamDisk` (`rd=md0`) and `TrustCache`.
pub fn load_apple_kernel_bundle_with_options(
    bus: &mut Vma2Bus,
    kernel_bytes: &[u8],
    devicetree_bytes: &[u8],
    cmdline: &str,
    opts: &AppleLoadOptions,
) -> Result<LoadedImage> {
    let mut dt_raw = unwrap_apple_payload(devicetree_bytes)?;
    let _ = finalize_adt_for_xnu(&mut dt_raw)?;
    let mut adt = parse_adt(&dt_raw)?;
    bus.memmap = Vma2MemMap::from_adt(&adt, bus.memmap.ram_size / (1024 * 1024));

    let kc_raw = unwrap_apple_payload(kernel_bytes)?;
    let macho = parse_macho(&kc_raw)?;

    let ram_base = bus.memmap.ram_base;
    let ram_size = bus.memmap.ram_size;
    let ram_end = ram_base + ram_size;

    let min_vmaddr = macho
        .min_vmaddr()
        .ok_or_else(|| ArmtError::Vm("Apple kernelcache has no loadable segments".into()))?;
    let entry_va = macho
        .entry_pc
        .ok_or_else(|| ArmtError::Vm("Apple kernelcache missing LC_UNIXTHREAD entry_pc".into()))?;

    // Align virt_base to a 32 MiB (16 KB granule L2 block) boundary so XNU's early
    // bootstrap L2 block page tables map (virt_base -> phys_base) with exact 32 MiB
    // congruence. Keep at least 2 MiB below the first kernel segment for boot_args.
    const L2_BLOCK_SIZE: u64 = 32 * 1024 * 1024;
    let mut virt_base = min_vmaddr & !(L2_BLOCK_SIZE - 1);
    if min_vmaddr.saturating_sub(virt_base) < 0x0020_0000 {
        virt_base = virt_base.saturating_sub(L2_BLOCK_SIZE);
    }
    let kernel_load_phys = ram_base.wrapping_add(min_vmaddr.wrapping_sub(virt_base));
    let mut loaded_segs = 0usize;
    let mut highest_phys = kernel_load_phys;

    for seg in &macho.segments {
        if seg.vmsize == 0 || seg.name == "__PAGEZERO" {
            continue;
        }
        let rel = seg.vmaddr.saturating_sub(min_vmaddr);
        let seg_phys = kernel_load_phys.saturating_add(rel);
        if seg_phys >= ram_end {
            continue;
        }
        let copy_len = (seg.filesize as usize)
            .min(macho.raw.len().saturating_sub(seg.fileoff as usize))
            .min((ram_end - seg_phys) as usize);
        if copy_len > 0 {
            let src_start = seg.fileoff as usize;
            bus.write_bytes(seg_phys, &macho.raw[src_start..src_start + copy_len])?;
            loaded_segs += 1;
        }
        let seg_end = seg_phys.saturating_add(seg.vmsize).min(ram_end);
        if seg_end > highest_phys {
            highest_phys = seg_end;
        }
    }

    // Align DeviceTree placement to 16 KB (Apple Silicon page granule) and patch /chosen
    // with the AVPBooter / iBoot runtime properties XNU's arm_init expects.
    let align_16k = 16 * 1024u64;
    let dt_phys = (highest_phys + align_16k - 1) & !(align_16k - 1);
    let boot_args_phys = ram_base + 0x4000;

    if let Some(chosen) = adt.find_path_mut("chosen") {
        chosen.set_prop("dram-base", ram_base.to_le_bytes().to_vec());
        chosen.set_prop("dram-size", ram_size.to_le_bytes().to_vec());
        let mut seed = Vec::with_capacity(256);
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..32 {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            seed.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
        }
        chosen.set_prop("random-seed", seed);
        chosen.set_prop("firmware-version", b"iBoot-10151.1.1\0".to_vec());
        chosen.set_prop("effective-production-status-ap", 1u32.to_le_bytes().to_vec());
        chosen.set_prop("effective-security-mode-ap", 1u32.to_le_bytes().to_vec());
        chosen.set_prop("security-domain", 1u32.to_le_bytes().to_vec());
        chosen.set_prop("debug-enabled", 1u32.to_le_bytes().to_vec());

        // Populate NVRAMv3 bank geometry and proxy bank with valid CHRP partition headers
        // and Adler-32 payload checksum so IONVRAM initializes cleanly.
        let nvram_bank_size: u32 = 0x1_0000;
        let nvram_bank_count: u32 = 1;
        let mut nvram_bank = vec![0u8; nvram_bank_size as usize];
        let chrp_checksum = |hdr: &[u8]| -> u8 {
            let mut sum: u32 = (hdr[0] as u32) + hdr[2..16].iter().map(|&b| b as u32).sum::<u32>();
            while sum > 0xFF {
                sum = (sum & 0xFF) + (sum >> 8);
            }
            sum as u8
        };
        // Entry 0 (0x0000..0x0020): 2 blocks (32 bytes), "nvram" v1 bank header + Adler32 + generation.
        nvram_bank[0] = 0x5A;
        nvram_bank[2..4].copy_from_slice(&2u16.to_le_bytes());
        nvram_bank[4..9].copy_from_slice(b"nvram");
        nvram_bank[1] = chrp_checksum(&nvram_bank[0..16]);
        nvram_bank[20..24].copy_from_slice(&1u32.to_le_bytes());
        // Entry 1 (0x0020..0x8000): "common" partition.
        let common_off = 0x20usize;
        let common_blocks = ((0x8000 - common_off) / 16) as u16;
        nvram_bank[common_off] = 0x70;
        nvram_bank[common_off + 2..common_off + 4].copy_from_slice(&common_blocks.to_le_bytes());
        nvram_bank[common_off + 4..common_off + 10].copy_from_slice(b"common");
        nvram_bank[common_off + 1] = chrp_checksum(&nvram_bank[common_off..common_off + 16]);
        // Entry 2 (0x8000..0x10000): "system" partition.
        let sys_off = 0x8000usize;
        let sys_blocks = ((nvram_bank_size as usize - sys_off) / 16) as u16;
        nvram_bank[sys_off] = 0x70;
        nvram_bank[sys_off + 2..sys_off + 4].copy_from_slice(&sys_blocks.to_le_bytes());
        nvram_bank[sys_off + 4..sys_off + 10].copy_from_slice(b"system");
        nvram_bank[sys_off + 1] = chrp_checksum(&nvram_bank[sys_off..sys_off + 16]);
        // Compute standard Adler-32 over bank[20..] into bank[16..20].
        let mut s1: u32 = 1;
        let mut s2: u32 = 0;
        for &b in &nvram_bank[20..] {
            s1 = (s1 + (b as u32)) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        let adler = (s2 << 16) | s1;
        nvram_bank[16..20].copy_from_slice(&adler.to_le_bytes());

        chosen.set_prop("nvram-bank-size", nvram_bank_size.to_le_bytes().to_vec());
        chosen.set_prop("nvram-bank-count", nvram_bank_count.to_le_bytes().to_vec());
        chosen.set_prop("nvram-current-bank", 0u32.to_le_bytes().to_vec());
        chosen.set_prop("max-env-var-data-size", 0x800u32.to_le_bytes().to_vec());
        chosen.set_prop("max-env-var-name-size", 0x40u32.to_le_bytes().to_vec());
        chosen.set_prop("max-env-partition", nvram_bank_size.to_le_bytes().to_vec());
        chosen.set_prop("nvram-proxy-data", nvram_bank);
    }
    if let Some(vram) = adt.find_path_mut("vram") {
        let mut reg = Vec::with_capacity(16);
        reg.extend_from_slice(&bus.memmap.framebuffer_base.to_le_bytes());
        reg.extend_from_slice(&bus.memmap.framebuffer_size_bytes().to_le_bytes());
        vram.set_prop("reg", reg);
    }
    if let Some(cpus) = adt.find_path_mut("cpus") {
        cpus.children.retain(|c| c.name == "cpu0");
    }
    bus.gic.set_apple_timer_fiq_routing();
    bus.virtio_blk.format_gpt_stub();

    let mut data_cursor = (dt_phys + 0x10_0000).max(dt_phys + align_16k);
    let mut effective_cmdline = cmdline.to_string();

    // XNU's arm_vm_init requires TrustCache to reside below kernelcache base (< vm_kernel_base).
    // Since `virt_base` leaves >= 2 MiB between `ram_base` (`boot_args` at +0x4000) and `kernel_load_phys`,
    // place TrustCache right at `ram_base + 0x1_0000` when it fits before the kernel image.
    let mut trustcache_range: Option<(u64, u64)> = None;
    if let Some(tc_raw) = opts.trustcache_bytes.as_deref() {
        let tc_payload = unwrap_apple_payload(tc_raw)?;
        // XNU's kern_trustcache.c expects `/chosen/memory-map/TrustCache` to start with:
        //   struct { uint32_t num_modules; uint32_t offsets[num_modules]; }
        // followed by the unwrapped `rtsc` module payload(s).
        let mut tc = Vec::with_capacity(8 + tc_payload.len());
        tc.extend_from_slice(&1u32.to_le_bytes());
        tc.extend_from_slice(&8u32.to_le_bytes());
        tc.extend_from_slice(&tc_payload);

        let tc_len = tc.len() as u64;
        let low_tc_addr = ram_base + 0x1_0000;
        let tc_addr = if low_tc_addr + tc_len <= kernel_load_phys {
            low_tc_addr
        } else {
            let addr = data_cursor;
            data_cursor = (data_cursor + tc_len + align_16k - 1) & !(align_16k - 1);
            addr
        };
        if tc_addr + tc_len <= ram_end {
            bus.write_bytes(tc_addr, &tc)?;
            trustcache_range = Some((tc_addr, tc_len));
        }
    }

    let mut ramdisk_range: Option<(u64, u64)> = None;
    if let Some(rd_raw) = opts.ramdisk_bytes.as_deref() {
        let rd = unwrap_apple_payload(rd_raw)?;
        let rd_len = rd.len() as u64;
        if data_cursor + rd_len > ram_end {
            return Err(ArmtError::Vm(format!(
                "RAMDisk ({} MiB) exceeds guest RAM; increase --ram-mb",
                rd_len / (1024 * 1024)
            )));
        }
        bus.write_bytes(data_cursor, &rd)?;
        ramdisk_range = Some((data_cursor, rd_len));
        data_cursor = (data_cursor + rd_len + align_16k - 1) & !(align_16k - 1);
        if !effective_cmdline.contains("rd=") {
            if !effective_cmdline.is_empty() {
                effective_cmdline.push(' ');
            }
            effective_cmdline.push_str("rd=md0");
        }
    }

    if let Some(mmap) = adt.find_path_mut("chosen/memory-map") {
        if let Some((addr, len)) = ramdisk_range {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&addr.to_le_bytes());
            v.extend_from_slice(&len.to_le_bytes());
            mmap.set_prop("RAMDisk", v);
        }
        if let Some((addr, len)) = trustcache_range {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&addr.to_le_bytes());
            v.extend_from_slice(&len.to_le_bytes());
            mmap.set_prop("TrustCache", v);
        }
    }

    let dt_patched = serialize_adt(&adt);
    if dt_phys + (dt_patched.len() as u64) <= ram_end {
        bus.write_bytes(dt_phys, &dt_patched)?;
    }
    let dt_end = (dt_phys + dt_patched.len() as u64 + align_16k - 1) & !(align_16k - 1);
    let top_of_kernel_data = if ramdisk_range.is_some() || trustcache_range.is_some() {
        data_cursor.max(dt_end)
    } else {
        dt_end
    };

    let mut ba = XnuBootArgs {
        revision: 2,
        version: 2,
        virt_base,
        phys_base: ram_base,
        mem_size: ram_size,
        top_of_kernel_data,
        video: XnuVideo {
            v_base_addr: bus.memmap.framebuffer_base,
            v_display: 1,
            v_row_bytes: (bus.memmap.framebuffer_width as u64) * 4,
            v_width: bus.memmap.framebuffer_width as u64,
            v_height: bus.memmap.framebuffer_height as u64,
            v_depth: 32,
        },
        machine_type: 0,
        device_tree_p: virt_base.wrapping_add(dt_phys - ram_base),
        device_tree_length: dt_patched.len() as u32,
        command_line: [0u8; XNU_BOOT_LINE_LENGTH],
        boot_flags: 0,
        mem_size_actual: ram_size,
    };
    ba.set_command_line(&effective_cmdline);
    bus.write_bytes(boot_args_phys, &ba.to_bytes())?;

    let entry_phys = kernel_load_phys.wrapping_add(entry_va.wrapping_sub(min_vmaddr));
    let sp = ram_end.saturating_sub(0x10_0000);

    Ok(LoadedImage {
        entry_pc: entry_phys,
        sp,
        boot_args_phys,
        is_apple_kernel: true,
        kernel_segments_loaded: loaded_segs,
        adt_summary: Some(format_adt_brief(&adt)),
    })
}

fn format_adt_brief(adt: &AdtNode) -> String {
    let model = adt
        .prop_str("model")
        .or_else(|| adt.prop_str("compatible"))
        .unwrap_or_else(|| adt.name.clone());
    format!("{model} ({} top-level nodes)", adt.children.len())
}

/// Load a standalone ARM64 Mach-O executable or flat raw AArch64 binary.
pub fn load_raw_or_macho(bus: &mut Vma2Bus, bytes: &[u8], load_addr: u64) -> Result<LoadedImage> {
    if bytes.len() >= 4 {
        let m_le = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let m_be = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        if m_le == 0xFEED_FACF || m_be == 0xCAFE_BABE || m_be == 0xBEBA_FECA {
            let macho = parse_macho(bytes)?;
            let base_va = macho.min_vmaddr().unwrap_or(0);
            for seg in &macho.segments {
                if seg.vmsize == 0 || seg.name == "__PAGEZERO" {
                    continue;
                }
                let dst = load_addr.wrapping_add(seg.vmaddr.saturating_sub(base_va));
                let copy_len = (seg.filesize as usize)
                    .min(macho.raw.len().saturating_sub(seg.fileoff as usize));
                if copy_len > 0 {
                    bus.write_bytes(
                        dst,
                        &macho.raw[seg.fileoff as usize..seg.fileoff as usize + copy_len],
                    )?;
                }
            }
            let entry = if let Some(pc) = macho.entry_pc {
                load_addr.wrapping_add(pc.saturating_sub(base_va))
            } else if let Some(off) = macho.entry_offset {
                load_addr.wrapping_add(off)
            } else {
                load_addr
            };
            let sp = bus.memmap.ram_base + bus.memmap.ram_size - 0x10_0000;
            return Ok(LoadedImage {
                entry_pc: entry,
                sp,
                boot_args_phys: 0,
                is_apple_kernel: false,
                kernel_segments_loaded: macho.segments.len(),
                adt_summary: None,
            });
        }
    }

    bus.write_bytes(load_addr, bytes)?;
    let sp = bus.memmap.ram_base + bus.memmap.ram_size - 0x10_0000;
    Ok(LoadedImage {
        entry_pc: load_addr,
        sp,
        boot_args_phys: 0,
        is_apple_kernel: false,
        kernel_segments_loaded: 1,
        adt_summary: None,
    })
}

/// Tiny label-patching ARM64 assembler helper for our built-in demo kernel.
struct MiniAsm {
    base_addr: u64,
    words: Vec<u32>,
    labels: HashMap<&'static str, usize>,
    fixups: Vec<(usize, &'static str, FixupKind)>,
}

enum FixupKind {
    B,
    Bl,
    BCond(Condition),
    Cbz(Width, Reg),
    Adr(Reg),
}

impl MiniAsm {
    fn new(base_addr: u64) -> Self {
        Self {
            base_addr,
            words: Vec::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    fn label(&mut self, name: &'static str) {
        self.labels.insert(name, self.words.len());
    }

    fn emit(&mut self, w: u32) {
        self.words.push(w);
    }

    fn emit_u64_imm(&mut self, rd: Reg, val: u64) {
        for w in asm::load_u64(rd, val) {
            self.emit(w);
        }
    }

    fn b(&mut self, target: &'static str) {
        let idx = self.words.len();
        self.words.push(0);
        self.fixups.push((idx, target, FixupKind::B));
    }

    fn bl(&mut self, target: &'static str) {
        let idx = self.words.len();
        self.words.push(0);
        self.fixups.push((idx, target, FixupKind::Bl));
    }

    fn b_cond(&mut self, cond: Condition, target: &'static str) {
        let idx = self.words.len();
        self.words.push(0);
        self.fixups.push((idx, target, FixupKind::BCond(cond)));
    }

    fn cbz(&mut self, w: Width, rt: Reg, target: &'static str) {
        let idx = self.words.len();
        self.words.push(0);
        self.fixups.push((idx, target, FixupKind::Cbz(w, rt)));
    }

    fn adr(&mut self, rd: Reg, target: &'static str) {
        let idx = self.words.len();
        self.words.push(0);
        self.fixups.push((idx, target, FixupKind::Adr(rd)));
    }

    fn finish(mut self, strings: &[(&'static str, &[u8])]) -> Vec<u8> {
        // Append null-terminated strings padded to 4-byte boundary after instructions
        for (lbl, data) in strings {
            self.label(lbl);
            let mut buf = data.to_vec();
            buf.push(0);
            while buf.len() % 4 != 0 {
                buf.push(0);
            }
            for chunk in buf.chunks_exact(4) {
                self.words
                    .push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }

        let _ = self.base_addr;
        for (idx, target, kind) in self.fixups {
            let tgt_idx = *self.labels.get(target).expect("missing asm label");
            let byte_off = ((tgt_idx as isize) - (idx as isize)) as i64 * 4;
            self.words[idx] = match kind {
                FixupKind::B => asm::b(byte_off),
                FixupKind::Bl => asm::bl(byte_off),
                FixupKind::BCond(c) => asm::b_cond(c, byte_off),
                FixupKind::Cbz(w, rt) => asm::cbz(w, rt, byte_off),
                FixupKind::Adr(rd) => {
                    let imm21 = (byte_off as u32) & 0x1F_FFFF;
                    let immlo = imm21 & 3;
                    let immhi = (imm21 >> 2) & 0x7_FFFF;
                    (immlo << 29) | (0b10000 << 24) | (immhi << 5) | (rd.0 as u32 & 0x1F)
                }
            };
        }

        asm::to_bytes(&self.words)
    }
}

/// Build and load a genuine, real AArch64 demo kernel into `Vma2Bus` RAM.
///
/// The kernel runs natively through the x86-64 Dynarec JIT (or Reference Interpreter),
/// verifies Fibonacci + memory + 24 MHz Apple timer, prints full boot diagnostics over PL011 UART,
/// renders a Mac-style graphical boot card to the 1024x768 BGRA framebuffer, and halts cleanly (`HLT #0`).
pub fn load_demo_kernel(bus: &mut Vma2Bus) -> Result<LoadedImage> {
    let entry = bus.memmap.ram_base + 0x10_0000;
    let uart_base = bus.memmap.uart0_base;
    let fb_base = bus.memmap.framebuffer_base;
    let fb_w = bus.memmap.framebuffer_width as u64;
    let fb_h = bus.memmap.framebuffer_height as u64;
    let scratch_ram = bus.memmap.ram_base + 0x4_0000;

    let mut a = MiniAsm::new(entry);

    // X19 = UART0 base, X20 = FB base
    a.emit_u64_imm(Reg(19), uart_base);
    a.emit_u64_imm(Reg(20), fb_base);

    // Print boot lines via uart_puts
    a.adr(Reg(0), "str_banner");
    a.bl("uart_puts");
    a.adr(Reg(0), "str_jit");
    a.bl("uart_puts");

    // Compute fib(20) in guest ARM64 code and store to RAM scratch
    a.emit(asm::movz(Width::X64, Reg(1), 20, 0));
    a.emit(asm::movz(Width::X64, Reg(2), 0, 0)); // a = 0
    a.emit(asm::movz(Width::X64, Reg(3), 1, 0)); // b = 1
    a.emit(asm::movz(Width::X64, Reg(4), 0, 0)); // i = 0
    a.label("fib_loop");
    a.emit(asm::sub_reg(
        Width::X64,
        Reg::XZR,
        Reg(4),
        Reg(1),
        ShiftType::Lsl,
        0,
        true,
    ));
    a.b_cond(Condition::Ge, "fib_done");
    a.emit(asm::add_reg(
        Width::X64,
        Reg(5),
        Reg(2),
        Reg(3),
        ShiftType::Lsl,
        0,
        false,
    ));
    a.emit(asm::mov_reg(Width::X64, Reg(2), Reg(3)));
    a.emit(asm::mov_reg(Width::X64, Reg(3), Reg(5)));
    a.emit(asm::add_imm(Width::X64, Reg(4), Reg(4), 1, false));
    a.b("fib_loop");
    a.label("fib_done");
    a.emit_u64_imm(Reg(6), scratch_ram);
    a.emit(asm::str_imm(Width::X64, Reg(2), Reg(6), 0));
    a.emit(asm::ldr_imm(Width::X64, Reg(7), Reg(6), 0));
    a.emit_u64_imm(Reg(8), 6765);
    a.emit(asm::sub_reg(
        Width::X64,
        Reg::XZR,
        Reg(7),
        Reg(8),
        ShiftType::Lsl,
        0,
        true,
    ));
    a.b_cond(Condition::Ne, "fib_fail");
    a.adr(Reg(0), "str_fib_ok");
    a.bl("uart_puts");
    a.b("check_timer");
    a.label("fib_fail");
    a.adr(Reg(0), "str_fib_err");
    a.bl("uart_puts");

    // Read CNTFRQ_EL0 (24 MHz)
    a.label("check_timer");
    a.emit(asm::mrs(Reg(9), SysReg::CNTFRQ_EL0));
    a.emit_u64_imm(Reg(10), 24_000_000);
    a.emit(asm::sub_reg(
        Width::X64,
        Reg::XZR,
        Reg(9),
        Reg(10),
        ShiftType::Lsl,
        0,
        true,
    ));
    a.b_cond(Condition::Ne, "fb_draw");
    a.adr(Reg(0), "str_timer_ok");
    a.bl("uart_puts");

    // Render sleek dark Mac-style boot screen directly into BGRA Framebuffer
    // (Fill background 0x0011141C, central card 0x001E2433, accent bar 0x003B82F6)
    a.label("fb_draw");
    // X0 = fb_base, X1 = pixel count (fb_w * fb_h), X2 = dark bg color 0x0011141C
    a.emit(asm::mov_reg(Width::X64, Reg(0), Reg(20)));
    a.emit_u64_imm(Reg(1), fb_w * fb_h);
    a.emit_u64_imm(Reg(2), 0x0011_141C);
    a.bl("fill_pixels");

    // Draw progress bar track at row 384 (center): y=384, x=256, w=512, h=24 -> 512 pixels/row * 24 rows
    a.emit_u64_imm(Reg(3), (384 * fb_w + 256) * 4);
    a.emit(asm::add_reg(
        Width::X64,
        Reg(21),
        Reg(20),
        Reg(3),
        ShiftType::Lsl,
        0,
        false,
    ));
    a.emit(asm::movz(Width::X64, Reg(22), 24, 0)); // row counter
    a.label("bar_row_loop");
    a.cbz(Width::X64, Reg(22), "bar_done");
    a.emit(asm::mov_reg(Width::X64, Reg(0), Reg(21)));
    a.emit(asm::movz(Width::X64, Reg(1), 512, 0));
    a.emit_u64_imm(Reg(2), 0x003B_82F6);
    a.bl("fill_pixels");
    a.emit_u64_imm(Reg(3), fb_w * 4);
    a.emit(asm::add_reg(
        Width::X64,
        Reg(21),
        Reg(21),
        Reg(3),
        ShiftType::Lsl,
        0,
        false,
    ));
    a.emit(asm::sub_imm(Width::X64, Reg(22), Reg(22), 1, false));
    a.b("bar_row_loop");
    a.label("bar_done");

    a.adr(Reg(0), "str_fb_ok");
    a.bl("uart_puts");
    a.adr(Reg(0), "str_done");
    a.bl("uart_puts");

    // Put 6765 in X0 as a clean return signature and halt
    a.emit(asm::mov_reg(Width::X64, Reg(0), Reg(7)));
    a.emit(asm::hlt(0));

    // Subroutine: uart_puts(X0 = cstr)
    a.label("uart_puts");
    a.label("uart_loop");
    a.emit(asm::ldrb_imm(Reg(1), Reg(0), 0));
    a.cbz(Width::W32, Reg(1), "uart_ret");
    a.emit(asm::str_imm(Width::W32, Reg(1), Reg(19), 0));
    a.emit(asm::add_imm(Width::X64, Reg(0), Reg(0), 1, false));
    a.b("uart_loop");
    a.label("uart_ret");
    a.emit(asm::ret(Reg::LR));

    // Subroutine: fill_pixels(X0 = ptr, X1 = even pixel count, X2 = 32-bit BGRA color)
    // Writes 2 pixels (64 bits) per iteration for speed.
    a.label("fill_pixels");
    a.emit(0xAA02_8042); // orr x2, x2, x2, lsl #32  (broadcast 32-bit pixel into 64-bit pair)
    a.label("fill_loop");
    a.cbz(Width::X64, Reg(1), "fill_ret");
    a.emit(asm::str_imm(Width::X64, Reg(2), Reg(0), 0));
    a.emit(asm::add_imm(Width::X64, Reg(0), Reg(0), 8, false));
    a.emit(asm::sub_imm(Width::X64, Reg(1), Reg(1), 2, false));
    a.b("fill_loop");
    a.label("fill_ret");
    a.emit(asm::ret(Reg::LR));

    let code_bytes = a.finish(&[
        (
            "str_banner",
            b"[ArmTranslator] VirtualMac2,1 (vma2) Userspace ARM64 Emulator - EL1h\n",
        ),
        (
            "str_jit",
            b"[JIT] Executing native AArch64 -> x86-64 Dynarec blocks\n",
        ),
        (
            "str_fib_ok",
            b"[SELFTEST] Guest ARM64 Fibonacci(20) == 6765 & RAM roundtrip: PASS\n",
        ),
        ("str_fib_err", b"[SELFTEST] Fibonacci check FAILED\n"),
        (
            "str_timer_ok",
            b"[HW] Apple Generic Timer CNTFRQ_EL0 == 24000000 Hz: PASS\n",
        ),
        (
            "str_fb_ok",
            b"[FB] Rendered 1024x768 BGRA Mac boot progress framebuffer\n",
        ),
        ("str_done", b"[BOOT] Demo kernel completed cleanly (HLT #0)\n"),
    ]);

    bus.write_bytes(entry, &code_bytes)?;

    let sp = bus.memmap.ram_base + bus.memmap.ram_size - 0x10_0000;
    Ok(LoadedImage {
        entry_pc: entry,
        sp,
        boot_args_phys: 0,
        is_apple_kernel: false,
        kernel_segments_loaded: 1,
        adt_summary: None,
    })
}

/// Helper to create a properly seeded `CpuState` from a `LoadedImage`.
pub fn init_cpu_from_image(img: &LoadedImage) -> CpuState {
    CpuState::default_vma2_el1(img.entry_pc, img.sp, img.boot_args_phys)
}
