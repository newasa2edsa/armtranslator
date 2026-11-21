use armt_apple::{
    decompress_payload, download_full_ipsw, dump_adt, fetch_boot_bundle, fetch_catalog,
    fetch_ramdisk_bundle, parse_adt, parse_img4_or_im4p, parse_macho,
};
use armt_core::isa::{Reg, ShiftType, Width};
use armt_core::{GuestMemory, HostCallArgs, HostCallResult, Result};
use armt_decoder::asm;
use armt_gpu::{launch_rtx_gui, probe_best_gpu, run_headless_gpu_self_test, LaunchMode, LaunchOptions};
use armt_jit::{invoke_arm64_fn, HostServiceTable, JitEngine, Mmu};
use armt_vm::{
    load_apple_kernel_bundle_with_options, load_demo_kernel, load_raw_or_macho, run_benchmark,
    AppleLoadOptions, ArmVm, ExecMode, VmConfig,
};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "armtranslator",
    version,
    about = "Userspace AArch64->x86-64 Dynarec Translator, RTX GPU Renderer & VirtualMac2,1 Emulator (zero Apple redistribution)"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ModeArg {
    Jit,
    Interp,
}

impl From<ModeArg> for ExecMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Jit => ExecMode::Jit,
            ModeArg::Interp => ExecMode::Interpreter,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the RTX GPU-accelerated VirtualMac2,1 Easy-Boot UI (default when run with no args)
    Launch {
        #[arg(long)]
        demo: bool,
        #[arg(long, default_value_t = 1024)]
        ram_mb: u64,
        #[arg(long, default_value = "./ipsw_cache")]
        cache_dir: PathBuf,
        #[arg(long, default_value = "VirtualMac2,1")]
        model: String,
        #[arg(long, default_value = "-v serial=3 debug=0x14e")]
        cmdline: String,
        #[arg(long)]
        disk: Option<PathBuf>,
        #[arg(long)]
        read_only: bool,
        #[arg(long)]
        ramdisk: Option<PathBuf>,
        #[arg(long)]
        trustcache: Option<PathBuf>,
    },
    /// One-command Easy-Boot: auto-fetch Apple VirtualMac2,1 from official CDN (if missing) & boot
    EasyBoot {
        #[arg(long, default_value = "VirtualMac2,1")]
        model: String,
        #[arg(long, default_value = "./ipsw_cache")]
        cache_dir: PathBuf,
        #[arg(long, default_value_t = 1024)]
        ram_mb: u64,
        #[arg(long)]
        headless: bool,
        #[arg(long, default_value_t = 500_000)]
        max_steps: u64,
        #[arg(long, default_value = "-v serial=3 debug=0x14e")]
        cmdline: String,
        #[arg(long)]
        with_ramdisk: bool,
        #[arg(long)]
        disk: Option<PathBuf>,
        #[arg(long)]
        read_only: bool,
        #[arg(long)]
        ramdisk: Option<PathBuf>,
        #[arg(long)]
        trustcache: Option<PathBuf>,
    },
    /// Inspect host NVIDIA RTX / wgpu adapter and optionally execute headless WGSL shader render test
    GpuInfo {
        #[arg(long)]
        test: bool,
    },
    /// Query Apple's official IPSW catalog live (mesu.apple.com)
    Catalog {
        #[arg(long, default_value = "VirtualMac2,1")]
        model: String,
        #[arg(long)]
        all: bool,
    },
    /// Fetch macOS VirtualMac2,1 assets directly from Apple CDN
    Fetch {
        #[arg(long, default_value = "VirtualMac2,1")]
        model: String,
        #[arg(long, default_value = "./ipsw_cache")]
        out: PathBuf,
        #[arg(long, default_value_t = true)]
        boot_only: bool,
        #[arg(long)]
        full: bool,
        #[arg(long)]
        ramdisk: bool,
    },
    /// Inspect an IM4P/IMG4, DeviceTree ADT, or Mach-O kernelcache file
    Inspect {
        path: PathBuf,
        #[arg(long)]
        dump_adt: bool,
        #[arg(long, value_delimiter = ',')]
        disasm: Vec<String>,
        #[arg(long, default_value_t = 40)]
        disasm_count: usize,
        #[arg(long, value_delimiter = ',')]
        dump_va: Vec<String>,
        #[arg(long)]
        find_str: Option<String>,
    },
    /// Boot a fetched Apple vma2 kernelcache or the ARM64 self-test demo kernel
    Boot {
        #[arg(long)]
        demo: bool,
        #[arg(long)]
        kernel: Option<PathBuf>,
        #[arg(long)]
        devicetree: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ModeArg::Jit)]
        mode: ModeArg,
        #[arg(long, default_value_t = 512)]
        ram_mb: u64,
        #[arg(long, default_value_t = 10_000_000)]
        max_steps: u64,
        #[arg(long)]
        gui: bool,
        #[arg(long, default_value = "-v serial=3 debug=0x14e")]
        cmdline: String,
        #[arg(long, default_value_t = 0)]
        trace_tail: usize,
        #[arg(long)]
        disk: Option<PathBuf>,
        #[arg(long)]
        read_only: bool,
        #[arg(long)]
        ramdisk: Option<PathBuf>,
        #[arg(long)]
        trustcache: Option<PathBuf>,
    },
    /// Execute a raw flat ARM64 binary or standalone Mach-O executable
    Run {
        binary: PathBuf,
        #[arg(long, value_enum, default_value_t = ModeArg::Jit)]
        mode: ModeArg,
        #[arg(long, default_value = "0x80000000")]
        load_addr: String,
        #[arg(long, default_value_t = 100_000)]
        max_steps: u64,
        #[arg(long)]
        trace: bool,
    },
    /// Run differential JIT vs Interpreter verification and performance benchmark
    Bench {
        #[arg(long, default_value_t = 200_000)]
        iterations: u64,
    },
    /// Demonstrate live x86-64 host <-> ARM64 guest ABI call translation
    AbiDemo,
}

fn parse_hex_or_dec(s: &str) -> std::result::Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse::<u64>().map_err(|e| e.to_string())
    }
}

fn resolve_cache_dir(cache_dir: &Path) -> PathBuf {
    let has_bundle = |dir: &Path| {
        dir.join("kernelcache.release.vma2").exists()
            && dir.join("DeviceTree.vma2macosap.im4p").exists()
    };
    if has_bundle(cache_dir) {
        return cache_dir.to_path_buf();
    }
    if cache_dir == Path::new("./ipsw_cache") || cache_dir == Path::new("ipsw_cache") {
        let rel = PathBuf::from("./target/release/ipsw_cache");
        if has_bundle(&rel) {
            return rel;
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let exe_cache = dir.join("ipsw_cache");
                if has_bundle(&exe_cache) {
                    return exe_cache;
                }
            }
        }
    }
    cache_dir.to_path_buf()
}

fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Commands::Launch {
        demo: false,
        ram_mb: 1024,
        cache_dir: PathBuf::from("./ipsw_cache"),
        model: "VirtualMac2,1".into(),
        cmdline: "-v serial=3 debug=0x14e".into(),
        disk: None,
        read_only: false,
        ramdisk: None,
        trustcache: None,
    });
    match command {
        Commands::Launch {
            demo,
            ram_mb,
            cache_dir,
            model,
            cmdline,
            disk,
            read_only,
            ramdisk,
            trustcache,
        } => {
            let cache_dir = resolve_cache_dir(&cache_dir);
            launch_rtx_gui(LaunchOptions {
                initial_mode: if demo {
                    LaunchMode::DemoBoot
                } else {
                    LaunchMode::AppleKernel
                },
                ram_mb,
                cache_dir,
                model,
                cmdline,
                kernel_path: None,
                devicetree_path: None,
                disk_path: disk,
                disk_read_only: read_only,
                ramdisk_path: ramdisk,
                trustcache_path: trustcache,
            })?;
        }
        Commands::EasyBoot {
            model,
            cache_dir,
            ram_mb,
            headless,
            max_steps,
            cmdline,
            with_ramdisk,
            disk,
            read_only,
            ramdisk,
            trustcache,
        } => {
            let cache_dir = resolve_cache_dir(&cache_dir);
            let kpath = cache_dir.join("kernelcache.release.vma2");
            let dpath = cache_dir.join("DeviceTree.vma2macosap.im4p");
            if !kpath.exists() || !dpath.exists() {
                println!(
                    "Easy-Boot: fetching official {model} boot assets from Apple CDN into {}...",
                    cache_dir.display()
                );
                let bundle = fetch_boot_bundle(&model, &cache_dir)?;
                println!(
                    "Fetched macOS {} ({}) -> {}",
                    bundle.build.product_version,
                    bundle.build.build_version,
                    cache_dir.display()
                );
            } else {
                println!("Easy-Boot: using cached {model} assets in {}", cache_dir.display());
            }

            let (rd_path, tc_path) = if with_ramdisk && ramdisk.is_none() {
                println!("Easy-Boot: ensuring RestoreRamDisk + RestoreTrustCache from Apple CDN...");
                let rb = fetch_ramdisk_bundle(&model, &cache_dir)?;
                (Some(rb.ramdisk_path), trustcache.or(rb.trustcache_path))
            } else {
                let auto_rd = ramdisk.or_else(|| {
                    let p = cache_dir.join("RestoreRamDisk.dmg");
                    p.exists().then_some(p)
                });
                let auto_tc = trustcache.or_else(|| {
                    let p = cache_dir.join("RestoreTrustCache.im4p");
                    p.exists().then_some(p)
                });
                (auto_rd, auto_tc)
            };

            if headless {
                let kb = std::fs::read(&kpath)?;
                let db = std::fs::read(&dpath)?;
                let rd_bytes = rd_path.as_ref().map(std::fs::read).transpose()?;
                let tc_bytes = tc_path.as_ref().map(std::fs::read).transpose()?;
                let effective_ram_mb = if rd_bytes.is_some() {
                    ram_mb.max(2048)
                } else {
                    ram_mb
                };
                let mut vm = ArmVm::new(VmConfig {
                    mode: ExecMode::Jit,
                    ram_mb: effective_ram_mb,
                    echo_uart: true,
                    max_steps,
                    trace: false,
                });
                if let Some(d) = &disk {
                    vm.attach_disk_image(d, read_only)?;
                }
                let loaded = load_apple_kernel_bundle_with_options(
                    &mut vm.bus,
                    &kb,
                    &db,
                    &cmdline,
                    &AppleLoadOptions {
                        ramdisk_bytes: rd_bytes,
                        trustcache_bytes: tc_bytes,
                    },
                )?;
                println!(
                    "Loaded Apple VirtualMac2,1 kernelcache ({} segments, entry=0x{:016x}, ADT={})",
                    loaded.kernel_segments_loaded,
                    loaded.entry_pc,
                    loaded.adt_summary.as_deref().unwrap_or("none")
                );
                vm.apply_image(&loaded);
                let stats = vm.run()?;
                print_vm_stats(&stats);
            } else {
                launch_rtx_gui(LaunchOptions {
                    initial_mode: LaunchMode::AppleKernel,
                    ram_mb,
                    cache_dir,
                    model,
                    cmdline,
                    kernel_path: Some(kpath),
                    devicetree_path: Some(dpath),
                    disk_path: disk,
                    disk_read_only: read_only,
                    ramdisk_path: rd_path,
                    trustcache_path: tc_path,
                })?;
            }
        }
        Commands::GpuInfo { test } => {
            let info = probe_best_gpu()?;
            println!("=== ArmTranslator Host GPU Adapter ===");
            println!("Adapter        : {}", info.name);
            println!("Vendor/Device  : 0x{:04x} / 0x{:04x}", info.vendor, info.device);
            println!("DeviceType     : {:?}", info.device_type);
            println!("Backend        : {:?}", info.backend);
            println!("Driver         : {} ({})", info.driver, info.driver_info);
            if test {
                let (_, non_zero) = run_headless_gpu_self_test()?;
                println!("WGSL Render    : PASS ({non_zero} non-zero GPU-composited pixels verified)");
            }
        }
        Commands::Catalog { model, all } => {
            let filter = if all { None } else { Some(model.as_str()) };
            let builds = fetch_catalog(filter)?;
            println!(
                "Official Apple IPSW Catalog ({} builds found):",
                builds.len()
            );
            for b in &builds {
                println!(
                    "  {:<14} macOS {:<8} ({:<8}) -> {}",
                    b.model, b.product_version, b.build_version, b.firmware_url
                );
            }
        }
        Commands::Fetch {
            model,
            out,
            boot_only,
            full,
            ramdisk,
        } => {
            if full {
                let builds = fetch_catalog(Some(&model))?;
                let b = builds.into_iter().next().ok_or_else(|| {
                    armt_core::ArmtError::AppleContainer(format!("Model {model} not found"))
                })?;
                std::fs::create_dir_all(&out)?;
                let filename = b
                    .firmware_url
                    .rsplit('/')
                    .next()
                    .unwrap_or("Restore.ipsw");
                let dest = out.join(filename);
                println!(
                    "Downloading full IPSW for {} {} ({}) -> {}",
                    b.model,
                    b.product_version,
                    b.build_version,
                    dest.display()
                );
                let mut last_pct = 999u64;
                download_full_ipsw(&b.firmware_url, &dest, b.sha1.as_deref(), |done, total| {
                    if total > 0 {
                        let pct = (done * 100) / total;
                        if pct != last_pct && pct % 5 == 0 {
                            last_pct = pct;
                            println!("  {pct}% ({} / {} MB)", done / 1_048_576, total / 1_048_576);
                        }
                    }
                })?;
                println!("Saved full IPSW: {}", dest.display());
            } else {
                let _ = boot_only;
                println!("Querying Apple catalog for {model} & extracting boot bundle via HTTP Range ZIP64...");
                let bundle = fetch_boot_bundle(&model, &out)?;
                println!(
                    "Extracted {} macOS {} ({}) from official Apple CDN:",
                    bundle.build.model, bundle.build.product_version, bundle.build.build_version
                );
                println!("  BuildManifest : {}", bundle.manifest_path.display());
                println!("  DeviceTree    : {}", bundle.devicetree_path.display());
                println!("  Kernelcache   : {}", bundle.kernelcache_path.display());
                if ramdisk {
                    println!("Fetching RestoreRamDisk + RestoreTrustCache via HTTP Range ZIP64...");
                    let rb = fetch_ramdisk_bundle(&model, &out)?;
                    println!("  RestoreRamDisk    : {}", rb.ramdisk_path.display());
                    if let Some(tc) = &rb.trustcache_path {
                        println!("  RestoreTrustCache : {}", tc.display());
                    }
                }
            }
        }
        Commands::Inspect {
            path,
            dump_adt: do_dump,
            disasm: disasm_vas,
            disasm_count,
            dump_va,
            find_str,
        } => {
            let bytes = std::fs::read(&path)?;
            println!("File: {} ({} bytes)", path.display(), bytes.len());
            let payload = if bytes.first() == Some(&0x30) {
                let im4p = parse_img4_or_im4p(&bytes)?;
                println!(
                    "IM4P container: tag='{}', desc='{}', compressed_payload={} bytes",
                    im4p.tag,
                    im4p.description,
                    im4p.payload.len()
                );
                decompress_payload(&im4p.payload)?
            } else {
                decompress_payload(&bytes)?
            };
            println!("Decompressed payload: {} bytes", payload.len());

            if payload.len() >= 4
                && u32::from_le_bytes(payload[0..4].try_into().unwrap()) == 0xFEED_FACF
            {
                let macho = parse_macho(&payload)?;
                println!(
                    "Mach-O 64-bit ARM64: filetype=0x{:x}, segments={}, fileset_entries={}, entry_pc={:?}",
                    macho.filetype,
                    macho.segments.len(),
                    macho.fileset_entries.len(),
                    macho.entry_pc.map(|p| format!("0x{p:016x}"))
                );
                for seg in macho.segments.iter().take(12) {
                    println!(
                        "  seg {:<16} vmaddr=0x{:016x} vmsize=0x{:08x} filesize=0x{:08x}",
                        seg.name, seg.vmaddr, seg.vmsize, seg.filesize
                    );
                }
                if let Some(needle_str) = find_str.as_deref() {
                    let needle = needle_str.as_bytes();
                    let mut targets = Vec::new();
                    for seg in &macho.segments {
                        let start = seg.fileoff as usize;
                        let end = (start + seg.filesize as usize).min(macho.raw.len());
                        if start >= end || needle.is_empty() {
                            continue;
                        }
                        let hay = &macho.raw[start..end];
                        for (pos, win) in hay.windows(needle.len()).enumerate() {
                            if win == needle {
                                targets.push(seg.vmaddr + pos as u64);
                            }
                        }
                    }
                    println!("\n--- String matches for {:?} ({}) ---", needle_str, targets.len());
                    for &tva in targets.iter().take(10) {
                        println!("  str VA = 0x{tva:016x}");
                        let tpage = tva & !0xFFF;
                        let toff = (tva & 0xFFF) as u32;
                        for seg in &macho.segments {
                            if seg.name != "__TEXT_EXEC" {
                                continue;
                            }
                            let start = seg.fileoff as usize;
                            let end = (start + seg.filesize as usize).min(macho.raw.len());
                            let code = &macho.raw[start..end];
                            for off in (0..code.len().saturating_sub(16)).step_by(4) {
                                let w0 = u32::from_le_bytes(code[off..off + 4].try_into().unwrap());
                                if (w0 & 0x9F00_0000) == 0x9000_0000 {
                                    let pc = seg.vmaddr + off as u64;
                                    let imm21 = (((w0 >> 5) & 0x7_FFFF) << 2) | ((w0 >> 29) & 3);
                                    let sext = ((imm21 as i32) << 11 >> 11) as i64;
                                    let page = ((pc & !0xFFF) as i64).wrapping_add(sext << 12) as u64;
                                    if page == tpage {
                                        let rd = w0 & 31;
                                        for k in 1..=4 {
                                            let w1 = u32::from_le_bytes(code[off + k * 4..off + k * 4 + 4].try_into().unwrap());
                                            if (w1 & 0xFFC0_0000) == 0x9100_0000
                                                && ((w1 >> 5) & 31) == rd
                                                && ((w1 >> 10) & 0xFFF) == toff
                                            {
                                                println!("    xref ADRP+ADD at VA = 0x{pc:016x}");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for va_str in disasm_vas {
                    let s = va_str.trim().trim_start_matches("0x").trim_start_matches("0X");
                    if let Ok(start_va) = u64::from_str_radix(s, 16) {
                        println!("\n--- Disassembly around 0x{start_va:016x} ({disasm_count} insns) ---");
                        for i in 0..disasm_count {
                            let va = start_va.wrapping_add((i as u64) * 4);
                            let mut found = None;
                            for seg in &macho.segments {
                                if va >= seg.vmaddr && va + 4 <= seg.vmaddr + seg.filesize {
                                    let off = (seg.fileoff + (va - seg.vmaddr)) as usize;
                                    if off + 4 <= macho.raw.len() {
                                        found = Some(u32::from_le_bytes(
                                            macho.raw[off..off + 4].try_into().unwrap(),
                                        ));
                                        break;
                                    }
                                }
                            }
                            if let Some(word) = found {
                                let insn = armt_decoder::decode(word);
                                let text = armt_decoder::disasm(&insn, va);
                                println!("  0x{va:016x}: {word:08x}  {text}");
                            } else {
                                println!("  0x{va:016x}: <unmapped in file segments>");
                                break;
                            }
                        }
                    }
                }
                for va_str in dump_va {
                    let s = va_str.trim().trim_start_matches("0x").trim_start_matches("0X");
                    if let Ok(start_va) = u64::from_str_radix(s, 16) {
                        println!("\n--- Data at 0x{start_va:016x} ---");
                        for seg in &macho.segments {
                            if start_va >= seg.vmaddr && start_va < seg.vmaddr + seg.filesize {
                                let off = (seg.fileoff + (start_va - seg.vmaddr)) as usize;
                                let end = (off + 128).min((seg.fileoff + seg.filesize) as usize).min(macho.raw.len());
                                let slice = &macho.raw[off..end];
                                for (row, chunk) in slice.chunks(16).enumerate() {
                                    let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
                                    let asc: String = chunk
                                        .iter()
                                        .map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '.' })
                                        .collect();
                                    println!("  0x{:016x}: {:<48} |{}|", start_va + (row * 16) as u64, hex.join(" "), asc);
                                }
                                let qwords: Vec<String> = slice
                                    .chunks_exact(8)
                                    .take(8)
                                    .map(|c| format!("0x{:016x}", u64::from_le_bytes(c.try_into().unwrap())))
                                    .collect();
                                println!("  qwords: {}", qwords.join(", "));
                                break;
                            }
                        }
                    }
                }
            } else {
                let adt = parse_adt(&payload)?;
                println!(
                    "Apple Device Tree (ADT): root='{}', model={:?}, top_children={}",
                    adt.name,
                    adt.prop_str("model"),
                    adt.children.len()
                );
                if do_dump {
                    println!("{}", dump_adt(&adt, 6));
                }
            }
        }
        Commands::Boot {
            demo,
            kernel,
            devicetree,
            mode,
            ram_mb,
            max_steps,
            gui,
            cmdline,
            trace_tail,
            disk,
            read_only,
            ramdisk,
            trustcache,
        } => {
            if gui {
                let launch_mode = if demo {
                    LaunchMode::DemoBoot
                } else {
                    LaunchMode::AppleKernel
                };
                launch_rtx_gui(LaunchOptions {
                    initial_mode: launch_mode,
                    ram_mb,
                    cache_dir: resolve_cache_dir(Path::new("./ipsw_cache")),
                    model: "VirtualMac2,1".into(),
                    cmdline,
                    kernel_path: kernel,
                    devicetree_path: devicetree,
                    disk_path: disk,
                    disk_read_only: read_only,
                    ramdisk_path: ramdisk,
                    trustcache_path: trustcache,
                })?;
                return Ok(());
            }

            let effective_ram_mb = if ramdisk.is_some() {
                ram_mb.max(2048)
            } else {
                ram_mb
            };
            let mut vm = ArmVm::new(VmConfig {
                mode: mode.into(),
                ram_mb: effective_ram_mb,
                echo_uart: true,
                max_steps,
                trace: false,
            });
            vm.trace_tail = trace_tail;
            if let Some(d) = &disk {
                vm.attach_disk_image(d, read_only)?;
            }

            let img = match (demo, kernel, devicetree) {
                (true, _, _) | (false, None, None) => load_demo_kernel(&mut vm.bus)?,
                (false, Some(kpath), Some(dpath)) => {
                    let kb = std::fs::read(&kpath)?;
                    let db = std::fs::read(&dpath)?;
                    let rd_bytes = ramdisk.as_ref().map(std::fs::read).transpose()?;
                    let tc_bytes = trustcache.as_ref().map(std::fs::read).transpose()?;
                    let loaded = load_apple_kernel_bundle_with_options(
                        &mut vm.bus,
                        &kb,
                        &db,
                        &cmdline,
                        &AppleLoadOptions {
                            ramdisk_bytes: rd_bytes,
                            trustcache_bytes: tc_bytes,
                        },
                    )?;
                    println!(
                        "Loaded Apple VirtualMac2,1 kernelcache ({} segments, entry=0x{:016x}, ADT={})",
                        loaded.kernel_segments_loaded,
                        loaded.entry_pc,
                        loaded.adt_summary.as_deref().unwrap_or("none")
                    );
                    loaded
                }
                _ => {
                    return Err(armt_core::ArmtError::Vm(
                        "Pass --demo or both --kernel <path> and --devicetree <path>".into(),
                    ));
                }
            };
            vm.apply_image(&img);
            let stats = vm.run()?;
            print_vm_stats(&stats);
        }
        Commands::Run {
            binary,
            mode,
            load_addr,
            max_steps,
            trace,
        } => {
            let addr = parse_hex_or_dec(&load_addr)
                .map_err(|e| armt_core::ArmtError::Vm(format!("Bad --load-addr: {e}")))?;
            let bytes = std::fs::read(&binary)?;
            let mut vm = ArmVm::new(VmConfig {
                mode: mode.into(),
                ram_mb: 64,
                echo_uart: true,
                max_steps,
                trace,
            });
            let img = load_raw_or_macho(&mut vm.bus, &bytes, addr)?;
            vm.apply_image(&img);
            let stats = vm.run()?;
            println!(
                "Done ({}) | insns={} | PC=0x{:016x} | X0=0x{:x}",
                stats.exit_reason, stats.insns_executed, stats.final_pc, stats.x0
            );
        }
        Commands::Bench { iterations } => {
            let rep = run_benchmark(iterations)?;
            println!("=== ArmTranslator x86-64 Dynarec Benchmark ({} iters) ===", rep.iterations);
            println!(
                "Differential check  : {}",
                if rep.differential_pass { "PASS (bit-exact match)" } else { "FAIL" }
            );
            println!(
                "Reference Interp    : {} insns in {:.2} ms ({:.2} MIPS)",
                rep.interp_insns, rep.interp_ms, rep.interp_mips
            );
            println!(
                "x86-64 Dynarec JIT  : {} insns in {:.2} ms ({:.2} MIPS, {} blocks)",
                rep.jit_insns, rep.jit_ms, rep.jit_mips, rep.jit_blocks_compiled
            );
            println!("JIT Speedup         : {:.2}x", rep.speedup);
            if !rep.differential_pass {
                return Err(armt_core::ArmtError::Vm("Differential check failed".into()));
            }
        }
        Commands::AbiDemo => {
            let mut vm = ArmVm::new(VmConfig {
                mode: ExecMode::Jit,
                ram_mb: 16,
                echo_uart: false,
                max_steps: 50_000,
                trace: false,
            });
            let fn_addr = vm.bus.memmap.ram_base + 0x2000;
            // Guest ARM64 function: computes (X0 * 3 + X1), passes to host via SVC (X16=1001), returns
            let words = [
                asm::movz(Width::X64, Reg(2), 3, 0),
                asm::mul(Width::X64, Reg(0), Reg(0), Reg(2)),
                asm::add_reg(Width::X64, Reg(0), Reg(0), Reg(1), ShiftType::Lsl, 0, false),
                asm::movz(Width::X64, Reg(16), 1001, 0),
                asm::svc(0x80),
                asm::ret(Reg::LR),
            ];
            vm.bus.write_bytes(fn_addr, &asm::to_bytes(&words))?;

            let mut jit = JitEngine::new();
            let mut mmu = Mmu::new();
            let mut table = HostServiceTable::new();
            table.register(1001, |args, _| {
                println!("  [Host x64 callback] received ARM64 X0 = {}", args[0]);
                HostCallResult {
                    x0: args[0] + 1000,
                    x1: 42,
                }
            });

            println!("Invoking ARM64 guest function from x86-64 host with args (15, 7)...");
            let res = invoke_arm64_fn(
                &mut jit,
                &mut vm.cpu,
                &mut vm.bus,
                &mut mmu,
                &mut table,
                fn_addr,
                HostCallArgs::from_slice(&[15, 7]),
                10_000,
            )?;
            println!(
                "Returned from ARM64 -> x86-64 host: X0={}, X1={} (expected X0=1052, X1=42)",
                res.x0, res.x1
            );
            assert_eq!(res.x0, 1052);
            assert_eq!(res.x1, 42);
        }
    }
    Ok(())
}

fn print_vm_stats(stats: &armt_vm::VmStats) {
    println!("\n=== ArmTranslator VM Summary ===");
    println!("Exit reason         : {}", stats.exit_reason);
    println!("Final PC            : 0x{:016x}", stats.final_pc);
    println!("X0                  : 0x{:016x} ({})", stats.x0, stats.x0);
    println!(
        "Executed            : {} insns in {:.3} ms ({:.2} MIPS)",
        stats.insns_executed,
        stats.elapsed_secs * 1000.0,
        stats.mips
    );
    println!(
        "JIT telemetry       : {} blocks compiled, {} hits, {} misses, {} interp fallbacks",
        stats.blocks_compiled, stats.jit_hits, stats.jit_misses, stats.interp_fallbacks
    );
    println!(
        "Devices             : UART0 {} bytes TX, Framebuffer {} non-zero pixels",
        stats.uart_bytes, stats.fb_non_zero_pixels
    );
    if !stats.hvc_calls.is_empty() {
        println!(
            "Hypercalls (HVC)    : {:?}",
            stats
                .hvc_calls
                .iter()
                .take(8)
                .map(|(fid, c)| format!("0x{fid:x}({c})"))
                .collect::<Vec<_>>()
        );
    }
    if !stats.unknown_sysregs.is_empty() {
        println!(
            "Apple/Unknown sysregs observed ({}): {:?}",
            stats.unknown_sysregs.len(),
            &stats.unknown_sysregs[..stats.unknown_sysregs.len().min(32)]
        );
    }
    if !stats.unknown_mmio_pages.is_empty() {
        println!(
            "Unmapped MMIO probe pages ({}): {:?}",
            stats.unknown_mmio_pages.len(),
            stats
                .unknown_mmio_pages
                .iter()
                .take(64)
                .map(|(a, c)| format!("0x{a:x}({c})"))
                .collect::<Vec<_>>()
        );
    }
}

fn main() {
    if let Err(e) = run_cli() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
