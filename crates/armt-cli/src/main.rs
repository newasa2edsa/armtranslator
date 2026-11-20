use armt_apple::{
    decompress_payload, download_full_ipsw, dump_adt, fetch_boot_bundle, fetch_catalog, parse_adt,
    parse_img4_or_im4p, parse_macho,
};
use armt_core::isa::{Reg, ShiftType, Width};
use armt_core::{GuestMemory, HostCallArgs, HostCallResult, Result};
use armt_decoder::asm;
use armt_jit::{invoke_arm64_fn, HostServiceTable, JitEngine, Mmu};
use armt_vm::{
    load_apple_kernel_bundle, load_demo_kernel, load_raw_or_macho, run_benchmark, ArmVm, ExecMode,
    VmConfig,
};
use clap::{Parser, Subcommand, ValueEnum};
use minifb::{Key, Window, WindowOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "armtranslator",
    version,
    about = "Userspace AArch64->x86-64 Dynarec Translator & VirtualMac2,1 Emulator (legal Apple asset fetcher, zero redistribution)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
    },
    /// Inspect an IM4P/IMG4, DeviceTree ADT, or Mach-O kernelcache file
    Inspect {
        path: PathBuf,
        #[arg(long)]
        dump_adt: bool,
    },
    /// Boot the built-in ARM64 demo kernel or a fetched Apple vma2 kernelcache + DeviceTree
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

fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
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
            }
        }
        Commands::Inspect { path, dump_adt: do_dump } => {
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
            } else {
                let adt = parse_adt(&payload)?;
                println!(
                    "Apple Device Tree (ADT): root='{}', model={:?}, top_children={}",
                    adt.name,
                    adt.prop_str("model"),
                    adt.children.len()
                );
                if do_dump {
                    println!("{}", dump_adt(&adt, 3));
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
        } => {
            let mut vm = ArmVm::new(VmConfig {
                mode: mode.into(),
                ram_mb,
                echo_uart: true,
                max_steps,
                trace: false,
            });

            let img = match (demo, kernel, devicetree) {
                (true, _, _) | (false, None, None) => load_demo_kernel(&mut vm.bus)?,
                (false, Some(kpath), Some(dpath)) => {
                    let kb = std::fs::read(&kpath)?;
                    let db = std::fs::read(&dpath)?;
                    let loaded = load_apple_kernel_bundle(&mut vm.bus, &kb, &db, &cmdline)?;
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

            let stats = if gui {
                let fb = vm.framebuffer();
                let done = Arc::new(AtomicBool::new(false));
                let done_bg = done.clone();
                let handle = std::thread::spawn(move || {
                    let res = vm.run();
                    done_bg.store(true, Ordering::SeqCst);
                    res
                });

                let mut win = Window::new(
                    "ArmTranslator - VirtualMac2,1 Display (ESC to close)",
                    fb.width as usize,
                    fb.height as usize,
                    WindowOptions::default(),
                )
                .map_err(|e| armt_core::ArmtError::Device(format!("GUI window failed: {e}")))?;
                win.set_target_fps(60);

                while win.is_open() && !win.is_key_down(Key::Escape) {
                    let guard = fb.pixels.lock().unwrap();
                    let _ = win.update_with_buffer(&guard, fb.width as usize, fb.height as usize);
                }
                handle
                    .join()
                    .map_err(|_| armt_core::ArmtError::Vm("VM thread panicked".into()))??
            } else {
                vm.run()?
            };

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
            if !stats.unknown_sysregs.is_empty() {
                println!(
                    "Apple/Unknown sysregs observed ({}): {:?}",
                    stats.unknown_sysregs.len(),
                    &stats.unknown_sysregs[..stats.unknown_sysregs.len().min(8)]
                );
            }
            if !stats.unknown_mmio_pages.is_empty() {
                println!(
                    "Unmapped MMIO probe pages ({}): {:?}",
                    stats.unknown_mmio_pages.len(),
                    stats
                        .unknown_mmio_pages
                        .iter()
                        .take(8)
                        .map(|(a, c)| format!("0x{a:x}({c})"))
                        .collect::<Vec<_>>()
                );
            }
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

fn main() {
    if let Err(e) = run_cli() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
