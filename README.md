<img src="assets/icon.svg" width="64" height="64" alt="">

# ArmTranslator

**ArmTranslator** is a fast, memory-safe, userspace **AArch64 (ARM64) → x86-64 Dynamic Binary Translator (Dynarec JIT)** and sandboxed **`VirtualMac2,1` (`vma2`)** macOS emulator written in pure Rust.

Inspired by [`tylertms/Stela`](https://github.com/tylertms/Stela), **ArmTranslator runs 100% in userspace** on Windows and Linux x86-64 PCs — it never installs custom UEFI bootloaders, never modifies your GPT partition table, and carries **zero risk of making your PC unbootable**.

## Legal Notice — Zero Apple Redistribution

This repository contains **zero Apple code, firmware, kernels, or binaries**. Every macOS component is fetched live by the user directly from Apple's official update catalog (`https://mesu.apple.com/assets/macos/com_apple_macOSIPSW/com_apple_macOSIPSW.xml`) and CDN (`updates.cdn-apple.com`).

## Architecture

```text
+----------------------------------------------------------------------+
|  armt-cli  (CLI + minifb GUI Display Window)                         |
+----------------------------------------------------------------------+
|  armt-vm   (VM Orchestrator, XNU boot_args builder, Demo & Benchmarks)|
+-----------------+-----------------+------------------+---------------+
|   armt-decoder  |    armt-jit     |    armt-apple    |    armt-hw    |
| AArch64 decode, | W^X x86-64 JIT, | Live Catalog XML,| VirtualMac2,1:|
| disasm & mini   | Ref Interpreter,| HTTP-Range ZIP64,| PL011 UART,   |
| ARM64 assembler | 4KB/16KB MMU,   | IM4P/IMG4 DER,   | GICv3, Timer, |
| (incl. ARM64e   | x64 <-> ARM64   | LZFSE/LZSS, ADT, | VMApple AES/  |
| PAC hints)      | ABI Bridge      | 64-bit Mach-O    | CFG/BDIF,FB   |
+-----------------+-----------------+------------------+---------------+
|  armt-core (Shared ISA IR, CpuState, GuestMemory/MMIO/IRQ, Boot/ABI) |
+----------------------------------------------------------------------+
```

## Quick Start

```powershell
# Build release binary
cargo build --release

# 1. Differential JIT vs Interpreter check + MIPS benchmark
.\target\release\armtranslator.exe bench

# 2. x86-64 <-> ARM64 ABI call bridge & SVC callback demo
.\target\release\armtranslator.exe abi-demo

# 3. Boot built-in ARM64 diagnostic kernel (add --gui for live 1024x768 window)
.\target\release\armtranslator.exe boot --demo --mode jit

# 4. Query Apple's live IPSW catalog for VirtualMac2,1
.\target\release\armtranslator.exe catalog --model VirtualMac2,1

# 5. Fast HTTP-Range extraction of DeviceTree + kernelcache (no 25 GB download!)
.\target\release\armtranslator.exe fetch --model VirtualMac2,1 --boot-only --out ./ipsw_cache

# 6. Inspect real Apple DeviceTree (ADT) and Mach-O kernelcache
.\target\release\armtranslator.exe inspect .\ipsw_cache\DeviceTree.vma2macosap.im4p --dump-adt
.\target\release\armtranslator.exe inspect .\ipsw_cache\kernelcache.release.vma2

# 7. Boot real Apple XNU kernelcache + DeviceTree inside the sandboxed emulator
.\target\release\armtranslator.exe boot --kernel .\ipsw_cache\kernelcache.release.vma2 --devicetree .\ipsw_cache\DeviceTree.vma2macosap.im4p --mode jit --max-steps 50000
```

## Current Capabilities & Roadmap

**Implemented today:**
- Live Apple IPSW catalog querying & HTTP-Range ZIP64 extraction of `BuildManifest.plist`, `DeviceTree.vma2macosap.im4p`, and `kernelcache.release.vma2` (plus full resumable IPSW downloads).
- Pure-Rust IM4P/IMG4 ASN.1 DER parser, LZFSE (`bvx2`) / Apple LZSS (`complzss`) decompression, Apple Device Tree (ADT) parser, and 64-bit Mach-O / `MH_FILESET` loader.
- AArch64 decoder (including ARMv8.3 PAC hints) + reference interpreter + native W^X x86-64 Dynarec JIT with inline bounds-checked RAM access, 4KB/16KB Stage-1 MMU, and Win64/SysV ↔ AAPCS64 ABI bridge.
- Sandboxed `VirtualMac2,1` platform: PL011 UART, GICv3, 24 MHz Apple Generic Timer, `vmapple-cfg`, `vmapple-bdif`, `vmapple-aes`, `pvpanic-mmio`, `virtio-blk`, `virtio-input`, and `1024x768` BGRA framebuffer with optional `minifb` GUI.

**Roadmap toward full interactive macOS desktop GUI:**
- Full `MH_FILESET` multi-kext relocation & `TTBR1` early boot page-table handoff compatibility for latest XNU versions.
- Complete Apple-private `S3_*` system register & `AIC`/PMGR/IODART MMIO modeling.
- Multi-vCPU SMP execution, VirtIO-GPU 3D/Metal AIR→host shader translation, APFS/AEA disk image decryption & mounting, and full WindowServer boot.
