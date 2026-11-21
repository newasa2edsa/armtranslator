<img src="assets/icon.svg" width="64" height="64" alt="">

# ArmTranslator

**ArmTranslator** is a fast, memory-safe, userspace **AArch64 (ARM64) → x86-64 Dynamic Binary Translator (Dynarec JIT)** and sandboxed **`VirtualMac2,1` (`vma2`)** macOS emulator written in pure Rust.

Inspired by [`tylertms/Stela`](https://github.com/tylertms/Stela), **ArmTranslator runs 100% in userspace** on Windows and Linux x86-64 PCs — it never installs custom UEFI bootloaders, never modifies your GPT partition table, and carries **zero risk of making your PC unbootable**.

## Legal Notice — Zero Apple Redistribution

This repository contains **zero Apple code, firmware, kernels, or binaries**. Every macOS component is fetched live by the user directly from Apple's official update catalog (`https://mesu.apple.com/assets/macos/com_apple_macOSIPSW/com_apple_macOSIPSW.xml`) and CDN (`updates.cdn-apple.com`).

## Architecture

```text
+------------------------------------------------------------------------+
|  armt-cli  (One-Click Launch, Easy-Boot, GPU Probe & CLI Tools)        |
+------------------------------------------------------------------------+
|  armt-gpu  (NVIDIA RTX GPU Hardware Renderer via eframe/wgpu WGSL)     |
+------------------------------------------------------------------------+
|  armt-vm   (VM Orchestrator, VMApple PAC HVC, Real XNU Kernel Loader)    |
+-----------------+-----------------+------------------+-----------------+
|   armt-decoder  |    armt-jit     |    armt-apple    |     armt-hw     |
| AArch64 decode, | W^X x86-64 JIT, | Live Catalog XML,| VirtualMac2,1:  |
| disasm & ARM64  | Ref Interpreter,| HTTP-Range ZIP64,| AppleParavirtGpu|
| assembler (incl.| 4KB/16KB MMU,   | IM4P/IMG4 DER,   | VirtIO-GPU v2,  |
| PAC & LDXP/STXP)| x64 <-> ARM64   | LZFSE/LZSS, ADT, | Live VirtioInput|
|                 | ABI Bridge      | 64-bit Mach-O    | UART/GIC/Timer  |
+-----------------+-----------------+------------------+-----------------+
|  armt-core (Shared ISA IR, CpuState, GuestMemory/MMIO/IRQ, Boot/ABI)   |
+------------------------------------------------------------------------+
```

## Quick Start & Easy Boot

```powershell
# Build release binary
cargo build --release

# 0. One-click Easy-Boot RTX GUI (Real Apple VirtualMac2,1 XNU launcher)
.\target\release\armtranslator.exe
# or:
.\target\release\armtranslator.exe launch

# 1. Verify NVIDIA RTX adapter & run headless WGSL display shader test
.\target\release\armtranslator.exe gpu-info --test

# 2. One-command Easy-Boot real Apple VirtualMac2,1 (auto-fetches from Apple CDN & boots)
.\target\release\armtranslator.exe easy-boot

# 3. Differential JIT vs Interpreter check + MIPS benchmark
.\target\release\armtranslator.exe bench

# 4. x86-64 <-> ARM64 ABI call bridge & SVC callback demo
.\target\release\armtranslator.exe abi-demo

# 5. Boot built-in ARM64 self-test kernel
.\target\release\armtranslator.exe boot --demo --mode jit

# 6. Query Apple's live IPSW catalog & HTTP-Range extract DeviceTree + kernelcache
.\target\release\armtranslator.exe catalog --model VirtualMac2,1
.\target\release\armtranslator.exe fetch --model VirtualMac2,1 --boot-only --out ./ipsw_cache
.\target\release\armtranslator.exe inspect .\ipsw_cache\DeviceTree.vma2macosap.im4p --dump-adt
.\target\release\armtranslator.exe inspect .\ipsw_cache\kernelcache.release.vma2
```

## Current Capabilities & Roadmap

**Implemented today:**
- **NVIDIA RTX Hardware Display Pipeline (`armt-gpu`)**: `eframe 0.37` + `wgpu 30` (DirectX 12 / Vulkan) with explicit discrete GPU selection, custom WGSL scanout + hardware cursor shader (`display.wgsl`), and live mouse/keyboard forwarding.
- **VirtIO-GPU v2 & `AppleParavirtGpu` (`armt-hw`)**: Full VirtIO-GPU v2 MMIO split-virtqueue 2D scanout + cursor commands alongside paravirt 2D compositor acceleration (`0x200b_0000`).
- **One-Click Easy-Boot & Live Apple CDN Pipeline**: Auto-queries `mesu.apple.com` and extracts `BuildManifest.plist`, `DeviceTree.vma2macosap.im4p`, and `kernelcache.release.vma2` via HTTP-Range ZIP64 (no 25 GB download required), with pure-Rust IM4P/IMG4 DER, LZFSE/LZSS, ADT, and 64-bit Mach-O/`MH_FILESET` parsing.
- **VMApple PAC & SMCCC/PSCI Hypercalls + 16 KB MMU + AdvSIMD**: Handles `0xC100_0000..0xC100_0005` PAC HVCs, SMCCC/PSCI, 32 MiB L2-aligned macOS `boot_args` (`BOOT_LINE_LENGTH=1024`), 4 KB & Apple 16 KB `TTBR1` 48-bit page walks, `LDXP`/`STXP`/`CAS`/`DC ZVA`/`AT`, Advanced SIMD (`MOVI`, `UMINV`, `UMOV`, `DUP`, `EXT`, etc.), and MMU-aware JIT code caching.

**Roadmap toward full unmodified Apple WindowServer desktop:**
- Complete Apple-private `S3_*` system register, `AIC` v2, PMGR, and IODART MMIO modeling for late XNU kernel phases.
- Multi-vCPU SMP execution, VirtIO-GPU 3D / Metal AIR→SPIR-V/DXIL host RTX shader translation, APFS/AEA root volume decryption & mounting, and full Apple `WindowServer` boot.
