use crate::mmu::Mmu;
use armt_core::isa::{
    AddrMode, ExtendType, Instruction, PacOp, Reg, ShiftType, SysReg, Width,
};
use armt_core::{CpuState, GuestMemory, Result, SysRegBus};
use armt_decoder::decode;

/// Execution outcome of stepping a single AArch64 instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Continue,
    Branch(u64),
    Svc(u16),
    Hvc(u16),
    Smc(u16),
    Brk(u16),
    Hlt(u16),
    Wfi,
    Wfe,
    Eret,
    Exception { ec: u8, iss: u32, far: u64 },
    Unknown(u32),
}

/// Complete reference AArch64 interpreter (used for verification & complex system instruction fallback).
pub struct Interpreter;

#[inline]
pub fn strip_pac(addr: u64) -> u64 {
    if ((addr >> 55) & 1) != 0 {
        addr | 0xFFFF_0000_0000_0000
    } else {
        addr & 0x0000_FFFF_FFFF_FFFF
    }
}

#[inline]
fn shift_val(val: u64, shift: ShiftType, amount: u8, w: Width) -> u64 {
    let bits = w.bits();
    let amt = (amount as u32) & (bits - 1);
    let masked = val & w.mask();
    if amt == 0 {
        return masked;
    }
    let res = match shift {
        ShiftType::Lsl => masked.wrapping_shl(amt),
        ShiftType::Lsr => masked.wrapping_shr(amt),
        ShiftType::Asr => match w {
            Width::W32 => ((masked as u32 as i32).wrapping_shr(amt)) as u32 as u64,
            Width::X64 => ((masked as i64).wrapping_shr(amt)) as u64,
        },
        ShiftType::Ror => match w {
            Width::W32 => (masked as u32).rotate_right(amt) as u64,
            Width::X64 => masked.rotate_right(amt),
        },
    };
    res & w.mask()
}

#[inline]
fn extend_val(val: u64, ext: ExtendType, shift: u8) -> u64 {
    let base: u64 = match ext {
        ExtendType::Uxtb => (val as u8) as u64,
        ExtendType::Uxth => (val as u16) as u64,
        ExtendType::Uxtw => (val as u32) as u64,
        ExtendType::Uxtx => val,
        ExtendType::Sxtb => ((val as i8) as i64) as u64,
        ExtendType::Sxth => ((val as i16) as i64) as u64,
        ExtendType::Sxtw => ((val as i32) as i64) as u64,
        ExtendType::Sxtx => val,
    };
    base.wrapping_shl(shift as u32)
}

#[inline]
pub fn add_with_carry(a: u64, b: u64, carry_in: bool, w: Width) -> (u64, bool, bool, bool, bool) {
    let mask = w.mask();
    let ua = a & mask;
    let ub = b & mask;
    let c_in = carry_in as u128;
    let usum = (ua as u128) + (ub as u128) + c_in;
    let result = (usum as u64) & mask;

    let bits = w.bits();
    let sign_bit = 1u64 << (bits - 1);
    let n = (result & sign_bit) != 0;
    let z = result == 0;
    let c = (usum >> bits) != 0;
    let sa = (ua & sign_bit) != 0;
    let sb = (ub & sign_bit) != 0;
    let sr = n;
    let v = (sa == sb) && (sr != sa);
    (result, n, z, c, v)
}

#[inline]
fn write_gpr(cpu: &mut CpuState, w: Width, rd: Reg, val: u64, sp_allowed: bool) {
    let v = val & w.mask();
    if sp_allowed {
        cpu.write_sp_or_x(rd, v);
    } else {
        cpu.write_x(rd, v);
    }
}

#[inline]
fn read_gpr(cpu: &CpuState, w: Width, r: Reg, sp_allowed: bool) -> u64 {
    let v = if sp_allowed {
        cpu.read_sp_or_x(r)
    } else {
        cpu.read_x(r)
    };
    v & w.mask()
}

fn compute_addr(cpu: &mut CpuState, rn: Reg, mode: AddrMode) -> (u64, Option<u64>) {
    let base = cpu.read_sp_or_x(rn);
    match mode {
        AddrMode::UnsignedOffset(off) => (base.wrapping_add(off as u64), None),
        AddrMode::PreIndex(off) => {
            let ea = base.wrapping_add(off as u64);
            (ea, Some(ea))
        }
        AddrMode::PostIndex(off) => (base, Some(base.wrapping_add(off as u64))),
        AddrMode::RegOffset { rm, extend, shift } => {
            let off = extend_val(cpu.read_x(rm), extend, shift);
            (base.wrapping_add(off), None)
        }
    }
}

fn decode_fp_imm8(imm8: u8, is_64: bool) -> u64 {
    let sign = ((imm8 >> 7) & 1) as u64;
    let exp_hi = ((imm8 >> 6) & 1) as u64;
    let exp_lo = ((imm8 >> 4) & 3) as u64;
    let frac = (imm8 & 0xF) as u64;
    if is_64 {
        let exp = if exp_hi == 0 {
            0x400 | exp_lo
        } else {
            0x3FC | exp_lo
        };
        (sign << 63) | (exp << 52) | (frac << 48)
    } else {
        let exp = if exp_hi == 0 {
            0x80 | exp_lo
        } else {
            0x7C | exp_lo
        };
        ((sign << 31) | (exp << 23) | (frac << 19)) & 0xFFFF_FFFF
    }
}

impl Interpreter {
    pub fn step<M: GuestMemory + SysRegBus + ?Sized>(
        cpu: &mut CpuState,
        mem: &mut M,
        mmu: &mut Mmu,
    ) -> Result<StepOutcome> {
        let pc = cpu.pc;
        let pa = mmu.translate(cpu, mem, pc, false)?;
        let word = mem.read_u32(pa)?;
        let outcome = Self::exec_word(cpu, mem, mmu, pc, word)?;
        match outcome {
            StepOutcome::Continue => cpu.pc = pc.wrapping_add(4),
            StepOutcome::Branch(target) => cpu.pc = target,
            StepOutcome::Eret => Mmu::eret(cpu),
            _ => {}
        }
        cpu.insn_count += 1;
        Ok(outcome)
    }

    /// Execute a pre-fetched instruction word at `pc` WITHOUT modifying `cpu.pc` on `Continue`
    /// (on `Branch(t)`, returns `Branch(t)` so callers or JIT can apply `pc = t`).
    pub fn exec_word<M: GuestMemory + SysRegBus + ?Sized>(
        cpu: &mut CpuState,
        mem: &mut M,
        mmu: &mut Mmu,
        pc: u64,
        word: u32,
    ) -> Result<StepOutcome> {
        let inst = decode(word);
        Self::exec_inst(cpu, mem, mmu, pc, &inst)
    }

    pub fn exec_inst<M: GuestMemory + SysRegBus + ?Sized>(
        cpu: &mut CpuState,
        mem: &mut M,
        mmu: &mut Mmu,
        pc: u64,
        inst: &Instruction,
    ) -> Result<StepOutcome> {
        match *inst {
            Instruction::Hint { imm: 2 } => Ok(StepOutcome::Wfe),
            Instruction::Nop | Instruction::Hint { .. } | Instruction::Barrier { .. } | Instruction::Prfm => {
                Ok(StepOutcome::Continue)
            }
            Instruction::Clrex => {
                cpu.exclusive_valid = 0;
                Ok(StepOutcome::Continue)
            }

            // --- ALU Immediate & Register ---
            Instruction::AddImm {
                w,
                rd,
                rn,
                imm,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, true);
                let (res, n, z, c, v) = add_with_carry(a, imm, false, w);
                write_gpr(cpu, w, rd, res, !set_flags);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SubImm {
                w,
                rd,
                rn,
                imm,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, true);
                let (res, n, z, c, v) = add_with_carry(a, !imm, true, w);
                write_gpr(cpu, w, rd, res, !set_flags);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::AddReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                let (res, n, z, c, v) = add_with_carry(a, b, false, w);
                write_gpr(cpu, w, rd, res, false);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SubReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                let (res, n, z, c, v) = add_with_carry(a, !b, true, w);
                write_gpr(cpu, w, rd, res, false);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::AddExt {
                w,
                rd,
                rn,
                rm,
                extend,
                amount,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, true);
                let b = extend_val(cpu.read_x(rm), extend, amount) & w.mask();
                let (res, n, z, c, v) = add_with_carry(a, b, false, w);
                write_gpr(cpu, w, rd, res, !set_flags);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SubExt {
                w,
                rd,
                rn,
                rm,
                extend,
                amount,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, true);
                let b = extend_val(cpu.read_x(rm), extend, amount) & w.mask();
                let (res, n, z, c, v) = add_with_carry(a, !b, true, w);
                write_gpr(cpu, w, rd, res, !set_flags);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Adc {
                w,
                rd,
                rn,
                rm,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = read_gpr(cpu, w, rm, false);
                let (res, n, z, c, v) = add_with_carry(a, b, cpu.c(), w);
                write_gpr(cpu, w, rd, res, false);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Sbc {
                w,
                rd,
                rn,
                rm,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = read_gpr(cpu, w, rm, false);
                let (res, n, z, c, v) = add_with_carry(a, !b, cpu.c(), w);
                write_gpr(cpu, w, rd, res, false);
                if set_flags {
                    cpu.set_nzcv(n, z, c, v);
                }
                Ok(StepOutcome::Continue)
            }

            // --- Logical ---
            Instruction::AndImm {
                w,
                rd,
                rn,
                imm,
                set_flags,
            } => {
                let res = (read_gpr(cpu, w, rn, false) & imm) & w.mask();
                write_gpr(cpu, w, rd, res, !set_flags);
                if set_flags {
                    let n = (res & (1u64 << (w.bits() - 1))) != 0;
                    cpu.set_nzcv(n, res == 0, false, false);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::OrrImm { w, rd, rn, imm } => {
                let res = (read_gpr(cpu, w, rn, false) | imm) & w.mask();
                write_gpr(cpu, w, rd, res, true);
                Ok(StepOutcome::Continue)
            }
            Instruction::EorImm { w, rd, rn, imm } => {
                let res = (read_gpr(cpu, w, rn, false) ^ imm) & w.mask();
                write_gpr(cpu, w, rd, res, true);
                Ok(StepOutcome::Continue)
            }
            Instruction::AndReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                let res = (a & b) & w.mask();
                write_gpr(cpu, w, rd, res, false);
                if set_flags {
                    let n = (res & (1u64 << (w.bits() - 1))) != 0;
                    cpu.set_nzcv(n, res == 0, false, false);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::OrrReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                write_gpr(cpu, w, rd, a | b, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::EorReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                write_gpr(cpu, w, rd, a ^ b, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::BicReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
                set_flags,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                let res = (a & !b) & w.mask();
                write_gpr(cpu, w, rd, res, false);
                if set_flags {
                    let n = (res & (1u64 << (w.bits() - 1))) != 0;
                    cpu.set_nzcv(n, res == 0, false, false);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::OrnReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                write_gpr(cpu, w, rd, (a | !b) & w.mask(), false);
                Ok(StepOutcome::Continue)
            }
            Instruction::EonReg {
                w,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = shift_val(read_gpr(cpu, w, rm, false), shift, amount, w);
                write_gpr(cpu, w, rd, (a ^ !b) & w.mask(), false);
                Ok(StepOutcome::Continue)
            }

            // --- Move wide & PC-rel ---
            Instruction::Movz {
                w,
                rd,
                imm16,
                shift,
            } => {
                let v = ((imm16 as u64) << shift) & w.mask();
                cpu.write_x(rd, v);
                Ok(StepOutcome::Continue)
            }
            Instruction::Movn {
                w,
                rd,
                imm16,
                shift,
            } => {
                let v = (!((imm16 as u64) << shift)) & w.mask();
                cpu.write_x(rd, v);
                Ok(StepOutcome::Continue)
            }
            Instruction::Movk {
                w,
                rd,
                imm16,
                shift,
            } => {
                let cur = cpu.read_x(rd);
                let mask = !(0xFFFFu64 << shift);
                let v = ((cur & mask) | ((imm16 as u64) << shift)) & w.mask();
                cpu.write_x(rd, v);
                Ok(StepOutcome::Continue)
            }
            Instruction::Adr { rd, offset } => {
                cpu.write_x(rd, (pc as i64).wrapping_add(offset) as u64);
                Ok(StepOutcome::Continue)
            }
            Instruction::Adrp { rd, offset } => {
                let page = (pc & !0xFFF) as i64;
                cpu.write_x(rd, page.wrapping_add(offset) as u64);
                Ok(StepOutcome::Continue)
            }

            // --- Shifts, Bitfields, Extract, Bit-ops ---
            Instruction::Lslv { w, rd, rn, rm } => {
                let s = (cpu.read_x(rm) as u8) & (w.bits() as u8 - 1);
                write_gpr(cpu, w, rd, shift_val(cpu.read_x(rn), ShiftType::Lsl, s, w), false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Lsrv { w, rd, rn, rm } => {
                let s = (cpu.read_x(rm) as u8) & (w.bits() as u8 - 1);
                write_gpr(cpu, w, rd, shift_val(cpu.read_x(rn), ShiftType::Lsr, s, w), false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Asrv { w, rd, rn, rm } => {
                let s = (cpu.read_x(rm) as u8) & (w.bits() as u8 - 1);
                write_gpr(cpu, w, rd, shift_val(cpu.read_x(rn), ShiftType::Asr, s, w), false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Rorv { w, rd, rn, rm } => {
                let s = (cpu.read_x(rm) as u8) & (w.bits() as u8 - 1);
                write_gpr(cpu, w, rd, shift_val(cpu.read_x(rn), ShiftType::Ror, s, w), false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Sbfm {
                w,
                rd,
                rn,
                immr,
                imms,
            } => {
                let res = exec_bitfield(cpu.read_x(rn), w, immr, imms, BitfieldKind::Sbfm);
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Bfm {
                w,
                rd,
                rn,
                immr,
                imms,
            } => {
                let dst = cpu.read_x(rd);
                let src = cpu.read_x(rn);
                let res = exec_bfm(dst, src, w, immr, imms);
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Ubfm {
                w,
                rd,
                rn,
                immr,
                imms,
            } => {
                let res = exec_bitfield(cpu.read_x(rn), w, immr, imms, BitfieldKind::Ubfm);
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Extr { w, rd, rn, rm, lsb } => {
                let bits = w.bits();
                let hi = cpu.read_x(rn) & w.mask();
                let lo = cpu.read_x(rm) & w.mask();
                let l = (lsb as u32) & (bits - 1);
                let res = if l == 0 {
                    lo
                } else {
                    ((lo >> l) | (hi << (bits - l))) & w.mask()
                };
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Clz { w, rd, rn } => {
                let v = read_gpr(cpu, w, rn, false);
                let res = match w {
                    Width::W32 => (v as u32).leading_zeros() as u64,
                    Width::X64 => v.leading_zeros() as u64,
                };
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Cls { w, rd, rn } => {
                let v = read_gpr(cpu, w, rn, false);
                let bits = w.bits();
                let sign = (v >> (bits - 1)) & 1;
                let flipped = if sign == 1 { !v & w.mask() } else { v };
                let lz = match w {
                    Width::W32 => (flipped as u32).leading_zeros(),
                    Width::X64 => flipped.leading_zeros(),
                };
                cpu.write_x(rd, lz.saturating_sub(1) as u64);
                Ok(StepOutcome::Continue)
            }
            Instruction::Rbit { w, rd, rn } => {
                let v = read_gpr(cpu, w, rn, false);
                let res = match w {
                    Width::W32 => (v as u32).reverse_bits() as u64,
                    Width::X64 => v.reverse_bits(),
                };
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Rev16 { w, rd, rn } => {
                let v = read_gpr(cpu, w, rn, false);
                let mut res = 0u64;
                let count = match w {
                    Width::W32 => 2,
                    Width::X64 => 4,
                };
                for i in 0..count {
                    let h = ((v >> (i * 16)) & 0xFFFF) as u16;
                    res |= (h.swap_bytes() as u64) << (i * 16);
                }
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Rev32 { w, rd, rn } => {
                let v = read_gpr(cpu, w, rn, false);
                let lo = (v as u32).swap_bytes() as u64;
                let res = match w {
                    Width::W32 => lo,
                    Width::X64 => lo | (((((v >> 32) as u32).swap_bytes()) as u64) << 32),
                };
                cpu.write_x(rd, res);
                Ok(StepOutcome::Continue)
            }
            Instruction::Rev64 { rd, rn } => {
                cpu.write_x(rd, cpu.read_x(rn).swap_bytes());
                Ok(StepOutcome::Continue)
            }

            // --- Multiply / Divide ---
            Instruction::Madd { w, rd, rn, rm, ra } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = read_gpr(cpu, w, rm, false);
                let c = read_gpr(cpu, w, ra, false);
                write_gpr(cpu, w, rd, c.wrapping_add(a.wrapping_mul(b)), false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Msub { w, rd, rn, rm, ra } => {
                let a = read_gpr(cpu, w, rn, false);
                let b = read_gpr(cpu, w, rm, false);
                let c = read_gpr(cpu, w, ra, false);
                write_gpr(cpu, w, rd, c.wrapping_sub(a.wrapping_mul(b)), false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Smaddl { rd, rn, rm, ra } => {
                let a = (cpu.read_w(rn) as i32) as i64;
                let b = (cpu.read_w(rm) as i32) as i64;
                let c = cpu.read_x(ra) as i64;
                cpu.write_x(rd, c.wrapping_add(a.wrapping_mul(b)) as u64);
                Ok(StepOutcome::Continue)
            }
            Instruction::Smsubl { rd, rn, rm, ra } => {
                let a = (cpu.read_w(rn) as i32) as i64;
                let b = (cpu.read_w(rm) as i32) as i64;
                let c = cpu.read_x(ra) as i64;
                cpu.write_x(rd, c.wrapping_sub(a.wrapping_mul(b)) as u64);
                Ok(StepOutcome::Continue)
            }
            Instruction::Umaddl { rd, rn, rm, ra } => {
                let a = cpu.read_w(rn) as u64;
                let b = cpu.read_w(rm) as u64;
                let c = cpu.read_x(ra);
                cpu.write_x(rd, c.wrapping_add(a.wrapping_mul(b)));
                Ok(StepOutcome::Continue)
            }
            Instruction::Umsubl { rd, rn, rm, ra } => {
                let a = cpu.read_w(rn) as u64;
                let b = cpu.read_w(rm) as u64;
                let c = cpu.read_x(ra);
                cpu.write_x(rd, c.wrapping_sub(a.wrapping_mul(b)));
                Ok(StepOutcome::Continue)
            }
            Instruction::Smulh { rd, rn, rm } => {
                let a = (cpu.read_x(rn) as i64) as i128;
                let b = (cpu.read_x(rm) as i64) as i128;
                cpu.write_x(rd, ((a * b) >> 64) as u64);
                Ok(StepOutcome::Continue)
            }
            Instruction::Umulh { rd, rn, rm } => {
                let a = cpu.read_x(rn) as u128;
                let b = cpu.read_x(rm) as u128;
                cpu.write_x(rd, ((a * b) >> 64) as u64);
                Ok(StepOutcome::Continue)
            }
            Instruction::Udiv { w, rd, rn, rm } => {
                let n = read_gpr(cpu, w, rn, false);
                let d = read_gpr(cpu, w, rm, false);
                let res = if d == 0 { 0 } else { n / d };
                write_gpr(cpu, w, rd, res, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Sdiv { w, rd, rn, rm } => {
                let res = match w {
                    Width::W32 => {
                        let n = cpu.read_w(rn) as i32;
                        let d = cpu.read_w(rm) as i32;
                        if d == 0 {
                            0
                        } else {
                            n.wrapping_div(d) as u32 as u64
                        }
                    }
                    Width::X64 => {
                        let n = cpu.read_x(rn) as i64;
                        let d = cpu.read_x(rm) as i64;
                        if d == 0 {
                            0
                        } else {
                            n.wrapping_div(d) as u64
                        }
                    }
                };
                write_gpr(cpu, w, rd, res, false);
                Ok(StepOutcome::Continue)
            }

            // --- Conditional Select & Compare ---
            Instruction::Csel { w, rd, rn, rm, cond } => {
                let v = if cpu.cond_holds(cond) {
                    read_gpr(cpu, w, rn, false)
                } else {
                    read_gpr(cpu, w, rm, false)
                };
                write_gpr(cpu, w, rd, v, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Csinc { w, rd, rn, rm, cond } => {
                let v = if cpu.cond_holds(cond) {
                    read_gpr(cpu, w, rn, false)
                } else {
                    read_gpr(cpu, w, rm, false).wrapping_add(1)
                };
                write_gpr(cpu, w, rd, v, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Csinv { w, rd, rn, rm, cond } => {
                let v = if cpu.cond_holds(cond) {
                    read_gpr(cpu, w, rn, false)
                } else {
                    !read_gpr(cpu, w, rm, false)
                };
                write_gpr(cpu, w, rd, v, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Csneg { w, rd, rn, rm, cond } => {
                let v = if cpu.cond_holds(cond) {
                    read_gpr(cpu, w, rn, false)
                } else {
                    read_gpr(cpu, w, rm, false).wrapping_neg()
                };
                write_gpr(cpu, w, rd, v, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::CcmpReg {
                w,
                rn,
                rm,
                nzcv,
                cond,
            } => {
                if cpu.cond_holds(cond) {
                    let a = read_gpr(cpu, w, rn, false);
                    let b = read_gpr(cpu, w, rm, false);
                    let (_, n, z, c, v) = add_with_carry(a, !b, true, w);
                    cpu.set_nzcv(n, z, c, v);
                } else {
                    cpu.nzcv = (nzcv as u64) << 28;
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::CcmpImm {
                w,
                rn,
                imm5,
                nzcv,
                cond,
            } => {
                if cpu.cond_holds(cond) {
                    let a = read_gpr(cpu, w, rn, false);
                    let (_, n, z, c, v) = add_with_carry(a, !(imm5 as u64), true, w);
                    cpu.set_nzcv(n, z, c, v);
                } else {
                    cpu.nzcv = (nzcv as u64) << 28;
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::CcmnReg {
                w,
                rn,
                rm,
                nzcv,
                cond,
            } => {
                if cpu.cond_holds(cond) {
                    let a = read_gpr(cpu, w, rn, false);
                    let b = read_gpr(cpu, w, rm, false);
                    let (_, n, z, c, v) = add_with_carry(a, b, false, w);
                    cpu.set_nzcv(n, z, c, v);
                } else {
                    cpu.nzcv = (nzcv as u64) << 28;
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::CcmnImm {
                w,
                rn,
                imm5,
                nzcv,
                cond,
            } => {
                if cpu.cond_holds(cond) {
                    let a = read_gpr(cpu, w, rn, false);
                    let (_, n, z, c, v) = add_with_carry(a, imm5 as u64, false, w);
                    cpu.set_nzcv(n, z, c, v);
                } else {
                    cpu.nzcv = (nzcv as u64) << 28;
                }
                Ok(StepOutcome::Continue)
            }

            // --- Branches & PAC ---
            Instruction::B { offset } => Ok(StepOutcome::Branch((pc as i64).wrapping_add(offset) as u64)),
            Instruction::Bl { offset } => {
                cpu.write_x(Reg::LR, pc.wrapping_add(4));
                Ok(StepOutcome::Branch((pc as i64).wrapping_add(offset) as u64))
            }
            Instruction::BCond { cond, offset } => {
                if cpu.cond_holds(cond) {
                    Ok(StepOutcome::Branch((pc as i64).wrapping_add(offset) as u64))
                } else {
                    Ok(StepOutcome::Continue)
                }
            }
            Instruction::Cbz { w, rt, offset } => {
                if read_gpr(cpu, w, rt, false) == 0 {
                    Ok(StepOutcome::Branch((pc as i64).wrapping_add(offset) as u64))
                } else {
                    Ok(StepOutcome::Continue)
                }
            }
            Instruction::Cbnz { w, rt, offset } => {
                if read_gpr(cpu, w, rt, false) != 0 {
                    Ok(StepOutcome::Branch((pc as i64).wrapping_add(offset) as u64))
                } else {
                    Ok(StepOutcome::Continue)
                }
            }
            Instruction::Tbz { rt, bit, offset } => {
                if ((cpu.read_x(rt) >> bit) & 1) == 0 {
                    Ok(StepOutcome::Branch((pc as i64).wrapping_add(offset) as u64))
                } else {
                    Ok(StepOutcome::Continue)
                }
            }
            Instruction::Tbnz { rt, bit, offset } => {
                if ((cpu.read_x(rt) >> bit) & 1) != 0 {
                    Ok(StepOutcome::Branch((pc as i64).wrapping_add(offset) as u64))
                } else {
                    Ok(StepOutcome::Continue)
                }
            }
            Instruction::Br { rn } => Ok(StepOutcome::Branch(strip_pac(cpu.read_x(rn)))),
            Instruction::Blr { rn } => {
                let target = strip_pac(cpu.read_x(rn));
                cpu.write_x(Reg::LR, pc.wrapping_add(4));
                Ok(StepOutcome::Branch(target))
            }
            Instruction::Ret { rn } => Ok(StepOutcome::Branch(strip_pac(cpu.read_x(rn)))),
            Instruction::PacHint { op, rd, rn } => match op {
                PacOp::Retaa | PacOp::Retab => {
                    Ok(StepOutcome::Branch(strip_pac(cpu.read_x(Reg::LR))))
                }
                PacOp::Braa | PacOp::Brab => Ok(StepOutcome::Branch(strip_pac(cpu.read_x(rn)))),
                PacOp::Blraa | PacOp::Blrab => {
                    let target = strip_pac(cpu.read_x(rn));
                    cpu.write_x(Reg::LR, pc.wrapping_add(4));
                    Ok(StepOutcome::Branch(target))
                }
                PacOp::Autia | PacOp::Autib | PacOp::Autda | PacOp::Autdb | PacOp::Xpaci | PacOp::Xpacd => {
                    let v = strip_pac(cpu.read_x(rd));
                    cpu.write_x(rd, v);
                    Ok(StepOutcome::Continue)
                }
                PacOp::Pacia | PacOp::Pacib | PacOp::Pacda | PacOp::Pacdb | PacOp::Paciasp | PacOp::Autiasp => {
                    Ok(StepOutcome::Continue)
                }
            },
            Instruction::Pacga { rd, rn, rm } => {
                // Deterministic 32-bit PACGA tag in bits [63:32] (low 32 bits = 0)
                let a = cpu.read_x(rn) ^ cpu.pac_keys[6];
                let b = cpu.read_sp_or_x(rm) ^ cpu.pac_keys[7];
                let mut h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.rotate_left(17);
                h ^= h >> 30;
                h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                h ^= h >> 27;
                cpu.write_x(rd, h & 0xFFFF_FFFF_0000_0000);
                Ok(StepOutcome::Continue)
            }

            // --- Loads & Stores ---
            Instruction::Ldr { w, rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let val = match w {
                    Width::W32 => mem.read_u32(pa)? as u64,
                    Width::X64 => mem.read_u64(pa)?,
                };
                cpu.write_x(rt, val);
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Str { w, rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                match w {
                    Width::W32 => mem.write_u32(pa, cpu.read_w(rt))?,
                    Width::X64 => mem.write_u64(pa, cpu.read_x(rt))?,
                }
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldrb { rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                cpu.write_x(rt, mem.read_u8(pa)? as u64);
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Strb { rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                mem.write_u8(pa, cpu.read_x(rt) as u8)?;
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldrh { rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                cpu.write_x(rt, mem.read_u16(pa)? as u64);
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Strh { rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                mem.write_u16(pa, cpu.read_x(rt) as u16)?;
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldrsb { w, rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let b = mem.read_u8(pa)? as i8 as i64 as u64;
                cpu.write_x(rt, b & w.mask());
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldrsh { w, rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let h = mem.read_u16(pa)? as i16 as i64 as u64;
                cpu.write_x(rt, h & w.mask());
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldrsw { rt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let v = mem.read_u32(pa)? as i32 as i64 as u64;
                cpu.write_x(rt, v);
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::LdrLiteral {
                w,
                rt,
                offset,
                signed,
            } => {
                let ea = (pc as i64).wrapping_add(offset) as u64;
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let val = match (w, signed) {
                    (Width::W32, false) => mem.read_u32(pa)? as u64,
                    (Width::X64, true) => mem.read_u32(pa)? as i32 as i64 as u64,
                    (Width::X64, false) => mem.read_u64(pa)?,
                    _ => mem.read_u32(pa)? as u64,
                };
                cpu.write_x(rt, val);
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldp {
                w,
                rt,
                rt2,
                rn,
                mode,
            } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa1 = mmu.translate(cpu, mem, ea, false)?;
                let stride = match w {
                    Width::W32 => 4u64,
                    Width::X64 => 8u64,
                };
                let pa2 = mmu.translate(cpu, mem, ea + stride, false)?;
                let (v1, v2) = match w {
                    Width::W32 => (mem.read_u32(pa1)? as u64, mem.read_u32(pa2)? as u64),
                    Width::X64 => (mem.read_u64(pa1)?, mem.read_u64(pa2)?),
                };
                cpu.write_x(rt, v1);
                cpu.write_x(rt2, v2);
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Stp {
                w,
                rt,
                rt2,
                rn,
                mode,
            } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa1 = mmu.translate(cpu, mem, ea, true)?;
                let stride = match w {
                    Width::W32 => 4u64,
                    Width::X64 => 8u64,
                };
                let pa2 = mmu.translate(cpu, mem, ea + stride, true)?;
                match w {
                    Width::W32 => {
                        mem.write_u32(pa1, cpu.read_w(rt))?;
                        mem.write_u32(pa2, cpu.read_w(rt2))?;
                    }
                    Width::X64 => {
                        mem.write_u64(pa1, cpu.read_x(rt))?;
                        mem.write_u64(pa2, cpu.read_x(rt2))?;
                    }
                }
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldpsw { rt, rt2, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa1 = mmu.translate(cpu, mem, ea, false)?;
                let pa2 = mmu.translate(cpu, mem, ea + 4, false)?;
                cpu.write_x(rt, mem.read_u32(pa1)? as i32 as i64 as u64);
                cpu.write_x(rt2, mem.read_u32(pa2)? as i32 as i64 as u64);
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldxr { size, rt, rn, .. } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let val = read_sized(mem, pa, size)?;
                cpu.write_x(rt, val);
                cpu.exclusive_addr = ea;
                cpu.exclusive_valid = 1;
                Ok(StepOutcome::Continue)
            }
            Instruction::Stxr {
                size, rs, rt, rn, ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                if cpu.exclusive_valid != 0 && cpu.exclusive_addr == ea {
                    let pa = mmu.translate(cpu, mem, ea, true)?;
                    write_sized(mem, pa, size, cpu.read_x(rt))?;
                    cpu.exclusive_valid = 0;
                    cpu.write_w(rs, 0);
                } else {
                    cpu.exclusive_valid = 0;
                    cpu.write_w(rs, 1);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldxp {
                is_64, rt, rt2, rn, ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                let stride = if is_64 { 8 } else { 4 };
                let pa1 = mmu.translate(cpu, mem, ea, false)?;
                let pa2 = mmu.translate(cpu, mem, ea + stride, false)?;
                let (v1, v2) = if is_64 {
                    (mem.read_u64(pa1)?, mem.read_u64(pa2)?)
                } else {
                    (mem.read_u32(pa1)? as u64, mem.read_u32(pa2)? as u64)
                };
                cpu.write_x(rt, v1);
                cpu.write_x(rt2, v2);
                cpu.exclusive_addr = ea;
                cpu.exclusive_valid = 1;
                Ok(StepOutcome::Continue)
            }
            Instruction::Stxp {
                is_64,
                rs,
                rt,
                rt2,
                rn,
                ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                let stride = if is_64 { 8 } else { 4 };
                if cpu.exclusive_valid != 0 && cpu.exclusive_addr == ea {
                    let pa1 = mmu.translate(cpu, mem, ea, true)?;
                    let pa2 = mmu.translate(cpu, mem, ea + stride, true)?;
                    if is_64 {
                        mem.write_u64(pa1, cpu.read_x(rt))?;
                        mem.write_u64(pa2, cpu.read_x(rt2))?;
                    } else {
                        mem.write_u32(pa1, cpu.read_w(rt))?;
                        mem.write_u32(pa2, cpu.read_w(rt2))?;
                    }
                    cpu.exclusive_valid = 0;
                    cpu.write_w(rs, 0);
                } else {
                    cpu.exclusive_valid = 0;
                    cpu.write_w(rs, 1);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldar { size, rt, rn } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                cpu.write_x(rt, read_sized(mem, pa, size)?);
                Ok(StepOutcome::Continue)
            }
            Instruction::Casp {
                is_64,
                rs,
                rt,
                rn,
                ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                let rs2 = Reg((rs.0 + 1) & 31);
                let rt2 = Reg((rt.0 + 1) & 31);
                let stride = if is_64 { 8u64 } else { 4u64 };
                let pa1 = mmu.translate(cpu, mem, ea, true)?;
                let pa2 = mmu.translate(cpu, mem, ea + stride, true)?;
                if is_64 {
                    let old0 = mem.read_u64(pa1)?;
                    let old1 = mem.read_u64(pa2)?;
                    if old0 == cpu.read_x(rs) && old1 == cpu.read_x(rs2) {
                        mem.write_u64(pa1, cpu.read_x(rt))?;
                        mem.write_u64(pa2, cpu.read_x(rt2))?;
                    }
                    cpu.write_x(rs, old0);
                    cpu.write_x(rs2, old1);
                } else {
                    let old0 = mem.read_u32(pa1)?;
                    let old1 = mem.read_u32(pa2)?;
                    if old0 == cpu.read_w(rs) && old1 == cpu.read_w(rs2) {
                        mem.write_u32(pa1, cpu.read_w(rt))?;
                        mem.write_u32(pa2, cpu.read_w(rt2))?;
                    }
                    cpu.write_w(rs, old0);
                    cpu.write_w(rs2, old1);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Stlr { size, rt, rn } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                write_sized(mem, pa, size, cpu.read_x(rt))?;
                Ok(StepOutcome::Continue)
            }
            Instruction::Ldadd {
                size, rs, rt, rn, ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                let old = read_sized(mem, pa, size)?;
                let new = old.wrapping_add(cpu.read_x(rs));
                write_sized(mem, pa, size, new)?;
                cpu.write_x(rt, old);
                Ok(StepOutcome::Continue)
            }
            Instruction::LseAtomic {
                size,
                op,
                rs,
                rt,
                rn,
                ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                let old = read_sized(mem, pa, size)?;
                let mask = size_mask(size);
                let sval = cpu.read_x(rs) & mask;
                let oval = old & mask;
                let sign_ext = |v: u64| -> i64 {
                    match size {
                        0 => v as u8 as i8 as i64,
                        1 => v as u16 as i16 as i64,
                        2 => v as u32 as i32 as i64,
                        _ => v as i64,
                    }
                };
                let new_val = match op {
                    1 => oval & !sval,                                  // LDCLR
                    2 => oval ^ sval,                                   // LDEOR
                    3 => oval | sval,                                   // LDSET
                    4 => sign_ext(oval).max(sign_ext(sval)) as u64,     // LDSMAX
                    5 => sign_ext(oval).min(sign_ext(sval)) as u64,     // LDSMIN
                    6 => oval.max(sval),                                // LDUMAX
                    7 => oval.min(sval),                                // LDUMIN
                    _ => oval,
                };
                write_sized(mem, pa, size, new_val)?;
                cpu.write_x(rt, old);
                Ok(StepOutcome::Continue)
            }
            Instruction::Cas {
                size, rs, rt, rn, ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                let old = read_sized(mem, pa, size)?;
                let mask = size_mask(size);
                if (old & mask) == (cpu.read_x(rs) & mask) {
                    write_sized(mem, pa, size, cpu.read_x(rt))?;
                }
                cpu.write_x(rs, old);
                Ok(StepOutcome::Continue)
            }
            Instruction::Swp {
                size, rs, rt, rn, ..
            } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                let old = read_sized(mem, pa, size)?;
                write_sized(mem, pa, size, cpu.read_x(rs))?;
                cpu.write_x(rt, old);
                Ok(StepOutcome::Continue)
            }

            // --- System & Exceptions ---
            Instruction::Mrs { rt, sysreg } => {
                let val = read_cpu_sysreg(cpu, mem, sysreg);
                cpu.write_x(rt, val);
                Ok(StepOutcome::Continue)
            }
            Instruction::Msr { sysreg, rt } => {
                let val = cpu.read_x(rt);
                write_cpu_sysreg(cpu, mem, mmu, sysreg, val);
                Ok(StepOutcome::Continue)
            }
            Instruction::MsrImm { pstate_field, crm } => {
                match pstate_field {
                    0b000_100 => cpu.pan = crm & 1,
                    0b000_101 => cpu.set_sp_sel(crm & 1),
                    0b011_110 => cpu.daif |= crm & 0xF,
                    0b011_111 => cpu.daif &= !(crm & 0xF),
                    _ => {}
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Tlbi { .. } => {
                mmu.tlbi_all();
                Ok(StepOutcome::Continue)
            }
            Instruction::Dc { op1, crm, op2, rt } => {
                // DC ZVA (op1=3, crm=4, op2=1): zero 64-byte cache line
                if op1 == 3 && crm == 4 && op2 == 1 {
                    let va = cpu.read_sp_or_x(rt) & !63;
                    let pa = mmu.translate(cpu, mem, va, true)?;
                    for off in (0..64u64).step_by(8) {
                        mem.write_u64(pa + off, 0)?;
                    }
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::At { op1: 0, crm: 8, op2, rt } => {
                let va = cpu.read_x(rt);
                let is_write = (op2 & 1) != 0;
                match mmu.translate(cpu, mem, va, is_write) {
                    Ok(pa) => {
                        cpu.par_el1 = pa & 0x0000_FFFF_FFFF_F000;
                    }
                    Err(_) => {
                        cpu.par_el1 = 1; // F bit = 1 (translation failed)
                    }
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Ic { .. } | Instruction::At { .. } | Instruction::Sys { .. } => {
                Ok(StepOutcome::Continue)
            }
            Instruction::Svc { imm } => Ok(StepOutcome::Svc(imm)),
            Instruction::Hvc { imm } => Ok(StepOutcome::Hvc(imm)),
            Instruction::Smc { imm } => Ok(StepOutcome::Smc(imm)),
            Instruction::Brk { imm } => Ok(StepOutcome::Brk(imm)),
            Instruction::Hlt { imm } => Ok(StepOutcome::Hlt(imm)),
            Instruction::Eret => Ok(StepOutcome::Eret),

            // --- Scalar FP & SIMD essentials ---
            Instruction::FmovGprToFp { is_64, rd, rn } => {
                let v = cpu.read_x(rn);
                cpu.v[rd as usize][0] = if is_64 { v } else { v & 0xFFFF_FFFF };
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::FmovFpToGpr { is_64, rd, rn } => {
                let v = cpu.v[rn as usize][0];
                cpu.write_x(rd, if is_64 { v } else { v & 0xFFFF_FFFF });
                Ok(StepOutcome::Continue)
            }
            Instruction::FmovImm { is_64, rd, imm8 } => {
                cpu.v[rd as usize][0] = decode_fp_imm8(imm8, is_64);
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::FmovReg { is_64, rd, rn } => {
                let v = cpu.v[rn as usize][0];
                cpu.v[rd as usize][0] = if is_64 { v } else { v & 0xFFFF_FFFF };
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::Fadd { is_64, rd, rn, rm } => {
                fp_binop(cpu, is_64, rd, rn, rm, |a, b| a + b, |a, b| a + b);
                Ok(StepOutcome::Continue)
            }
            Instruction::Fsub { is_64, rd, rn, rm } => {
                fp_binop(cpu, is_64, rd, rn, rm, |a, b| a - b, |a, b| a - b);
                Ok(StepOutcome::Continue)
            }
            Instruction::Fmul { is_64, rd, rn, rm } => {
                fp_binop(cpu, is_64, rd, rn, rm, |a, b| a * b, |a, b| a * b);
                Ok(StepOutcome::Continue)
            }
            Instruction::Fdiv { is_64, rd, rn, rm } => {
                fp_binop(cpu, is_64, rd, rn, rm, |a, b| a / b, |a, b| a / b);
                Ok(StepOutcome::Continue)
            }
            Instruction::Fneg { is_64, rd, rn } => {
                if is_64 {
                    cpu.v[rd as usize][0] = cpu.v[rn as usize][0] ^ (1u64 << 63);
                } else {
                    cpu.v[rd as usize][0] = (cpu.v[rn as usize][0] ^ (1u64 << 31)) & 0xFFFF_FFFF;
                }
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::Fabs { is_64, rd, rn } => {
                if is_64 {
                    cpu.v[rd as usize][0] = cpu.v[rn as usize][0] & !(1u64 << 63);
                } else {
                    cpu.v[rd as usize][0] = cpu.v[rn as usize][0] & 0x7FFF_FFFF;
                }
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::Fsqrt { is_64, rd, rn } => {
                if is_64 {
                    let f = f64::from_bits(cpu.v[rn as usize][0]);
                    cpu.v[rd as usize][0] = f.sqrt().to_bits();
                } else {
                    let f = f32::from_bits(cpu.v[rn as usize][0] as u32);
                    cpu.v[rd as usize][0] = f.sqrt().to_bits() as u64;
                }
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::Fcmp {
                is_64,
                rn,
                rm,
                with_zero,
            } => {
                if is_64 {
                    let a = f64::from_bits(cpu.v[rn as usize][0]);
                    let b = if with_zero {
                        0.0
                    } else {
                        f64::from_bits(cpu.v[rm as usize][0])
                    };
                    set_fp_nzcv(cpu, a.partial_cmp(&b));
                } else {
                    let a = f32::from_bits(cpu.v[rn as usize][0] as u32);
                    let b = if with_zero {
                        0.0f32
                    } else {
                        f32::from_bits(cpu.v[rm as usize][0] as u32)
                    };
                    set_fp_nzcv(cpu, a.partial_cmp(&b));
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::Fcvt {
                dst_64,
                src_64,
                rd,
                rn,
            } => {
                let val = match (src_64, dst_64) {
                    (false, true) => (f32::from_bits(cpu.v[rn as usize][0] as u32) as f64).to_bits(),
                    (true, false) => (f64::from_bits(cpu.v[rn as usize][0]) as f32).to_bits() as u64,
                    _ => cpu.v[rn as usize][0],
                };
                cpu.v[rd as usize] = [val, 0];
                Ok(StepOutcome::Continue)
            }
            Instruction::Scvtf { is_64, w, rd, rn } => {
                let s = match w {
                    Width::W32 => (cpu.read_w(rn) as i32) as i64,
                    Width::X64 => cpu.read_x(rn) as i64,
                };
                cpu.v[rd as usize][0] = if is_64 {
                    (s as f64).to_bits()
                } else {
                    (s as f32).to_bits() as u64
                };
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::Ucvtf { is_64, w, rd, rn } => {
                let u = read_gpr(cpu, w, rn, false);
                cpu.v[rd as usize][0] = if is_64 {
                    (u as f64).to_bits()
                } else {
                    (u as f32).to_bits() as u64
                };
                cpu.v[rd as usize][1] = 0;
                Ok(StepOutcome::Continue)
            }
            Instruction::Fcvtzs { is_64, w, rd, rn } => {
                let out: i64 = if is_64 {
                    f64::from_bits(cpu.v[rn as usize][0]) as i64
                } else {
                    f32::from_bits(cpu.v[rn as usize][0] as u32) as i64
                };
                write_gpr(cpu, w, rd, out as u64, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::Fcvtzu { is_64, w, rd, rn } => {
                let out: u64 = if is_64 {
                    f64::from_bits(cpu.v[rn as usize][0]).max(0.0) as u64
                } else {
                    f32::from_bits(cpu.v[rn as usize][0] as u32).max(0.0) as u64
                };
                write_gpr(cpu, w, rd, out, false);
                Ok(StepOutcome::Continue)
            }
            Instruction::LdrFpLiteral { size, vt, offset } => {
                let ea = (pc as i64).wrapping_add(offset) as u64;
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let mut v = [0u64; 2];
                match size {
                    4 => v[0] = mem.read_u32(pa)? as u64,
                    8 => v[0] = mem.read_u64(pa)?,
                    16 => {
                        let pa2 = mmu.translate(cpu, mem, ea + 8, false)?;
                        v[0] = mem.read_u64(pa)?;
                        v[1] = mem.read_u64(pa2)?;
                    }
                    _ => {}
                }
                cpu.v[vt as usize] = v;
                Ok(StepOutcome::Continue)
            }
            Instruction::LdrFp { size, vt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                let mut v = [0u64; 2];
                match size {
                    1 => v[0] = mem.read_u8(pa)? as u64,
                    2 => v[0] = mem.read_u16(pa)? as u64,
                    4 => v[0] = mem.read_u32(pa)? as u64,
                    8 => v[0] = mem.read_u64(pa)?,
                    16 => {
                        let pa2 = mmu.translate(cpu, mem, ea + 8, false)?;
                        v[0] = mem.read_u64(pa)?;
                        v[1] = mem.read_u64(pa2)?;
                    }
                    _ => {}
                }
                cpu.v[vt as usize] = v;
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::StrFp { size, vt, rn, mode } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let pa = mmu.translate(cpu, mem, ea, true)?;
                let v = cpu.v[vt as usize];
                match size {
                    1 => mem.write_u8(pa, v[0] as u8)?,
                    2 => mem.write_u16(pa, v[0] as u16)?,
                    4 => mem.write_u32(pa, v[0] as u32)?,
                    8 => mem.write_u64(pa, v[0])?,
                    16 => {
                        let pa2 = mmu.translate(cpu, mem, ea + 8, true)?;
                        mem.write_u64(pa, v[0])?;
                        mem.write_u64(pa2, v[1])?;
                    }
                    _ => {}
                }
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::LdpFp {
                size,
                vt,
                vt2,
                rn,
                mode,
            } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let stride = size as u64;
                for (idx, reg) in [(0u64, vt), (1u64, vt2)] {
                    let a = ea + idx * stride;
                    let pa = mmu.translate(cpu, mem, a, false)?;
                    let mut val = [0u64; 2];
                    match size {
                        4 => val[0] = mem.read_u32(pa)? as u64,
                        8 => val[0] = mem.read_u64(pa)?,
                        16 => {
                            let pa2 = mmu.translate(cpu, mem, a + 8, false)?;
                            val[0] = mem.read_u64(pa)?;
                            val[1] = mem.read_u64(pa2)?;
                        }
                        _ => {}
                    }
                    cpu.v[reg as usize] = val;
                }
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::StpFp {
                size,
                vt,
                vt2,
                rn,
                mode,
            } => {
                let (ea, wb) = compute_addr(cpu, rn, mode);
                let stride = size as u64;
                for (idx, reg) in [(0u64, vt), (1u64, vt2)] {
                    let a = ea + idx * stride;
                    let pa = mmu.translate(cpu, mem, a, true)?;
                    let val = cpu.v[reg as usize];
                    match size {
                        4 => mem.write_u32(pa, val[0] as u32)?,
                        8 => mem.write_u64(pa, val[0])?,
                        16 => {
                            let pa2 = mmu.translate(cpu, mem, a + 8, true)?;
                            mem.write_u64(pa, val[0])?;
                            mem.write_u64(pa2, val[1])?;
                        }
                        _ => {}
                    }
                }
                if let Some(new_rn) = wb {
                    cpu.write_sp_or_x(rn, new_rn);
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdLdStSingle {
                q,
                is_load,
                replicate,
                post_index,
                eb,
                selem,
                index,
                rm,
                rn,
                vt,
            } => {
                let base = cpu.read_sp_or_x(rn);
                let mut addr = base;
                let eb_us = eb as usize;
                let bytes = if q { 16usize } else { 8usize };
                for s in 0..selem {
                    let reg = ((vt + s) & 31) as usize;
                    if is_load {
                        let mut elem = [0u8; 8];
                        for b in 0..eb_us {
                            let pa = mmu.translate(cpu, mem, addr.wrapping_add(b as u64), false)?;
                            elem[b] = mem.read_u8(pa)?;
                        }
                        if replicate {
                            let mut out = [0u8; 16];
                            for off in (0..bytes).step_by(eb_us) {
                                out[off..off + eb_us].copy_from_slice(&elem[..eb_us]);
                            }
                            cpu.v[reg] = bytes_to_vec(out);
                        } else {
                            let mut out = vec_to_bytes(cpu.v[reg]);
                            let dst = ((index as usize) * eb_us).min(16 - eb_us);
                            out[dst..dst + eb_us].copy_from_slice(&elem[..eb_us]);
                            cpu.v[reg] = bytes_to_vec(out);
                        }
                    } else {
                        let src_bytes = vec_to_bytes(cpu.v[reg]);
                        let src = ((index as usize) * eb_us).min(16 - eb_us);
                        for b in 0..eb_us {
                            let pa = mmu.translate(cpu, mem, addr.wrapping_add(b as u64), true)?;
                            mem.write_u8(pa, src_bytes[src + b])?;
                        }
                    }
                    addr = addr.wrapping_add(eb as u64);
                }
                if post_index {
                    let inc = if rm == Reg::XZR {
                        (selem as u64) * (eb as u64)
                    } else {
                        cpu.read_x(rm)
                    };
                    cpu.write_sp_or_x(rn, base.wrapping_add(inc));
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdLdStMulti {
                q,
                is_load,
                post_index,
                eb,
                rpt,
                selem,
                rm,
                rn,
                vt,
            } => {
                let base = cpu.read_sp_or_x(rn);
                let mut addr = base;
                let eb_us = eb as usize;
                let datasize = if q { 16usize } else { 8usize };
                let elems = datasize / eb_us;
                let mut tt = vt;
                for _ in 0..rpt {
                    let mut regs = [[0u8; 16]; 4];
                    if !is_load {
                        for s in 0..(selem as usize) {
                            regs[s] = vec_to_bytes(cpu.v[((tt + s as u8) & 31) as usize]);
                        }
                    }
                    for e in 0..elems {
                        let off = e * eb_us;
                        for s in 0..(selem as usize) {
                            if is_load {
                                for b in 0..eb_us {
                                    let pa = mmu.translate(cpu, mem, addr.wrapping_add(b as u64), false)?;
                                    regs[s][off + b] = mem.read_u8(pa)?;
                                }
                            } else {
                                for b in 0..eb_us {
                                    let pa = mmu.translate(cpu, mem, addr.wrapping_add(b as u64), true)?;
                                    mem.write_u8(pa, regs[s][off + b])?;
                                }
                            }
                            addr = addr.wrapping_add(eb as u64);
                        }
                    }
                    if is_load {
                        for s in 0..(selem as usize) {
                            cpu.v[((tt + s as u8) & 31) as usize] = bytes_to_vec(regs[s]);
                        }
                    }
                    tt = (tt + 1) & 31;
                }
                if post_index {
                    let inc = if rm == Reg::XZR {
                        (rpt as u64) * (selem as u64) * (datasize as u64)
                    } else {
                        cpu.read_x(rm)
                    };
                    cpu.write_sp_or_x(rn, base.wrapping_add(inc));
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdMov { q, vd, vn } => {
                cpu.v[vd as usize][0] = cpu.v[vn as usize][0];
                cpu.v[vd as usize][1] = if q { cpu.v[vn as usize][1] } else { 0 };
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdMovi {
                q,
                op,
                cmode,
                imm8,
                vd,
            } => {
                let imm64 = adv_simd_expand_imm(op, cmode, imm8);
                let is_orr_bic = cmode < 12 && (cmode & 1) == 1;
                let is_mvni = !is_orr_bic && op == 1 && cmode < 14;
                let lanes = if q { 2 } else { 1 };
                for lane in 0..lanes {
                    cpu.v[vd as usize][lane] = if is_orr_bic {
                        if op == 0 {
                            cpu.v[vd as usize][lane] | imm64
                        } else {
                            cpu.v[vd as usize][lane] & !imm64
                        }
                    } else if is_mvni {
                        !imm64
                    } else {
                        imm64
                    };
                }
                if !q {
                    cpu.v[vd as usize][1] = 0;
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdBitwise { q, op, vd, vn, vm } => {
                let a = cpu.v[vn as usize];
                let b = cpu.v[vm as usize];
                let d = cpu.v[vd as usize];
                for lane in 0..if q { 2 } else { 1 } {
                    cpu.v[vd as usize][lane] = match op {
                        0 => a[lane] & b[lane],
                        1 => a[lane] & !b[lane],
                        2 => a[lane] | b[lane],
                        3 => a[lane] | !b[lane],
                        4 => a[lane] ^ b[lane],
                        5 => (a[lane] & d[lane]) | (b[lane] & !d[lane]), // BSL
                        6 => (a[lane] & b[lane]) | (d[lane] & !b[lane]), // BIT
                        7 => (a[lane] & !b[lane]) | (d[lane] & b[lane]), // BIF
                        _ => a[lane],
                    };
                }
                if !q {
                    cpu.v[vd as usize][1] = 0;
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdAddSub {
                q,
                esize,
                is_sub,
                vd,
                vn,
                vm,
            } => {
                let a = cpu.v[vn as usize];
                let b = cpu.v[vm as usize];
                for lane in 0..if q { 2 } else { 1 } {
                    cpu.v[vd as usize][lane] = match esize {
                        64 => {
                            if is_sub {
                                a[lane].wrapping_sub(b[lane])
                            } else {
                                a[lane].wrapping_add(b[lane])
                            }
                        }
                        32 => {
                            let lo = if is_sub {
                                (a[lane] as u32).wrapping_sub(b[lane] as u32)
                            } else {
                                (a[lane] as u32).wrapping_add(b[lane] as u32)
                            };
                            let hi = if is_sub {
                                ((a[lane] >> 32) as u32).wrapping_sub((b[lane] >> 32) as u32)
                            } else {
                                ((a[lane] >> 32) as u32).wrapping_add((b[lane] >> 32) as u32)
                            };
                            (lo as u64) | ((hi as u64) << 32)
                        }
                        _ => a[lane],
                    };
                }
                if !q {
                    cpu.v[vd as usize][1] = 0;
                }
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdDup { q, esize, vd, rn } => {
                let raw = cpu.read_x(rn);
                let lane64 = match esize {
                    8 => u64::from_le_bytes([raw as u8; 8]),
                    16 => {
                        let v = raw as u64 & 0xFFFF;
                        v | (v << 16) | (v << 32) | (v << 48)
                    }
                    32 => {
                        let v = raw as u64 & 0xFFFF_FFFF;
                        v | (v << 32)
                    }
                    _ => raw,
                };
                cpu.v[vd as usize][0] = lane64;
                cpu.v[vd as usize][1] = if q { lane64 } else { 0 };
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdThreeSame {
                q,
                u,
                size,
                opcode,
                vd,
                vn,
                vm,
            } => {
                let bytes = if q { 16usize } else { 8usize };
                let eb = 1usize << size;
                let ebits = (eb * 8) as u32;
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let b = vec_to_bytes(cpu.v[vm as usize]);
                let d = vec_to_bytes(cpu.v[vd as usize]);
                let mut out = [0u8; 16];
                if opcode == 0b10111 && !u {
                    // ADDP (vector pairwise)
                    let count = bytes / eb;
                    let half = count / 2;
                    for i in 0..half {
                        let s0 = read_elem_u64(&a[i * 2 * eb..(i * 2 + 1) * eb]);
                        let s1 = read_elem_u64(&a[(i * 2 + 1) * eb..(i * 2 + 2) * eb]);
                        write_elem_u64(&mut out[i * eb..(i + 1) * eb], s0.wrapping_add(s1));
                        let b0 = read_elem_u64(&b[i * 2 * eb..(i * 2 + 1) * eb]);
                        let b1 = read_elem_u64(&b[(i * 2 + 1) * eb..(i * 2 + 2) * eb]);
                        write_elem_u64(
                            &mut out[(half + i) * eb..(half + i + 1) * eb],
                            b0.wrapping_add(b1),
                        );
                    }
                    cpu.v[vd as usize] = bytes_to_vec(out);
                    return Ok(StepOutcome::Continue);
                }
                for off in (0..bytes).step_by(eb) {
                    let av = read_elem_u64(&a[off..off + eb]);
                    let bv = read_elem_u64(&b[off..off + eb]);
                    let dv = read_elem_u64(&d[off..off + eb]);
                    let as_ = read_elem_i64(&a[off..off + eb]);
                    let bs_ = read_elem_i64(&b[off..off + eb]);
                    let res: u64 = match (opcode, u) {
                        (0b10001, true) => if av == bv { u64::MAX } else { 0 }, // CMEQ
                        (0b10001, false) => if (av & bv) != 0 { u64::MAX } else { 0 }, // CMTST
                        (0b00110, false) => if as_ > bs_ { u64::MAX } else { 0 }, // CMGT
                        (0b00110, true) => if av > bv { u64::MAX } else { 0 }, // CMHI
                        (0b00111, false) => if as_ >= bs_ { u64::MAX } else { 0 }, // CMGE
                        (0b00111, true) => if av >= bv { u64::MAX } else { 0 }, // CMHS
                        (0b01000, is_u) => {
                            // SSHL (u=0) / USHL (u=1)
                            let shift = (bv as u8 as i8) as i32;
                            if shift >= 0 {
                                let s = shift as u32;
                                if s >= ebits { 0 } else { av << s }
                            } else {
                                let r = (-shift) as u32;
                                if is_u {
                                    if r >= ebits { 0 } else { av >> r }
                                } else if r >= ebits {
                                    if as_ < 0 { u64::MAX } else { 0 }
                                } else {
                                    (as_ >> r) as u64
                                }
                            }
                        }
                        (0b01100, false) => as_.max(bs_) as u64, // SMAX
                        (0b01100, true) => av.max(bv), // UMAX
                        (0b01101, false) => as_.min(bs_) as u64, // SMIN
                        (0b01101, true) => av.min(bv), // UMIN
                        (0b10011, false) => av.wrapping_mul(bv), // MUL
                        (0b10010, false) => dv.wrapping_add(av.wrapping_mul(bv)), // MLA
                        (0b10010, true) => dv.wrapping_sub(av.wrapping_mul(bv)),  // MLS
                        _ => av,
                    };
                    write_elem_u64(&mut out[off..off + eb], res);
                }
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdShiftImm {
                q,
                u,
                immh,
                immb,
                opcode,
                vd,
                vn,
            } => {
                let (eb, highest_bit) = if (immh & 8) != 0 {
                    (8usize, 64u32)
                } else if (immh & 4) != 0 {
                    (4usize, 32u32)
                } else if (immh & 2) != 0 {
                    (2usize, 16u32)
                } else {
                    (1usize, 8u32)
                };
                let imm7 = ((immh as u32) << 3) | (immb as u32);
                let shift_left = imm7.saturating_sub(highest_bit);
                let shift_right = (highest_bit * 2).saturating_sub(imm7);
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let d = vec_to_bytes(cpu.v[vd as usize]);

                if opcode == 0b10100 {
                    // SSHLL/USHLL/SSHLL2/USHLL2 (widen eb -> 2*eb and shift left)
                    let dst_eb = (eb * 2).min(8);
                    let src_start = if q { 8usize } else { 0usize };
                    let count = 8 / eb;
                    let mut out = [0u8; 16];
                    for i in 0..count {
                        let s_off = src_start + i * eb;
                        let val = if u {
                            read_elem_u64(&a[s_off..s_off + eb])
                        } else {
                            read_elem_i64(&a[s_off..s_off + eb]) as u64
                        };
                        let shifted = if shift_left >= 64 { 0 } else { val << shift_left };
                        write_elem_u64(&mut out[i * dst_eb..(i + 1) * dst_eb], shifted);
                    }
                    cpu.v[vd as usize] = bytes_to_vec(out);
                    return Ok(StepOutcome::Continue);
                }
                if opcode == 0b10000 && !u {
                    // SHRN / SHRN2 (narrow 2*eb -> eb after right shift)
                    let src_eb = (eb * 2).min(8);
                    let dst_start = if q { 8usize } else { 0usize };
                    let mut out = if q { d } else { [0u8; 16] };
                    let count = 8 / eb;
                    for i in 0..count {
                        let val = read_elem_u64(&a[i * src_eb..(i + 1) * src_eb]);
                        let shifted = if shift_right >= 64 { 0 } else { val >> shift_right };
                        write_elem_u64(&mut out[dst_start + i * eb..dst_start + (i + 1) * eb], shifted);
                    }
                    cpu.v[vd as usize] = bytes_to_vec(out);
                    return Ok(StepOutcome::Continue);
                }

                let bytes = if q { 16usize } else { 8usize };
                let mut out = [0u8; 16];
                for off in (0..bytes).step_by(eb) {
                    let av = read_elem_u64(&a[off..off + eb]);
                    let as_ = read_elem_i64(&a[off..off + eb]);
                    let dv = read_elem_u64(&d[off..off + eb]);
                    let res = match (opcode, u) {
                        (0b01010, false) => if shift_left >= highest_bit { 0 } else { av << shift_left }, // SHL
                        (0b01010, true) => {
                            // SLI
                            let mask = if shift_left >= highest_bit { 0 } else { !0u64 << shift_left };
                            (dv & !mask) | (if shift_left >= highest_bit { 0 } else { av << shift_left })
                        }
                        (0b00000, false) => {
                            // SSHR
                            if shift_right >= highest_bit {
                                if as_ < 0 { u64::MAX } else { 0 }
                            } else {
                                (as_ >> shift_right) as u64
                            }
                        }
                        (0b00000, true) => {
                            // USHR
                            if shift_right >= highest_bit { 0 } else { av >> shift_right }
                        }
                        (0b00010, false) => {
                            // SSRA
                            let s = if shift_right >= highest_bit {
                                if as_ < 0 { u64::MAX } else { 0 }
                            } else {
                                (as_ >> shift_right) as u64
                            };
                            dv.wrapping_add(s)
                        }
                        (0b00010, true) => {
                            // USRA
                            let s = if shift_right >= highest_bit { 0 } else { av >> shift_right };
                            dv.wrapping_add(s)
                        }
                        (0b00100, false) => {
                            // SRSHR
                            let round = if shift_right > 0 && shift_right < 64 {
                                1i128 << (shift_right - 1)
                            } else {
                                0
                            };
                            (((as_ as i128) + round) >> shift_right.min(63)) as u64
                        }
                        (0b00100, true) => {
                            // URSHR
                            let round = if shift_right > 0 && shift_right < 64 {
                                1u128 << (shift_right - 1)
                            } else {
                                0
                            };
                            (((av as u128) + round) >> shift_right.min(63)) as u64
                        }
                        (0b01000, true) => {
                            // SRI
                            let shifted = if shift_right >= highest_bit { 0 } else { av >> shift_right };
                            let keep_bits = highest_bit.saturating_sub(shift_right);
                            let keep_mask = if keep_bits == 64 { u64::MAX } else { (1u64 << keep_bits) - 1 };
                            (dv & !keep_mask) | (shifted & keep_mask)
                        }
                        _ => av,
                    };
                    write_elem_u64(&mut out[off..off + eb], res);
                }
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdAcross {
                q,
                u,
                size,
                opcode,
                vd,
                vn,
            } => {
                let bytes = if q { 16usize } else { 8usize };
                let eb = 1usize << size;
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let mut res: u64 = match (opcode, u) {
                    (0x03, _) | (0x1B, _) => 0, // SADDLV / UADDLV / ADDV
                    (0x0A, false) => i64::MIN as u64, // SMAXV
                    (0x0A, true) => 0,                // UMAXV
                    (0x1A, false) => i64::MAX as u64, // SMINV
                    (0x1A, true) => u64::MAX,         // UMINV
                    _ => 0,
                };
                for off in (0..bytes).step_by(eb) {
                    let av = read_elem_u64(&a[off..off + eb]);
                    let as_ = read_elem_i64(&a[off..off + eb]);
                    match (opcode, u) {
                        (0x03, false) => res = (res as i64).wrapping_add(as_) as u64,
                        (0x03, true) | (0x1B, _) => res = res.wrapping_add(av),
                        (0x0A, false) => res = (res as i64).max(as_) as u64,
                        (0x0A, true) => res = res.max(av),
                        (0x1A, false) => res = (res as i64).min(as_) as u64,
                        (0x1A, true) => res = res.min(av),
                        _ => {}
                    }
                }
                let mask = if opcode == 0x03 {
                    let out_bits = (eb * 16).min(64);
                    if out_bits == 64 { u64::MAX } else { (1u64 << out_bits) - 1 }
                } else {
                    let out_bits = eb * 8;
                    if out_bits == 64 { u64::MAX } else { (1u64 << out_bits) - 1 }
                };
                cpu.v[vd as usize] = [res & mask, 0];
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdTwoRegMisc {
                q,
                u,
                size,
                opcode,
                vd,
                vn,
            } => {
                let bytes = if q { 16usize } else { 8usize };
                let eb = 1usize << size;
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let d = vec_to_bytes(cpu.v[vd as usize]);
                let mut out = [0u8; 16];
                match (opcode, u, size) {
                    (0x02 | 0x06, _, 0..=2) => {
                        // SADDLP / UADDLP (0x02) and SADALP / UADALP (0x06)
                        let dst_eb = eb * 2;
                        let accumulate = opcode == 0x06;
                        for off in (0..bytes).step_by(dst_eb) {
                            let half_off = off + eb;
                            let pair_sum: u64 = if u {
                                read_elem_u64(&a[off..half_off])
                                    .wrapping_add(read_elem_u64(&a[half_off..off + dst_eb]))
                            } else {
                                read_elem_i64(&a[off..half_off])
                                    .wrapping_add(read_elem_i64(&a[half_off..off + dst_eb]))
                                    as u64
                            };
                            let val = if accumulate {
                                if u {
                                    read_elem_u64(&d[off..off + dst_eb]).wrapping_add(pair_sum)
                                } else {
                                    read_elem_i64(&d[off..off + dst_eb])
                                        .wrapping_add(pair_sum as i64)
                                        as u64
                                }
                            } else {
                                pair_sum
                            };
                            write_elem_u64(&mut out[off..off + dst_eb], val);
                        }
                    }
                    (0x05, true, 0) => {
                        // NOT (MVN)
                        for i in 0..bytes {
                            out[i] = !a[i];
                        }
                    }
                    (0x05, false, 0) => {
                        // CNT
                        for i in 0..bytes {
                            out[i] = a[i].count_ones() as u8;
                        }
                    }
                    (0x05, true, 1) => {
                        // RBIT
                        for i in 0..bytes {
                            out[i] = a[i].reverse_bits();
                        }
                    }
                    (0x00, false, _) => {
                        // REV64
                        for chunk in (0..bytes).step_by(8) {
                            let n = 8 / eb;
                            for i in 0..n {
                                let src = chunk + i * eb;
                                let dst = chunk + (n - 1 - i) * eb;
                                out[dst..dst + eb].copy_from_slice(&a[src..src + eb]);
                            }
                        }
                    }
                    (0x00, true, _) => {
                        // REV32
                        for chunk in (0..bytes).step_by(4) {
                            let n = 4 / eb;
                            for i in 0..n {
                                let src = chunk + i * eb;
                                let dst = chunk + (n - 1 - i) * eb;
                                out[dst..dst + eb].copy_from_slice(&a[src..src + eb]);
                            }
                        }
                    }
                    (0x01, false, _) => {
                        // REV16
                        for chunk in (0..bytes).step_by(2) {
                            out[chunk] = a[chunk + 1];
                            out[chunk + 1] = a[chunk];
                        }
                    }
                    _ => {
                        for off in (0..bytes).step_by(eb) {
                            let av = read_elem_u64(&a[off..off + eb]);
                            let as_ = read_elem_i64(&a[off..off + eb]);
                            let r: u64 = match (opcode, u) {
                                (0x08, false) => if as_ > 0 { u64::MAX } else { 0 },  // CMGT #0
                                (0x08, true) => if as_ >= 0 { u64::MAX } else { 0 },  // CMGE #0
                                (0x09, false) => if as_ == 0 { u64::MAX } else { 0 }, // CMEQ #0
                                (0x09, true) => if as_ <= 0 { u64::MAX } else { 0 },  // CMLE #0
                                (0x0A, false) => if as_ < 0 { u64::MAX } else { 0 },  // CMLT #0
                                (0x0B, false) => as_.wrapping_abs() as u64,           // ABS
                                (0x0B, true) => as_.wrapping_neg() as u64,            // NEG
                                (0x04, true) => (av.leading_zeros() - (64 - (eb * 8) as u32)) as u64, // CLZ
                                _ => av,
                            };
                            write_elem_u64(&mut out[off..off + eb], r);
                        }
                    }
                }
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdDupElem { q, imm5, vd, vn } => {
                let tz = imm5.trailing_zeros().min(3) as usize;
                let eb = 1usize << tz;
                let idx = (imm5 as usize) >> (tz + 1);
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let src = (idx * eb).min(16 - eb);
                let bytes = if q { 16usize } else { 8usize };
                let mut out = [0u8; 16];
                for off in (0..bytes).step_by(eb) {
                    out[off..off + eb].copy_from_slice(&a[src..src + eb]);
                }
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdScalarDupElem { imm5, vd, vn } => {
                let tz = imm5.trailing_zeros().min(3) as usize;
                let eb = 1usize << tz;
                let idx = (imm5 as usize) >> (tz + 1);
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let src = (idx * eb).min(16 - eb);
                let mut out = [0u8; 16];
                out[..eb].copy_from_slice(&a[src..src + eb]);
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdUmov {
                q,
                imm5,
                signed,
                rd,
                vn,
            } => {
                let tz = imm5.trailing_zeros().min(3) as usize;
                let eb = 1usize << tz;
                let idx = (imm5 as usize) >> (tz + 1);
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let src = (idx * eb).min(16 - eb);
                let val = if signed {
                    let s = read_elem_i64(&a[src..src + eb]) as u64;
                    if q { s } else { s & 0xFFFF_FFFF }
                } else {
                    read_elem_u64(&a[src..src + eb])
                };
                cpu.write_x(rd, val);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdInsGpr { imm5, vd, rn } => {
                let tz = imm5.trailing_zeros().min(3) as usize;
                let eb = 1usize << tz;
                let idx = (imm5 as usize) >> (tz + 1);
                let mut out = vec_to_bytes(cpu.v[vd as usize]);
                let dst = (idx * eb).min(16 - eb);
                write_elem_u64(&mut out[dst..dst + eb], cpu.read_x(rn));
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdInsElem {
                imm5,
                imm4,
                vd,
                vn,
            } => {
                let tz = imm5.trailing_zeros().min(3) as usize;
                let eb = 1usize << tz;
                let dst_idx = (imm5 as usize) >> (tz + 1);
                let src_idx = (imm4 as usize) >> tz;
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let mut out = vec_to_bytes(cpu.v[vd as usize]);
                let src = (src_idx * eb).min(16 - eb);
                let dst = (dst_idx * eb).min(16 - eb);
                out[dst..dst + eb].copy_from_slice(&a[src..src + eb]);
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdExt {
                q,
                imm4,
                vd,
                vn,
                vm,
            } => {
                let bytes = if q { 16usize } else { 8usize };
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let b = vec_to_bytes(cpu.v[vm as usize]);
                let mut cat = [0u8; 32];
                cat[..bytes].copy_from_slice(&a[..bytes]);
                cat[bytes..bytes * 2].copy_from_slice(&b[..bytes]);
                let pos = (imm4 as usize) & (bytes - 1);
                let mut out = [0u8; 16];
                out[..bytes].copy_from_slice(&cat[pos..pos + bytes]);
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdTbl {
                q,
                is_tbx,
                len,
                vd,
                vn,
                vm,
            } => {
                let bytes = if q { 16usize } else { 8usize };
                let table_len = (len as usize) * 16;
                let mut table = [0u8; 64];
                for t in 0..(len as usize) {
                    let b = vec_to_bytes(cpu.v[((vn as usize) + t) & 31]);
                    table[t * 16..(t + 1) * 16].copy_from_slice(&b);
                }
                let idx = vec_to_bytes(cpu.v[vm as usize]);
                let mut out = if is_tbx {
                    vec_to_bytes(cpu.v[vd as usize])
                } else {
                    [0u8; 16]
                };
                if is_tbx && !q {
                    out[8..16].fill(0);
                }
                for i in 0..bytes {
                    let ix = idx[i] as usize;
                    if ix < table_len {
                        out[i] = table[ix];
                    } else if !is_tbx {
                        out[i] = 0;
                    }
                }
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdPermute {
                q,
                size,
                op,
                vd,
                vn,
                vm,
            } => {
                let bytes = if q { 16usize } else { 8usize };
                let eb = 1usize << size;
                let count = bytes / eb;
                let a = vec_to_bytes(cpu.v[vn as usize]);
                let b = vec_to_bytes(cpu.v[vm as usize]);
                let mut out = [0u8; 16];
                match op {
                    1 | 5 => {
                        // UZP1 (part 0) / UZP2 (part 1)
                        let part = if op == 5 { 1usize } else { 0usize };
                        for i in 0..count {
                            let idx = 2 * i + part;
                            let src = if idx < count {
                                &a[idx * eb..(idx + 1) * eb]
                            } else {
                                let bi = idx - count;
                                &b[bi * eb..(bi + 1) * eb]
                            };
                            out[i * eb..(i + 1) * eb].copy_from_slice(src);
                        }
                    }
                    2 | 6 => {
                        // TRN1 (part 0) / TRN2 (part 1)
                        let part = if op == 6 { 1usize } else { 0usize };
                        for pair in 0..(count / 2) {
                            let i0 = 2 * pair;
                            let i1 = 2 * pair + 1;
                            let s0 = (2 * pair + part) * eb;
                            out[i0 * eb..(i0 + 1) * eb].copy_from_slice(&a[s0..s0 + eb]);
                            out[i1 * eb..(i1 + 1) * eb].copy_from_slice(&b[s0..s0 + eb]);
                        }
                    }
                    3 | 7 => {
                        // ZIP1 (part 0) / ZIP2 (part 1)
                        let base_idx = if op == 7 { count / 2 } else { 0usize };
                        for pair in 0..(count / 2) {
                            let s = (base_idx + pair) * eb;
                            let d0 = (2 * pair) * eb;
                            let d1 = (2 * pair + 1) * eb;
                            out[d0..d0 + eb].copy_from_slice(&a[s..s + eb]);
                            out[d1..d1 + eb].copy_from_slice(&b[s..s + eb]);
                        }
                    }
                    _ => {}
                }
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdScalarAddp { vd, vn } => {
                let sum = cpu.v[vn as usize][0].wrapping_add(cpu.v[vn as usize][1]);
                cpu.v[vd as usize] = [sum, 0];
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdScalarTwoRegMisc {
                u,
                size,
                opcode,
                vd,
                vn,
            } => {
                cpu.v[vd as usize] = exec_scalar_two_reg_misc(u, size, opcode, cpu.v[vn as usize]);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdElemMul {
                q,
                u,
                size,
                l,
                m,
                h,
                rm,
                opcode,
                vd,
                vn,
            } => {
                let eb = 1usize << size;
                let (vm, idx) = if size == 1 {
                    (rm as usize, ((h as usize) << 2) | ((l as usize) << 1) | (m as usize))
                } else {
                    (((m as usize) << 4) | (rm as usize), ((h as usize) << 1) | (l as usize))
                };
                let b_arr = vec_to_bytes(cpu.v[vm]);
                let elem_off = idx * eb;
                let bu = read_elem_u64(&b_arr[elem_off..elem_off + eb]);
                let bs = read_elem_i64(&b_arr[elem_off..elem_off + eb]);
                let a_arr = vec_to_bytes(cpu.v[vn as usize]);
                let d_arr = vec_to_bytes(cpu.v[vd as usize]);

                let mut out = [0u8; 16];
                if matches!(opcode, 0b0010 | 0b0110 | 0b1010) {
                    // Long multiply: SMLAL/UMLAL (0010), SMLSL/UMLSL (0110), SMULL/UMULL (1010)
                    let dst_eb = eb * 2;
                    let part_off = if q { 8usize } else { 0usize };
                    let lanes = 8 / eb;
                    for i in 0..lanes {
                        let s_off = part_off + i * eb;
                        let prod: u64 = if u {
                            read_elem_u64(&a_arr[s_off..s_off + eb]).wrapping_mul(bu)
                        } else {
                            (read_elem_i64(&a_arr[s_off..s_off + eb]).wrapping_mul(bs)) as u64
                        };
                        let dv = read_elem_u64(&d_arr[i * dst_eb..(i + 1) * dst_eb]);
                        let res = match opcode {
                            0b0010 => dv.wrapping_add(prod),
                            0b0110 => dv.wrapping_sub(prod),
                            _ => prod,
                        };
                        write_elem_u64(&mut out[i * dst_eb..(i + 1) * dst_eb], res);
                    }
                } else {
                    // Normal multiply: MLA (0000), MLS (0100), MUL (1000)
                    let bytes = if q { 16usize } else { 8usize };
                    for off in (0..bytes).step_by(eb) {
                        let prod = read_elem_u64(&a_arr[off..off + eb]).wrapping_mul(bu);
                        let dv = read_elem_u64(&d_arr[off..off + eb]);
                        let res = match opcode {
                            0b0000 => dv.wrapping_add(prod),
                            0b0100 => dv.wrapping_sub(prod),
                            _ => prod,
                        };
                        write_elem_u64(&mut out[off..off + eb], res);
                    }
                }
                cpu.v[vd as usize] = bytes_to_vec(out);
                Ok(StepOutcome::Continue)
            }
            Instruction::CryptoAes { op, vd, vn } => {
                cpu.v[vd as usize] = exec_crypto_aes(op, cpu.v[vd as usize], cpu.v[vn as usize]);
                Ok(StepOutcome::Continue)
            }
            Instruction::CryptoSha2 { op, vd, vn } => {
                cpu.v[vd as usize] = exec_crypto_sha2(op, cpu.v[vd as usize], cpu.v[vn as usize]);
                Ok(StepOutcome::Continue)
            }
            Instruction::CryptoSha3 { op, vd, vn, vm } => {
                cpu.v[vd as usize] = exec_crypto_sha3(
                    op,
                    cpu.v[vd as usize],
                    cpu.v[vn as usize],
                    cpu.v[vm as usize],
                );
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdPmull { q, size, vd, vn, vm } => {
                cpu.v[vd as usize] = exec_pmull(q, size, cpu.v[vn as usize], cpu.v[vm as usize]);
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdThreeDiff { q, u, size, opcode, vd, vn, vm } => {
                cpu.v[vd as usize] = exec_simd_three_diff(
                    q,
                    u,
                    size,
                    opcode,
                    cpu.v[vd as usize],
                    cpu.v[vn as usize],
                    cpu.v[vm as usize],
                );
                Ok(StepOutcome::Continue)
            }
            Instruction::CryptoEor3 { bcax, vd, vn, vm, va } => {
                let n = cpu.v[vn as usize];
                let m = cpu.v[vm as usize];
                let a = cpu.v[va as usize];
                cpu.v[vd as usize] = if bcax {
                    [n[0] ^ (m[0] & !a[0]), n[1] ^ (m[1] & !a[1])]
                } else {
                    [n[0] ^ m[0] ^ a[0], n[1] ^ m[1] ^ a[1]]
                };
                Ok(StepOutcome::Continue)
            }
            Instruction::CryptoRax1 { vd, vn, vm } => {
                let n = cpu.v[vn as usize];
                let m = cpu.v[vm as usize];
                cpu.v[vd as usize] = [n[0] ^ m[0].rotate_left(1), n[1] ^ m[1].rotate_left(1)];
                Ok(StepOutcome::Continue)
            }
            Instruction::CryptoXar { imm6, vd, vn, vm } => {
                let n = cpu.v[vn as usize];
                let m = cpu.v[vm as usize];
                let rot = (imm6 & 63) as u32;
                cpu.v[vd as usize] = [
                    (n[0] ^ m[0]).rotate_right(rot),
                    (n[1] ^ m[1]).rotate_right(rot),
                ];
                Ok(StepOutcome::Continue)
            }
            Instruction::Unknown(w) => Ok(StepOutcome::Unknown(w)),
        }
    }
}

#[derive(Clone, Copy)]
enum BitfieldKind {
    Sbfm,
    Ubfm,
}

fn exec_bitfield(src: u64, w: Width, immr: u8, imms: u8, kind: BitfieldKind) -> u64 {
    let datasize = w.bits();
    let s = (imms as u32) & (datasize - 1);
    let r = (immr as u32) & (datasize - 1);
    let src_m = src & w.mask();

    let res = if s >= r {
        let len = s - r + 1;
        let field = (src_m >> r) & if len == 64 { u64::MAX } else { (1u64 << len) - 1 };
        match kind {
            BitfieldKind::Ubfm => field,
            BitfieldKind::Sbfm => {
                let sign = (field >> (len - 1)) & 1;
                if sign != 0 && len < 64 {
                    field | (!((1u64 << len) - 1))
                } else {
                    field
                }
            }
        }
    } else {
        // Insert bottom (s + 1) bits of src at bit (datasize - r)
        let len = s + 1;
        let width_mask = if len == 64 { u64::MAX } else { (1u64 << len) - 1 };
        let field = src_m & width_mask;
        let pos = datasize - r;
        let shifted = field << pos;
        match kind {
            BitfieldKind::Ubfm => shifted,
            BitfieldKind::Sbfm => {
                let sign = (field >> (len - 1)) & 1;
                let top_bit = pos + len;
                if sign != 0 && top_bit < 64 {
                    shifted | (!((1u64 << top_bit) - 1))
                } else {
                    shifted
                }
            }
        }
    };
    res & w.mask()
}

fn exec_bfm(dst: u64, src: u64, w: Width, immr: u8, imms: u8) -> u64 {
    let datasize = w.bits();
    let s = (imms as u32) & (datasize - 1);
    let r = (immr as u32) & (datasize - 1);
    let dst_m = dst & w.mask();
    let src_m = src & w.mask();

    let res = if s >= r {
        let len = s - r + 1;
        let mask = if len == 64 { u64::MAX } else { (1u64 << len) - 1 };
        let field = (src_m >> r) & mask;
        (dst_m & !mask) | field
    } else {
        let len = s + 1;
        let mask = if len == 64 { u64::MAX } else { (1u64 << len) - 1 };
        let field = src_m & mask;
        let pos = datasize - r;
        (dst_m & !(mask << pos)) | (field << pos)
    };
    res & w.mask()
}

fn vec_to_bytes(v: [u64; 2]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&v[0].to_le_bytes());
    out[8..16].copy_from_slice(&v[1].to_le_bytes());
    out
}

fn bytes_to_vec(b: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
    ]
}

fn read_elem_u64(slice: &[u8]) -> u64 {
    match slice.len() {
        1 => slice[0] as u64,
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes(slice.try_into().unwrap()),
        _ => 0,
    }
}

fn read_elem_i64(slice: &[u8]) -> i64 {
    match slice.len() {
        1 => slice[0] as i8 as i64,
        2 => i16::from_le_bytes([slice[0], slice[1]]) as i64,
        4 => i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as i64,
        8 => i64::from_le_bytes(slice.try_into().unwrap()),
        _ => 0,
    }
}

fn write_elem_u64(slice: &mut [u8], val: u64) {
    match slice.len() {
        1 => slice[0] = val as u8,
        2 => slice.copy_from_slice(&(val as u16).to_le_bytes()),
        4 => slice.copy_from_slice(&(val as u32).to_le_bytes()),
        8 => slice.copy_from_slice(&val.to_le_bytes()),
        _ => {}
    }
}

fn adv_simd_expand_imm(op: u8, cmode: u8, imm8: u8) -> u64 {
    let i8 = imm8 as u64;
    match cmode >> 1 {
        0 => {
            let v = i8;
            v | (v << 32)
        }
        1 => {
            let v = i8 << 8;
            v | (v << 32)
        }
        2 => {
            let v = i8 << 16;
            v | (v << 32)
        }
        3 => {
            let v = i8 << 24;
            v | (v << 32)
        }
        4 => {
            let v = i8;
            v | (v << 16) | (v << 32) | (v << 48)
        }
        5 => {
            let v = i8 << 8;
            v | (v << 16) | (v << 32) | (v << 48)
        }
        6 => {
            let v = if (cmode & 1) == 0 {
                (i8 << 8) | 0xFF
            } else {
                (i8 << 16) | 0xFFFF
            };
            v | (v << 32)
        }
        7 => match (cmode & 1, op) {
            (0, 0) => u64::from_le_bytes([imm8; 8]),
            (0, 1) => {
                let mut out = 0u64;
                for bit in 0..8 {
                    if ((imm8 >> bit) & 1) != 0 {
                        out |= 0xFFu64 << (bit * 8);
                    }
                }
                out
            }
            (1, 0) => {
                let sign = ((imm8 >> 7) as u32) << 31;
                let b6 = (imm8 >> 6) & 1;
                let exp = (((b6 ^ 1) as u32) << 7)
                    | (if b6 != 0 { 0x1F } else { 0 } << 2)
                    | (((imm8 >> 4) & 0x3) as u32);
                let frac = ((imm8 & 0xF) as u32) << 19;
                let f = (sign | (exp << 23) | frac) as u64;
                f | (f << 32)
            }
            _ => {
                let sign = ((imm8 >> 7) as u64) << 63;
                let b6 = (imm8 >> 6) & 1;
                let exp = (((b6 ^ 1) as u64) << 10)
                    | (if b6 != 0 { 0xFF } else { 0 } << 2)
                    | (((imm8 >> 4) & 0x3) as u64);
                let frac = ((imm8 & 0xF) as u64) << 48;
                sign | (exp << 52) | frac
            }
        },
        _ => 0,
    }
}

fn fp_binop<F32, F64>(cpu: &mut CpuState, is_64: bool, rd: u8, rn: u8, rm: u8, f32_op: F32, f64_op: F64)
where
    F32: FnOnce(f32, f32) -> f32,
    F64: FnOnce(f64, f64) -> f64,
{
    if is_64 {
        let a = f64::from_bits(cpu.v[rn as usize][0]);
        let b = f64::from_bits(cpu.v[rm as usize][0]);
        cpu.v[rd as usize][0] = f64_op(a, b).to_bits();
    } else {
        let a = f32::from_bits(cpu.v[rn as usize][0] as u32);
        let b = f32::from_bits(cpu.v[rm as usize][0] as u32);
        cpu.v[rd as usize][0] = f32_op(a, b).to_bits() as u64;
    }
    cpu.v[rd as usize][1] = 0;
}

fn set_fp_nzcv(cpu: &mut CpuState, ord: Option<std::cmp::Ordering>) {
    match ord {
        Some(std::cmp::Ordering::Less) => cpu.set_nzcv(true, false, false, false),
        Some(std::cmp::Ordering::Equal) => cpu.set_nzcv(false, true, true, false),
        Some(std::cmp::Ordering::Greater) => cpu.set_nzcv(false, false, true, false),
        None => cpu.set_nzcv(false, false, true, true),
    }
}

#[inline]
fn size_mask(size: u8) -> u64 {
    match size {
        0 => 0xFF,
        1 => 0xFFFF,
        2 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

fn read_sized<M: GuestMemory + ?Sized>(mem: &mut M, pa: u64, size: u8) -> Result<u64> {
    Ok(match size {
        0 => mem.read_u8(pa)? as u64,
        1 => mem.read_u16(pa)? as u64,
        2 => mem.read_u32(pa)? as u64,
        _ => mem.read_u64(pa)?,
    })
}

fn write_sized<M: GuestMemory + ?Sized>(mem: &mut M, pa: u64, size: u8, val: u64) -> Result<()> {
    match size {
        0 => mem.write_u8(pa, val as u8),
        1 => mem.write_u16(pa, val as u16),
        2 => mem.write_u32(pa, val as u32),
        _ => mem.write_u64(pa, val),
    }
}

fn read_cpu_sysreg<M: SysRegBus + ?Sized>(cpu: &CpuState, bus: &mut M, reg: SysReg) -> u64 {
    match reg {
        SysReg::MIDR_EL1 => 0x611F_0220,
        SysReg::MPIDR_EL1 => 0x8000_0000,
        SysReg::CTR_EL0 => 0x8444_C004,
        SysReg::DCZID_EL0 => 4,
        SysReg::ID_AA64PFR0_EL1 => 0x0000_0011,
        SysReg::ID_AA64PFR1_EL1 => 0x0000_0000,
        SysReg::ID_AA64DFR0_EL1 => 0x1030_5106,
        SysReg::ID_AA64ISAR0_EL1 => 0x0000_1111,
        SysReg::ID_AA64ISAR1_EL1 => 0x0000_0000,
        SysReg::ID_AA64MMFR0_EL1 => 0x0010_0005, // 16KB & 4KB granules, 48-bit PA
        SysReg::ID_AA64MMFR1_EL1 => 0x0010_0000, // FEAT_PAN
        SysReg::ID_AA64MMFR2_EL1 => 0x0000_0000,
        SysReg::CCSIDR_EL1 => 0x700F_E01A,
        SysReg::CLIDR_EL1 => 0x0A20_0023,
        SysReg::CSSELR_EL1 => cpu.csselr_el1,
        SysReg::SCTLR_EL1 => cpu.sctlr_el1,
        SysReg::CPACR_EL1 => cpu.cpacr_el1,
        SysReg::TTBR0_EL1 => cpu.ttbr0_el1,
        SysReg::TTBR1_EL1 => cpu.ttbr1_el1,
        SysReg::TCR_EL1 => cpu.tcr_el1,
        SysReg::MDSCR_EL1 => cpu.mdscr_el1,
        SysReg::MAIR_EL1 => cpu.mair_el1,
        SysReg::AMAIR_EL1 => cpu.amair_el1,
        SysReg::VBAR_EL1 => cpu.vbar_el1,
        SysReg::ELR_EL1 => cpu.elr_el1,
        SysReg::SPSR_EL1 => cpu.spsr_el1,
        SysReg::ESR_EL1 => cpu.esr_el1,
        SysReg::FAR_EL1 => cpu.far_el1,
        SysReg::PAR_EL1 => cpu.par_el1,
        SysReg::SP_EL0 => cpu.sp_el0,
        SysReg::SPSEL => cpu.sp_sel as u64,
        SysReg::CONTEXTIDR_EL1 => cpu.contextidr_el1,
        SysReg::TPIDR_EL0 => cpu.tpidr_el0,
        SysReg::TPIDR_EL1 => cpu.tpidr_el1,
        SysReg::TPIDRRO_EL0 => cpu.tpidrro_el0,
        SysReg::NZCV => cpu.nzcv,
        SysReg::DAIF => (cpu.daif as u64) << 6,
        SysReg::PAN => (cpu.pan as u64) << 22,
        SysReg::CURRENTEL => (cpu.el as u64) << 2,
        _ => bus.read_sysreg(reg).unwrap_or(0),
    }
}

fn write_cpu_sysreg<M: SysRegBus + ?Sized>(
    cpu: &mut CpuState,
    bus: &mut M,
    mmu: &mut Mmu,
    reg: SysReg,
    val: u64,
) {
    match reg {
        SysReg::SCTLR_EL1 => {
            cpu.sctlr_el1 = val;
            mmu.tlbi_all();
        }
        SysReg::CPACR_EL1 => cpu.cpacr_el1 = val,
        SysReg::TTBR0_EL1 => {
            cpu.ttbr0_el1 = val;
            mmu.tlbi_all();
        }
        SysReg::TTBR1_EL1 => {
            cpu.ttbr1_el1 = val;
            mmu.tlbi_all();
        }
        SysReg::TCR_EL1 => {
            cpu.tcr_el1 = val;
            mmu.tlbi_all();
        }
        SysReg::MAIR_EL1 => cpu.mair_el1 = val,
        SysReg::AMAIR_EL1 => cpu.amair_el1 = val,
        SysReg::CSSELR_EL1 => cpu.csselr_el1 = val,
        SysReg::MDSCR_EL1 => cpu.mdscr_el1 = val,
        SysReg::PAR_EL1 => cpu.par_el1 = val,
        SysReg::CONTEXTIDR_EL1 => cpu.contextidr_el1 = val,
        SysReg::VBAR_EL1 => cpu.vbar_el1 = val,
        SysReg::ELR_EL1 => cpu.elr_el1 = val,
        SysReg::SPSR_EL1 => cpu.spsr_el1 = val,
        SysReg::ESR_EL1 => cpu.esr_el1 = val,
        SysReg::FAR_EL1 => cpu.far_el1 = val,
        SysReg::SP_EL0 => cpu.sp_el0 = val,
        SysReg::SPSEL => cpu.set_sp_sel((val & 1) as u8),
        SysReg::TPIDR_EL0 => cpu.tpidr_el0 = val,
        SysReg::TPIDR_EL1 => cpu.tpidr_el1 = val,
        SysReg::TPIDRRO_EL0 => cpu.tpidrro_el0 = val,
        SysReg::NZCV => cpu.nzcv = val & 0xF000_0000,
        SysReg::DAIF => cpu.daif = ((val >> 6) & 0xF) as u8,
        SysReg::PAN => cpu.pan = ((val >> 22) & 1) as u8,
        _ => {
            let _ = bus.write_sysreg(reg, val);
        }
    }
}

fn vec_to_u32s(v: [u64; 2]) -> [u32; 4] {
    [
        v[0] as u32,
        (v[0] >> 32) as u32,
        v[1] as u32,
        (v[1] >> 32) as u32,
    ]
}

fn u32s_to_vec(w: [u32; 4]) -> [u64; 2] {
    [
        (w[0] as u64) | ((w[1] as u64) << 32),
        (w[2] as u64) | ((w[3] as u64) << 32),
    ]
}

fn exec_crypto_sha2(op: u8, vd: [u64; 2], vn: [u64; 2]) -> [u64; 2] {
    let d = vec_to_u32s(vd);
    let n = vec_to_u32s(vn);
    match op {
        0 => {
            // SHA1H Sd, Sn
            [n[0].rotate_left(30) as u64, 0]
        }
        1 => {
            // SHA1SU1 Vd.4S, Vn.4S
            let mut t = [
                d[0] ^ n[1],
                d[1] ^ n[2],
                d[2] ^ n[3],
                d[3],
            ];
            t[0] = t[0].rotate_left(1);
            t[1] = t[1].rotate_left(1);
            t[2] = t[2].rotate_left(1);
            t[3] = (t[3] ^ t[0]).rotate_left(1);
            u32s_to_vec(t)
        }
        2 => {
            // SHA256SU0 Vd.4S, Vn.4S
            let t = [d[1], d[2], d[3], n[0]];
            let mut out = [0u32; 4];
            for e in 0..4 {
                let s0 = t[e].rotate_right(7) ^ t[e].rotate_right(18) ^ (t[e] >> 3);
                out[e] = d[e].wrapping_add(s0);
            }
            u32s_to_vec(out)
        }
        _ => vd,
    }
}

fn exec_crypto_sha3(op: u8, vd: [u64; 2], vn: [u64; 2], vm: [u64; 2]) -> [u64; 2] {
    let d = vec_to_u32s(vd);
    let n = vec_to_u32s(vn);
    let m = vec_to_u32s(vm);
    match op {
        0..=2 => {
            // SHA1C / SHA1P / SHA1M
            let mut x = d;
            let mut y = n[0];
            for e in 0..4 {
                let t = match op {
                    0 => ((x[2] ^ x[3]) & x[1]) ^ x[3],
                    1 => x[1] ^ x[2] ^ x[3],
                    _ => (x[1] & x[2]) | ((x[1] | x[2]) & x[3]),
                };
                let new_a = y
                    .wrapping_add(x[0].rotate_left(5))
                    .wrapping_add(t)
                    .wrapping_add(m[e]);
                let rot_b = x[1].rotate_left(30);
                y = x[3];
                x = [new_a, x[0], rot_b, x[2]];
            }
            u32s_to_vec(x)
        }
        3 => {
            // SHA1SU0 Vd.4S, Vn.4S, Vm.4S
            let op_v = [d[2], d[3], n[0], n[1]];
            u32s_to_vec([
                op_v[0] ^ d[0] ^ m[0],
                op_v[1] ^ d[1] ^ m[1],
                op_v[2] ^ d[2] ^ m[2],
                op_v[3] ^ d[3] ^ m[3],
            ])
        }
        4 | 5 => {
            // SHA256H (op=4, part1=true) / SHA256H2 (op=5, part1=false)
            let part1 = op == 4;
            let (mut x, mut y) = if part1 { (d, n) } else { (n, d) };
            for e in 0..4 {
                let s1 = y[0].rotate_right(6) ^ y[0].rotate_right(11) ^ y[0].rotate_right(25);
                let ch = ((y[1] ^ y[2]) & y[0]) ^ y[2];
                let t1 = y[3].wrapping_add(s1).wrapping_add(ch).wrapping_add(m[e]);
                let s0 = x[0].rotate_right(2) ^ x[0].rotate_right(13) ^ x[0].rotate_right(22);
                let maj = (x[0] & x[1]) | ((x[0] | x[1]) & x[2]);
                let t2 = s0.wrapping_add(maj);
                let new_e = t1.wrapping_add(x[3]);
                let new_a = t1.wrapping_add(t2);
                y = [new_e, y[0], y[1], y[2]];
                x = [new_a, x[0], x[1], x[2]];
            }
            u32s_to_vec(if part1 { x } else { y })
        }
        6 => {
            // SHA256SU1 Vd.4S, Vn.4S, Vm.4S
            let t0 = [n[1], n[2], n[3], m[0]];
            let sig1 = |x: u32| x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10);
            let mut out = [0u32; 4];
            out[0] = d[0].wrapping_add(t0[0]).wrapping_add(sig1(m[2]));
            out[1] = d[1].wrapping_add(t0[1]).wrapping_add(sig1(m[3]));
            out[2] = d[2].wrapping_add(t0[2]).wrapping_add(sig1(out[0]));
            out[3] = d[3].wrapping_add(t0[3]).wrapping_add(sig1(out[1]));
            u32s_to_vec(out)
        }
        _ => vd,
    }
}

#[inline]
fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if (b & 1) != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1B;
        }
        b >>= 1;
    }
    p
}

const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

const AES_INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

fn exec_crypto_aes(op: u8, vd: [u64; 2], vn: [u64; 2]) -> [u64; 2] {
    let d = vec_to_bytes(vd);
    let n = vec_to_bytes(vn);
    let mut out = [0u8; 16];
    match op {
        4 => {
            // AESE: AddRoundKey -> ShiftRows -> SubBytes
            let mut s = [0u8; 16];
            for i in 0..16 {
                s[i] = d[i] ^ n[i];
            }
            for c in 0..4 {
                for r in 0..4 {
                    out[c * 4 + r] = AES_SBOX[s[((c + r) % 4) * 4 + r] as usize];
                }
            }
        }
        5 => {
            // AESD: AddRoundKey -> InvShiftRows -> InvSubBytes
            let mut s = [0u8; 16];
            for i in 0..16 {
                s[i] = d[i] ^ n[i];
            }
            for c in 0..4 {
                for r in 0..4 {
                    out[c * 4 + r] = AES_INV_SBOX[s[((c + 4 - r) % 4) * 4 + r] as usize];
                }
            }
        }
        6 => {
            // AESMC: MixColumns(Vn)
            for c in 0..4 {
                let b = &n[c * 4..c * 4 + 4];
                out[c * 4] = gf_mul(2, b[0]) ^ gf_mul(3, b[1]) ^ b[2] ^ b[3];
                out[c * 4 + 1] = b[0] ^ gf_mul(2, b[1]) ^ gf_mul(3, b[2]) ^ b[3];
                out[c * 4 + 2] = b[0] ^ b[1] ^ gf_mul(2, b[2]) ^ gf_mul(3, b[3]);
                out[c * 4 + 3] = gf_mul(3, b[0]) ^ b[1] ^ b[2] ^ gf_mul(2, b[3]);
            }
        }
        7 => {
            // AESIMC: InvMixColumns(Vn)
            for c in 0..4 {
                let b = &n[c * 4..c * 4 + 4];
                out[c * 4] = gf_mul(0x0e, b[0]) ^ gf_mul(0x0b, b[1]) ^ gf_mul(0x0d, b[2]) ^ gf_mul(0x09, b[3]);
                out[c * 4 + 1] = gf_mul(0x09, b[0]) ^ gf_mul(0x0e, b[1]) ^ gf_mul(0x0b, b[2]) ^ gf_mul(0x0d, b[3]);
                out[c * 4 + 2] = gf_mul(0x0d, b[0]) ^ gf_mul(0x09, b[1]) ^ gf_mul(0x0e, b[2]) ^ gf_mul(0x0b, b[3]);
                out[c * 4 + 3] = gf_mul(0x0b, b[0]) ^ gf_mul(0x0d, b[1]) ^ gf_mul(0x09, b[2]) ^ gf_mul(0x0e, b[3]);
            }
        }
        _ => return vd,
    }
    bytes_to_vec(out)
}

fn exec_pmull(q: bool, size: u8, vn: [u64; 2], vm: [u64; 2]) -> [u64; 2] {
    if size == 3 {
        let a = if q { vn[1] } else { vn[0] };
        let b = if q { vm[1] } else { vm[0] };
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..64 {
            if ((b >> i) & 1) != 0 {
                lo ^= a << i;
                if i > 0 {
                    hi ^= a >> (64 - i);
                }
            }
        }
        [lo, hi]
    } else {
        // size == 0: 8x8 -> 16-bit polynomial multiply
        let a_b = vec_to_bytes(vn);
        let b_b = vec_to_bytes(vm);
        let off = if q { 8 } else { 0 };
        let mut out = [0u8; 16];
        for lane in 0..8 {
            let a = a_b[off + lane] as u16;
            let b = b_b[off + lane] as u16;
            let mut p = 0u16;
            for i in 0..8 {
                if ((b >> i) & 1) != 0 {
                    p ^= a << i;
                }
            }
            out[lane * 2..lane * 2 + 2].copy_from_slice(&p.to_le_bytes());
        }
        bytes_to_vec(out)
    }
}

/// Executes ARMv8 Advanced SIMD Three-Different widening instructions (`SADDL`, `UADDL`, `SADDW`,
/// `UADDW`, `SSUBL`, `USUBL`, `SSUBW`, `USUBW`, `SABAL`, `UABAL`, `SABDL`, `UABDL`, `SMLAL`,
/// `UMLAL`, `SMLSL`, `UMLSL`, `SMULL`, `UMULL` and `2`-suffixed upper-half variants).
fn exec_simd_three_diff(
    q: bool,
    u: bool,
    size: u8,
    opcode: u8,
    vd: [u64; 2],
    vn: [u64; 2],
    vm: [u64; 2],
) -> [u64; 2] {
    let esize = 1usize << size;
    let dsize = esize * 2;
    let count = 16 / dsize;
    let off = if q { 8 } else { 0 };
    let signed = !u;
    let d_b = vec_to_bytes(vd);
    let n_b = vec_to_bytes(vn);
    let m_b = vec_to_bytes(vm);
    let read_int = |bytes: &[u8; 16], byte_off: usize, width: usize, sign: bool| -> i128 {
        let mut buf = [if sign && (bytes[byte_off + width - 1] & 0x80) != 0 {
            0xFF
        } else {
            0u8
        }; 16];
        buf[..width].copy_from_slice(&bytes[byte_off..byte_off + width]);
        i128::from_le_bytes(buf)
    };
    let mut out = [0u8; 16];
    for lane in 0..count {
        let m = read_int(&m_b, off + lane * esize, esize, signed);
        let res = match opcode {
            0 => read_int(&n_b, off + lane * esize, esize, signed).wrapping_add(m),
            1 => read_int(&n_b, lane * dsize, dsize, signed).wrapping_add(m),
            2 => read_int(&n_b, off + lane * esize, esize, signed).wrapping_sub(m),
            3 => read_int(&n_b, lane * dsize, dsize, signed).wrapping_sub(m),
            5 => read_int(&d_b, lane * dsize, dsize, false)
                .wrapping_add((read_int(&n_b, off + lane * esize, esize, signed) - m).abs()),
            7 => (read_int(&n_b, off + lane * esize, esize, signed) - m).abs(),
            8 => read_int(&d_b, lane * dsize, dsize, signed)
                .wrapping_add(read_int(&n_b, off + lane * esize, esize, signed).wrapping_mul(m)),
            10 => read_int(&d_b, lane * dsize, dsize, signed)
                .wrapping_sub(read_int(&n_b, off + lane * esize, esize, signed).wrapping_mul(m)),
            12 => read_int(&n_b, off + lane * esize, esize, signed).wrapping_mul(m),
            _ => 0,
        };
        let bytes = res.to_le_bytes();
        out[lane * dsize..lane * dsize + dsize].copy_from_slice(&bytes[..dsize]);
    }
    bytes_to_vec(out)
}

#[cfg(test)]
mod crypto_tests {
    use super::*;

    #[test]
    fn test_sha1_sha256_aes_pmull() {
        // AESMC followed by AESIMC must be identity
        let v = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        let mc = exec_crypto_aes(6, [0, 0], v);
        let imc = exec_crypto_aes(7, [0, 0], mc);
        assert_eq!(imc, v);

        // SHA1SU0 decode check on 0x5e023020
        assert_eq!(
            armt_decoder::decode(0x5e023020),
            Instruction::CryptoSha3 {
                op: 3,
                vd: 0,
                vn: 1,
                vm: 2
            }
        );

        // PMULL 64x64: 0x0000_0000_0000_0003 * 0x0000_0000_0000_0005 = 0xF
        assert_eq!(exec_pmull(false, 3, [3, 0], [5, 0]), [15, 0]);

        // UADDL V0.8H, V2.8B, V0.8B (0x2e200040)
        assert_eq!(
            armt_decoder::decode(0x2e200040),
            Instruction::SimdThreeDiff {
                q: false,
                u: true,
                size: 0,
                opcode: 0,
                vd: 0,
                vn: 2,
                vm: 0,
            }
        );
        let added = exec_simd_three_diff(
            false,
            true,
            0,
            0,
            [0, 0],
            [0x0102_0304_0506_0708, 0],
            [0x1020_3040_5060_7080, 0],
        );
        assert_eq!(added, [0x0055_0066_0077_0088, 0x0011_0022_0033_0044]);

        // EOR3 V23.16B, V6.16B, V1.16B, V18.16B (0xce0148d7)
        assert_eq!(
            armt_decoder::decode(0xce0148d7),
            Instruction::CryptoEor3 {
                bcax: false,
                vd: 23,
                vn: 6,
                vm: 1,
                va: 18,
            }
        );

        // 0x7e61d821 = UCVTF D1, D1
        assert_eq!(
            armt_decoder::decode(0x7e61d821),
            Instruction::SimdScalarTwoRegMisc {
                u: true,
                size: 1,
                opcode: 0x1D,
                vd: 1,
                vn: 1
            }
        );
        let out = exec_scalar_two_reg_misc(true, 1, 0x1D, [42, 0]);
        assert_eq!(f64::from_bits(out[0]), 42.0);

        // 0x6ea06a80 = UADALP V0.2D, V20.4S (APFS fletcher64 pairwise accumulate)
        struct SingleInsnMem(u32);
        impl GuestMemory for SingleInsnMem {
            fn read_u8(&mut self, _: u64) -> Result<u8> { Ok(0) }
            fn read_u16(&mut self, _: u64) -> Result<u16> { Ok(0) }
            fn read_u32(&mut self, _: u64) -> Result<u32> { Ok(self.0) }
            fn read_u64(&mut self, _: u64) -> Result<u64> { Ok(0) }
            fn write_u8(&mut self, _: u64, _: u8) -> Result<()> { Ok(()) }
            fn write_u16(&mut self, _: u64, _: u16) -> Result<()> { Ok(()) }
            fn write_u32(&mut self, _: u64, _: u32) -> Result<()> { Ok(()) }
            fn write_u64(&mut self, _: u64, _: u64) -> Result<()> { Ok(()) }
        }
        impl SysRegBus for SingleInsnMem {
            fn read_sysreg(&mut self, _: SysReg) -> Option<u64> { None }
            fn write_sysreg(&mut self, _: SysReg, _: u64) -> bool { false }
        }
        let mut cpu = CpuState::default_vma2_el1(0x8000_0000, 0x8001_0000, 0);
        let mut mem = SingleInsnMem(0x6ea0_6a80);
        let mut mmu = Mmu::new();
        cpu.sctlr_el1 &= !1;
        cpu.pc = 0x8000_2000;
        cpu.v[0] = [10, 20];
        cpu.v[20] = [(2u64 << 32) | 1u64, (4u64 << 32) | 3u64];
        Interpreter::step(&mut cpu, &mut mem, &mut mmu).unwrap();
        assert_eq!(cpu.v[0], [10 + 1 + 2, 20 + 3 + 4]);
    }
}

fn exec_scalar_two_reg_misc(u: bool, size: u8, opcode: u8, vn: [u64; 2]) -> [u64; 2] {
    let sz_is_64 = (size & 1) == 1;
    let hi_bit = (size >> 1) & 1;
    let val = vn[0];
    let f_val = if sz_is_64 {
        f64::from_bits(val)
    } else {
        f32::from_bits(val as u32) as f64
    };
    let out0 = match (opcode, hi_bit, u) {
        (0x1D, 0, false) => {
            // SCVTF Sd/Dd, Sn/Dn
            if sz_is_64 {
                ((val as i64) as f64).to_bits()
            } else {
                (((val as u32) as i32) as f32).to_bits() as u64
            }
        }
        (0x1D, 0, true) => {
            // UCVTF Sd/Dd, Sn/Dn
            if sz_is_64 {
                (val as f64).to_bits()
            } else {
                ((val as u32) as f32).to_bits() as u64
            }
        }
        (0x1B, 1, false) => {
            // FCVTZS
            if sz_is_64 {
                (f_val.trunc() as i64) as u64
            } else {
                (f_val.trunc() as i32) as u32 as u64
            }
        }
        (0x1B, 1, true) => {
            // FCVTZU
            if sz_is_64 {
                f_val.trunc().max(0.0) as u64
            } else {
                (f_val.trunc().max(0.0) as u32) as u64
            }
        }
        (0x1A, 0, false) => {
            // FCVTNS
            let r = f_val.round_ties_even();
            if sz_is_64 { (r as i64) as u64 } else { (r as i32) as u32 as u64 }
        }
        (0x1A, 0, true) => {
            // FCVTNU
            let r = f_val.round_ties_even().max(0.0);
            if sz_is_64 { r as u64 } else { (r as u32) as u64 }
        }
        (0x1A, 1, false) => {
            let r = f_val.ceil();
            if sz_is_64 { (r as i64) as u64 } else { (r as i32) as u32 as u64 }
        }
        (0x1A, 1, true) => {
            let r = f_val.ceil().max(0.0);
            if sz_is_64 { r as u64 } else { (r as u32) as u64 }
        }
        (0x1B, 0, false) => {
            let r = f_val.floor();
            if sz_is_64 { (r as i64) as u64 } else { (r as i32) as u32 as u64 }
        }
        (0x1B, 0, true) => {
            let r = f_val.floor().max(0.0);
            if sz_is_64 { r as u64 } else { (r as u32) as u64 }
        }
        (0x1C, 0, false) => {
            let r = f_val.round();
            if sz_is_64 { (r as i64) as u64 } else { (r as i32) as u32 as u64 }
        }
        (0x1C, 0, true) => {
            let r = f_val.round().max(0.0);
            if sz_is_64 { r as u64 } else { (r as u32) as u64 }
        }
        _ => val,
    };
    [out0, 0]
}
