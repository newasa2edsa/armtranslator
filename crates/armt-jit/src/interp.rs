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
            StepOutcome::Eret => cpu.pc = cpu.elr_el1,
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
            Instruction::Umaddl { rd, rn, rm, ra } => {
                let a = cpu.read_w(rn) as u64;
                let b = cpu.read_w(rm) as u64;
                let c = cpu.read_x(ra);
                cpu.write_x(rd, c.wrapping_add(a.wrapping_mul(b)));
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
            Instruction::Ldar { size, rt, rn } => {
                let ea = cpu.read_sp_or_x(rn);
                let pa = mmu.translate(cpu, mem, ea, false)?;
                cpu.write_x(rt, read_sized(mem, pa, size)?);
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
            Instruction::Dc { .. } | Instruction::Ic { .. } | Instruction::At { .. } | Instruction::Sys { .. } => {
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
            Instruction::SimdMov { q, vd, vn } => {
                cpu.v[vd as usize][0] = cpu.v[vn as usize][0];
                cpu.v[vd as usize][1] = if q { cpu.v[vn as usize][1] } else { 0 };
                Ok(StepOutcome::Continue)
            }
            Instruction::SimdBitwise { q, op, vd, vn, vm } => {
                let a = cpu.v[vn as usize];
                let b = cpu.v[vm as usize];
                for lane in 0..if q { 2 } else { 1 } {
                    cpu.v[vd as usize][lane] = match op {
                        0 => a[lane] & b[lane],
                        1 => a[lane] & !b[lane],
                        2 => a[lane] | b[lane],
                        3 => a[lane] | !b[lane],
                        4 => a[lane] ^ b[lane],
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
        SysReg::ID_AA64ISAR0_EL1 => 0x0000_1111,
        SysReg::ID_AA64ISAR1_EL1 => 0x0000_0000,
        SysReg::ID_AA64MMFR0_EL1 => 0x0010_0005, // 16KB & 4KB granules, 48-bit PA
        SysReg::SCTLR_EL1 => cpu.sctlr_el1,
        SysReg::CPACR_EL1 => cpu.cpacr_el1,
        SysReg::TTBR0_EL1 => cpu.ttbr0_el1,
        SysReg::TTBR1_EL1 => cpu.ttbr1_el1,
        SysReg::TCR_EL1 => cpu.tcr_el1,
        SysReg::MAIR_EL1 => cpu.mair_el1,
        SysReg::VBAR_EL1 => cpu.vbar_el1,
        SysReg::ELR_EL1 => cpu.elr_el1,
        SysReg::SPSR_EL1 => cpu.spsr_el1,
        SysReg::ESR_EL1 => cpu.esr_el1,
        SysReg::FAR_EL1 => cpu.far_el1,
        SysReg::SP_EL0 => cpu.sp_el0,
        SysReg::TPIDR_EL0 => cpu.tpidr_el0,
        SysReg::TPIDR_EL1 => cpu.tpidr_el1,
        SysReg::TPIDRRO_EL0 => cpu.tpidrro_el0,
        SysReg::NZCV => cpu.nzcv,
        SysReg::DAIF => (cpu.daif as u64) << 6,
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
        SysReg::VBAR_EL1 => cpu.vbar_el1 = val,
        SysReg::ELR_EL1 => cpu.elr_el1 = val,
        SysReg::SPSR_EL1 => cpu.spsr_el1 = val,
        SysReg::ESR_EL1 => cpu.esr_el1 = val,
        SysReg::FAR_EL1 => cpu.far_el1 = val,
        SysReg::SP_EL0 => cpu.sp_el0 = val,
        SysReg::TPIDR_EL0 => cpu.tpidr_el0 = val,
        SysReg::TPIDR_EL1 => cpu.tpidr_el1 = val,
        SysReg::TPIDRRO_EL0 => cpu.tpidrro_el0 = val,
        SysReg::NZCV => cpu.nzcv = val & 0xF000_0000,
        SysReg::DAIF => cpu.daif = ((val >> 6) & 0xF) as u8,
        _ => {
            let _ = bus.write_sysreg(reg, val);
        }
    }
}
