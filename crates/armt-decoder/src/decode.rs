use armt_core::isa::{
    AddrMode, BarrierOp, Condition, ExtendType, Instruction, PacOp, Reg, ShiftType, SysReg, Width,
};

#[inline]
const fn sign_ext(val: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    (((val as u64) << shift) as i64) >> shift
}

/// Exact ARMv8 `DecodeBitMasks(N, imms, immr)` for logical immediate instructions.
pub fn decode_bit_masks(is_64: bool, n: u32, imms: u32, immr: u32) -> Option<u64> {
    if !is_64 && n != 0 {
        return None;
    }
    let comb = ((n & 1) << 6) | ((!imms) & 0x3F);
    if comb == 0 {
        return None;
    }
    let len = 31 - comb.leading_zeros();
    if len < 1 || len > 6 {
        return None;
    }
    let esize = 1u32 << len;
    let levels = esize - 1;
    let s = imms & levels;
    let r = immr & levels;
    if s == levels {
        return None;
    }
    let elem_mask: u64 = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let welem: u64 = if s + 1 == 64 {
        u64::MAX
    } else {
        (1u64 << (s + 1)) - 1
    };
    let rotated = if r == 0 {
        welem & elem_mask
    } else {
        ((welem >> r) | (welem << (esize - r))) & elem_mask
    };

    let mut result = 0u64;
    if esize == 64 {
        result = rotated;
    } else {
        let mut bit = 0;
        while bit < 64 {
            result |= rotated << bit;
            bit += esize;
        }
    }
    if !is_64 {
        result &= 0xFFFF_FFFF;
    }
    Some(result)
}

/// Decode a single 32-bit little-endian AArch64 instruction word into IR.
pub fn decode(word: u32) -> Instruction {
    let op0 = (word >> 25) & 0xF;

    // 1. Data Processing - Immediate: op0 == 100x
    if (op0 & 0b1110) == 0b1000 {
        return decode_dp_imm(word);
    }

    // 2. Branches, Exception Generating and System: op0 == 101x
    if (op0 & 0b1110) == 0b1010 {
        return decode_branch_sys(word);
    }

    // 3. Loads and Stores: op0[2:0] == x1x0 -> (op0 & 0b0101) == 0b0100
    if (op0 & 0b0101) == 0b0100 {
        return decode_load_store(word);
    }

    // 4. Data Processing - Register: op0 == x101
    if (op0 & 0b0111) == 0b0101 {
        return decode_dp_reg(word);
    }

    // 5. Data Processing - Scalar FP and SIMD: op0 == x111
    if (op0 & 0b0111) == 0b0111 {
        return decode_fp_simd(word);
    }

    Instruction::Unknown(word)
}

fn decode_dp_imm(word: u32) -> Instruction {
    let sf = (word >> 31) & 1;
    let w = if sf == 1 { Width::X64 } else { Width::W32 };
    let rd = Reg((word & 0x1F) as u8);
    let op_group = (word >> 23) & 0x7;

    match op_group {
        // PC-relative addressing: ADR / ADRP
        0b000 | 0b001 => {
            let immlo = (word >> 29) & 0x3;
            let immhi = (word >> 5) & 0x7FFFF;
            let imm21 = sign_ext((immhi << 2) | immlo, 21);
            if sf == 0 {
                Instruction::Adr { rd, offset: imm21 }
            } else {
                Instruction::Adrp {
                    rd,
                    offset: imm21 << 12,
                }
            }
        }
        // Add/Subtract (immediate)
        0b010 | 0b011 => {
            let sub = ((word >> 30) & 1) == 1;
            let set_flags = ((word >> 29) & 1) == 1;
            let sh = (word >> 22) & 1;
            let imm12 = ((word >> 10) & 0xFFF) as u64;
            let imm = if sh == 1 { imm12 << 12 } else { imm12 };
            let rn = Reg(((word >> 5) & 0x1F) as u8);
            if sub {
                Instruction::SubImm {
                    w,
                    rd,
                    rn,
                    imm,
                    set_flags,
                }
            } else {
                Instruction::AddImm {
                    w,
                    rd,
                    rn,
                    imm,
                    set_flags,
                }
            }
        }
        // Logical (immediate)
        0b100 => {
            let opc = (word >> 29) & 0x3;
            let n = (word >> 22) & 1;
            let immr = (word >> 16) & 0x3F;
            let imms = (word >> 10) & 0x3F;
            let rn = Reg(((word >> 5) & 0x1F) as u8);
            let Some(imm) = decode_bit_masks(sf == 1, n, imms, immr) else {
                return Instruction::Unknown(word);
            };
            match opc {
                0 => Instruction::AndImm {
                    w,
                    rd,
                    rn,
                    imm,
                    set_flags: false,
                },
                1 => Instruction::OrrImm { w, rd, rn, imm },
                2 => Instruction::EorImm { w, rd, rn, imm },
                _ => Instruction::AndImm {
                    w,
                    rd,
                    rn,
                    imm,
                    set_flags: true,
                },
            }
        }
        // Move wide (immediate)
        0b101 => {
            let opc = (word >> 29) & 0x3;
            let hw = ((word >> 21) & 0x3) as u8;
            if sf == 0 && hw >= 2 {
                return Instruction::Unknown(word);
            }
            let shift = hw * 16;
            let imm16 = ((word >> 5) & 0xFFFF) as u16;
            match opc {
                0 => Instruction::Movn {
                    w,
                    rd,
                    imm16,
                    shift,
                },
                2 => Instruction::Movz {
                    w,
                    rd,
                    imm16,
                    shift,
                },
                3 => Instruction::Movk {
                    w,
                    rd,
                    imm16,
                    shift,
                },
                _ => Instruction::Unknown(word),
            }
        }
        // Bitfield
        0b110 => {
            let opc = (word >> 29) & 0x3;
            let immr = ((word >> 16) & 0x3F) as u8;
            let imms = ((word >> 10) & 0x3F) as u8;
            let rn = Reg(((word >> 5) & 0x1F) as u8);
            match opc {
                0 => Instruction::Sbfm {
                    w,
                    rd,
                    rn,
                    immr,
                    imms,
                },
                1 => Instruction::Bfm {
                    w,
                    rd,
                    rn,
                    immr,
                    imms,
                },
                2 => Instruction::Ubfm {
                    w,
                    rd,
                    rn,
                    immr,
                    imms,
                },
                _ => Instruction::Unknown(word),
            }
        }
        // Extract
        0b111 => {
            let op21 = (word >> 29) & 0x3;
            let o0 = (word >> 21) & 1;
            if op21 != 0 || o0 != 0 {
                return Instruction::Unknown(word);
            }
            let rm = Reg(((word >> 16) & 0x1F) as u8);
            let imms = ((word >> 10) & 0x3F) as u8;
            let rn = Reg(((word >> 5) & 0x1F) as u8);
            Instruction::Extr {
                w,
                rd,
                rn,
                rm,
                lsb: imms,
            }
        }
        _ => Instruction::Unknown(word),
    }
}

fn decode_branch_sys(word: u32) -> Instruction {
    // Unconditional branch immediate: B / BL
    if (word & 0x7C00_0000) == 0x1400_0000 {
        let offset = sign_ext(word & 0x03FF_FFFF, 26) << 2;
        return if (word >> 31) & 1 == 1 {
            Instruction::Bl { offset }
        } else {
            Instruction::B { offset }
        };
    }

    // Conditional branch immediate: B.cond
    if (word & 0xFF00_0010) == 0x5400_0000 {
        let cond = Condition::from_u8((word & 0xF) as u8);
        let offset = sign_ext((word >> 5) & 0x7FFFF, 19) << 2;
        return Instruction::BCond { cond, offset };
    }

    // Compare and branch: CBZ / CBNZ
    if (word & 0x7E00_0000) == 0x3400_0000 {
        let w = if (word >> 31) & 1 == 1 {
            Width::X64
        } else {
            Width::W32
        };
        let is_nz = ((word >> 24) & 1) == 1;
        let offset = sign_ext((word >> 5) & 0x7FFFF, 19) << 2;
        let rt = Reg((word & 0x1F) as u8);
        return if is_nz {
            Instruction::Cbnz { w, rt, offset }
        } else {
            Instruction::Cbz { w, rt, offset }
        };
    }

    // Test and branch: TBZ / TBNZ
    if (word & 0x7E00_0000) == 0x3600_0000 {
        let b5 = (word >> 31) & 1;
        let is_nz = ((word >> 24) & 1) == 1;
        let b40 = (word >> 19) & 0x1F;
        let bit = ((b5 << 5) | b40) as u8;
        let offset = sign_ext((word >> 5) & 0x3FFF, 14) << 2;
        let rt = Reg((word & 0x1F) as u8);
        return if is_nz {
            Instruction::Tbnz { rt, bit, offset }
        } else {
            Instruction::Tbz { rt, bit, offset }
        };
    }

    // Exception generation
    if (word & 0xFF00_0000) == 0xD400_0000 {
        let opc = (word >> 21) & 0x7;
        let imm = ((word >> 5) & 0xFFFF) as u16;
        let op2 = (word >> 2) & 0x7;
        let ll = word & 0x3;
        if op2 == 0 {
            return match (opc, ll) {
                (0, 1) => Instruction::Svc { imm },
                (0, 2) => Instruction::Hvc { imm },
                (0, 3) => Instruction::Smc { imm },
                (1, 0) => Instruction::Brk { imm },
                (2, 0) => Instruction::Hlt { imm },
                _ => Instruction::Unknown(word),
            };
        }
        return Instruction::Unknown(word);
    }

    // System instructions (0xD5xx_xxxx)
    if (word & 0xFFC0_0000) == 0xD500_0000 {
        // HINT family
        if (word & 0xFFFF_F01F) == 0xD503_201F {
            let crm = ((word >> 8) & 0xF) as u8;
            let op2 = ((word >> 5) & 0x7) as u8;
            let imm = (crm << 3) | op2;
            return match imm {
                0 => Instruction::Nop,
                25 | 27 => Instruction::PacHint {
                    op: PacOp::Paciasp,
                    rd: Reg::LR,
                    rn: Reg::SP,
                },
                29 | 31 => Instruction::PacHint {
                    op: PacOp::Autiasp,
                    rd: Reg::LR,
                    rn: Reg::SP,
                },
                _ => Instruction::Hint { imm },
            };
        }

        // Barriers & CLREX
        if (word & 0xFFFF_F01F) == 0xD503_301F {
            let crm = ((word >> 8) & 0xF) as u8;
            let op2 = (word >> 5) & 0x7;
            return match op2 {
                2 => Instruction::Clrex,
                4 => Instruction::Barrier {
                    op: BarrierOp::Dsb(crm),
                },
                5 => Instruction::Barrier {
                    op: BarrierOp::Dmb(crm),
                },
                6 => Instruction::Barrier {
                    op: BarrierOp::Isb(crm),
                },
                _ => Instruction::Nop,
            };
        }

        // MSR (immediate)
        if (word & 0xFFF8_F01F) == 0xD500_401F {
            let op1 = ((word >> 16) & 0x7) as u8;
            let crm = ((word >> 8) & 0xF) as u8;
            let op2 = ((word >> 5) & 0x7) as u8;
            return Instruction::MsrImm {
                pstate_field: (op1 << 3) | op2,
                crm,
            };
        }

        // MRS (register) -> [31:20] == 0xD53
        if (word >> 20) == 0xD53 {
            let rt = Reg((word & 0x1F) as u8);
            let sysreg = SysReg::from_mrs_word(word);
            return Instruction::Mrs { rt, sysreg };
        }

        // MSR (register) -> [31:20] == 0xD51
        if (word >> 20) == 0xD51 {
            let rt = Reg((word & 0x1F) as u8);
            let sysreg = SysReg::from_mrs_word(word);
            return Instruction::Msr { sysreg, rt };
        }

        // SYS -> [31:19] == 0b1101_0101_0000_1 (0xD508_0000..0xD50F_FFFF)
        if (word & 0xFFF8_0000) == 0xD508_0000 {
            let op1 = ((word >> 16) & 0x7) as u8;
            let crn = ((word >> 12) & 0xF) as u8;
            let crm = ((word >> 8) & 0xF) as u8;
            let op2 = ((word >> 5) & 0x7) as u8;
            let rt = Reg((word & 0x1F) as u8);
            return match crn {
                7 => match crm {
                    1 | 5 => Instruction::Ic { op1, crm, op2, rt },
                    8 | 9 => Instruction::At { op1, crm, op2, rt },
                    _ => Instruction::Dc { op1, crm, op2, rt },
                },
                8 => Instruction::Tlbi { op1, crm, op2, rt },
                _ => Instruction::Sys {
                    op1,
                    crn,
                    crm,
                    op2,
                    rt,
                },
            };
        }

        return Instruction::Unknown(word);
    }

    // Unconditional branch (register): BR, BLR, RET, ERET, PAC branches
    if (word & 0xFE00_0000) == 0xD600_0000 {
        let opc = (word >> 21) & 0xF;
        let op2 = (word >> 16) & 0x1F;
        let op3 = (word >> 10) & 0x3F;
        let rn = Reg(((word >> 5) & 0x1F) as u8);
        let op4 = Reg((word & 0x1F) as u8);

        if op2 == 0x1F && op3 == 0 && op4.0 == 0 {
            return match opc {
                0 => Instruction::Br { rn },
                1 => Instruction::Blr { rn },
                2 => Instruction::Ret { rn },
                4 if rn.0 == 31 => Instruction::Eret,
                _ => Instruction::Unknown(word),
            };
        }

        // ARMv8.3 PAC branches and authenticated returns
        if op2 == 0x1F {
            return match (opc, op3, rn.0, op4.0) {
                (2, 2, 31, 31) => Instruction::PacHint {
                    op: PacOp::Retaa,
                    rd: Reg::LR,
                    rn: Reg::SP,
                },
                (2, 3, 31, 31) => Instruction::PacHint {
                    op: PacOp::Retab,
                    rd: Reg::LR,
                    rn: Reg::SP,
                },
                (0, 2, _, _) | (8, 2, _, _) => Instruction::PacHint {
                    op: PacOp::Braa,
                    rd: op4,
                    rn,
                },
                (0, 3, _, _) | (8, 3, _, _) => Instruction::PacHint {
                    op: PacOp::Brab,
                    rd: op4,
                    rn,
                },
                (1, 2, _, _) | (9, 2, _, _) => Instruction::PacHint {
                    op: PacOp::Blraa,
                    rd: op4,
                    rn,
                },
                (1, 3, _, _) | (9, 3, _, _) => Instruction::PacHint {
                    op: PacOp::Blrab,
                    rd: op4,
                    rn,
                },
                _ => Instruction::Unknown(word),
            };
        }
    }

    Instruction::Unknown(word)
}

fn decode_load_store(word: u32) -> Instruction {
    let op0 = (word >> 28) & 0xF;
    let op1 = (word >> 26) & 1; // V bit (FP/SIMD when 1)
    let op2 = (word >> 23) & 0x3;
    let op3 = (word >> 16) & 0x3F;
    let op4 = (word >> 10) & 0x3;

    // 1. Load/Store Exclusive / Ordered / CAS: op0[1:0]==00 && op1==0 && op2[1]==0
    if (word & 0x3F00_0000) == 0x0800_0000 {
        let size = ((word >> 30) & 0x3) as u8;
        let o2 = (word >> 23) & 1;
        let l = (word >> 22) & 1;
        let o1 = (word >> 21) & 1;
        let rs = Reg(((word >> 16) & 0x1F) as u8);
        let o0 = (word >> 15) & 1;
        let rn = Reg(((word >> 5) & 0x1F) as u8);
        let rt = Reg((word & 0x1F) as u8);

        return match (o2, l, o1, o0) {
            (0, 1, 0, _) => Instruction::Ldxr {
                size,
                acquire: o0 == 1,
                rt,
                rn,
            },
            (0, 0, 0, _) => Instruction::Stxr {
                size,
                release: o0 == 1,
                rs,
                rt,
                rn,
            },
            (1, 1, 0, 1) => Instruction::Ldar { size, rt, rn },
            (1, 0, 0, 1) => Instruction::Stlr { size, rt, rn },
            (0, _, 1, _) => Instruction::Cas {
                size,
                acquire: l == 1,
                release: o0 == 1,
                rs,
                rt,
                rn,
            },
            _ => Instruction::Unknown(word),
        };
    }

    // 2. Load register (literal): (word & 0x3B00_0000) == 0x1800_0000
    if (word & 0x3B00_0000) == 0x1800_0000 {
        let opc = (word >> 30) & 0x3;
        let offset = sign_ext((word >> 5) & 0x7FFFF, 19) << 2;
        let rt = Reg((word & 0x1F) as u8);
        if op1 == 0 {
            return match opc {
                0 => Instruction::LdrLiteral {
                    w: Width::W32,
                    rt,
                    offset,
                    signed: false,
                },
                1 => Instruction::LdrLiteral {
                    w: Width::X64,
                    rt,
                    offset,
                    signed: false,
                },
                2 => Instruction::LdrLiteral {
                    w: Width::X64,
                    rt,
                    offset,
                    signed: true,
                },
                3 => Instruction::Prfm,
                _ => Instruction::Unknown(word),
            };
        } else {
            // FP literal load (ignore or treat as Unknown if unsupported)
            let fp_size = match opc {
                0 => 4u8,
                1 => 8u8,
                2 => 16u8,
                _ => return Instruction::Unknown(word),
            };
            let _ = fp_size;
            return Instruction::Unknown(word);
        }
    }

    // 3. Load/Store Pair: (word & 0x3A00_0000) == 0x2800_0000
    if (word & 0x3A00_0000) == 0x2800_0000 {
        let opc = (word >> 30) & 0x3;
        let mode_bits = op2; // 0=no-alloc, 1=post, 2=signed-offset, 3=pre
        let is_load = ((word >> 22) & 1) == 1;
        let imm7 = sign_ext((word >> 15) & 0x7F, 7);
        let rt2 = Reg(((word >> 10) & 0x1F) as u8);
        let rn = Reg(((word >> 5) & 0x1F) as u8);
        let rt = Reg((word & 0x1F) as u8);

        if op1 == 0 {
            let scale = if opc >= 2 { 8 } else { 4 };
            let off = imm7 * scale;
            let mode = match mode_bits {
                1 => AddrMode::PostIndex(off),
                3 => AddrMode::PreIndex(off),
                _ => AddrMode::UnsignedOffset(off),
            };
            return if opc == 1 && is_load {
                Instruction::Ldpsw { rt, rt2, rn, mode }
            } else {
                let w = if opc >= 2 { Width::X64 } else { Width::W32 };
                if is_load {
                    Instruction::Ldp {
                        w,
                        rt,
                        rt2,
                        rn,
                        mode,
                    }
                } else {
                    Instruction::Stp {
                        w,
                        rt,
                        rt2,
                        rn,
                        mode,
                    }
                }
            };
        } else {
            let size = 4u8 << (opc & 0x3);
            if size > 16 {
                return Instruction::Unknown(word);
            }
            let off = imm7 * (size as i64);
            let mode = match mode_bits {
                1 => AddrMode::PostIndex(off),
                3 => AddrMode::PreIndex(off),
                _ => AddrMode::UnsignedOffset(off),
            };
            return if is_load {
                Instruction::LdpFp {
                    size,
                    vt: rt.0,
                    vt2: rt2.0,
                    rn,
                    mode,
                }
            } else {
                Instruction::StpFp {
                    size,
                    vt: rt.0,
                    vt2: rt2.0,
                    rn,
                    mode,
                }
            };
        }
    }

    // 4. Load/Store Register (unscaled, pre, post, reg-offset, atomics, unsigned-imm)
    if (op0 & 0x3) == 0b11 && ((word >> 25) & 1) == 0 {
        let size = ((word >> 30) & 0x3) as u8;
        let opc = ((word >> 22) & 0x3) as u8;
        let rn = Reg(((word >> 5) & 0x1F) as u8);
        let rt = Reg((word & 0x1F) as u8);

        let mode = if ((word >> 24) & 1) == 1 {
            // Unsigned scaled immediate
            let imm12 = ((word >> 10) & 0xFFF) as i64;
            let shift = if op1 == 1 && (opc & 2) != 0 && size == 0 {
                4
            } else {
                size
            };
            AddrMode::UnsignedOffset(imm12 << shift)
        } else if (op3 & 0x20) == 0 {
            // Unscaled / Post-index / Pre-index
            let imm9 = sign_ext((word >> 12) & 0x1FF, 9);
            match op4 {
                0 | 2 => AddrMode::UnsignedOffset(imm9),
                1 => AddrMode::PostIndex(imm9),
                3 => AddrMode::PreIndex(imm9),
                _ => return Instruction::Unknown(word),
            }
        } else if op4 == 2 {
            // Register offset
            let rm = Reg(((word >> 16) & 0x1F) as u8);
            let option = ((word >> 13) & 0x7) as u8;
            let s_bit = ((word >> 12) & 1) as u8;
            let shift_amt = if s_bit == 1 {
                if op1 == 1 && (opc & 2) != 0 && size == 0 {
                    4
                } else {
                    size
                }
            } else {
                0
            };
            AddrMode::RegOffset {
                rm,
                extend: ExtendType::from_u8(option),
                shift: shift_amt,
            }
        } else if op4 == 0 && op1 == 0 {
            // LSE Atomics
            let a = ((word >> 23) & 1) == 1;
            let r = ((word >> 22) & 1) == 1;
            let rs = Reg(((word >> 16) & 0x1F) as u8);
            let atomic_opc = (word >> 12) & 0xF;
            return match atomic_opc {
                0 => Instruction::Ldadd {
                    size,
                    acquire: a,
                    release: r,
                    rs,
                    rt,
                    rn,
                },
                8 => Instruction::Swp {
                    size,
                    acquire: a,
                    release: r,
                    rs,
                    rt,
                    rn,
                },
                _ => Instruction::Unknown(word),
            };
        } else {
            return Instruction::Unknown(word);
        };

        if op1 == 0 {
            return match (size, opc) {
                (0, 0) => Instruction::Strb { rt, rn, mode },
                (0, 1) => Instruction::Ldrb { rt, rn, mode },
                (0, 2) => Instruction::Ldrsb {
                    w: Width::X64,
                    rt,
                    rn,
                    mode,
                },
                (0, 3) => Instruction::Ldrsb {
                    w: Width::W32,
                    rt,
                    rn,
                    mode,
                },
                (1, 0) => Instruction::Strh { rt, rn, mode },
                (1, 1) => Instruction::Ldrh { rt, rn, mode },
                (1, 2) => Instruction::Ldrsh {
                    w: Width::X64,
                    rt,
                    rn,
                    mode,
                },
                (1, 3) => Instruction::Ldrsh {
                    w: Width::W32,
                    rt,
                    rn,
                    mode,
                },
                (2, 0) => Instruction::Str {
                    w: Width::W32,
                    rt,
                    rn,
                    mode,
                },
                (2, 1) => Instruction::Ldr {
                    w: Width::W32,
                    rt,
                    rn,
                    mode,
                },
                (2, 2) => Instruction::Ldrsw { rt, rn, mode },
                (3, 0) => Instruction::Str {
                    w: Width::X64,
                    rt,
                    rn,
                    mode,
                },
                (3, 1) => Instruction::Ldr {
                    w: Width::X64,
                    rt,
                    rn,
                    mode,
                },
                (3, 2) => Instruction::Prfm,
                _ => Instruction::Unknown(word),
            };
        } else {
            let fp_size = match (size, opc & 2) {
                (0, 0) => 1u8,
                (1, 0) => 2u8,
                (2, 0) => 4u8,
                (3, 0) => 8u8,
                (0, 2) => 16u8,
                _ => return Instruction::Unknown(word),
            };
            return if (opc & 1) == 1 {
                Instruction::LdrFp {
                    size: fp_size,
                    vt: rt.0,
                    rn,
                    mode,
                }
            } else {
                Instruction::StrFp {
                    size: fp_size,
                    vt: rt.0,
                    rn,
                    mode,
                }
            };
        }
    }

    Instruction::Unknown(word)
}

fn decode_dp_reg(word: u32) -> Instruction {
    let sf = (word >> 31) & 1;
    let w = if sf == 1 { Width::X64 } else { Width::W32 };
    let op0 = (word >> 30) & 1;
    let op1 = (word >> 28) & 1;
    let op2 = (word >> 21) & 0xF;
    let rd = Reg((word & 0x1F) as u8);
    let rn = Reg(((word >> 5) & 0x1F) as u8);
    let rm = Reg(((word >> 16) & 0x1F) as u8);

    if op1 == 0 {
        if (op2 & 0b1000) == 0 {
            // Logical (shifted register)
            let opc = (word >> 29) & 0x3;
            let shift = ShiftType::from_u8(((word >> 22) & 0x3) as u8);
            let n = (word >> 21) & 1;
            let amount = ((word >> 10) & 0x3F) as u8;
            return match (opc, n) {
                (0, 0) => Instruction::AndReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                    set_flags: false,
                },
                (0, 1) => Instruction::BicReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                    set_flags: false,
                },
                (1, 0) => Instruction::OrrReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                (1, 1) => Instruction::OrnReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                (2, 0) => Instruction::EorReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                (2, 1) => Instruction::EonReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                },
                (3, 0) => Instruction::AndReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                    set_flags: true,
                },
                _ => Instruction::BicReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                    set_flags: true,
                },
            };
        } else if (op2 & 0b0001) == 0 {
            // Add/Sub (shifted register)
            let sub = ((word >> 30) & 1) == 1;
            let set_flags = ((word >> 29) & 1) == 1;
            let shift = ShiftType::from_u8(((word >> 22) & 0x3) as u8);
            let amount = ((word >> 10) & 0x3F) as u8;
            return if sub {
                Instruction::SubReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                    set_flags,
                }
            } else {
                Instruction::AddReg {
                    w,
                    rd,
                    rn,
                    rm,
                    shift,
                    amount,
                    set_flags,
                }
            };
        } else {
            // Add/Sub (extended register)
            let sub = ((word >> 30) & 1) == 1;
            let set_flags = ((word >> 29) & 1) == 1;
            let extend = ExtendType::from_u8(((word >> 13) & 0x7) as u8);
            let amount = ((word >> 10) & 0x7) as u8;
            return if sub {
                Instruction::SubExt {
                    w,
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                    set_flags,
                }
            } else {
                Instruction::AddExt {
                    w,
                    rd,
                    rn,
                    rm,
                    extend,
                    amount,
                    set_flags,
                }
            };
        }
    }

    // op1 == 1: Data processing 3-source when op2[3] == 1
    if (op2 & 0b1000) != 0 {
        let op31 = (word >> 21) & 0x7;
        let o0 = (word >> 15) & 1;
        let ra = Reg(((word >> 10) & 0x1F) as u8);
        return match (sf, op31, o0) {
            (_, 0, 0) => Instruction::Madd { w, rd, rn, rm, ra },
            (_, 0, 1) => Instruction::Msub { w, rd, rn, rm, ra },
            (1, 1, 0) => Instruction::Smaddl { rd, rn, rm, ra },
            (1, 2, 0) => Instruction::Smulh { rd, rn, rm },
            (1, 5, 0) => Instruction::Umaddl { rd, rn, rm, ra },
            (1, 6, 0) => Instruction::Umulh { rd, rn, rm },
            _ => Instruction::Unknown(word),
        };
    }

    match op2 {
        // Add/Sub with carry
        0b0000 => {
            let set_flags = ((word >> 29) & 1) == 1;
            if op0 == 0 {
                Instruction::Adc {
                    w,
                    rd,
                    rn,
                    rm,
                    set_flags,
                }
            } else {
                Instruction::Sbc {
                    w,
                    rd,
                    rn,
                    rm,
                    set_flags,
                }
            }
        }
        // Conditional compare (register / immediate)
        0b0010 => {
            let is_imm = ((word >> 11) & 1) == 1;
            let cond = Condition::from_u8(((word >> 12) & 0xF) as u8);
            let nzcv = (word & 0xF) as u8;
            if op0 == 0 {
                if is_imm {
                    Instruction::CcmnImm {
                        w,
                        rn,
                        imm5: rm.0,
                        nzcv,
                        cond,
                    }
                } else {
                    Instruction::CcmnReg {
                        w,
                        rn,
                        rm,
                        nzcv,
                        cond,
                    }
                }
            } else if is_imm {
                Instruction::CcmpImm {
                    w,
                    rn,
                    imm5: rm.0,
                    nzcv,
                    cond,
                }
            } else {
                Instruction::CcmpReg {
                    w,
                    rn,
                    rm,
                    nzcv,
                    cond,
                }
            }
        }
        // Conditional select
        0b0100 => {
            let cond = Condition::from_u8(((word >> 12) & 0xF) as u8);
            let o2 = (word >> 10) & 0x3;
            match (op0, o2) {
                (0, 0) => Instruction::Csel {
                    w,
                    rd,
                    rn,
                    rm,
                    cond,
                },
                (0, 1) => Instruction::Csinc {
                    w,
                    rd,
                    rn,
                    rm,
                    cond,
                },
                (1, 0) => Instruction::Csinv {
                    w,
                    rd,
                    rn,
                    rm,
                    cond,
                },
                (1, 1) => Instruction::Csneg {
                    w,
                    rd,
                    rn,
                    rm,
                    cond,
                },
                _ => Instruction::Unknown(word),
            }
        }
        // Data-processing 1-source / 2-source
        0b0110 => {
            let opcode = (word >> 10) & 0x3F;
            if op0 == 0 {
                // 2-source
                match opcode {
                    2 => Instruction::Udiv { w, rd, rn, rm },
                    3 => Instruction::Sdiv { w, rd, rn, rm },
                    8 => Instruction::Lslv { w, rd, rn, rm },
                    9 => Instruction::Lsrv { w, rd, rn, rm },
                    10 => Instruction::Asrv { w, rd, rn, rm },
                    11 => Instruction::Rorv { w, rd, rn, rm },
                    _ => Instruction::Unknown(word),
                }
            } else {
                // 1-source
                let opcode2 = rm.0;
                if opcode2 == 0 {
                    match opcode {
                        0 => Instruction::Rbit { w, rd, rn },
                        1 => Instruction::Rev16 { w, rd, rn },
                        2 => Instruction::Rev32 { w, rd, rn },
                        3 if sf == 1 => Instruction::Rev64 { rd, rn },
                        4 => Instruction::Clz { w, rd, rn },
                        5 => Instruction::Cls { w, rd, rn },
                        _ => Instruction::Unknown(word),
                    }
                } else if opcode2 == 1 {
                    let pac = match opcode {
                        0 => PacOp::Pacia,
                        1 => PacOp::Pacib,
                        2 => PacOp::Pacda,
                        3 => PacOp::Pacdb,
                        4 => PacOp::Autia,
                        5 => PacOp::Autib,
                        6 => PacOp::Autda,
                        7 => PacOp::Autdb,
                        8..=15 => PacOp::Pacia,
                        16 => PacOp::Xpaci,
                        17 => PacOp::Xpacd,
                        _ => return Instruction::Unknown(word),
                    };
                    Instruction::PacHint { op: pac, rd, rn }
                } else {
                    Instruction::Unknown(word)
                }
            }
        }
        _ => Instruction::Unknown(word),
    }
}

fn decode_fp_simd(word: u32) -> Instruction {
    // Scalar Floating-Point: bits [28:24] == 0b11110 (0x1E) and bit 21 == 1
    if ((word >> 24) & 0x1F) == 0b11110 && ((word >> 21) & 1) == 1 {
        let sf = ((word >> 31) & 1) == 1;
        let ftype = (word >> 22) & 0x3;
        if ftype > 1 {
            return Instruction::Unknown(word);
        }
        let is_64 = ftype == 1;
        let rd = (word & 0x1F) as u8;
        let rn = ((word >> 5) & 0x1F) as u8;
        let rm = ((word >> 16) & 0x1F) as u8;

        // FP <-> Integer conversion & FMOV GPR<->FP: bits [15:10] == 000000
        if ((word >> 10) & 0x3F) == 0 && ((word >> 29) & 1) == 0 {
            let rmode = (word >> 19) & 0x3;
            let opcode = (word >> 16) & 0x7;
            let w = if sf { Width::X64 } else { Width::W32 };
            return match (rmode, opcode) {
                (0, 2) => Instruction::Scvtf {
                    is_64,
                    w,
                    rd,
                    rn: Reg(rn),
                },
                (0, 3) => Instruction::Ucvtf {
                    is_64,
                    w,
                    rd,
                    rn: Reg(rn),
                },
                (3, 0) => Instruction::Fcvtzs {
                    is_64,
                    w,
                    rd: Reg(rd),
                    rn,
                },
                (3, 1) => Instruction::Fcvtzu {
                    is_64,
                    w,
                    rd: Reg(rd),
                    rn,
                },
                (0, 6) => Instruction::FmovFpToGpr {
                    is_64,
                    rd: Reg(rd),
                    rn,
                },
                (0, 7) => Instruction::FmovGprToFp {
                    is_64,
                    rd,
                    rn: Reg(rn),
                },
                _ => Instruction::Unknown(word),
            };
        }

        // FP immediate: [12:10] == 100
        if ((word >> 10) & 0x7) == 0b100 {
            let imm8 = ((word >> 13) & 0xFF) as u8;
            return Instruction::FmovImm { is_64, rd, imm8 };
        }

        // FP compare: [13:10] == 1000
        if ((word >> 10) & 0xF) == 0b1000 {
            let with_zero = ((word >> 3) & 1) == 1;
            return Instruction::Fcmp {
                is_64,
                rn,
                rm,
                with_zero,
            };
        }

        // FP 2-source: [11:10] == 10
        if ((word >> 10) & 0x3) == 0b10 {
            let opc = (word >> 12) & 0xF;
            return match opc {
                0 => Instruction::Fmul { is_64, rd, rn, rm },
                1 => Instruction::Fdiv { is_64, rd, rn, rm },
                2 => Instruction::Fadd { is_64, rd, rn, rm },
                3 => Instruction::Fsub { is_64, rd, rn, rm },
                _ => Instruction::Unknown(word),
            };
        }

        // FP 1-source: [14:10] == 10000
        if ((word >> 10) & 0x1F) == 0b10000 {
            let opc = (word >> 15) & 0x3F;
            return match opc {
                0 => Instruction::FmovReg { is_64, rd, rn },
                1 => Instruction::Fabs { is_64, rd, rn },
                2 => Instruction::Fneg { is_64, rd, rn },
                3 => Instruction::Fsqrt { is_64, rd, rn },
                4 => Instruction::Fcvt {
                    dst_64: false,
                    src_64: is_64,
                    rd,
                    rn,
                },
                5 => Instruction::Fcvt {
                    dst_64: true,
                    src_64: is_64,
                    rd,
                    rn,
                },
                _ => Instruction::Unknown(word),
            };
        }
    }

    // Advanced SIMD 3-same (ORR / MOV Vd, Vn; AND/EOR/BIC; ADD/SUB vector)
    if (word & 0x9F20_0400) == 0x0E20_0400 {
        let q = ((word >> 30) & 1) == 1;
        let u = ((word >> 29) & 1) == 1;
        let size = ((word >> 22) & 0x3) as u8;
        let vm = ((word >> 16) & 0x1F) as u8;
        let opcode = (word >> 11) & 0x1F;
        let vn = ((word >> 5) & 0x1F) as u8;
        let vd = (word & 0x1F) as u8;

        // Logical bitwise & ORR-alias MOV
        if opcode == 0b00011 {
            if !u && size == 2 && vn == vm {
                return Instruction::SimdMov { q, vd, vn };
            }
            let op_code = ((u as u8) << 2) | size;
            return Instruction::SimdBitwise {
                q,
                op: op_code,
                vd,
                vn,
                vm,
            };
        }

        // ADD / SUB integer vectors
        if opcode == 0b10000 {
            let esize = 8u8 << size;
            return Instruction::SimdAddSub {
                q,
                esize,
                is_sub: u,
                vd,
                vn,
                vm,
            };
        }
    }

    // SIMD DUP Vd, Rn (general): 0Q00_1110_000_imm5_0000_11_Rn_Rd
    if (word & 0xBFE0_FC00) == 0x0E00_0C00 {
        let q = ((word >> 30) & 1) == 1;
        let imm5 = (word >> 16) & 0x1F;
        let esize = if imm5 & 1 != 0 {
            8
        } else if imm5 & 2 != 0 {
            16
        } else if imm5 & 4 != 0 {
            32
        } else if imm5 & 8 != 0 {
            64
        } else {
            return Instruction::Unknown(word);
        };
        let rn = Reg(((word >> 5) & 0x1F) as u8);
        let vd = (word & 0x1F) as u8;
        return Instruction::SimdDup { q, esize, vd, rn };
    }

    Instruction::Unknown(word)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm;

    #[test]
    fn test_decode_bit_masks() {
        assert_eq!(decode_bit_masks(true, 1, 0x0F, 0), Some(0xFFFF));
        assert_eq!(decode_bit_masks(false, 0, 0x07, 0), Some(0xFF));
        assert_eq!(decode_bit_masks(true, 1, 0x3F, 0), None);
        // 0x5555555555555555: esize=2, s=0, r=0 -> N=0, imms=0b111100 (len=1)
        assert_eq!(
            decode_bit_masks(true, 0, 0b111100, 0),
            Some(0x5555_5555_5555_5555)
        );
    }

    #[test]
    fn test_roundtrip_alu_branch_mem() {
        let words = [
            asm::movz(Width::X64, Reg(0), 0x1234, 0),
            asm::movk(Width::X64, Reg(0), 0xABCD, 16),
            asm::add_imm(Width::X64, Reg(1), Reg(0), 64, false),
            asm::sub_imm(Width::X64, Reg(2), Reg(1), 16, true),
            asm::add_reg(Width::X64, Reg(3), Reg(1), Reg(2), ShiftType::Lsl, 1, false),
            asm::ldr_imm(Width::X64, Reg(4), Reg::SP, 16),
            asm::str_imm(Width::X64, Reg(4), Reg::SP, 24),
            asm::stp_pre(Width::X64, Reg(29), Reg::LR, Reg::SP, -16),
            asm::ldp_post(Width::X64, Reg(29), Reg::LR, Reg::SP, 16),
            asm::b(16),
            asm::bl(-8),
            asm::b_cond(Condition::Ne, -12),
            asm::cbz(Width::X64, Reg(0), 24),
            asm::ret(Reg::LR),
            asm::svc(0x80),
            asm::hlt(0),
            asm::nop(),
        ];

        for &w in &words {
            let inst = decode(w);
            assert!(
                !matches!(inst, Instruction::Unknown(_)),
                "unexpected Unknown for 0x{w:08x}: {inst:?}"
            );
        }
    }

    #[test]
    fn test_pac_and_apple_sysreg() {
        // PACIASP = 0xD503_233F, AUTIASP = 0xD503_23BF
        assert!(matches!(
            decode(0xD503_233F),
            Instruction::PacHint {
                op: PacOp::Paciasp,
                ..
            }
        ));
        assert!(matches!(
            decode(0xD503_23BF),
            Instruction::PacHint {
                op: PacOp::Autiasp,
                ..
            }
        ));
        // RETAA = 0xD65F_0BFF
        assert!(matches!(
            decode(0xD65F_0BFF),
            Instruction::PacHint {
                op: PacOp::Retaa,
                ..
            }
        ));
        // MRS x0, SCTLR_EL1 = 0xD538_1000
        assert_eq!(
            decode(0xD538_1000),
            Instruction::Mrs {
                rt: Reg(0),
                sysreg: SysReg::SCTLR_EL1
            }
        );
        // MSR SCTLR_EL1, x0 = 0xD518_1000
        assert_eq!(
            decode(0xD518_1000),
            Instruction::Msr {
                sysreg: SysReg::SCTLR_EL1,
                rt: Reg(0)
            }
        );
    }
}
