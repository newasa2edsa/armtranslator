use armt_core::isa::{AddrMode, BarrierOp, Instruction, Reg, Width};

fn reg_name(r: Reg, w: Width, sp: bool) -> String {
    let idx = r.index();
    if idx == 31 {
        return match (sp, w) {
            (true, Width::X64) => "sp".into(),
            (true, Width::W32) => "wsp".into(),
            (false, Width::X64) => "xzr".into(),
            (false, Width::W32) => "wzr".into(),
        };
    }
    match w {
        Width::X64 => format!("x{idx}"),
        Width::W32 => format!("w{idx}"),
    }
}

fn fmt_addr(rn: Reg, mode: AddrMode) -> String {
    let base = reg_name(rn, Width::X64, true);
    match mode {
        AddrMode::UnsignedOffset(0) => format!("[{base}]"),
        AddrMode::UnsignedOffset(off) => format!("[{base}, #{off}]"),
        AddrMode::PreIndex(off) => format!("[{base}, #{off}]!"),
        AddrMode::PostIndex(off) => format!("[{base}], #{off}"),
        AddrMode::RegOffset { rm, extend, shift } => {
            let rm_s = reg_name(rm, Width::X64, false);
            if shift > 0 {
                format!("[{base}, {rm_s}, {} #{shift}]", extend.name())
            } else {
                format!("[{base}, {rm_s}, {}]", extend.name())
            }
        }
    }
}

/// Human-readable AArch64 disassembly for diagnostics & trace logs.
pub fn disasm(inst: &Instruction, pc: u64) -> String {
    match *inst {
        Instruction::AddImm {
            w,
            rd,
            rn,
            imm,
            set_flags,
        } => {
            let op = if set_flags { "adds" } else { "add" };
            format!(
                "{op} {}, {}, #0x{imm:x}",
                reg_name(rd, w, !set_flags),
                reg_name(rn, w, true)
            )
        }
        Instruction::SubImm {
            w,
            rd,
            rn,
            imm,
            set_flags,
        } => {
            let op = if set_flags { "subs" } else { "sub" };
            format!(
                "{op} {}, {}, #0x{imm:x}",
                reg_name(rd, w, !set_flags),
                reg_name(rn, w, true)
            )
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
            let op = if set_flags { "adds" } else { "add" };
            format!(
                "{op} {}, {}, {}, {} #{amount}",
                reg_name(rd, w, false),
                reg_name(rn, w, false),
                reg_name(rm, w, false),
                shift.name()
            )
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
            let op = if set_flags { "subs" } else { "sub" };
            format!(
                "{op} {}, {}, {}, {} #{amount}",
                reg_name(rd, w, false),
                reg_name(rn, w, false),
                reg_name(rm, w, false),
                shift.name()
            )
        }
        Instruction::Movz {
            w,
            rd,
            imm16,
            shift,
        } => {
            format!(
                "movz {}, #0x{imm16:x}, lsl #{shift}",
                reg_name(rd, w, false)
            )
        }
        Instruction::Movn {
            w,
            rd,
            imm16,
            shift,
        } => {
            format!(
                "movn {}, #0x{imm16:x}, lsl #{shift}",
                reg_name(rd, w, false)
            )
        }
        Instruction::Movk {
            w,
            rd,
            imm16,
            shift,
        } => {
            format!(
                "movk {}, #0x{imm16:x}, lsl #{shift}",
                reg_name(rd, w, false)
            )
        }
        Instruction::Adr { rd, offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            format!("adr {}, 0x{target:x}", reg_name(rd, Width::X64, false))
        }
        Instruction::Adrp { rd, offset } => {
            let base = (pc & !0xFFF) as i64;
            let target = base.wrapping_add(offset) as u64;
            format!("adrp {}, 0x{target:x}", reg_name(rd, Width::X64, false))
        }
        Instruction::B { offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            format!("b 0x{target:x}")
        }
        Instruction::Bl { offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            format!("bl 0x{target:x}")
        }
        Instruction::BCond { cond, offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            format!("b.{} 0x{target:x}", cond.name())
        }
        Instruction::Cbz { w, rt, offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            format!("cbz {}, 0x{target:x}", reg_name(rt, w, false))
        }
        Instruction::Cbnz { w, rt, offset } => {
            let target = (pc as i64).wrapping_add(offset) as u64;
            format!("cbnz {}, 0x{target:x}", reg_name(rt, w, false))
        }
        Instruction::Br { rn } => format!("br {}", reg_name(rn, Width::X64, false)),
        Instruction::Blr { rn } => format!("blr {}", reg_name(rn, Width::X64, false)),
        Instruction::Ret { rn } => format!("ret {}", reg_name(rn, Width::X64, false)),
        Instruction::Ldr { w, rt, rn, mode } => {
            format!("ldr {}, {}", reg_name(rt, w, false), fmt_addr(rn, mode))
        }
        Instruction::Str { w, rt, rn, mode } => {
            format!("str {}, {}", reg_name(rt, w, false), fmt_addr(rn, mode))
        }
        Instruction::Ldrb { rt, rn, mode } => {
            format!(
                "ldrb {}, {}",
                reg_name(rt, Width::W32, false),
                fmt_addr(rn, mode)
            )
        }
        Instruction::Strb { rt, rn, mode } => {
            format!(
                "strb {}, {}",
                reg_name(rt, Width::W32, false),
                fmt_addr(rn, mode)
            )
        }
        Instruction::Ldp {
            w,
            rt,
            rt2,
            rn,
            mode,
        } => format!(
            "ldp {}, {}, {}",
            reg_name(rt, w, false),
            reg_name(rt2, w, false),
            fmt_addr(rn, mode)
        ),
        Instruction::Stp {
            w,
            rt,
            rt2,
            rn,
            mode,
        } => format!(
            "stp {}, {}, {}",
            reg_name(rt, w, false),
            reg_name(rt2, w, false),
            fmt_addr(rn, mode)
        ),
        Instruction::Ldxp {
            is_64,
            acquire,
            rt,
            rt2,
            rn,
        } => {
            let w = if is_64 { Width::X64 } else { Width::W32 };
            let op = if acquire { "ldaxp" } else { "ldxp" };
            format!(
                "{op} {}, {}, [{}]",
                reg_name(rt, w, false),
                reg_name(rt2, w, false),
                reg_name(rn, Width::X64, true)
            )
        }
        Instruction::Stxp {
            is_64,
            release,
            rs,
            rt,
            rt2,
            rn,
        } => {
            let w = if is_64 { Width::X64 } else { Width::W32 };
            let op = if release { "stlxp" } else { "stxp" };
            format!(
                "{op} {}, {}, {}, [{}]",
                reg_name(rs, Width::W32, false),
                reg_name(rt, w, false),
                reg_name(rt2, w, false),
                reg_name(rn, Width::X64, true)
            )
        }
        Instruction::Mrs { rt, sysreg } => {
            format!("mrs {}, {sysreg}", reg_name(rt, Width::X64, false))
        }
        Instruction::Msr { sysreg, rt } => {
            format!("msr {sysreg}, {}", reg_name(rt, Width::X64, false))
        }
        Instruction::Svc { imm } => format!("svc #0x{imm:x}"),
        Instruction::Hvc { imm } => format!("hvc #0x{imm:x}"),
        Instruction::Brk { imm } => format!("brk #0x{imm:x}"),
        Instruction::Hlt { imm } => format!("hlt #0x{imm:x}"),
        Instruction::Eret => "eret".into(),
        Instruction::Nop => "nop".into(),
        Instruction::Barrier { op } => match op {
            BarrierOp::Dsb(v) => format!("dsb #{v}"),
            BarrierOp::Dmb(v) => format!("dmb #{v}"),
            BarrierOp::Isb(v) => format!("isb #{v}"),
        },
        Instruction::PacHint { op, .. } => format!("{op:?}").to_lowercase(),
        Instruction::Unknown(word) => format!(".word 0x{word:08x}"),
        other => format!("{other:?}"),
    }
}
