use armt_core::isa::{Condition, Reg, ShiftType, SysReg, Width};

#[inline]
const fn sf_bit(w: Width) -> u32 {
    match w {
        Width::W32 => 0,
        Width::X64 => 1,
    }
}

pub const fn nop() -> u32 {
    0xD503_201F
}

pub const fn ret(rn: Reg) -> u32 {
    0xD65F_0000 | (((rn.0 & 0x1F) as u32) << 5)
}

pub const fn svc(imm16: u16) -> u32 {
    0xD400_0001 | ((imm16 as u32) << 5)
}

pub const fn hlt(imm16: u16) -> u32 {
    0xD440_0000 | ((imm16 as u32) << 5)
}

pub const fn brk(imm16: u16) -> u32 {
    0xD420_0000 | ((imm16 as u32) << 5)
}

pub const fn movz(w: Width, rd: Reg, imm16: u16, shift: u8) -> u32 {
    let hw = ((shift / 16) & 3) as u32;
    (sf_bit(w) << 31)
        | (0b10 << 29)
        | (0b100101 << 23)
        | (hw << 21)
        | ((imm16 as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn movn(w: Width, rd: Reg, imm16: u16, shift: u8) -> u32 {
    let hw = ((shift / 16) & 3) as u32;
    (sf_bit(w) << 31)
        | (0b00 << 29)
        | (0b100101 << 23)
        | (hw << 21)
        | ((imm16 as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn movk(w: Width, rd: Reg, imm16: u16, shift: u8) -> u32 {
    let hw = ((shift / 16) & 3) as u32;
    (sf_bit(w) << 31)
        | (0b11 << 29)
        | (0b100101 << 23)
        | (hw << 21)
        | ((imm16 as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

/// Emit `MOVZ` + up to 3 `MOVK` instructions to load any 64-bit constant into `rd`.
pub fn load_u64(rd: Reg, val: u64) -> Vec<u32> {
    let mut out = vec![movz(Width::X64, rd, (val & 0xFFFF) as u16, 0)];
    for chunk in 1..4u8 {
        let part = ((val >> (chunk * 16)) & 0xFFFF) as u16;
        if part != 0 {
            out.push(movk(Width::X64, rd, part, chunk * 16));
        }
    }
    out
}

pub const fn add_imm(w: Width, rd: Reg, rn: Reg, imm12: u16, set_flags: bool) -> u32 {
    (sf_bit(w) << 31)
        | ((set_flags as u32) << 29)
        | (0b100010 << 23)
        | (((imm12 & 0xFFF) as u32) << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn sub_imm(w: Width, rd: Reg, rn: Reg, imm12: u16, set_flags: bool) -> u32 {
    (sf_bit(w) << 31)
        | (1 << 30)
        | ((set_flags as u32) << 29)
        | (0b100010 << 23)
        | (((imm12 & 0xFFF) as u32) << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn cmp_imm(w: Width, rn: Reg, imm12: u16) -> u32 {
    sub_imm(w, Reg::XZR, rn, imm12, true)
}

pub const fn add_reg(
    w: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    shift: ShiftType,
    amount: u8,
    set_flags: bool,
) -> u32 {
    (sf_bit(w) << 31)
        | ((set_flags as u32) << 29)
        | (0b01011 << 24)
        | (((shift as u32) & 3) << 22)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (((amount & 0x3F) as u32) << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn sub_reg(
    w: Width,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    shift: ShiftType,
    amount: u8,
    set_flags: bool,
) -> u32 {
    (sf_bit(w) << 31)
        | (1 << 30)
        | ((set_flags as u32) << 29)
        | (0b01011 << 24)
        | (((shift as u32) & 3) << 22)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (((amount & 0x3F) as u32) << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn orr_reg(w: Width, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf_bit(w) << 31)
        | (0b01 << 29)
        | (0b01010 << 24)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn mov_reg(w: Width, rd: Reg, rm: Reg) -> u32 {
    orr_reg(w, rd, Reg::XZR, rm)
}

pub const fn and_reg(w: Width, rd: Reg, rn: Reg, rm: Reg, set_flags: bool) -> u32 {
    let opc = if set_flags { 0b11 } else { 0b00 };
    (sf_bit(w) << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn eor_reg(w: Width, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf_bit(w) << 31)
        | (0b10 << 29)
        | (0b01010 << 24)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn mul(w: Width, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf_bit(w) << 31)
        | (0b0011011000 << 21)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (0b011111 << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn lslv(w: Width, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf_bit(w) << 31)
        | (0b0011010110 << 21)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (0b001000 << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn lsrv(w: Width, rd: Reg, rn: Reg, rm: Reg) -> u32 {
    (sf_bit(w) << 31)
        | (0b0011010110 << 21)
        | (((rm.0 & 0x1F) as u32) << 16)
        | (0b001001 << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rd.0 & 0x1F) as u32)
}

pub const fn b(byte_offset: i64) -> u32 {
    let imm26 = ((byte_offset >> 2) as u32) & 0x03FF_FFFF;
    0x1400_0000 | imm26
}

pub const fn bl(byte_offset: i64) -> u32 {
    let imm26 = ((byte_offset >> 2) as u32) & 0x03FF_FFFF;
    0x9400_0000 | imm26
}

pub const fn b_cond(cond: Condition, byte_offset: i64) -> u32 {
    let imm19 = ((byte_offset >> 2) as u32) & 0x7FFFF;
    0x5400_0000 | (imm19 << 5) | ((cond as u32) & 0xF)
}

pub const fn cbz(w: Width, rt: Reg, byte_offset: i64) -> u32 {
    let imm19 = ((byte_offset >> 2) as u32) & 0x7FFFF;
    (sf_bit(w) << 31) | (0b0110100 << 24) | (imm19 << 5) | ((rt.0 & 0x1F) as u32)
}

pub const fn cbnz(w: Width, rt: Reg, byte_offset: i64) -> u32 {
    let imm19 = ((byte_offset >> 2) as u32) & 0x7FFFF;
    (sf_bit(w) << 31) | (0b0110101 << 24) | (imm19 << 5) | ((rt.0 & 0x1F) as u32)
}

pub const fn ldr_imm(w: Width, rt: Reg, rn: Reg, byte_offset: u32) -> u32 {
    let (size, shift) = match w {
        Width::W32 => (2u32, 2u32),
        Width::X64 => (3u32, 3u32),
    };
    let imm12 = (byte_offset >> shift) & 0xFFF;
    (size << 30)
        | (0b11100101 << 22)
        | (imm12 << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rt.0 & 0x1F) as u32)
}

pub const fn str_imm(w: Width, rt: Reg, rn: Reg, byte_offset: u32) -> u32 {
    let (size, shift) = match w {
        Width::W32 => (2u32, 2u32),
        Width::X64 => (3u32, 3u32),
    };
    let imm12 = (byte_offset >> shift) & 0xFFF;
    (size << 30)
        | (0b11100100 << 22)
        | (imm12 << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rt.0 & 0x1F) as u32)
}

pub const fn strb_imm(rt: Reg, rn: Reg, byte_offset: u32) -> u32 {
    let imm12 = byte_offset & 0xFFF;
    (0b00 << 30)
        | (0b11100100 << 22)
        | (imm12 << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rt.0 & 0x1F) as u32)
}

pub const fn ldrb_imm(rt: Reg, rn: Reg, byte_offset: u32) -> u32 {
    let imm12 = byte_offset & 0xFFF;
    (0b00 << 30)
        | (0b11100101 << 22)
        | (imm12 << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rt.0 & 0x1F) as u32)
}

pub const fn stp_pre(w: Width, rt: Reg, rt2: Reg, rn: Reg, byte_offset: i32) -> u32 {
    let (opc, scale) = match w {
        Width::W32 => (0b00u32, 4),
        Width::X64 => (0b10u32, 8),
    };
    let imm7 = ((byte_offset / scale) as u32) & 0x7F;
    (opc << 30)
        | (0b10100110 << 22)
        | (imm7 << 15)
        | (((rt2.0 & 0x1F) as u32) << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rt.0 & 0x1F) as u32)
}

pub const fn ldp_post(w: Width, rt: Reg, rt2: Reg, rn: Reg, byte_offset: i32) -> u32 {
    let (opc, scale) = match w {
        Width::W32 => (0b00u32, 4),
        Width::X64 => (0b10u32, 8),
    };
    let imm7 = ((byte_offset / scale) as u32) & 0x7F;
    (opc << 30)
        | (0b10100011 << 22)
        | (imm7 << 15)
        | (((rt2.0 & 0x1F) as u32) << 10)
        | (((rn.0 & 0x1F) as u32) << 5)
        | ((rt.0 & 0x1F) as u32)
}

pub const fn mrs(rt: Reg, sysreg: SysReg) -> u32 {
    let (op0, op1, crn, crm, op2) = sysreg.fields();
    let o0 = ((op0.saturating_sub(2)) & 1) as u32;
    0xD530_0000
        | (o0 << 19)
        | (((op1 as u32) & 7) << 16)
        | (((crn as u32) & 0xF) << 12)
        | (((crm as u32) & 0xF) << 8)
        | (((op2 as u32) & 7) << 5)
        | ((rt.0 & 0x1F) as u32)
}

pub const fn msr(sysreg: SysReg, rt: Reg) -> u32 {
    let (op0, op1, crn, crm, op2) = sysreg.fields();
    let o0 = ((op0.saturating_sub(2)) & 1) as u32;
    0xD510_0000
        | (o0 << 19)
        | (((op1 as u32) & 7) << 16)
        | (((crn as u32) & 0xF) << 12)
        | (((crm as u32) & 0xF) << 8)
        | (((op2 as u32) & 7) << 5)
        | ((rt.0 & 0x1F) as u32)
}

/// Encode a slice of 32-bit AArch64 words into little-endian bytes.
pub fn to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
