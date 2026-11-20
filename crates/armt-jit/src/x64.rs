/// x86-64 general-purpose register encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Gpr {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Gpr {
    #[inline]
    pub const fn low3(self) -> u8 {
        (self as u8) & 7
    }

    #[inline]
    pub const fn hi(self) -> u8 {
        ((self as u8) >> 3) & 1
    }
}

/// x86-64 condition code encodings (`0x0..=0xF`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum X64Cc {
    O = 0x0,
    No = 0x1,
    B = 0x2,  // C / NAe
    Ae = 0x3, // NC / NB
    E = 0x4,  // Z
    Ne = 0x5, // NZ
    Be = 0x6,
    A = 0x7,
    S = 0x8,
    Ns = 0x9,
    L = 0xC,
    Ge = 0xD,
    Le = 0xE,
    G = 0xF,
}

/// Minimal, zero-dependency x86-64 native machine code assembler for the Dynarec JIT.
pub struct X64Emitter {
    pub code: Vec<u8>,
}

impl Default for X64Emitter {
    fn default() -> Self {
        Self::new()
    }
}

impl X64Emitter {
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(512),
        }
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.code.len()
    }

    #[inline]
    pub fn emit_u8(&mut self, b: u8) {
        self.code.push(b);
    }

    #[inline]
    pub fn emit_u32(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn emit_u64(&mut self, v: u64) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    fn emit_rex(&mut self, w: bool, r: u8, x: u8, b: u8) {
        if w || r != 0 || x != 0 || b != 0 {
            self.emit_u8(0x40 | ((w as u8) << 3) | ((r & 1) << 2) | ((x & 1) << 1) | (b & 1));
        }
    }

    /// Emit REX ensuring 8-bit register encoding works cleanly for SPL/BPL/SIL/DIL too.
    fn emit_rex_byte(&mut self, r: Gpr, b: Gpr) {
        let need = r.hi() != 0 || b.hi() != 0 || (r as u8) >= 4 || (b as u8) >= 4;
        if need {
            self.emit_u8(0x40 | (r.hi() << 2) | b.hi());
        }
    }

    fn emit_modrm_rr(&mut self, reg_field: u8, rm: Gpr) {
        self.emit_u8(0xC0 | ((reg_field & 7) << 3) | rm.low3());
    }

    fn emit_modrm_disp(&mut self, reg_field: u8, base: Gpr, disp: i32) {
        let b_low = base.low3();
        let reg_low = reg_field & 7;
        if disp == 0 && b_low != 5 {
            self.emit_u8((reg_low << 3) | b_low);
            if b_low == 4 {
                self.emit_u8(0x24);
            }
        } else if (-128..=127).contains(&disp) {
            self.emit_u8(0x40 | (reg_low << 3) | b_low);
            if b_low == 4 {
                self.emit_u8(0x24);
            }
            self.emit_u8(disp as i8 as u8);
        } else {
            self.emit_u8(0x80 | (reg_low << 3) | b_low);
            if b_low == 4 {
                self.emit_u8(0x24);
            }
            self.emit_u32(disp as u32);
        }
    }

    fn emit_modrm_sib(&mut self, reg_field: u8, base: Gpr, index: Gpr, scale: u8, disp: i32) {
        assert_ne!(index, Gpr::Rsp, "RSP cannot be SIB index");
        let s = match scale {
            1 => 0u8,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => panic!("invalid scale"),
        };
        let b_low = base.low3();
        let reg_low = reg_field & 7;
        let sib = (s << 6) | (index.low3() << 3) | b_low;
        if disp == 0 && b_low != 5 {
            self.emit_u8((reg_low << 3) | 4);
            self.emit_u8(sib);
        } else if (-128..=127).contains(&disp) {
            self.emit_u8(0x40 | (reg_low << 3) | 4);
            self.emit_u8(sib);
            self.emit_u8(disp as i8 as u8);
        } else {
            self.emit_u8(0x80 | (reg_low << 3) | 4);
            self.emit_u8(sib);
            self.emit_u32(disp as u32);
        }
    }

    pub fn push(&mut self, r: Gpr) {
        self.emit_rex(false, 0, 0, r.hi());
        self.emit_u8(0x50 + r.low3());
    }

    pub fn pop(&mut self, r: Gpr) {
        self.emit_rex(false, 0, 0, r.hi());
        self.emit_u8(0x58 + r.low3());
    }

    pub fn ret(&mut self) {
        self.emit_u8(0xC3);
    }

    pub fn mov64_imm(&mut self, rd: Gpr, imm: u64) {
        if imm <= u32::MAX as u64 {
            self.mov32_imm(rd, imm as u32);
            return;
        }
        self.emit_rex(true, 0, 0, rd.hi());
        self.emit_u8(0xB8 + rd.low3());
        self.emit_u64(imm);
    }

    pub fn mov32_imm(&mut self, rd: Gpr, imm: u32) {
        self.emit_rex(false, 0, 0, rd.hi());
        self.emit_u8(0xB8 + rd.low3());
        self.emit_u32(imm);
    }

    pub fn mov64_rr(&mut self, rd: Gpr, rs: Gpr) {
        self.emit_rex(true, rs.hi(), 0, rd.hi());
        self.emit_u8(0x89);
        self.emit_modrm_rr(rs as u8, rd);
    }

    pub fn mov32_rr(&mut self, rd: Gpr, rs: Gpr) {
        self.emit_rex(false, rs.hi(), 0, rd.hi());
        self.emit_u8(0x89);
        self.emit_modrm_rr(rs as u8, rd);
    }

    pub fn mov64_rm(&mut self, rd: Gpr, base: Gpr, disp: i32) {
        self.emit_rex(true, rd.hi(), 0, base.hi());
        self.emit_u8(0x8B);
        self.emit_modrm_disp(rd as u8, base, disp);
    }

    pub fn mov64_mr(&mut self, base: Gpr, disp: i32, rs: Gpr) {
        self.emit_rex(true, rs.hi(), 0, base.hi());
        self.emit_u8(0x89);
        self.emit_modrm_disp(rs as u8, base, disp);
    }

    pub fn mov32_rm(&mut self, rd: Gpr, base: Gpr, disp: i32) {
        self.emit_rex(false, rd.hi(), 0, base.hi());
        self.emit_u8(0x8B);
        self.emit_modrm_disp(rd as u8, base, disp);
    }

    pub fn mov32_mr(&mut self, base: Gpr, disp: i32, rs: Gpr) {
        self.emit_rex(false, rs.hi(), 0, base.hi());
        self.emit_u8(0x89);
        self.emit_modrm_disp(rs as u8, base, disp);
    }

    pub fn mov64_sib_rm(&mut self, rd: Gpr, base: Gpr, index: Gpr, scale: u8, disp: i32) {
        self.emit_rex(true, rd.hi(), index.hi(), base.hi());
        self.emit_u8(0x8B);
        self.emit_modrm_sib(rd as u8, base, index, scale, disp);
    }

    pub fn mov64_sib_mr(&mut self, base: Gpr, index: Gpr, scale: u8, disp: i32, rs: Gpr) {
        self.emit_rex(true, rs.hi(), index.hi(), base.hi());
        self.emit_u8(0x89);
        self.emit_modrm_sib(rs as u8, base, index, scale, disp);
    }

    pub fn mov32_sib_rm(&mut self, rd: Gpr, base: Gpr, index: Gpr, scale: u8, disp: i32) {
        self.emit_rex(false, rd.hi(), index.hi(), base.hi());
        self.emit_u8(0x8B);
        self.emit_modrm_sib(rd as u8, base, index, scale, disp);
    }

    pub fn mov32_sib_mr(&mut self, base: Gpr, index: Gpr, scale: u8, disp: i32, rs: Gpr) {
        self.emit_rex(false, rs.hi(), index.hi(), base.hi());
        self.emit_u8(0x89);
        self.emit_modrm_sib(rs as u8, base, index, scale, disp);
    }

    pub fn movzx8_sib_rm(&mut self, rd: Gpr, base: Gpr, index: Gpr) {
        self.emit_rex(false, rd.hi(), index.hi(), base.hi());
        self.emit_u8(0x0F);
        self.emit_u8(0xB6);
        self.emit_modrm_sib(rd as u8, base, index, 1, 0);
    }

    pub fn mov8_sib_mr(&mut self, base: Gpr, index: Gpr, rs: Gpr) {
        let need = rs.hi() != 0 || index.hi() != 0 || base.hi() != 0 || (rs as u8) >= 4;
        if need {
            self.emit_u8(0x40 | (rs.hi() << 2) | (index.hi() << 1) | base.hi());
        }
        self.emit_u8(0x88);
        self.emit_modrm_sib(rs as u8, base, index, 1, 0);
    }

    pub fn alu64_rr(&mut self, opcode: u8, rd: Gpr, rs: Gpr) {
        // 0x01=ADD, 0x29=SUB, 0x21=AND, 0x09=OR, 0x31=XOR, 0x39=CMP, 0x85=TEST
        self.emit_rex(true, rs.hi(), 0, rd.hi());
        self.emit_u8(opcode);
        self.emit_modrm_rr(rs as u8, rd);
    }

    pub fn alu32_rr(&mut self, opcode: u8, rd: Gpr, rs: Gpr) {
        self.emit_rex(false, rs.hi(), 0, rd.hi());
        self.emit_u8(opcode);
        self.emit_modrm_rr(rs as u8, rd);
    }

    pub fn alu64_ri(&mut self, subop: u8, rd: Gpr, imm: i32) {
        // subop: 0=ADD, 4=AND, 5=SUB, 6=XOR, 7=CMP, 1=OR
        self.emit_rex(true, 0, 0, rd.hi());
        if (-128..=127).contains(&imm) {
            self.emit_u8(0x83);
            self.emit_modrm_rr(subop, rd);
            self.emit_u8(imm as i8 as u8);
        } else {
            self.emit_u8(0x81);
            self.emit_modrm_rr(subop, rd);
            self.emit_u32(imm as u32);
        }
    }

    pub fn alu32_ri(&mut self, subop: u8, rd: Gpr, imm: i32) {
        self.emit_rex(false, 0, 0, rd.hi());
        if (-128..=127).contains(&imm) {
            self.emit_u8(0x83);
            self.emit_modrm_rr(subop, rd);
            self.emit_u8(imm as i8 as u8);
        } else {
            self.emit_u8(0x81);
            self.emit_modrm_rr(subop, rd);
            self.emit_u32(imm as u32);
        }
    }

    pub fn add64_mr_imm(&mut self, base: Gpr, disp: i32, imm: i32) {
        self.emit_rex(true, 0, 0, base.hi());
        if (-128..=127).contains(&imm) {
            self.emit_u8(0x83);
            self.emit_modrm_disp(0, base, disp);
            self.emit_u8(imm as i8 as u8);
        } else {
            self.emit_u8(0x81);
            self.emit_modrm_disp(0, base, disp);
            self.emit_u32(imm as u32);
        }
    }

    pub fn not64(&mut self, rd: Gpr) {
        self.emit_rex(true, 0, 0, rd.hi());
        self.emit_u8(0xF7);
        self.emit_modrm_rr(2, rd);
    }

    pub fn not32(&mut self, rd: Gpr) {
        self.emit_rex(false, 0, 0, rd.hi());
        self.emit_u8(0xF7);
        self.emit_modrm_rr(2, rd);
    }

    pub fn neg64(&mut self, rd: Gpr) {
        self.emit_rex(true, 0, 0, rd.hi());
        self.emit_u8(0xF7);
        self.emit_modrm_rr(3, rd);
    }

    pub fn neg32(&mut self, rd: Gpr) {
        self.emit_rex(false, 0, 0, rd.hi());
        self.emit_u8(0xF7);
        self.emit_modrm_rr(3, rd);
    }

    /// Shift `rd` by `imm` (`subop`: 4=SHL, 5=SHR, 7=SAR, 1=ROR).
    pub fn shift64_imm(&mut self, subop: u8, rd: Gpr, imm: u8) {
        self.emit_rex(true, 0, 0, rd.hi());
        self.emit_u8(0xC1);
        self.emit_modrm_rr(subop, rd);
        self.emit_u8(imm & 63);
    }

    pub fn shift32_imm(&mut self, subop: u8, rd: Gpr, imm: u8) {
        self.emit_rex(false, 0, 0, rd.hi());
        self.emit_u8(0xC1);
        self.emit_modrm_rr(subop, rd);
        self.emit_u8(imm & 31);
    }

    pub fn shift64_cl(&mut self, subop: u8, rd: Gpr) {
        self.emit_rex(true, 0, 0, rd.hi());
        self.emit_u8(0xD3);
        self.emit_modrm_rr(subop, rd);
    }

    pub fn shift32_cl(&mut self, subop: u8, rd: Gpr) {
        self.emit_rex(false, 0, 0, rd.hi());
        self.emit_u8(0xD3);
        self.emit_modrm_rr(subop, rd);
    }

    pub fn imul64_rr(&mut self, rd: Gpr, rs: Gpr) {
        self.emit_rex(true, rd.hi(), 0, rs.hi());
        self.emit_u8(0x0F);
        self.emit_u8(0xAF);
        self.emit_modrm_rr(rd as u8, rs);
    }

    pub fn imul32_rr(&mut self, rd: Gpr, rs: Gpr) {
        self.emit_rex(false, rd.hi(), 0, rs.hi());
        self.emit_u8(0x0F);
        self.emit_u8(0xAF);
        self.emit_modrm_rr(rd as u8, rs);
    }

    pub fn cqo(&mut self) {
        self.emit_u8(0x48);
        self.emit_u8(0x99);
    }

    pub fn cdq(&mut self) {
        self.emit_u8(0x99);
    }

    pub fn div64(&mut self, rm: Gpr, signed: bool) {
        self.emit_rex(true, 0, 0, rm.hi());
        self.emit_u8(0xF7);
        self.emit_modrm_rr(if signed { 7 } else { 6 }, rm);
    }

    pub fn div32(&mut self, rm: Gpr, signed: bool) {
        self.emit_rex(false, 0, 0, rm.hi());
        self.emit_u8(0xF7);
        self.emit_modrm_rr(if signed { 7 } else { 6 }, rm);
    }

    pub fn setcc(&mut self, cc: X64Cc, rd: Gpr) {
        self.emit_rex_byte(Gpr::Rax, rd);
        self.emit_u8(0x0F);
        self.emit_u8(0x90 + (cc as u8));
        self.emit_modrm_rr(0, rd);
    }

    pub fn movzx_r32_r8(&mut self, rd: Gpr, rs: Gpr) {
        self.emit_rex_byte(rd, rs);
        self.emit_u8(0x0F);
        self.emit_u8(0xB6);
        self.emit_modrm_rr(rd as u8, rs);
    }

    pub fn cmovcc64(&mut self, cc: X64Cc, rd: Gpr, rs: Gpr) {
        self.emit_rex(true, rd.hi(), 0, rs.hi());
        self.emit_u8(0x0F);
        self.emit_u8(0x40 + (cc as u8));
        self.emit_modrm_rr(rd as u8, rs);
    }

    pub fn call_r(&mut self, r: Gpr) {
        self.emit_rex(false, 0, 0, r.hi());
        self.emit_u8(0xFF);
        self.emit_modrm_rr(2, r);
    }

    pub fn jmp_rel32_placeholder(&mut self) -> usize {
        self.emit_u8(0xE9);
        let patch_at = self.pos();
        self.emit_u32(0);
        patch_at
    }

    pub fn jcc_rel32_placeholder(&mut self, cc: X64Cc) -> usize {
        self.emit_u8(0x0F);
        self.emit_u8(0x80 + (cc as u8));
        let patch_at = self.pos();
        self.emit_u32(0);
        patch_at
    }

    pub fn patch_rel32(&mut self, patch_at: usize, target_pos: usize) {
        let rel = (target_pos as isize) - ((patch_at + 4) as isize);
        let bytes = (rel as i32).to_le_bytes();
        self.code[patch_at..patch_at + 4].copy_from_slice(&bytes);
    }
}
