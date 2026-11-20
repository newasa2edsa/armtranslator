use std::fmt;

/// General-purpose AArch64 register index (0..=30, 31 = XZR or SP depending on instruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u8);

impl Reg {
    pub const XZR: Reg = Reg(31);
    pub const SP: Reg = Reg(31);
    pub const LR: Reg = Reg(30);

    #[inline]
    pub const fn index(self) -> usize {
        (self.0 & 0x1F) as usize
    }

    #[inline]
    pub const fn is_zr_or_sp(self) -> bool {
        (self.0 & 0x1F) == 31
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0 & 0x1F)
    }
}

/// Register / operation bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Width {
    W32,
    X64,
}

impl Width {
    #[inline]
    pub const fn bits(self) -> u32 {
        match self {
            Width::W32 => 32,
            Width::X64 => 64,
        }
    }

    #[inline]
    pub const fn mask(self) -> u64 {
        match self {
            Width::W32 => 0xFFFF_FFFF,
            Width::X64 => u64::MAX,
        }
    }
}

/// AArch64 condition codes (encoding 0..=15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Condition {
    Eq = 0,
    Ne = 1,
    Cs = 2,
    Cc = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
    Al = 14,
    Nv = 15,
}

impl Condition {
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val & 0xF {
            0 => Condition::Eq,
            1 => Condition::Ne,
            2 => Condition::Cs,
            3 => Condition::Cc,
            4 => Condition::Mi,
            5 => Condition::Pl,
            6 => Condition::Vs,
            7 => Condition::Vc,
            8 => Condition::Hi,
            9 => Condition::Ls,
            10 => Condition::Ge,
            11 => Condition::Lt,
            12 => Condition::Gt,
            13 => Condition::Le,
            14 => Condition::Al,
            _ => Condition::Nv,
        }
    }

    #[inline]
    pub const fn invert(self) -> Self {
        Condition::from_u8((self as u8) ^ 1)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Condition::Eq => "eq",
            Condition::Ne => "ne",
            Condition::Cs => "cs",
            Condition::Cc => "cc",
            Condition::Mi => "mi",
            Condition::Pl => "pl",
            Condition::Vs => "vs",
            Condition::Vc => "vc",
            Condition::Hi => "hi",
            Condition::Ls => "ls",
            Condition::Ge => "ge",
            Condition::Lt => "lt",
            Condition::Gt => "gt",
            Condition::Le => "le",
            Condition::Al => "al",
            Condition::Nv => "nv",
        }
    }
}

/// Shift operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShiftType {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

impl ShiftType {
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val & 0x3 {
            0 => ShiftType::Lsl,
            1 => ShiftType::Lsr,
            2 => ShiftType::Asr,
            _ => ShiftType::Ror,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            ShiftType::Lsl => "lsl",
            ShiftType::Lsr => "lsr",
            ShiftType::Asr => "asr",
            ShiftType::Ror => "ror",
        }
    }
}

/// Extend operation type for extended-register ALU and register-offset addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExtendType {
    Uxtb = 0,
    Uxth = 1,
    Uxtw = 2,
    Uxtx = 3,
    Sxtb = 4,
    Sxth = 5,
    Sxtw = 6,
    Sxtx = 7,
}

impl ExtendType {
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val & 0x7 {
            0 => ExtendType::Uxtb,
            1 => ExtendType::Uxth,
            2 => ExtendType::Uxtw,
            3 => ExtendType::Uxtx,
            4 => ExtendType::Sxtb,
            5 => ExtendType::Sxth,
            6 => ExtendType::Sxtw,
            _ => ExtendType::Sxtx,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            ExtendType::Uxtb => "uxtb",
            ExtendType::Uxth => "uxth",
            ExtendType::Uxtw => "uxtw",
            ExtendType::Uxtx => "uxtx",
            ExtendType::Sxtb => "sxtb",
            ExtendType::Sxth => "sxth",
            ExtendType::Sxtw => "sxtw",
            ExtendType::Sxtx => "sxtx",
        }
    }
}

/// Addressing modes for load/store instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddrMode {
    UnsignedOffset(i64),
    PreIndex(i64),
    PostIndex(i64),
    RegOffset {
        rm: Reg,
        extend: ExtendType,
        shift: u8,
    },
}

/// Memory/instruction barrier operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BarrierOp {
    Dsb(u8),
    Dmb(u8),
    Isb(u8),
}

/// Pointer Authentication (ARMv8.3 PAC) hint/instruction variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PacOp {
    Pacia,
    Pacib,
    Pacda,
    Pacdb,
    Autia,
    Autib,
    Autda,
    Autdb,
    Xpaci,
    Xpacd,
    Paciasp,
    Autiasp,
    Retaa,
    Retab,
    Blraa,
    Blrab,
    Braa,
    Brab,
}

/// Packed 16-bit ARMv8 System Register encoding: `(op0[15:14], op1[13:11], crn[10:7], crm[6:3], op2[2:0])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SysReg(pub u16);

impl SysReg {
    #[inline]
    pub const fn from_fields(op0: u8, op1: u8, crn: u8, crm: u8, op2: u8) -> Self {
        let val = (((op0 as u16) & 0x3) << 14)
            | (((op1 as u16) & 0x7) << 11)
            | (((crn as u16) & 0xF) << 7)
            | (((crm as u16) & 0xF) << 3)
            | ((op2 as u16) & 0x7);
        Self(val)
    }

    #[inline]
    pub const fn fields(self) -> (u8, u8, u8, u8, u8) {
        let op0 = ((self.0 >> 14) & 0x3) as u8;
        let op1 = ((self.0 >> 11) & 0x7) as u8;
        let crn = ((self.0 >> 7) & 0xF) as u8;
        let crm = ((self.0 >> 3) & 0xF) as u8;
        let op2 = (self.0 & 0x7) as u8;
        (op0, op1, crn, crm, op2)
    }

    #[inline]
    pub const fn from_mrs_word(word: u32) -> Self {
        let o0 = ((word >> 19) & 1) as u8;
        let op1 = ((word >> 16) & 0x7) as u8;
        let crn = ((word >> 12) & 0xF) as u8;
        let crm = ((word >> 8) & 0xF) as u8;
        let op2 = ((word >> 5) & 0x7) as u8;
        Self::from_fields(2 + o0, op1, crn, crm, op2)
    }

    #[inline]
    pub const fn is_apple_impl_defined(self) -> bool {
        let (op0, _, crn, _, _) = self.fields();
        op0 == 3 && (crn == 11 || crn == 15)
    }

    pub const MIDR_EL1: SysReg = SysReg::from_fields(3, 0, 0, 0, 0);
    pub const MPIDR_EL1: SysReg = SysReg::from_fields(3, 0, 0, 0, 5);
    pub const CTR_EL0: SysReg = SysReg::from_fields(3, 3, 0, 0, 1);
    pub const DCZID_EL0: SysReg = SysReg::from_fields(3, 3, 0, 0, 7);
    pub const ID_AA64PFR0_EL1: SysReg = SysReg::from_fields(3, 0, 0, 4, 0);
    pub const ID_AA64ISAR0_EL1: SysReg = SysReg::from_fields(3, 0, 0, 6, 0);
    pub const ID_AA64ISAR1_EL1: SysReg = SysReg::from_fields(3, 0, 0, 6, 1);
    pub const ID_AA64MMFR0_EL1: SysReg = SysReg::from_fields(3, 0, 0, 7, 0);
    pub const SCTLR_EL1: SysReg = SysReg::from_fields(3, 0, 1, 0, 0);
    pub const CPACR_EL1: SysReg = SysReg::from_fields(3, 0, 1, 0, 2);
    pub const TTBR0_EL1: SysReg = SysReg::from_fields(3, 0, 2, 0, 0);
    pub const TTBR1_EL1: SysReg = SysReg::from_fields(3, 0, 2, 0, 1);
    pub const TCR_EL1: SysReg = SysReg::from_fields(3, 0, 2, 0, 2);
    pub const SPSR_EL1: SysReg = SysReg::from_fields(3, 0, 4, 0, 0);
    pub const ELR_EL1: SysReg = SysReg::from_fields(3, 0, 4, 0, 1);
    pub const SP_EL0: SysReg = SysReg::from_fields(3, 0, 4, 1, 0);
    pub const NZCV: SysReg = SysReg::from_fields(3, 3, 4, 2, 0);
    pub const DAIF: SysReg = SysReg::from_fields(3, 3, 4, 2, 1);
    pub const CURRENTEL: SysReg = SysReg::from_fields(3, 0, 4, 2, 2);
    pub const ICC_PMR_EL1: SysReg = SysReg::from_fields(3, 0, 4, 6, 0);
    pub const ESR_EL1: SysReg = SysReg::from_fields(3, 0, 5, 2, 0);
    pub const FAR_EL1: SysReg = SysReg::from_fields(3, 0, 6, 0, 0);
    pub const MAIR_EL1: SysReg = SysReg::from_fields(3, 0, 10, 2, 0);
    pub const VBAR_EL1: SysReg = SysReg::from_fields(3, 0, 12, 0, 0);
    pub const ICC_IAR1_EL1: SysReg = SysReg::from_fields(3, 0, 12, 12, 0);
    pub const ICC_EOIR1_EL1: SysReg = SysReg::from_fields(3, 0, 12, 12, 1);
    pub const ICC_BPR1_EL1: SysReg = SysReg::from_fields(3, 0, 12, 12, 3);
    pub const ICC_SRE_EL1: SysReg = SysReg::from_fields(3, 0, 12, 12, 5);
    pub const ICC_IGRPEN1_EL1: SysReg = SysReg::from_fields(3, 0, 12, 12, 7);
    pub const TPIDR_EL0: SysReg = SysReg::from_fields(3, 3, 13, 0, 2);
    pub const TPIDRRO_EL0: SysReg = SysReg::from_fields(3, 3, 13, 0, 3);
    pub const TPIDR_EL1: SysReg = SysReg::from_fields(3, 0, 13, 0, 4);
    pub const CNTFRQ_EL0: SysReg = SysReg::from_fields(3, 3, 14, 0, 0);
    pub const CNTPCT_EL0: SysReg = SysReg::from_fields(3, 3, 14, 0, 1);
    pub const CNTVCT_EL0: SysReg = SysReg::from_fields(3, 3, 14, 0, 2);
    pub const CNTP_TVAL_EL0: SysReg = SysReg::from_fields(3, 3, 14, 2, 0);
    pub const CNTP_CTL_EL0: SysReg = SysReg::from_fields(3, 3, 14, 2, 1);
    pub const CNTP_CVAL_EL0: SysReg = SysReg::from_fields(3, 3, 14, 2, 2);
    pub const CNTV_TVAL_EL0: SysReg = SysReg::from_fields(3, 3, 14, 3, 0);
    pub const CNTV_CTL_EL0: SysReg = SysReg::from_fields(3, 3, 14, 3, 1);
    pub const CNTV_CVAL_EL0: SysReg = SysReg::from_fields(3, 3, 14, 3, 2);

    pub fn name(self) -> String {
        match self {
            Self::MIDR_EL1 => "midr_el1".into(),
            Self::MPIDR_EL1 => "mpidr_el1".into(),
            Self::CTR_EL0 => "ctr_el0".into(),
            Self::DCZID_EL0 => "dczid_el0".into(),
            Self::ID_AA64PFR0_EL1 => "id_aa64pfr0_el1".into(),
            Self::ID_AA64ISAR0_EL1 => "id_aa64isar0_el1".into(),
            Self::ID_AA64ISAR1_EL1 => "id_aa64isar1_el1".into(),
            Self::ID_AA64MMFR0_EL1 => "id_aa64mmfr0_el1".into(),
            Self::SCTLR_EL1 => "sctlr_el1".into(),
            Self::CPACR_EL1 => "cpacr_el1".into(),
            Self::TTBR0_EL1 => "ttbr0_el1".into(),
            Self::TTBR1_EL1 => "ttbr1_el1".into(),
            Self::TCR_EL1 => "tcr_el1".into(),
            Self::SPSR_EL1 => "spsr_el1".into(),
            Self::ELR_EL1 => "elr_el1".into(),
            Self::SP_EL0 => "sp_el0".into(),
            Self::NZCV => "nzcv".into(),
            Self::DAIF => "daif".into(),
            Self::CURRENTEL => "currentel".into(),
            Self::ICC_PMR_EL1 => "icc_pmr_el1".into(),
            Self::ESR_EL1 => "esr_el1".into(),
            Self::FAR_EL1 => "far_el1".into(),
            Self::MAIR_EL1 => "mair_el1".into(),
            Self::VBAR_EL1 => "vbar_el1".into(),
            Self::ICC_IAR1_EL1 => "icc_iar1_el1".into(),
            Self::ICC_EOIR1_EL1 => "icc_eoir1_el1".into(),
            Self::ICC_BPR1_EL1 => "icc_bpr1_el1".into(),
            Self::ICC_SRE_EL1 => "icc_sre_el1".into(),
            Self::ICC_IGRPEN1_EL1 => "icc_igrpen1_el1".into(),
            Self::TPIDR_EL0 => "tpidr_el0".into(),
            Self::TPIDRRO_EL0 => "tpidrro_el0".into(),
            Self::TPIDR_EL1 => "tpidr_el1".into(),
            Self::CNTFRQ_EL0 => "cntfrq_el0".into(),
            Self::CNTPCT_EL0 => "cntpct_el0".into(),
            Self::CNTVCT_EL0 => "cntvct_el0".into(),
            Self::CNTP_TVAL_EL0 => "cntp_tval_el0".into(),
            Self::CNTP_CTL_EL0 => "cntp_ctl_el0".into(),
            Self::CNTP_CVAL_EL0 => "cntp_cval_el0".into(),
            Self::CNTV_TVAL_EL0 => "cntv_tval_el0".into(),
            Self::CNTV_CTL_EL0 => "cntv_ctl_el0".into(),
            Self::CNTV_CVAL_EL0 => "cntv_cval_el0".into(),
            _ => {
                let (op0, op1, crn, crm, op2) = self.fields();
                format!("s{op0}_{op1}_c{crn}_c{crm}_{op2}")
            }
        }
    }
}

impl fmt::Display for SysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// Complete AArch64 Instruction IR shared by decoder, interpreter, and x86-64 JIT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    // --- ALU Immediate / Register / Extended / Carry ---
    AddImm { w: Width, rd: Reg, rn: Reg, imm: u64, set_flags: bool },
    SubImm { w: Width, rd: Reg, rn: Reg, imm: u64, set_flags: bool },
    AddReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8, set_flags: bool },
    SubReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8, set_flags: bool },
    AddExt { w: Width, rd: Reg, rn: Reg, rm: Reg, extend: ExtendType, amount: u8, set_flags: bool },
    SubExt { w: Width, rd: Reg, rn: Reg, rm: Reg, extend: ExtendType, amount: u8, set_flags: bool },
    Adc { w: Width, rd: Reg, rn: Reg, rm: Reg, set_flags: bool },
    Sbc { w: Width, rd: Reg, rn: Reg, rm: Reg, set_flags: bool },

    // --- Logical Immediate / Register ---
    AndImm { w: Width, rd: Reg, rn: Reg, imm: u64, set_flags: bool },
    OrrImm { w: Width, rd: Reg, rn: Reg, imm: u64 },
    EorImm { w: Width, rd: Reg, rn: Reg, imm: u64 },
    AndReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8, set_flags: bool },
    OrrReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8 },
    EorReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8 },
    BicReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8, set_flags: bool },
    OrnReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8 },
    EonReg { w: Width, rd: Reg, rn: Reg, rm: Reg, shift: ShiftType, amount: u8 },

    // --- Move Wide & PC-relative ---
    Movz { w: Width, rd: Reg, imm16: u16, shift: u8 },
    Movn { w: Width, rd: Reg, imm16: u16, shift: u8 },
    Movk { w: Width, rd: Reg, imm16: u16, shift: u8 },
    Adr { rd: Reg, offset: i64 },
    Adrp { rd: Reg, offset: i64 },

    // --- Shifts, Bitfields, Extract, Bit-ops ---
    Lslv { w: Width, rd: Reg, rn: Reg, rm: Reg },
    Lsrv { w: Width, rd: Reg, rn: Reg, rm: Reg },
    Asrv { w: Width, rd: Reg, rn: Reg, rm: Reg },
    Rorv { w: Width, rd: Reg, rn: Reg, rm: Reg },
    Sbfm { w: Width, rd: Reg, rn: Reg, immr: u8, imms: u8 },
    Bfm { w: Width, rd: Reg, rn: Reg, immr: u8, imms: u8 },
    Ubfm { w: Width, rd: Reg, rn: Reg, immr: u8, imms: u8 },
    Extr { w: Width, rd: Reg, rn: Reg, rm: Reg, lsb: u8 },
    Clz { w: Width, rd: Reg, rn: Reg },
    Cls { w: Width, rd: Reg, rn: Reg },
    Rbit { w: Width, rd: Reg, rn: Reg },
    Rev16 { w: Width, rd: Reg, rn: Reg },
    Rev32 { w: Width, rd: Reg, rn: Reg },
    Rev64 { rd: Reg, rn: Reg },

    // --- Multiply / Divide ---
    Madd { w: Width, rd: Reg, rn: Reg, rm: Reg, ra: Reg },
    Msub { w: Width, rd: Reg, rn: Reg, rm: Reg, ra: Reg },
    Smaddl { rd: Reg, rn: Reg, rm: Reg, ra: Reg },
    Umaddl { rd: Reg, rn: Reg, rm: Reg, ra: Reg },
    Smulh { rd: Reg, rn: Reg, rm: Reg },
    Umulh { rd: Reg, rn: Reg, rm: Reg },
    Udiv { w: Width, rd: Reg, rn: Reg, rm: Reg },
    Sdiv { w: Width, rd: Reg, rn: Reg, rm: Reg },

    // --- Conditional Select & Compare ---
    Csel { w: Width, rd: Reg, rn: Reg, rm: Reg, cond: Condition },
    Csinc { w: Width, rd: Reg, rn: Reg, rm: Reg, cond: Condition },
    Csinv { w: Width, rd: Reg, rn: Reg, rm: Reg, cond: Condition },
    Csneg { w: Width, rd: Reg, rn: Reg, rm: Reg, cond: Condition },
    CcmpReg { w: Width, rn: Reg, rm: Reg, nzcv: u8, cond: Condition },
    CcmpImm { w: Width, rn: Reg, imm5: u8, nzcv: u8, cond: Condition },
    CcmnReg { w: Width, rn: Reg, rm: Reg, nzcv: u8, cond: Condition },
    CcmnImm { w: Width, rn: Reg, imm5: u8, nzcv: u8, cond: Condition },

    // --- Branches & PAC ---
    B { offset: i64 },
    Bl { offset: i64 },
    BCond { cond: Condition, offset: i64 },
    Cbz { w: Width, rt: Reg, offset: i64 },
    Cbnz { w: Width, rt: Reg, offset: i64 },
    Tbz { rt: Reg, bit: u8, offset: i64 },
    Tbnz { rt: Reg, bit: u8, offset: i64 },
    Br { rn: Reg },
    Blr { rn: Reg },
    Ret { rn: Reg },
    PacHint { op: PacOp, rd: Reg, rn: Reg },

    // --- Loads & Stores ---
    Ldr { w: Width, rt: Reg, rn: Reg, mode: AddrMode },
    Str { w: Width, rt: Reg, rn: Reg, mode: AddrMode },
    Ldrb { rt: Reg, rn: Reg, mode: AddrMode },
    Strb { rt: Reg, rn: Reg, mode: AddrMode },
    Ldrh { rt: Reg, rn: Reg, mode: AddrMode },
    Strh { rt: Reg, rn: Reg, mode: AddrMode },
    Ldrsb { w: Width, rt: Reg, rn: Reg, mode: AddrMode },
    Ldrsh { w: Width, rt: Reg, rn: Reg, mode: AddrMode },
    Ldrsw { rt: Reg, rn: Reg, mode: AddrMode },
    LdrLiteral { w: Width, rt: Reg, offset: i64, signed: bool },
    Ldp { w: Width, rt: Reg, rt2: Reg, rn: Reg, mode: AddrMode },
    Stp { w: Width, rt: Reg, rt2: Reg, rn: Reg, mode: AddrMode },
    Ldpsw { rt: Reg, rt2: Reg, rn: Reg, mode: AddrMode },
    Ldxr { size: u8, acquire: bool, rt: Reg, rn: Reg },
    Stxr { size: u8, release: bool, rs: Reg, rt: Reg, rn: Reg },
    Ldar { size: u8, rt: Reg, rn: Reg },
    Stlr { size: u8, rt: Reg, rn: Reg },
    Ldadd { size: u8, acquire: bool, release: bool, rs: Reg, rt: Reg, rn: Reg },
    Cas { size: u8, acquire: bool, release: bool, rs: Reg, rt: Reg, rn: Reg },
    Swp { size: u8, acquire: bool, release: bool, rs: Reg, rt: Reg, rn: Reg },
    Prfm,

    // --- System & Exceptions ---
    Mrs { rt: Reg, sysreg: SysReg },
    Msr { sysreg: SysReg, rt: Reg },
    MsrImm { pstate_field: u8, crm: u8 },
    Sys { op1: u8, crn: u8, crm: u8, op2: u8, rt: Reg },
    Svc { imm: u16 },
    Hvc { imm: u16 },
    Smc { imm: u16 },
    Eret,
    Brk { imm: u16 },
    Hlt { imm: u16 },
    Nop,
    Hint { imm: u8 },
    Barrier { op: BarrierOp },
    Clrex,
    Tlbi { op1: u8, crm: u8, op2: u8, rt: Reg },
    Dc { op1: u8, crm: u8, op2: u8, rt: Reg },
    Ic { op1: u8, crm: u8, op2: u8, rt: Reg },
    At { op1: u8, crm: u8, op2: u8, rt: Reg },

    // --- Floating-Point & SIMD essentials ---
    FmovGprToFp { is_64: bool, rd: u8, rn: Reg },
    FmovFpToGpr { is_64: bool, rd: Reg, rn: u8 },
    FmovImm { is_64: bool, rd: u8, imm8: u8 },
    FmovReg { is_64: bool, rd: u8, rn: u8 },
    Fadd { is_64: bool, rd: u8, rn: u8, rm: u8 },
    Fsub { is_64: bool, rd: u8, rn: u8, rm: u8 },
    Fmul { is_64: bool, rd: u8, rn: u8, rm: u8 },
    Fdiv { is_64: bool, rd: u8, rn: u8, rm: u8 },
    Fneg { is_64: bool, rd: u8, rn: u8 },
    Fabs { is_64: bool, rd: u8, rn: u8 },
    Fsqrt { is_64: bool, rd: u8, rn: u8 },
    Fcmp { is_64: bool, rn: u8, rm: u8, with_zero: bool },
    Fcvt { dst_64: bool, src_64: bool, rd: u8, rn: u8 },
    Scvtf { is_64: bool, w: Width, rd: u8, rn: Reg },
    Ucvtf { is_64: bool, w: Width, rd: u8, rn: Reg },
    Fcvtzs { is_64: bool, w: Width, rd: Reg, rn: u8 },
    Fcvtzu { is_64: bool, w: Width, rd: Reg, rn: u8 },
    LdrFp { size: u8, vt: u8, rn: Reg, mode: AddrMode },
    StrFp { size: u8, vt: u8, rn: Reg, mode: AddrMode },
    LdpFp { size: u8, vt: u8, vt2: u8, rn: Reg, mode: AddrMode },
    StpFp { size: u8, vt: u8, vt2: u8, rn: Reg, mode: AddrMode },
    SimdMov { q: bool, vd: u8, vn: u8 },
    SimdBitwise { q: bool, op: u8, vd: u8, vn: u8, vm: u8 },
    SimdAddSub { q: bool, esize: u8, is_sub: bool, vd: u8, vn: u8, vm: u8 },
    SimdDup { q: bool, esize: u8, vd: u8, rn: Reg },

    /// Unrecognized raw 32-bit instruction word.
    Unknown(u32),
}
