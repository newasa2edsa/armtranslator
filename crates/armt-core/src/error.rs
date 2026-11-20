use thiserror::Error;

/// Unified error type across all ArmTranslator crates.
#[derive(Debug, Error)]
pub enum ArmtError {
    #[error("AArch64 decode error at PC=0x{pc:016x} (word=0x{word:08x}): {reason}")]
    Decode {
        pc: u64,
        word: u32,
        reason: String,
    },

    #[error("Memory access fault at 0x{addr:016x} (size={size}, write={write}): {reason}")]
    MemoryFault {
        addr: u64,
        size: u8,
        write: bool,
        reason: String,
    },

    #[error("Alignment fault at 0x{addr:016x} (required {align}-byte alignment)")]
    AlignmentFault { addr: u64, align: usize },

    #[error("MMU translation fault at VA=0x{va:016x} (ISS=0x{iss:08x}): {reason}")]
    MmuFault { va: u64, iss: u32, reason: String },

    #[error("JIT error: {0}")]
    Jit(String),

    #[error("Apple container error: {0}")]
    AppleContainer(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Decompression error: {0}")]
    Decompress(String),

    #[error("Mach-O parse error: {0}")]
    MachO(String),

    #[error("ADT parse error: {0}")]
    Adt(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("VM error: {0}")]
    Vm(String),

    #[error("ABI bridge error: {0}")]
    Abi(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ArmtError>;
