use crate::runnable::CompilerType;
use anyhow::{anyhow, Result};
use std::io::{Read, Write};

use crate::utils::Storage;

pub const USE_SIMD: u32 = 0x01;
pub const USE_THREADS: u32 = 0x02;
pub const CSE: u32 = 0x04;
pub const FASTMATH: u32 = 0x08;
// pub const SANITIZE: u32 = 0x10;
pub const COMPLEX: u32 = 0x20;
pub const SYMBOLICA: u32 = 0x40;
pub const OPT_LEVEL_MASK: u32 = 0x0f00;
pub const OPT_LEVEL_SHIFT: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub opt: u32,
    pub ty: CompilerType,
}

impl Config {
    const MAGIC: usize = 0x802c3c77c7422e70;

    pub fn new(ty: CompilerType, opt: u32) -> Result<Config> {
        Ok(Config { opt, ty })
    }

    pub fn from_name(ty: &str, opt: u32) -> Result<Config> {
        let ty = match ty {
            "bytecode" => CompilerType::ByteCode,
            "arm" => CompilerType::Arm,
            "riscv" => CompilerType::RiscV,
            "amd" => CompilerType::Amd,
            "amd-avx" => CompilerType::AmdAVX,
            "amd-sse" => CompilerType::AmdSSE,
            "native" => CompilerType::Native,
            "debug" => CompilerType::Debug,
            _ => return Err(anyhow!("invalid ty")),
        };
        Self::new(ty, opt)
    }

    fn test(&self, mask: u32) -> bool {
        self.opt & mask != 0
    }

    pub fn cross_compiled(&self) -> bool {
        (self.is_amd64() && !cfg!(target_arch = "x86_64"))
            || (self.is_arm64() && !cfg!(target_arch = "aarch64"))
            || (self.is_riscv64() && !cfg!(target_arch = "riscv64"))
    }

    pub fn is_amd64(&self) -> bool {
        (matches!(self.ty, CompilerType::Native) && cfg!(target_arch = "x86_64"))
            || matches!(self.ty, CompilerType::Amd)
            || matches!(self.ty, CompilerType::AmdSSE)
            || matches!(self.ty, CompilerType::AmdAVX)
    }

    pub fn is_arm64(&self) -> bool {
        (matches!(self.ty, CompilerType::Native) && cfg!(target_arch = "aarch64"))
            || matches!(self.ty, CompilerType::Arm)
    }

    pub fn is_riscv64(&self) -> bool {
        (matches!(self.ty, CompilerType::Native) && cfg!(target_arch = "riscv64"))
            || matches!(self.ty, CompilerType::RiscV)
    }

    fn cpu_has_avx() -> bool {
        #[cfg(target_arch = "x86_64")]
        return is_x86_feature_detected!("avx");
        #[cfg(not(target_arch = "x86_64"))]
        return false;
    }

    pub fn has_avx(&self) -> bool {
        self.is_amd64() && !matches!(self.ty, CompilerType::AmdSSE) && Self::cpu_has_avx()
    }

    pub fn is_sse(&self) -> bool {
        self.is_amd64() && !self.has_avx()
    }

    pub fn is_bytecode(&self) -> bool {
        matches!(self.ty, CompilerType::ByteCode)
    }

    pub fn is_debug(&self) -> bool {
        matches!(self.ty, CompilerType::Debug)
    }

    pub fn may_fast(&self) -> bool {
        self.is_amd64() || self.is_arm64() || self.is_riscv64()
    }

    pub fn cse(&self) -> bool {
        self.test(CSE)
    }

    pub fn use_simd(&self) -> bool {
        // TODO! To implement Symbolica arm64 SIMD.
        if self.symbolica() && self.is_arm64() {
            return false;
        }
        self.test(USE_SIMD) && (self.has_avx() || self.is_arm64())
    }

    pub fn use_threads(&self) -> bool {
        self.test(USE_THREADS)
    }

    pub fn fastmath(&self) -> bool {
        self.test(FASTMATH) && (self.has_avx() || self.is_arm64() || self.is_riscv64())
    }

    pub fn opt_level(&self) -> u8 {
        let level = ((self.opt & OPT_LEVEL_MASK) >> OPT_LEVEL_SHIFT) as u8;

        if self.is_sse() {
            level.min(2)
        } else {
            level
        }
    }

    pub fn compiler_type(&self) -> CompilerType {
        if self.has_avx() {
            CompilerType::AmdAVX
        } else if self.is_amd64() {
            CompilerType::AmdSSE
        } else if self.is_arm64() {
            CompilerType::Arm
        } else if self.is_riscv64() {
            CompilerType::RiscV
        } else if self.is_bytecode() {
            CompilerType::ByteCode
        } else if self.is_debug() {
            CompilerType::Debug
        } else {
            unreachable!()
        }
    }

    pub fn native_compiler_type(&self) -> CompilerType {
        let config = Config::new(CompilerType::Native, self.opt).unwrap();
        config.compiler_type()
    }

    fn available_registers(&self) -> u8 {
        if self.is_arm64() || self.is_riscv64() {
            32
        } else {
            16
        }
    }

    pub fn count_scratch(&self) -> u8 {
        if self.is_complex() {
            (self.available_registers() - 6) / 2
        } else {
            self.available_registers() - 2
        }
    }

    pub fn symbolica(&self) -> bool {
        self.test(SYMBOLICA)
    }

    pub fn is_complex(&self) -> bool {
        self.test(COMPLEX)
    }

    /// Sets of optimization level. The valid values are 0, 1, 2, which roughly correspond to gcc O0, O1, and O2 levels.
    pub fn set_opt_level(&mut self, opt_level: u8) {
        self.opt = (self.opt & !OPT_LEVEL_MASK) | ((opt_level as u32) << OPT_LEVEL_SHIFT);
    }

    /// Enables Common-Subexpression-Elimination.
    pub fn set_cse(&mut self, enabled: bool) {
        self.opt = (self.opt & !CSE) | if enabled { CSE } else { 0 };
    }

    /// Enables fastmath mode. The main effect is to generate fused-multiply-addition
    /// instructions if possible.
    pub fn set_fastmath(&mut self, enabled: bool) {
        self.opt = (self.opt & !FASTMATH) | if enabled { FASTMATH } else { 0 };
    }

    /// Enables SIMD mode.
    pub fn set_simd(&mut self, enabled: bool) {
        self.opt = (self.opt & !USE_SIMD) | if enabled { USE_SIMD } else { 0 };
    }

    /// Enables Complex Numbers.
    pub fn set_complex(&mut self, enabled: bool) {
        self.opt = (self.opt & !COMPLEX) | if enabled { COMPLEX } else { 0 };
    }

    /// Enables Symbolica Mode.
    pub fn set_symbolica(&mut self, enabled: bool) {
        self.opt = (self.opt & !SYMBOLICA) | if enabled { SYMBOLICA } else { 0 };
    }
}

impl Default for Config {
    fn default() -> Config {
        Config::new(
            CompilerType::Native,
            USE_SIMD | USE_THREADS | (2 << OPT_LEVEL_SHIFT),
        )
        .unwrap()
    }
}

// the list of intrinsic unary ops, i.e., operations that can be implemented directly in
// machine code
const UNARY: &[&str] = &[
    "abs",
    "not",
    "neg",
    "root",
    "square",
    "cube",
    "recip",
    "round",
    "floor",
    "ceiling",
    "trunc",
    "frac",
    "_powi_",
    "_call_",
    "real",
    "imaginary",
    "conjugate",
];

// the list of intrinsic binary ops, i.e., operations that can be implemented directly in
// machine code
const BINARY: &[&str] = &[
    "plus",
    "minus",
    "times",
    "divide",
    "rem",
    "gt",
    "geq",
    "lt",
    "leq",
    "eq",
    "neq",
    "and",
    "or",
    "xor",
    "_ifelse_",
    "_powi_mod_",
    "_call_",
    "min",
    "max",
    "heaviside",
    "complex",
];

impl Config {
    pub fn is_intrinsic_unary(&self, op: &str) -> bool {
        if op == "root" {
            !self.is_complex()
        } else {
            UNARY.contains(&op)
        }
    }

    pub fn is_intrinsic_binary(&self, op: &str) -> bool {
        BINARY.contains(&op)
    }
}

impl Storage for Config {
    fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&Self::MAGIC.to_le_bytes())?;

        let ty: usize = match self.ty {
            CompilerType::Native => 0,
            CompilerType::Amd => 1,
            CompilerType::AmdAVX => 2,
            CompilerType::AmdSSE => 3,
            CompilerType::Arm => 4,
            CompilerType::RiscV => 5,
            CompilerType::ByteCode => 6,
            CompilerType::Debug => 7,
        };

        let val: usize = (self.opt as usize) | (ty << 32);
        stream.write_all(&val.to_le_bytes())?;
        Ok(())
    }

    fn load(stream: &mut impl Read) -> Result<Self> {
        let mut bytes: [u8; 8] = [0; 8];

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != Self::MAGIC {
            return Err(anyhow!("invalid magic number"));
        }

        stream.read_exact(&mut bytes)?;
        let val = usize::from_le_bytes(bytes);
        let opt: u32 = (val & 0xffffffff) as u32;
        let ty: u32 = (val >> 32) as u32;

        let ty: CompilerType = match ty {
            0 => CompilerType::Native,
            1 => CompilerType::Amd,
            2 => CompilerType::AmdAVX,
            3 => CompilerType::AmdSSE,
            4 => CompilerType::Arm,
            5 => CompilerType::RiscV,
            6 => CompilerType::ByteCode,
            7 => CompilerType::Debug,
            _ => return Err(anyhow!("invalid compiler type value.")),
        };

        Ok(Config { opt, ty })
    }
}
