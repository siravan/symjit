use super::runnable::CompilerType;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::Arc;

use super::code::{Func, VirtualTable};
use super::defuns::Defuns;
use super::operation::Operation;
use super::symbol::Loc;
use super::utils::Storage;

pub const USE_SIMD: u32 = 0x0000_0001;
pub const USE_THREADS: u32 = 0x0000_0002;
pub const CSE: u32 = 0x0000_0004;
pub const FASTMATH: u32 = 0x0000_0008;

pub const ENABLE_SIMD512: u32 = 0x0000_0010;
pub const COMPLEX: u32 = 0x0000_0020;
pub const SYMBOLICA: u32 = 0x0000_0040;
pub const SIMD_BRANCH: u32 = 0x0000_0080;

pub const COMPACT: u32 = 0x0000_1000;
pub const COMPRESS: u32 = 0x0000_2000;
pub const DIRECT: u32 = 0x0000_4000;
pub const FAST_COMPLEX: u32 = 0x0000_8000;

pub const DEBUG_BYTECODE: u32 = 0x0001_0000;
pub const DEBUG_SCALAR: u32 = 0x0002_0000;
pub const DEBUG_SIMD: u32 = 0x0004_0000;
pub const DEBUG_STATS: u32 = 0x0008_0000;
pub const DEBUG_LOCK: u32 = 0x0100_0000;
pub const DEBUG_TOPOLOGY: u32 = 0x1000_0000;
pub const DEBUG_INSTRUCTIONS: u32 = 0x2000_0000;

pub const HUGE: u32 = 0x0010_0000;
pub const PARALLEL_MUL: u32 = 0x0020_0000;

pub const DIRECT_ARENA: u32 = 0x0040_0000;
pub const DIRECT_ARENA_IDENTITY_OUTPUT: u32 = 0x0080_0000;
pub const DIRECT_ARENA_OPERATION_MASK: u32 = 0x0600_0000;
pub const DIRECT_ARENA_OPERATION_SHIFT: usize = 25;

pub const OPT_LEVEL_MASK: u32 = 0x0000_0f00;
pub const OPT_LEVEL_SHIFT: usize = 8;

pub const SPILL_AREA: usize = 16;
pub const ABI_AREA: usize = 16;

#[cfg(feature = "symbolica")]
pub const SLICE_CAP: usize = 256;

#[cfg(not(feature = "symbolica"))]
pub const SLICE_CAP: usize = 32;

pub const DEFAULT_STACK_LIMIT: usize = 1 << 20;

#[derive(Clone)]
pub struct Config {
    pub opt: u32,
    pub ty: CompilerType,
    pub df: Option<Arc<Defuns>>,
    pub stack: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct ConfigToml {
    ty: String,
    options: Options,
    debug: DebugOptions,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Options {
    use_simd: bool,
    enable_simd512: bool,
    use_threads: bool,
    cse: bool,
    fastmath: bool,
    complex: bool,
    symbolica: bool,
    simd_branch: bool,
    compact: bool,
    compress: bool,
    direct: bool,
    fast_complex: bool,
    huge: bool,
    parallel_mul: bool,
    direct_arena: bool,
    direct_arena_identity_output: bool,
    opt_level: u8,
    stack_limit: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct DebugOptions {
    bytecode: bool,
    scalar: bool,
    simd: bool,
    stats: bool,
    topology: bool,
    lock: bool,
    instructions: bool,
}

#[derive(Debug)]
pub enum KernelType {
    ColumnFirst(bool),
    RowFirst,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Config {{")?;

        if self.use_simd() {
            write!(f, "simd, ")?;
        }
        if self.use_simd512() {
            write!(f, "simd512, ")?;
        }
        if self.use_threads() {
            write!(f, "threads, ")?;
        }
        if self.cse() {
            write!(f, "cse, ")?;
        }
        if self.fastmath() {
            write!(f, "fastmath, ")?;
        }
        if self.is_complex() {
            write!(f, "complex, ")?;
        }
        if self.symbolica() {
            write!(f, "symbolica, ")?;
        }
        if self.simd_branch() {
            write!(f, "simd_branch, ")?;
        }
        if self.compact() {
            write!(f, "compact, ")?;
        }
        if self.compress() {
            write!(f, "compress, ")?;
        }
        if self.direct() {
            write!(f, "direct, ")?;
        }
        if self.fast_complex() {
            write!(f, "fast_complex, ")?;
        }
        if self.huge() {
            write!(f, "huge, ")?;
        }
        if self.parallel_mul() {
            write!(f, "parallel_mul, ")?;
        }
        if self.debug_bytecode() {
            write!(f, "debug_bytecode, ")?;
        }
        if self.debug_scalar() {
            write!(f, "debug_scalar, ")?;
        }
        if self.debug_simd() {
            write!(f, "debug_simd, ")?;
        }
        if self.debug_stats() {
            write!(f, "debug_stats, ")?;
        }
        if self.debug_topology() {
            write!(f, "debug_topology, ")?;
        }
        if self.debug_instructions() {
            write!(f, "debug_instructions, ")?;
        }
        if self.debug_lock() {
            write!(f, "debug_lock, ")?;
        }

        write!(
            f,
            "opt_level = {}, stack limit = {}}}",
            self.opt_level(),
            self.stack_limit()
        )
    }
}

impl Config {
    const MAGIC: usize = 0x802c3c77c7422e70;

    pub fn new(ty: CompilerType, opt: u32) -> Result<Config> {
        Ok(Config {
            opt,
            ty,
            df: None,
            stack: DEFAULT_STACK_LIMIT,
        })
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
            _ => {
                if ty.ends_with(".toml") {
                    return Self::from_toml(ty, opt);
                } else {
                    return Err(anyhow!("invalid ty"));
                }
            }
        };
        Self::new(ty, opt)
    }

    pub fn from_toml(path: &str, mut opt: u32) -> Result<Config> {
        let toml = std::fs::read_to_string(path)?;
        let c: ConfigToml = toml::from_str(&toml)?;

        opt &= COMPLEX | SYMBOLICA;
        let mut config = Self::from_name(&c.ty, opt)?;

        config.set_simd(c.options.use_simd);
        config.enable_simd512(c.options.enable_simd512);
        config.set_threads(c.options.use_threads);
        config.set_cse(c.options.cse);
        config.set_fastmath(c.options.fastmath);
        config.set_complex(c.options.complex | config.is_complex());
        config.set_symbolica(c.options.symbolica | config.symbolica());
        config.set_simd_branch(c.options.simd_branch);
        config.set_compact(c.options.compact);
        config.set_compress(c.options.compress);
        config.set_direct(c.options.direct);
        config.set_fast_complex(c.options.fast_complex);
        config.set_huge(c.options.huge);
        config.set_parallel_mul(c.options.parallel_mul);
        config.set_direct_arena(c.options.direct_arena);
        config.set_direct_arena_identity_output(c.options.direct_arena_identity_output);

        config.set_opt_level(c.options.opt_level);
        config.set_stack_limit(c.options.stack_limit);

        config.set_debug_bytecode(c.debug.bytecode);
        config.set_debug_scalar(c.debug.scalar);
        config.set_debug_simd(c.debug.simd);
        config.set_debug_stats(c.debug.stats);
        config.set_debug_topology(c.debug.topology);
        config.set_debug_instructions(c.debug.instructions);
        config.set_debug_lock(c.debug.lock);

        Ok(config)
    }

    pub fn to_toml(&self, path: &str) {
        let ty = match self.ty {
            CompilerType::ByteCode => "bytecode",
            CompilerType::Arm => "arm",
            CompilerType::RiscV => "riscv",
            CompilerType::Amd => "amd",
            CompilerType::AmdAVX => "amd-avx",
            CompilerType::AmdSSE => "amd-sse",
            CompilerType::Native => "native",
            CompilerType::Debug => "debug",
        }
        .into();

        let options: Options = Options {
            use_simd: self.use_simd(),
            enable_simd512: self.is_enabled_simd512(),
            use_threads: self.use_threads(),
            cse: self.cse(),
            fastmath: self.fastmath(),
            complex: self.is_complex(),
            symbolica: self.symbolica(),
            simd_branch: self.simd_branch(),
            compact: self.compact(),
            compress: self.compress(),
            direct: self.direct(),
            fast_complex: self.fast_complex(),
            opt_level: self.opt_level(),
            stack_limit: self.stack_limit(),
            huge: self.huge(),
            parallel_mul: self.parallel_mul(),
            direct_arena: self.direct_arena(),
            direct_arena_identity_output: self.direct_arena_identity_output(),
        };

        let debug: DebugOptions = DebugOptions {
            bytecode: self.debug_bytecode(),
            scalar: self.debug_scalar(),
            simd: self.debug_simd(),
            stats: self.debug_stats(),
            topology: self.debug_topology(),
            instructions: self.debug_instructions(),
            lock: self.debug_lock(),
        };

        let c: ConfigToml = ConfigToml { ty, options, debug };
        let toml = toml::to_string(&c).unwrap();
        let _ = std::fs::write(path, toml);
    }

    pub fn from_defuns(df: Defuns) -> Result<Config> {
        let mut config = Config::default();
        config.set_defuns(df);
        Ok(config)
    }

    pub fn set_defuns(&mut self, df: Defuns) {
        match self.df {
            None => self.df = Some(Arc::new(df)),
            Some(_) => panic!("Config defuns can only be set once."),
        }
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

    fn cpu_has_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        return is_x86_feature_detected!("avx512f");
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
        self.test(USE_SIMD) && (self.has_avx() || self.is_arm64())
    }

    pub fn is_enabled_simd512(&self) -> bool {
        self.test(ENABLE_SIMD512)
    }

    pub fn use_simd512(&self) -> bool {
        self.test(ENABLE_SIMD512) && self.test(USE_SIMD) && Self::cpu_has_avx512()
    }

    pub fn simd_branch(&self) -> bool {
        self.test(SIMD_BRANCH) && (self.has_avx() || self.is_arm64())
    }

    pub fn use_threads(&self) -> bool {
        self.test(USE_THREADS)
    }

    pub fn fastmath(&self) -> bool {
        self.test(FASTMATH) && (self.has_avx() || self.is_arm64() || self.is_riscv64())
    }

    pub fn compact(&self) -> bool {
        self.test(COMPACT)
    }

    pub fn compress(&self) -> bool {
        self.test(COMPRESS)
    }

    pub fn direct(&self) -> bool {
        self.test(DIRECT)
    }

    pub fn huge(&self) -> bool {
        self.test(HUGE)
    }

    pub fn parallel_mul(&self) -> bool {
        self.test(PARALLEL_MUL)
    }

    /// Uses pointer descriptors for scalar inputs to a direct arena application.
    pub fn direct_arena(&self) -> bool {
        self.test(DIRECT_ARENA)
    }

    /// Destination operation used by a direct arena application.
    pub fn direct_arena_operation(&self) -> u8 {
        ((self.opt & DIRECT_ARENA_OPERATION_MASK) >> DIRECT_ARENA_OPERATION_SHIFT) as u8
    }

    /// Stores outputs without complex-scalar scaling.
    pub fn direct_arena_identity_output(&self) -> bool {
        true
        //self.test(DIRECT_ARENA_IDENTITY_OUTPUT)
    }

    pub fn debug_bytecode(&self) -> bool {
        self.test(DEBUG_BYTECODE)
    }

    pub fn debug_scalar(&self) -> bool {
        self.test(DEBUG_SCALAR)
    }

    pub fn debug_simd(&self) -> bool {
        self.test(DEBUG_SIMD)
    }

    pub fn debug_stats(&self) -> bool {
        self.test(DEBUG_STATS)
    }

    pub fn debug_topology(&self) -> bool {
        self.test(DEBUG_TOPOLOGY)
    }

    pub fn debug_instructions(&self) -> bool {
        self.test(DEBUG_INSTRUCTIONS)
    }

    pub fn debug_lock(&self) -> bool {
        self.test(DEBUG_LOCK)
    }

    pub fn opt_level(&self) -> u8 {
        let level = ((self.opt & OPT_LEVEL_MASK) >> OPT_LEVEL_SHIFT) as u8;

        if self.is_sse() {
            level.min(2)
        } else {
            level
        }
    }

    pub fn stack_limit(&self) -> usize {
        self.stack
    }

    pub fn compiler_type(&self) -> CompilerType {
        if self.has_avx() {
            CompilerType::AmdAVX
        } else if self.is_amd64() {
            println!("warning: fallback to SSE instructions is deprecated and will be removed in a future version.");
            println!("warning (cont.): the minimum x86-64 profile will be raised to x86-64-v3.");
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

    pub fn kernel_type(&self) -> KernelType {
        if self.symbolica() && !self.direct_arena() {
            KernelType::RowFirst
        } else {
            KernelType::ColumnFirst(false)
        }
    }

    pub fn native_compiler_type(&self) -> CompilerType {
        let config = Config::new(CompilerType::Native, self.opt).unwrap();
        config.compiler_type()
    }

    pub fn available_registers(&self) -> u8 {
        16
        /*
        if (self.is_arm64() || self.is_riscv64()) && self.opt_level() == 3 {
            32
        } else {
            16
        }
        */
    }

    pub fn count_scratch(&self) -> u8 {
        if !self.is_complex() {
            self.available_registers() - 2
        } else if self.fast_complex() && !self.use_simd() && (self.is_arm64() || self.is_riscv64())
        {
            self.available_registers() - 5
        } else {
            (self.available_registers() - 6) / 2
        }
    }

    pub fn is_external_func(&self, op: &str) -> bool {
        if let Some(df) = &self.df {
            matches!(df.funcs.get(op), Some(Func::Slice { .. }))
                || matches!(df.funcs.get(op), Some(Func::App { .. }))
        } else {
            false
        }
    }

    pub fn is_kernel_func(&self, op: &str) -> bool {
        if let Some(df) = &self.df {
            matches!(df.funcs.get(op), Some(Func::App { .. }))
        } else {
            false
        }
    }

    pub fn location(&self, arg: u8) -> Loc {
        let k = if self.is_complex() { 2 } else { 1 };
        Loc::Stack(SPILL_AREA as u32 + k * arg as u32)
    }

    pub fn symbolica(&self) -> bool {
        self.test(SYMBOLICA)
    }

    pub fn is_complex(&self) -> bool {
        self.test(COMPLEX)
    }

    pub fn fast_complex(&self) -> bool {
        self.test(FAST_COMPLEX) && (self.has_avx() || self.is_arm64())
    }

    /// Sets of optimization level. The valid values are 0, 1, 2, which roughly correspond to gcc O0, O1, and O2 levels.
    pub fn set_opt_level(&mut self, opt_level: u8) {
        if !self.debug_lock() {
            self.opt = (self.opt & !OPT_LEVEL_MASK) | ((opt_level as u32) << OPT_LEVEL_SHIFT);
        }
    }

    /// Enables Common-Subexpression-Elimination.
    pub fn set_cse(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !CSE) | if enabled { CSE } else { 0 };
        }
    }

    /// Enables fastmath mode. The main effect is to generate fused-multiply-addition
    /// instructions if possible.
    pub fn set_fastmath(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !FASTMATH) | if enabled { FASTMATH } else { 0 };
        }
    }

    /// Enables SIMD mode.
    pub fn set_simd(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !USE_SIMD) | if enabled { USE_SIMD } else { 0 };
        }
    }

    /// Enables SIMD mode.
    pub fn enable_simd512(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !ENABLE_SIMD512) | if enabled { ENABLE_SIMD512 } else { 0 };
        }
    }

    /// Enables forced SIMD branching mode.
    pub fn set_simd_branch(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !SIMD_BRANCH) | if enabled { SIMD_BRANCH } else { 0 };
        }
    }

    /// Enables Complex Numbers.
    pub fn set_complex(&mut self, enabled: bool) {
        self.opt = (self.opt & !COMPLEX) | if enabled { COMPLEX } else { 0 };
    }

    /// Enables Fast Complex (using SIMD instructions in the scalar code).
    pub fn set_fast_complex(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !FAST_COMPLEX) | if enabled { FAST_COMPLEX } else { 0 };
        }
    }

    /// Enables Multi-threading.
    pub fn set_threads(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !USE_THREADS) | if enabled { USE_THREADS } else { 0 };
        }
    }

    /// Enables Symbolica Mode.
    pub fn set_symbolica(&mut self, enabled: bool) {
        self.opt = (self.opt & !SYMBOLICA) | if enabled { SYMBOLICA } else { 0 };
    }

    /// Compact stack frame.
    pub fn set_compact(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !COMPACT) | if enabled { COMPACT } else { 0 };
        }
    }

    /// Memory-saver mode for very large inputs.
    pub fn set_compress(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !COMPRESS) | if enabled { COMPRESS } else { 0 };
        }
    }

    /// Direct translation from Symbolica IR to Symjit IR.
    /// This is a typo but is kept for compatibility.
    pub fn set_dicect(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !DIRECT) | if enabled { DIRECT } else { 0 };
        }
    }

    /// Direct translation from Symbolica IR to Symjit IR.
    pub fn set_direct(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !DIRECT) | if enabled { DIRECT } else { 0 };
        }
    }

    /// Huge paged to reduce TLB pressure.
    pub fn set_huge(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !HUGE) | if enabled { HUGE } else { 0 };
        }
    }

    /// Merge serial complex multiplications into parallel operation.
    pub fn set_parallel_mul(&mut self, enabled: bool) {
        if !self.debug_lock() {
            self.opt = (self.opt & !PARALLEL_MUL) | if enabled { PARALLEL_MUL } else { 0 };
        }
    }

    /// Enables the direct arena scalar-input ABI.
    pub fn set_direct_arena(&mut self, enabled: bool) {
        self.opt = (self.opt & !DIRECT_ARENA) | if enabled { DIRECT_ARENA } else { 0 };
    }

    /// Sets the direct arena destination operation encoded in portable MIR.
    pub fn set_direct_arena_operation(&mut self, operation: u8) {
        assert!(operation <= 1, "invalid direct arena destination operation");
        self.opt = (self.opt & !DIRECT_ARENA_OPERATION_MASK)
            | ((operation as u32) << DIRECT_ARENA_OPERATION_SHIFT);
    }

    /// Selects identity output scaling independently of the operation.
    pub fn set_direct_arena_identity_output(&mut self, enabled: bool) {
        self.opt = (self.opt & !DIRECT_ARENA_IDENTITY_OUTPUT)
            | if enabled {
                DIRECT_ARENA_IDENTITY_OUTPUT
            } else {
                0
            };
    }

    /// Dump bytecode for debugging
    pub fn set_debug_bytecode(&mut self, enabled: bool) {
        self.opt = (self.opt & !DEBUG_BYTECODE) | if enabled { DEBUG_BYTECODE } else { 0 };
    }

    /// Dump scalar binary for debugging
    pub fn set_debug_scalar(&mut self, enabled: bool) {
        self.opt = (self.opt & !DEBUG_SCALAR) | if enabled { DEBUG_SCALAR } else { 0 };
    }

    /// Dump simd binary for debugging
    pub fn set_debug_simd(&mut self, enabled: bool) {
        self.opt = (self.opt & !DEBUG_SIMD) | if enabled { DEBUG_SIMD } else { 0 };
    }

    /// Print stats for debugging
    pub fn set_debug_stats(&mut self, enabled: bool) {
        self.opt = (self.opt & !DEBUG_STATS) | if enabled { DEBUG_STATS } else { 0 };
    }

    /// Print topology information for debugging
    pub fn set_debug_topology(&mut self, enabled: bool) {
        self.opt = (self.opt & !DEBUG_TOPOLOGY) | if enabled { DEBUG_TOPOLOGY } else { 0 };
    }

    /// Print instructions for deugging
    pub fn set_debug_instructions(&mut self, enabled: bool) {
        self.opt = (self.opt & !DEBUG_INSTRUCTIONS) | if enabled { DEBUG_INSTRUCTIONS } else { 0 };
    }

    /// Print stats for debugging
    pub fn set_debug_lock(&mut self, enabled: bool) {
        self.opt = (self.opt & !DEBUG_LOCK) | if enabled { DEBUG_LOCK } else { 0 };
    }

    pub fn set_stack_limit(&mut self, stack_limit: usize) {
        self.stack = stack_limit.max(DEFAULT_STACK_LIMIT);
    }

    pub fn max_lanes(&self) -> usize {
        if self.use_simd512() {
            8
        } else if self.use_simd() && self.has_avx() {
            4
        } else if self.use_simd() && self.is_arm64() {
            2
        } else {
            1
        }
    }

    pub fn sizeof(&self) -> usize {
        if self.is_complex() {
            2
        } else {
            1
        }
    }

    pub fn set_option(&mut self, option: &str, val: &str) -> Result<()> {
        match option {
            "use_simd" => {
                self.set_simd(val.parse::<bool>()?);
            }
            "enable_simd512" => {
                self.enable_simd512(val.parse::<bool>()?);
            }
            "use_threads" => {
                self.set_threads(val.parse::<bool>()?);
            }
            "cse" => {
                self.set_cse(val.parse::<bool>()?);
            }
            "fastmath" => {
                self.set_fastmath(val.parse::<bool>()?);
            }
            "complex" => {
                self.set_complex(val.parse::<bool>()?);
            }
            "symbolica" => {
                self.set_symbolica(val.parse::<bool>()?);
            }
            "simd_branch" => {
                self.set_simd_branch(val.parse::<bool>()?);
            }
            "compact" => {
                self.set_compact(val.parse::<bool>()?);
            }
            "compress" => {
                self.set_compress(val.parse::<bool>()?);
            }
            "direct" => {
                self.set_direct(val.parse::<bool>()?);
            }
            "fast_complex" => {
                self.set_fast_complex(val.parse::<bool>()?);
            }
            "huge" => {
                self.set_huge(val.parse::<bool>()?);
            }
            "parallel_mul" => {
                self.set_parallel_mul(val.parse::<bool>()?);
            }
            "opt_level" => {
                self.set_opt_level(val.parse::<u8>()?);
            }
            "stack_limit" => {
                self.set_stack_limit(val.parse::<usize>()?);
            }
            "debug_bytecode" => {
                self.set_debug_bytecode(val.parse::<bool>()?);
            }
            "debug_scalar" => {
                self.set_debug_scalar(val.parse::<bool>()?);
            }
            "debug_simd" => {
                self.set_debug_simd(val.parse::<bool>()?);
            }
            "debug_stats" => {
                self.set_debug_stats(val.parse::<bool>()?);
            }
            "debug_topology" => {
                self.set_debug_topology(val.parse::<bool>()?);
            }
            _ => return Err(anyhow!("option {} is not recognized.", option)),
        }

        Ok(())
    }

    pub fn can_compile(&self) -> Result<()> {
        if let Some(df) = &self.df {
            for (name, f) in &df.funcs {
                if let Func::App(app) = f {
                    let c = &app.config;

                    if c.is_complex() {
                        if !self.is_complex() {
                            return Err(anyhow!(
                                "Inner app {} is defined as complex, whereas the main app is real.",
                                name
                            ));
                        }

                        if app.count_obs != 2 {
                            return Err(anyhow!("Inner app {} should have only one output.", name));
                        }
                    } else {
                        if self.is_complex() {
                            return Err(anyhow!(
                                "Inner app {} is defined as real, whereas the main app is complex.",
                                name
                            ));
                        }

                        if app.count_obs != 1 {
                            return Err(anyhow!("Inner app {} should have only one output.", name));
                        }
                    }

                    if !c.use_simd() && self.use_simd() {
                        return Err(anyhow!(
                            "Inner app {} should use SIMD since the main app does.",
                            name
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Config {
        let toml = match std::env::var("SYMJIT_TOML") {
            Ok(toml) => toml,
            Err(_) => "symjit.toml".into(),
        };

        if std::fs::exists(&toml).unwrap() {
            Self::from_toml(&toml, 0).unwrap()
        } else {
            Config::new(
                CompilerType::Native,
                USE_SIMD
                    | CSE
                    | SYMBOLICA
                    | COMPACT
                    | FASTMATH
                    | FAST_COMPLEX
                    // | DIRECT
                    | PARALLEL_MUL
                    | (2 << OPT_LEVEL_SHIFT)
                    | DIRECT_ARENA_IDENTITY_OUTPUT,
            )
            .unwrap()
            // config.to_toml("symjit.toml");
        }
    }
}

// the list of intrinsic unary ops, i.e., operations that can be implemented directly in
// machine code
const UNARY: &[&str] = &[
    "abs",
    "not",
    "neg",
    "root",
    "real_root",
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
    "iszero",
    "isnotzero",
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
    pub fn is_intrinsic_unary(&self, op: &Operation) -> bool {
        match op {
            Operation::Op(s) => UNARY.contains(&s.as_str()),
            _ => false,
        }
    }

    pub fn is_intrinsic_binary(&self, op: &Operation) -> bool {
        match op {
            Operation::Plus | Operation::Minus | Operation::Times | Operation::Divide => true,
            Operation::Op(s) => BINARY.contains(&s.as_str()),
        }
    }

    pub fn symbolica_fun(&self, fun: &str, is_real: bool) -> String {
        if fun == "symbolica_sqrt" {
            if is_real {
                "real_root".into()
            } else {
                "root".into()
            }
        } else if let Some(op) = fun.strip_prefix("symbolica_") {
            let op = match op {
                "log" => "ln", // because of discrepency between real/complex log/ln
                "asin" => "arcsin",
                "acos" => "arccos",
                "atan" => "arctan",
                "asinh" => "arcsinh",
                "acosh" => "arccosh",
                "atanh" => "arctanh",
                op => op,
            };

            let oper = Operation::new(op);

            if self.is_intrinsic_unary(&oper)
                || self.is_intrinsic_binary(&oper)
                || (!self.is_complex() && VirtualTable::from_str(op).is_ok())
                || (self.is_complex() && VirtualTable::from_str(&format!("cplx_{}", op)).is_ok())
            {
                op.into()
            } else {
                fun.into()
            }
        } else {
            fun.into()
        }
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

    fn load(stream: &mut impl Read, config: &Self) -> Result<Self> {
        let mut bytes: [u8; 8] = [0; 8];

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != Self::MAGIC {
            return Err(anyhow!("invalid magic number (Config)"));
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

        Ok(Config {
            opt,
            ty,
            df: config.df.clone(),
            stack: DEFAULT_STACK_LIMIT,
        })
    }
}
