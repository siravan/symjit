// Relocatable object files (ELF64 for Linux, Mach-O for macOS) holding symjit kernels,
// to be linked into C/C++ programs (cargo feature `obj`). Modeled on the object emitter
// of the rue compiler (crates/rue-linker/src/emit.rs).
//
// An object holds one text section with one or more functions. Each function is
// position-independent machine code followed by the constants it reads PC-relative;
// its only references to the outside are calls to external functions (the C math
// library), recorded as relocations.
//
// Supported: x86-64 and AArch64, as ELF and Mach-O. The only relocation is a direct
// call: on x86-64 `call rel32` / `jmp rel32` (R_X86_64_PLT32 in ELF,
// X86_64_RELOC_BRANCH in Mach-O), on AArch64 `bl` (R_AARCH64_CALL26,
// ARM64_RELOC_BRANCH26).

mod elf;
mod header;
#[cfg(test)]
mod kernel_tests;
mod macho;
#[cfg(test)]
mod tests;

use anyhow::{anyhow, Result};
use std::collections::HashSet;

pub use header::{header, HeaderInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Elf,
    MachO,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub format: Format,
    pub arch: Arch,
}

impl Target {
    pub fn new(format: Format, arch: Arch) -> Target {
        Target { format, arch }
    }

    /// The object format and architecture of the host.
    pub fn host() -> Result<Target> {
        let format = if cfg!(target_os = "macos") {
            Format::MachO
        } else if cfg!(target_os = "linux") {
            Format::Elf
        } else {
            return Err(anyhow!("object files are supported on Linux (ELF) and macOS (Mach-O)"));
        };

        let arch = if cfg!(target_arch = "x86_64") {
            Arch::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else {
            return Err(anyhow!("object files are supported for x86-64 and AArch64"));
        };

        Ok(Target { format, arch })
    }

    /// The name of a symbol in the object (Mach-O prefixes C names with `_`).
    pub fn symbol_name(&self, name: &str) -> String {
        match self.format {
            Format::Elf => name.to_string(),
            Format::MachO => format!("_{}", name),
        }
    }

    // the filler between functions
    fn padding_byte(&self) -> u8 {
        match self.arch {
            Arch::X86_64 => 0xcc, // int3
            Arch::Aarch64 => 0,   // udf #0 (functions are 4-byte multiples)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocKind {
    /// A direct call to a function. x86-64: `call rel32` (e8) or `jmp rel32` (e9),
    /// `offset` is the position of the 4-byte rel32 field, which must hold 0. AArch64:
    /// `bl`, `offset` is the position of the instruction, whose offset must be 0.
    Call,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relocation {
    pub offset: usize,
    /// the C name of the target (without the Mach-O underscore)
    pub symbol: String,
    pub kind: RelocKind,
}

impl Relocation {
    pub fn call(offset: usize, symbol: &str) -> Relocation {
        Relocation {
            offset,
            symbol: symbol.to_string(),
            kind: RelocKind::Call,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

pub const TEXT_ALIGN: usize = 16;

pub struct ObjectBuilder {
    pub target: Target,
    pub code: Vec<u8>,
    pub functions: Vec<Function>,
    pub relocations: Vec<Relocation>,
}

impl ObjectBuilder {
    pub fn new(target: Target) -> ObjectBuilder {
        ObjectBuilder {
            target,
            code: Vec::new(),
            functions: Vec::new(),
            relocations: Vec::new(),
        }
    }

    /// Appends a function: `code` is its machine code (with any constants it reads
    /// PC-relative), `relocs` its calls to external functions, with offsets relative
    /// to the start of `code`. Functions start at multiples of TEXT_ALIGN.
    pub fn add_function(&mut self, name: &str, code: &[u8], relocs: &[Relocation]) -> Result<()> {
        if !is_c_identifier(name) {
            return Err(anyhow!("`{}` is not a valid C identifier", name));
        }
        if self.functions.iter().any(|f| f.name == name) {
            return Err(anyhow!("function `{}` is defined twice", name));
        }

        for r in relocs {
            if !is_c_identifier(&r.symbol) {
                return Err(anyhow!("`{}` is not a valid C identifier", r.symbol));
            }
            match r.kind {
                RelocKind::Call => match self.target.arch {
                    Arch::X86_64 => {
                        if r.offset < 1 || r.offset + 4 > code.len() {
                            return Err(anyhow!("call relocation at {} is outside `{}`", r.offset, name));
                        }
                        if !matches!(code[r.offset - 1], 0xe8 | 0xe9) {
                            return Err(anyhow!(
                                "call relocation at {} in `{}` does not follow a call/jmp rel32",
                                r.offset,
                                name
                            ));
                        }
                        if code[r.offset..r.offset + 4] != [0; 4] {
                            return Err(anyhow!(
                                "the rel32 field of the call at {} in `{}` must be 0",
                                r.offset,
                                name
                            ));
                        }
                    }
                    // the relocation is at the `bl` instruction, whose offset must be 0
                    Arch::Aarch64 => {
                        if r.offset % 4 != 0 || r.offset + 4 > code.len() {
                            return Err(anyhow!("call relocation at {} is outside `{}`", r.offset, name));
                        }
                        let word = u32::from_le_bytes(code[r.offset..r.offset + 4].try_into().unwrap());
                        if word != 0x9400_0000 {
                            return Err(anyhow!(
                                "call relocation at {} in `{}` is not at a `bl #0` ({:#010x})",
                                r.offset,
                                name,
                                word
                            ));
                        }
                    }
                },
            }
        }

        let pad = align_up(self.code.len(), TEXT_ALIGN) - self.code.len();
        let filler = self.target.padding_byte();
        self.code.extend(std::iter::repeat(filler).take(pad));

        let offset = self.code.len();
        self.code.extend_from_slice(code);
        self.functions.push(Function {
            name: name.to_string(),
            offset,
            size: code.len(),
        });
        self.relocations.extend(relocs.iter().map(|r| Relocation {
            offset: offset + r.offset,
            ..r.clone()
        }));

        Ok(())
    }

    /// The symbols called but not defined here, sorted.
    pub fn undefined_symbols(&self) -> Vec<String> {
        let defined: HashSet<&str> = self.functions.iter().map(|f| f.name.as_str()).collect();
        let mut undefined: Vec<String> = self
            .relocations
            .iter()
            .map(|r| r.symbol.clone())
            .filter(|s| !defined.contains(s.as_str()))
            .collect();
        undefined.sort();
        undefined.dedup();
        undefined
    }

    pub fn build(&self) -> Result<Vec<u8>> {
        if self.functions.is_empty() {
            return Err(anyhow!("an object file needs at least one function"));
        }
        match self.target.format {
            Format::Elf => Ok(elf::build(self)),
            Format::MachO => Ok(macho::build(self)),
        }
    }

    pub fn write(&self, path: &str) -> Result<()> {
        std::fs::write(path, self.build()?)?;
        Ok(())
    }
}

pub fn align_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

fn is_c_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// little-endian writers shared by the formats
pub(crate) trait Put {
    fn u8(&mut self, v: u8);
    fn u16(&mut self, v: u16);
    fn u32(&mut self, v: u32);
    fn u64(&mut self, v: u64);
    fn pad_to(&mut self, align: usize);
}

impl Put for Vec<u8> {
    fn u8(&mut self, v: u8) {
        self.push(v);
    }

    fn u16(&mut self, v: u16) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn pad_to(&mut self, align: usize) {
        let n = align_up(self.len(), align);
        self.resize(n, 0);
    }
}
