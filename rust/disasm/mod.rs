//! Disassembly of symjit's machine code: `Application::dump(name, "scalar-asm")` (and
//! `"simd-asm"`, `"fast-asm"`) writes a kernel as assembly text.
//!
//! The decoders read the finished bytes (after `apply_jumps`) and know only the instructions
//! symjit's encoders emit; anything else is listed as data (`.word`, `.byte`). The labels come
//! from the assembler (`Generator::take_labels`, kept in `MachineCode`): they name branch targets
//! and label references, and the label `@data` (`Generator::mark_data`, set by
//! `Builder::compile_from_mir` after the code) is where the data area starts: the constants
//! (`_const_N_`), the function table (`_func_op_`, `_simd_op_`, `_env_op_`) and the generators'
//! own constants. Code that was not compiled here (loaded `MachineCode`) has no labels and is
//! decoded to the end.
//!
//! Decoders: RISC-V (`riscv.rs`). x86-64 and AArch64 are to follow.

use anyhow::{anyhow, Result};
use std::collections::BTreeMap;
use std::fmt::Write;

pub mod riscv;

#[cfg(test)]
mod riscv_tests;

/// one decoded instruction (or a data item) of the listing
pub struct Decoded {
    /// its size in bytes
    pub len: usize,
    /// its bytes as the listing shows them
    pub hex: String,
    /// the assembly text
    pub text: String,
    /// the address it references (a branch target, a label reference), shown as a comment
    pub target: Option<usize>,
    /// a comment of its own (e.g. the value a `lui` + `addi` pair loads)
    pub note: Option<String>,
}

impl Decoded {
    /// bytes that are not an instruction
    pub fn bytes(code: &[u8]) -> Decoded {
        let hex: Vec<String> = code.iter().map(|b| format!("{:02x}", b)).collect();
        let text: Vec<String> = code.iter().map(|b| format!("0x{:02x}", b)).collect();

        Decoded {
            len: code.len().max(1),
            hex: hex.join(""),
            text: format!(".byte {}", text.join(", ")),
            target: None,
            note: None,
        }
    }
}

/// a decoder of one architecture
pub trait Decoder {
    /// decodes the instruction at the start of `code`, which is at offset `pc` of the kernel
    fn decode(&mut self, code: &[u8], pc: usize, symbols: &Symbols) -> Decoded;
}

/// the labels by offset
pub struct Symbols<'a> {
    by_offset: BTreeMap<usize, Vec<&'a str>>,
}

impl<'a> Symbols<'a> {
    pub fn new(labels: &'a [(usize, String)]) -> Symbols<'a> {
        let mut by_offset: BTreeMap<usize, Vec<&'a str>> = BTreeMap::new();
        for (ofs, name) in labels {
            by_offset.entry(*ofs).or_default().push(name.as_str());
        }
        Symbols { by_offset }
    }

    /// the labels at `ofs`
    pub fn at(&self, ofs: usize) -> &[&'a str] {
        self.by_offset.get(&ofs).map(|v| &v[..]).unwrap_or(&[])
    }

    /// an address as the listing writes it: `0x1c4 <@main>`; of several labels at one offset,
    /// the last in name order (`_const_0_` rather than `@data`)
    pub fn address(&self, ofs: i64) -> String {
        match self.at(ofs.max(0) as usize).last() {
            Some(name) if ofs >= 0 => format!("0x{:x} <{}>", ofs, name),
            _ => format!("0x{:x}", ofs),
        }
    }

    /// where the data area starts (the `@data` label), if known
    pub fn data_start(&self) -> Option<usize> {
        self.by_offset
            .iter()
            .find(|(_, names)| names.contains(&"@data"))
            .map(|(ofs, _)| *ofs)
    }
}

fn decoder(arch: &str) -> Result<Box<dyn Decoder>> {
    match arch {
        "riscv64" => Ok(Box::new(riscv::RiscvDecoder::default())),
        _ => Err(anyhow!("no disassembler for {} yet", arch)),
    }
}

/// whether a data label holds an address (the function table) rather than a double
fn is_address(name: &str) -> bool {
    name.starts_with("_func_") || name.starts_with("_simd_") || name.starts_with("_env_")
}

/// the 8 bytes at `ofs` as a double, if they are in the data area
fn double_at(code: &[u8], ofs: usize) -> Option<f64> {
    let b = code.get(ofs..ofs + 8)?;
    Some(f64::from_le_bytes(b.try_into().ok()?))
}

/// the listing of a kernel: the code, one instruction per line with its offset and bytes, the
/// labels at their offsets, then the data area as `.quad`s (doubles with their values, addresses
/// of the function table)
pub fn listing(arch: &str, code: &[u8], labels: &[(usize, String)]) -> Result<String> {
    let mut dec = decoder(arch)?;
    let symbols = Symbols::new(labels);
    let code_end = symbols.data_start().unwrap_or(code.len()).min(code.len());
    let mut out = String::new();

    let _ = writeln!(
        out,
        "; {} kernel, {} bytes: code 0x0..0x{:x}, data 0x{:x}..0x{:x}",
        arch,
        code.len(),
        code_end,
        code_end,
        code.len()
    );

    let labels_at = |out: &mut String, ofs: usize| {
        for name in symbols.at(ofs) {
            let _ = writeln!(out, "{}:", name);
        }
    };

    let mut pc = 0;
    while pc < code_end {
        labels_at(&mut out, pc);
        let d = dec.decode(&code[pc..code_end], pc, &symbols);
        let mut line = format!("{:>8x}:  {:<16}  {}", pc, d.hex, d.text);

        if let Some(t) = d.target {
            // a branch shows its target in the text already
            if !d.text.contains(&format!("0x{:x}", t)) {
                let _ = write!(line, "    # {}", symbols.address(t as i64));
                if t >= code_end {
                    if let Some(name) = symbols.at(t).last() {
                        if !is_address(name) {
                            if let Some(x) = double_at(code, t) {
                                let _ = write!(line, " = {:?}", x);
                            }
                        }
                    }
                }
            }
        }

        if let Some(note) = &d.note {
            let _ = write!(line, "    # {}", note);
        }

        let _ = writeln!(out, "{}", line);
        pc += d.len;
    }

    // the data area: 8-byte items, each under the last label before it
    let mut kind_address = false;
    let mut ofs = code_end;
    while ofs < code.len() {
        if let Some(name) = symbols.at(ofs).last() {
            kind_address = is_address(name);
        }
        labels_at(&mut out, ofs);

        if ofs + 8 <= code.len() {
            let b: [u8; 8] = code[ofs..ofs + 8].try_into().unwrap();
            let q = u64::from_le_bytes(b);
            let comment = if kind_address {
                "address".to_string()
            } else {
                format!("{:?}", f64::from_bits(q))
            };
            let _ = writeln!(
                out,
                "{:>8x}:  {:<16}  .quad 0x{:016x}    # {}",
                ofs,
                format!("{:016x}", q),
                q,
                comment
            );
            ofs += 8;
        } else {
            let d = Decoded::bytes(&code[ofs..]);
            let _ = writeln!(out, "{:>8x}:  {:<16}  {}", ofs, d.hex, d.text);
            ofs = code.len();
        }
    }

    for name in symbols.at(code.len()) {
        let _ = writeln!(out, "{}:", name);
    }

    Ok(out)
}
