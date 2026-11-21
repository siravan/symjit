//! Disassembly of symjit's machine code: `Application::dump(name, "scalar-asm")` (and
//! `"simd-asm"`, `"fast-asm"`) writes a kernel as assembly text that GNU `as` assembles back to
//! the same bytes; `"scalar-asm-hex"` (...) writes an objdump-like listing with the offset and the
//! bytes of each instruction.
//!
//! The decoders read the finished bytes (after `apply_jumps`) and know only the instructions
//! symjit's encoders emit, spelled so that the assembler encodes them as symjit did (e.g. RISC-V
//! `add rd, rs, zero`, not `mv`, which assembles to `addi`); anything else is written as raw data
//! of the same bytes (`.word`, `.inst`). The labels come from the assembler
//! (`Generator::take_labels`, kept in `MachineCode`): they name branch targets and label
//! references, and the label `@data` (`Generator::mark_data`, set by `Builder::compile_from_mir`
//! after the code) is where the data area starts: the constants (`_const_N_`), the function table
//! (`_func_op_`, `_simd_op_`, `_env_op_`) and the generators' own constants, written as `.quad`s.
//! Code that was not compiled here (loaded `MachineCode`) has no labels and is decoded to the end.
//!
//! The assembly form: the labels become local symbols (`@self` is `.L_self`, `_const_0_` is
//! `.L_const_0_`; targets without a label get `.L_at_<offset>`), branches and label references
//! use them, and the entry is the global function `symjit_<kind>` (`symjit_scalar`, ...). The
//! function table holds addresses in the process that compiled the kernel.
//!
//! Decoders: RISC-V (`riscv.rs`), AArch64 (`arm.rs`), x86-64 (`x86.rs`).

use anyhow::{anyhow, Result};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

pub mod arm;
pub mod riscv;
pub mod x86;

#[cfg(test)]
mod arm_tests;
#[cfg(test)]
mod riscv_tests;
#[cfg(test)]
mod x86_tests;

/// what the listing is for
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// assembler input: no offsets or bytes, symbolic labels and references
    Asm,
    /// for reading: the offset and the bytes of each line, numeric targets
    Hex,
}

/// a piece of an instruction's text
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Text(String),
    /// an address in the kernel (a branch target, a literal): its label, or the address
    Target(usize),
    /// the page an `adrp` computes (`page`), for the label of its pair (`target`)
    Page {
        page: usize,
        target: Option<usize>,
    },
    /// the low 12 bits of `target` (`:lo12:` of its label), `value` as encoded
    Lo12 {
        target: usize,
        value: i64,
    },
    /// a pc-relative reference to `target` (x86 `[rip+label]`): `+label`, or the displacement
    /// `value` (`+0x10`)
    Rel {
        target: usize,
        value: i64,
    },
}

/// one instruction (or data item) of the listing
#[derive(Clone, Debug)]
pub struct Item {
    pub pc: usize,
    /// the bytes as the hex listing shows them
    pub hex: String,
    pub pieces: Vec<Piece>,
    /// the same bytes as data, when a piece cannot be written symbolically
    pub raw: String,
    /// the address it references (shown as a comment, with the constant's value)
    pub refers: Option<usize>,
    /// a comment of its own (e.g. the value a `lui` + `addi` pair loads)
    pub note: Option<String>,
}

impl Item {
    pub fn text(pc: usize, hex: String, text: String, raw: String) -> Item {
        Item {
            pc,
            hex,
            pieces: vec![Piece::Text(text)],
            raw,
            refers: None,
            note: None,
        }
    }
}

/// a decoder of one architecture
pub trait Decoder {
    /// decodes `code` (the code area of a kernel, from offset 0) into items
    fn decode(&self, code: &[u8]) -> Vec<Item>;
    /// the comment marker of its assembler
    fn comment(&self) -> &'static str;
    /// the directives the assembly form starts with (after `.text`)
    fn directives(&self) -> Vec<&'static str>;
}

fn decoder(arch: &str) -> Result<Box<dyn Decoder>> {
    match arch {
        "riscv64" => Ok(Box::new(riscv::RiscvDecoder)),
        "aarch64" => Ok(Box::new(arm::ArmDecoder)),
        "x86_64" => Ok(Box::new(x86::X86Decoder)),
        _ => Err(anyhow!("no disassembler for {} yet", arch)),
    }
}

/// whether a data label holds an address (the function table) rather than a double
fn is_address(name: &str) -> bool {
    name.starts_with("_func_") || name.starts_with("_simd_") || name.starts_with("_env_")
}

/// the 8 bytes at `ofs` as a double
fn double_at(code: &[u8], ofs: usize) -> Option<f64> {
    let b = code.get(ofs..ofs + 8)?;
    Some(f64::from_le_bytes(b.try_into().ok()?))
}

/// a label as an assembler symbol: local (`.L`), with only the characters `as` accepts
fn asm_name(name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if clean.starts_with(".L") {
        clean
    } else if clean.starts_with('_') {
        format!(".L{}", clean)
    } else {
        format!(".L_{}", clean)
    }
}

/// the data area as items: `.quad`s, split where a label falls inside one
fn data_items(code: &[u8], code_end: usize, labels: &BTreeMap<usize, Vec<&str>>) -> Vec<Item> {
    let mut items = Vec::new();
    let mut ofs = code_end;
    let mut kind_address = false;

    while ofs < code.len() {
        if let Some(name) = labels.get(&ofs).and_then(|names| names.last()) {
            kind_address = is_address(name);
        }

        // the next label after `ofs` limits this item
        let next = labels
            .range(ofs + 1..)
            .next()
            .map(|(o, _)| *o)
            .unwrap_or(code.len())
            .min(code.len());

        if next - ofs >= 8 {
            let q = u64::from_le_bytes(code[ofs..ofs + 8].try_into().unwrap());
            let note = if kind_address {
                "address in the process that compiled the kernel".to_string()
            } else {
                format!("{:?}", f64::from_bits(q))
            };
            let text = format!(".quad 0x{:016x}", q);
            let mut item = Item::text(ofs, format!("{:016x}", q), text.clone(), text);
            item.note = Some(note);
            items.push(item);
            ofs += 8;
        } else {
            let b = &code[ofs..next];
            let hex: Vec<String> = b.iter().map(|x| format!("{:02x}", x)).collect();
            let bytes: Vec<String> = b.iter().map(|x| format!("0x{:02x}", x)).collect();
            let text = format!(".byte {}", bytes.join(", "));
            items.push(Item::text(ofs, hex.join(""), text.clone(), text));
            ofs = next;
        }
    }

    items
}

/// the listing of a kernel (`kind`: "scalar", "simd" or "fast") in the given mode
pub fn listing(
    arch: &str,
    kind: &str,
    code: &[u8],
    labels: &[(usize, String)],
    mode: Mode,
) -> Result<String> {
    let dec = decoder(arch)?;
    let cc = dec.comment();

    let mut by_offset: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for (ofs, name) in labels {
        by_offset.entry(*ofs).or_default().push(name.as_str());
    }

    let code_end = by_offset
        .iter()
        .find(|(_, names)| names.contains(&"@data"))
        .map(|(ofs, _)| *ofs)
        .unwrap_or(code.len())
        .min(code.len());

    let mut items = dec.decode(&code[..code_end]);
    items.extend(data_items(code, code_end, &by_offset));

    // the points a label can be written at: item starts and the end
    let mut points: HashSet<usize> = items.iter().map(|i| i.pc).collect();
    points.insert(code.len());

    // the names of the labels (unique assembler symbols in the assembly form), and one for every
    // other target; of several labels at one offset, references name the last (`_const_0_`
    // rather than `@data`)
    let mut names: HashMap<usize, String> = HashMap::new();
    let mut lines_of: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut taken: HashSet<String> = HashSet::new();
    let unique = |base: String, taken: &mut HashSet<String>| {
        let mut name = base.clone();
        let mut n = 1;
        while taken.contains(&name) {
            name = format!("{}_{}", base, n);
            n += 1;
        }
        taken.insert(name.clone());
        name
    };

    for (ofs, labs) in &by_offset {
        for lab in labs {
            let shown = match mode {
                Mode::Asm => unique(asm_name(lab), &mut taken),
                Mode::Hex => lab.to_string(),
            };
            lines_of.entry(*ofs).or_default().push(shown.clone());
            names.insert(*ofs, shown);
        }
    }

    let targets: Vec<usize> = items
        .iter()
        .flat_map(|i| i.pieces.iter())
        .filter_map(|p| match p {
            Piece::Target(t) => Some(*t),
            Piece::Page {
                target: Some(t), ..
            }
            | Piece::Lo12 { target: t, .. } => Some(*t),
            _ => None,
        })
        .collect();
    for t in targets {
        if mode == Mode::Asm && points.contains(&t) && !names.contains_key(&t) {
            let name = unique(format!(".L_at_{:x}", t), &mut taken);
            lines_of.entry(t).or_default().push(name.clone());
            names.insert(t, name);
        }
    }

    // a piece as text; None if the assembly form cannot write it
    let render = |p: &Piece| -> Option<String> {
        match (mode, p) {
            (_, Piece::Text(s)) => Some(s.clone()),
            (Mode::Asm, Piece::Target(t)) if points.contains(t) => names.get(t).cloned(),
            (
                Mode::Asm,
                Piece::Page {
                    target: Some(t), ..
                },
            ) if points.contains(t) => names.get(t).cloned(),
            (Mode::Asm, Piece::Lo12 { target, .. }) if points.contains(target) => {
                names.get(target).map(|n| format!(":lo12:{}", n))
            }
            (Mode::Asm, Piece::Rel { target, .. }) if points.contains(target) => {
                names.get(target).map(|n| format!("+{}", n))
            }
            (Mode::Asm, _) => None,
            (Mode::Hex, Piece::Target(t)) => Some(match names.get(t) {
                Some(n) => format!("0x{:x} <{}>", t, n),
                None => format!("0x{:x}", t),
            }),
            (Mode::Hex, Piece::Page { page, .. }) => Some(format!("0x{:x}", page)),
            (Mode::Hex, Piece::Lo12 { value, .. }) => Some(value.to_string()),
            (Mode::Hex, Piece::Rel { value, .. }) => Some(x86::signed_hex(*value)),
        }
    };

    let mut out = String::new();
    let entry = format!("symjit_{}", kind);

    let _ = writeln!(
        out,
        "{} symjit {} kernel, {}, {} bytes: code 0x0..0x{:x}, data 0x{:x}..0x{:x}",
        cc,
        kind,
        arch,
        code.len(),
        code_end,
        code_end,
        code.len()
    );

    if mode == Mode::Asm {
        let _ = writeln!(out, "\t.text");
        for d in dec.directives() {
            let _ = writeln!(out, "\t{}", d);
        }
        let _ = writeln!(out, "\t.globl {}", entry);
        let _ = writeln!(out, "\t.type {}, %function", entry);
        let _ = writeln!(out, "{}:", entry);
    }

    for item in &items {
        if mode == Mode::Asm && item.pc == code_end {
            let _ = writeln!(out, "\t.size {}, .-{}", entry, entry);
        }

        for name in lines_of.get(&item.pc).map(|v| &v[..]).unwrap_or(&[]) {
            let _ = writeln!(out, "{}:", name);
        }

        let text = item
            .pieces
            .iter()
            .map(render)
            .collect::<Option<Vec<String>>>()
            .map(|v| v.concat())
            .unwrap_or_else(|| item.raw.clone());

        let mut comment = Vec::new();
        if let Some(t) = item.refers {
            let name = names.get(&t).cloned();
            let mut c = match (mode, name) {
                (Mode::Asm, Some(n)) => n,
                (Mode::Hex, Some(n)) => format!("0x{:x} <{}>", t, n),
                (_, None) => format!("0x{:x}", t),
            };
            let label = by_offset.get(&t).and_then(|v| v.last());
            if t >= code_end && label.map(|l| !is_address(l)).unwrap_or(false) {
                if let Some(x) = double_at(code, t) {
                    let _ = write!(c, " = {:?}", x);
                }
            }
            comment.push(c);
        }
        if let Some(note) = &item.note {
            comment.push(note.clone());
        }

        let mut line = match mode {
            Mode::Asm => format!("\t{}", text),
            Mode::Hex => format!("{:>8x}:  {:<16}  {}", item.pc, item.hex, text),
        };
        if !comment.is_empty() {
            let _ = write!(line, "    {} {}", cc, comment.join("; "));
        }
        let _ = writeln!(out, "{}", line);
    }

    if mode == Mode::Asm && code_end == code.len() {
        let _ = writeln!(out, "\t.size {}, .-{}", entry, entry);
    }
    for name in lines_of.get(&code.len()).map(|v| &v[..]).unwrap_or(&[]) {
        let _ = writeln!(out, "{}:", name);
    }

    Ok(out)
}
