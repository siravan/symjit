//! RISC-V (RV64GC, little-endian) decoder for the instructions symjit emits: the `rvv!` macro
//! (`riscv64/macros.rs`). Anything else decodes as `.word`. The text follows GNU objdump's
//! (ABI register names, the pseudo-instructions `neg`, `not`, `nop`, `j`, `jr`, `ret`, `fmv.d`,
//! `fneg.d`, `fabs.d`; the rounding mode only when it is not `dyn`), except where the assembler
//! would encode the spelling differently: symjit's `mv` is `add rd, rs, zero` (`as` assembles
//! `mv` to `addi`). Pairs of `auipc` and `ld`/`fld`/`jalr` keep their numeric offsets (they are
//! relative, so the code assembles to the same bytes) and name the label they reach in a comment.

use super::{Decoder, Item, Piece};

const X: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

const F: [&str; 32] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2",
    "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9",
    "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

const RM: [&str; 8] = ["rne", "rtz", "rdn", "rup", "rmm", "rm5", "rm6", "dyn"];

/// an operand
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    X(u32),
    F(u32),
    Imm(i64),
    /// `imm(base)`
    Mem(i64, u32),
    /// a branch or jump target, relative to the instruction
    Rel(i64),
    /// a rounding mode other than `dyn`
    Rm(u32),
}

/// an instruction: its mnemonic and operands
#[derive(Debug, Clone, PartialEq)]
pub struct Ins {
    pub mnemonic: &'static str,
    pub ops: Vec<Op>,
}

fn ins(mnemonic: &'static str, ops: Vec<Op>) -> Option<Ins> {
    Some(Ins { mnemonic, ops })
}

fn imm_i(w: u32) -> i64 {
    ((w as i32) >> 20) as i64
}

fn imm_s(w: u32) -> i64 {
    ((((w as i32) >> 25) << 5) | ((w >> 7) & 0x1f) as i32) as i64
}

fn imm_b(w: u32) -> i64 {
    let imm = ((w >> 31) << 12)
        | (((w >> 7) & 1) << 11)
        | (((w >> 25) & 0x3f) << 5)
        | (((w >> 8) & 0x0f) << 1);
    (((imm << 19) as i32) >> 19) as i64
}

fn imm_j(w: u32) -> i64 {
    let imm = ((w >> 31) << 20)
        | (((w >> 12) & 0xff) << 12)
        | (((w >> 20) & 1) << 11)
        | (((w >> 21) & 0x3ff) << 1);
    (((imm << 11) as i32) >> 11) as i64
}

/// the rounding mode as an operand, or none for `dyn`
fn rm(f3: u32) -> Vec<Op> {
    if f3 == 7 {
        Vec::new()
    } else {
        vec![Op::Rm(f3)]
    }
}

/// decodes one instruction word; `None` if symjit does not emit it
pub fn decode_word(w: u32) -> Option<Ins> {
    use Op::*;

    let rd = (w >> 7) & 0x1f;
    let f3 = (w >> 12) & 7;
    let rs1 = (w >> 15) & 0x1f;
    let rs2 = (w >> 20) & 0x1f;
    let f7 = w >> 25;

    match w & 0x7f {
        0x33 => {
            let m = match (f7, f3) {
                (0x20, 0) if rs1 == 0 => return ins("neg", vec![X(rd), X(rs2)]),
                (0x00, 0) => "add",
                (0x20, 0) => "sub",
                (0x00, 1) => "sll",
                (0x00, 4) => "xor",
                (0x00, 5) => "srl",
                (0x20, 5) => "sra",
                (0x00, 6) => "or",
                (0x00, 7) => "and",
                _ => return None,
            };
            ins(m, vec![X(rd), X(rs1), X(rs2)])
        }
        0x13 => {
            let imm = imm_i(w);
            let shamt = ((w >> 20) & 0x3f) as i64;
            match f3 {
                0 if w == 0x13 => ins("nop", vec![]),
                0 => ins("addi", vec![X(rd), X(rs1), Imm(imm)]),
                4 if imm == -1 => ins("not", vec![X(rd), X(rs1)]),
                4 => ins("xori", vec![X(rd), X(rs1), Imm(imm)]),
                6 => ins("ori", vec![X(rd), X(rs1), Imm(imm)]),
                7 => ins("andi", vec![X(rd), X(rs1), Imm(imm)]),
                1 if w >> 26 == 0 => ins("slli", vec![X(rd), X(rs1), Imm(shamt)]),
                5 if w >> 26 == 0 => ins("srli", vec![X(rd), X(rs1), Imm(shamt)]),
                5 if w >> 26 == 0x10 => ins("srai", vec![X(rd), X(rs1), Imm(shamt)]),
                _ => None,
            }
        }
        0x03 if f3 == 3 => ins("ld", vec![X(rd), Mem(imm_i(w), rs1)]),
        0x07 if f3 == 3 => ins("fld", vec![F(rd), Mem(imm_i(w), rs1)]),
        0x23 if f3 == 3 => ins("sd", vec![X(rs2), Mem(imm_s(w), rs1)]),
        0x27 if f3 == 3 => ins("fsd", vec![F(rs2), Mem(imm_s(w), rs1)]),
        0x63 => {
            let m = match f3 {
                0 => "beq",
                1 => "bne",
                4 => "blt",
                5 => "bge",
                6 => "bltu",
                7 => "bgeu",
                _ => return None,
            };
            ins(m, vec![X(rs1), X(rs2), Rel(imm_b(w))])
        }
        0x6f if rd == 0 => ins("j", vec![Rel(imm_j(w))]),
        0x6f => ins("jal", vec![X(rd), Rel(imm_j(w))]),
        0x67 if f3 == 0 => {
            let imm = imm_i(w);
            match (rd, rs1, imm) {
                (0, 1, 0) => ins("ret", vec![]),
                (0, _, 0) => ins("jr", vec![X(rs1)]),
                _ => ins("jalr", vec![X(rd), Mem(imm, rs1)]),
            }
        }
        0x37 => ins("lui", vec![X(rd), Imm((w >> 12) as i64)]),
        0x17 => ins("auipc", vec![X(rd), Imm((w >> 12) as i64)]),
        0x53 => decode_fp(w, rd, f3, rs1, rs2, f7),
        0x43 | 0x47 | 0x4b | 0x4f if (w >> 25) & 3 == 1 => {
            let m = match w & 0x7f {
                0x43 => "fmadd.d",
                0x47 => "fmsub.d",
                0x4b => "fnmsub.d",
                _ => "fnmadd.d",
            };
            let mut ops = vec![F(rd), F(rs1), F(rs2), F(w >> 27)];
            ops.extend(rm(f3));
            ins(m, ops)
        }
        _ => None,
    }
}

/// OP-FP, the double-precision subset
fn decode_fp(w: u32, rd: u32, f3: u32, rs1: u32, rs2: u32, f7: u32) -> Option<Ins> {
    use Op::*;

    let arith = |m: &'static str| {
        let mut ops = vec![F(rd), F(rs1), F(rs2)];
        ops.extend(rm(f3));
        ins(m, ops)
    };

    match f7 {
        0x01 => arith("fadd.d"),
        0x05 => arith("fsub.d"),
        0x09 => arith("fmul.d"),
        0x0d => arith("fdiv.d"),
        0x2d if rs2 == 0 => {
            let mut ops = vec![F(rd), F(rs1)];
            ops.extend(rm(f3));
            ins("fsqrt.d", ops)
        }
        0x11 => match (f3, rs1 == rs2) {
            (0, true) => ins("fmv.d", vec![F(rd), F(rs1)]),
            (1, true) => ins("fneg.d", vec![F(rd), F(rs1)]),
            (2, true) => ins("fabs.d", vec![F(rd), F(rs1)]),
            (0, false) => ins("fsgnj.d", vec![F(rd), F(rs1), F(rs2)]),
            (1, false) => ins("fsgnjn.d", vec![F(rd), F(rs1), F(rs2)]),
            (2, false) => ins("fsgnjx.d", vec![F(rd), F(rs1), F(rs2)]),
            _ => None,
        },
        0x15 => match f3 {
            0 => ins("fmin.d", vec![F(rd), F(rs1), F(rs2)]),
            1 => ins("fmax.d", vec![F(rd), F(rs1), F(rs2)]),
            _ => None,
        },
        0x51 => {
            let m = match f3 {
                2 => "feq.d",
                1 => "flt.d",
                0 => "fle.d",
                _ => return None,
            };
            ins(m, vec![X(rd), F(rs1), F(rs2)])
        }
        0x61 => {
            let m = match rs2 {
                0 => "fcvt.w.d",
                2 => "fcvt.l.d",
                _ => return None,
            };
            let mut ops = vec![X(rd), F(rs1)];
            ops.extend(rm(f3));
            ins(m, ops)
        }
        // int -> double: fcvt.d.w is exact, and objdump omits its rounding mode
        0x69 => match rs2 {
            0 => ins("fcvt.d.w", vec![F(rd), X(rs1)]),
            2 => {
                let mut ops = vec![F(rd), X(rs1)];
                ops.extend(rm(f3));
                ins("fcvt.d.l", ops)
            }
            _ => None,
        },
        0x71 if f3 == 0 && rs2 == 0 => ins("fmv.x.d", vec![X(rd), F(rs1)]),
        0x79 if f3 == 0 && rs2 == 0 => ins("fmv.d.x", vec![F(rd), X(rs1)]),
        _ => {
            let _ = w;
            None
        }
    }
}

/// the instruction as assembly text; branch targets relative (`beq ra, sp, 16`), as an
/// assembler reads them
#[cfg(test)]
pub fn format(ins: &Ins) -> String {
    format_with(ins, &|rel| rel.to_string())
}

fn format_with(ins: &Ins, target: &dyn Fn(i64) -> String) -> String {
    let ops: Vec<String> = ins
        .ops
        .iter()
        .map(|op| match op {
            Op::X(r) => X[*r as usize].to_string(),
            Op::F(r) => F[*r as usize].to_string(),
            Op::Imm(i) => i.to_string(),
            Op::Mem(i, b) => format!("{}({})", i, X[*b as usize]),
            Op::Rel(rel) => target(*rel),
            Op::Rm(m) => RM[*m as usize].to_string(),
        })
        .collect();

    if ops.is_empty() {
        ins.mnemonic.to_string()
    } else {
        format!("{} {}", ins.mnemonic, ops.join(", "))
    }
}

/// the instruction as pieces of text: branch targets absolute (`pc` + offset)
fn pieces(ins: &Ins, pc: usize) -> Vec<Piece> {
    let mut out = vec![Piece::Text(ins.mnemonic.to_string())];

    for (k, op) in ins.ops.iter().enumerate() {
        let sep = if k == 0 { " " } else { ", " };
        out.push(Piece::Text(sep.to_string()));
        out.push(match op {
            Op::Rel(rel) => Piece::Target((pc as i64 + rel) as usize),
            _ => Piece::Text(format_with(
                &Ins {
                    mnemonic: "",
                    ops: vec![op.clone()],
                },
                &|rel| rel.to_string(),
            )),
        });
    }

    out
}

/// the listing's decoder: absolute branch targets, and the address an `auipc` + `ld`/`fld`/
/// `jalr`/`addi` pair forms (symjit's label references) or the value of `lui` + `addi` (`li`)
pub struct RiscvDecoder;

impl Decoder for RiscvDecoder {
    fn comment(&self) -> &'static str {
        "#"
    }

    // no compressed instructions, no linker relaxation: the same bytes as symjit's
    fn directives(&self) -> Vec<&'static str> {
        vec![".option norvc", ".option norelax", ".p2align 4"]
    }

    fn decode(&self, code: &[u8]) -> Vec<Item> {
        let mut items = Vec::new();
        // the register and value of the previous instruction's `auipc` (absolute) or `lui`
        let mut upper: Option<(u32, i64, bool)> = None;
        let mut pc = 0;

        while pc + 4 <= code.len() {
            let w = u32::from_le_bytes(code[pc..pc + 4].try_into().unwrap());
            let raw = format!(".word 0x{:08x}", w);
            let hex = format!("{:08x}", w);
            let prev = upper.take();

            let Some(ins) = decode_word(w) else {
                items.push(Item::text(pc, hex, raw.clone(), raw));
                pc += 4;
                continue;
            };

            let mut item = Item {
                pc,
                hex,
                pieces: pieces(&ins, pc),
                raw,
                refers: None,
                note: None,
            };

            match (ins.mnemonic, &ins.ops[..]) {
                ("auipc", [Op::X(rd), Op::Imm(imm)]) => {
                    let hi = (((*imm as u32) << 12) as i32) as i64;
                    upper = Some((*rd, pc as i64 + hi, true));
                }
                ("lui", [Op::X(rd), Op::Imm(imm)]) => {
                    let hi = (((*imm as u32) << 12) as i32) as i64;
                    upper = Some((*rd, hi, false));
                }
                (_, ops) => {
                    // the second half of a pair: its base register is the one `upper` set
                    if let Some((reg, value, pc_relative)) = prev {
                        let lo = match (ins.mnemonic, ops) {
                            ("ld" | "fld" | "jalr", [_, Op::Mem(lo, base)]) if *base == reg => {
                                Some(*lo)
                            }
                            ("addi", [_, Op::X(base), Op::Imm(lo)]) if *base == reg => Some(*lo),
                            ("jr", [Op::X(base)]) if *base == reg => Some(0),
                            _ => None,
                        };

                        match lo {
                            Some(lo) if pc_relative => item.refers = Some((value + lo) as usize),
                            Some(lo) => item.note = Some(format!("{}", (value + lo) as i32)),
                            None => {}
                        }
                    }
                }
            }

            items.push(item);
            pc += 4;
        }

        if pc < code.len() {
            let b = &code[pc..];
            let hex: Vec<String> = b.iter().map(|x| format!("{:02x}", x)).collect();
            let bytes: Vec<String> = b.iter().map(|x| format!("0x{:02x}", x)).collect();
            let text = format!(".byte {}", bytes.join(", "));
            items.push(Item::text(pc, hex.join(""), text.clone(), text));
        }

        items
    }
}
