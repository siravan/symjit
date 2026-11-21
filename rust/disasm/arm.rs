//! AArch64 decoder for the instructions symjit emits: the `arm!` macro (`arm/macros.rs`).
//! Anything else decodes as `.inst`. The text is GNU `as` syntax, spelled so that `as` encodes
//! it as symjit did: symjit's `nop` is `add x0, x0, #0`, its `mov xd, xn` is `orr` (or `add #0`
//! with `sp`), and the register forms of `fcmlt`/`fcmle` are `fcmgt`/`fcmge` with the operands
//! swapped. An `adrp` and the `ldr`/`add` after it (symjit's label references) become
//! `adrp xn, label` and `[xn, #:lo12:label]`.

use super::{Decoder, Item, Piece};

/// the offset field of a memory operand
#[derive(Debug, Clone, PartialEq)]
pub enum Off {
    None,
    Imm(i64),
    /// `xm, lsl #s`
    Reg(u32, u32),
}

/// an operand
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// a register, an immediate, a condition: as written
    Text(String),
    /// a branch or literal target, relative to the instruction
    Rel(i64),
    /// the page offset of an `adrp`, relative to the instruction's page
    Page(i64),
    /// `[base, offset]` (`base` 31 is `sp`)
    Mem(u32, Off),
}

/// an instruction: its mnemonic and operands
#[derive(Debug, Clone, PartialEq)]
pub struct Ins {
    pub mnemonic: String,
    pub ops: Vec<Op>,
}

const COND: [&str; 16] = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv",
];

/// x-register, 31 is `xzr`
fn x(r: u32) -> Op {
    Op::Text(if r == 31 {
        "xzr".to_string()
    } else {
        format!("x{}", r)
    })
}

/// x-register, 31 is `sp`
fn xs(r: u32) -> Op {
    Op::Text(if r == 31 {
        "sp".to_string()
    } else {
        format!("x{}", r)
    })
}

fn d(r: u32) -> Op {
    Op::Text(format!("d{}", r))
}

fn q(r: u32) -> Op {
    Op::Text(format!("q{}", r))
}

fn v(r: u32, arr: &str) -> Op {
    Op::Text(format!("v{}.{}", r, arr))
}

fn t(s: &str) -> Op {
    Op::Text(s.to_string())
}

fn imm(i: i64) -> Op {
    Op::Text(format!("#{}", i))
}

fn ins(mnemonic: &str, ops: Vec<Op>) -> Option<Ins> {
    Some(Ins {
        mnemonic: mnemonic.to_string(),
        ops,
    })
}

/// sign-extends the low `bits` bits
fn sext(x: u32, bits: u32) -> i64 {
    (((x << (32 - bits)) as i32) >> (32 - bits)) as i64
}

/// the double of an 8-bit floating-point immediate (VFPExpandImm)
fn fp8(imm8: u32) -> f64 {
    let sign = (imm8 >> 7) as u64;
    let b6 = ((imm8 >> 6) & 1) as u64;
    let repl: u64 = if b6 == 1 { 0xff } else { 0 };
    let exp = ((b6 ^ 1) << 10) | (repl << 2) | ((imm8 >> 4) & 3) as u64;
    let frac = ((imm8 & 15) as u64) << 48;
    f64::from_bits((sign << 63) | (exp << 52) | frac)
}

/// the double immediate as `as` writes it (`#0.5`, `#-1.0`)
fn fimm(imm8: u32) -> Op {
    Op::Text(format!("#{:?}", fp8(imm8)))
}

// masks of the register fields: rd, rn, rm, ra
const M_D: u32 = 0x1f;
const M_N: u32 = 0x1f << 5;
const M_M: u32 = 0x1f << 16;
const M_A: u32 = 0x1f << 10;

/// decodes one instruction word; `None` if symjit does not emit it
pub fn decode_word(w: u32) -> Option<Ins> {
    let rd = w & 0x1f;
    let rn = (w >> 5) & 0x1f;
    let rm = (w >> 16) & 0x1f;
    let ra = (w >> 10) & 0x1f;

    let is = |base: u32, fields: u32| w & !fields == base;
    let r3 = M_D | M_N | M_M;
    let r2 = M_D | M_N;

    // scalar double: three registers
    for (base, m) in [
        (0x1e602800, "fadd"),
        (0x1e603800, "fsub"),
        (0x1e600800, "fmul"),
        (0x1e601800, "fdiv"),
        (0x5e60e400, "fcmeq"),
        (0x7ee0e400, "fcmgt"),
        (0x7e60e400, "fcmge"),
    ] {
        if is(base, r3) {
            return ins(m, vec![d(rd), d(rn), d(rm)]);
        }
    }

    // scalar double: two registers
    for (base, m) in [
        (0x1e604000, "fmov"),
        (0x1e61c000, "fsqrt"),
        (0x1e614000, "fneg"),
        (0x1e60c000, "fabs"),
        (0x1e67c000, "frinti"),
        (0x1e654000, "frintm"),
        (0x1e64c000, "frintp"),
        (0x1e65c000, "frintz"),
    ] {
        if is(base, r2) {
            return ins(m, vec![d(rd), d(rn)]);
        }
    }

    for (base, m) in [
        (0x5ee0d800, "fcmeq"),
        (0x7ee0c800, "fcmge"),
        (0x5ee0e800, "fcmlt"),
    ] {
        if is(base, r2) {
            return ins(m, vec![d(rd), d(rn), t("#0.0")]);
        }
    }

    for (base, m) in [
        (0x1f400000, "fmadd"),
        (0x1f408000, "fmsub"),
        (0x1f600000, "fnmadd"),
        (0x1f608000, "fnmsub"),
    ] {
        if is(base, r3 | M_A) {
            return ins(m, vec![d(rd), d(rn), d(rm), d(ra)]);
        }
    }

    if is(0x9e670000, r2) {
        return ins("fmov", vec![d(rd), x(rn)]);
    }
    if is(0x9e660000, r2) {
        return ins("fmov", vec![x(rd), d(rn)]);
    }
    if is(0x7e70d800, r2) {
        return ins("faddp", vec![d(rd), v(rn, "2d")]);
    }
    if is(0x1e602008, M_N) {
        return ins("fcmp", vec![d(rn), t("#0.0")]);
    }
    if is(0x1e600c00, r3 | (0xf << 12)) {
        let cond = COND[((w >> 12) & 0xf) as usize];
        return ins("fcsel", vec![d(rd), d(rn), d(rm), t(cond)]);
    }
    if is(0x1e601000, M_D | (0xff << 13)) {
        return ins("fmov", vec![d(rd), fimm((w >> 13) & 0xff)]);
    }

    // immediates of vectors: fmov v.2d, #imm and the zero moves
    if is(0x6f00f400, M_D | (0x1f << 5) | (7 << 16)) {
        let imm8 = (((w >> 16) & 7) << 5) | ((w >> 5) & 0x1f);
        return ins("fmov", vec![v(rd, "2d"), fimm(imm8)]);
    }
    if is(0x4f00e400, M_D) {
        return ins("movi", vec![v(rd, "16b"), t("#0")]);
    }
    if is(0x2f00e400, M_D) {
        return ins("movi", vec![d(rd), t("#0")]);
    }
    if is(0x6f00e400, M_D) {
        return ins("movi", vec![v(rd, "2d"), t("#0")]);
    }

    // logical operations on bytes, 8b or 16b (bit 30)
    for (base, m) in [
        (0x0e201c00, "and"),
        (0x0ea01c00, "orr"),
        (0x2e201c00, "eor"),
        (0x2ea01c00, "bit"),
        (0x2ee01c00, "bif"),
        (0x0e601c00, "bic"),
        (0x2e601c00, "bsl"),
    ] {
        if is(base, r3 | (1 << 30)) {
            let arr = if w & (1 << 30) != 0 { "16b" } else { "8b" };
            if m == "orr" && rn == rm {
                return ins("mov", vec![v(rd, arr), v(rn, arr)]);
            }
            return ins(m, vec![v(rd, arr), v(rn, arr), v(rm, arr)]);
        }
    }
    if is(0x2e205800, r2 | (1 << 30)) {
        let arr = if w & (1 << 30) != 0 { "16b" } else { "8b" };
        return ins("not", vec![v(rd, arr), v(rn, arr)]);
    }

    // packed doubles: three registers
    for (base, m, rot) in [
        (0x4e60d400, "fadd", None),
        (0x4ee0d400, "fsub", None),
        (0x6e60dc00, "fmul", None),
        (0x6e60fc00, "fdiv", None),
        (0x4e60cc00, "fmla", None),
        (0x4ee0cc00, "fmls", None),
        (0x4e60e400, "fcmeq", None),
        (0x6ee0e400, "fcmgt", None),
        (0x6e60e400, "fcmge", None),
        (0x4ec03800, "zip1", None),
        (0x4ec07800, "zip2", None),
        (0x4ec01800, "uzp1", None),
        (0x4ec05800, "uzp2", None),
        (0x6ec0c400, "fcmla", Some("#0")),
        (0x6ec0cc00, "fcmla", Some("#90")),
        (0x6ec0d400, "fcmla", Some("#180")),
        (0x6ec0dc00, "fcmla", Some("#270")),
        (0x6ec0e400, "fcadd", Some("#90")),
        (0x6ec0f400, "fcadd", Some("#270")),
    ] {
        if is(base, r3) {
            let mut ops = vec![v(rd, "2d"), v(rn, "2d"), v(rm, "2d")];
            if let Some(r) = rot {
                ops.push(t(r));
            }
            return ins(m, ops);
        }
    }
    if is(0x6e004000, r3) {
        return ins(
            "ext",
            vec![v(rd, "16b"), v(rn, "16b"), v(rm, "16b"), t("#8")],
        );
    }

    // packed doubles: two registers
    for (base, m) in [
        (0x6ee1f800, "fsqrt"),
        (0x6ee0f800, "fneg"),
        (0x4ee0f800, "fabs"),
        (0x6ee19800, "frinti"),
        (0x4e619800, "frintm"),
        (0x4ee18800, "frintp"),
        (0x4ee19800, "frintz"),
    ] {
        if is(base, r2) {
            return ins(m, vec![v(rd, "2d"), v(rn, "2d")]);
        }
    }
    if is(0x4ee0d800, r2) {
        return ins("fcmeq", vec![v(rd, "2d"), v(rn, "2d"), t("#0.0")]);
    }
    if is(0x4e200800, r2) {
        return ins("rev64", vec![v(rd, "16b"), v(rn, "16b")]);
    }

    // by element: v.2d, v.2d, v.d[index] (index in bit 11)
    for (base, m) in [
        (0x4fc01000, "fmla"),
        (0x4fc05000, "fmls"),
        (0x4fc09000, "fmul"),
    ] {
        if is(base, r3 | (1 << 11)) {
            let index = (w >> 11) & 1;
            return ins(
                m,
                vec![
                    v(rd, "2d"),
                    v(rn, "2d"),
                    Op::Text(format!("v{}.d[{}]", rm, index)),
                ],
            );
        }
    }
    for (base, index) in [(0x4e080400, 0), (0x4e180400, 1)] {
        if is(base, r2) {
            return ins(
                "dup",
                vec![v(rd, "2d"), Op::Text(format!("v{}.d[{}]", rn, index))],
            );
        }
    }
    for (base, index) in [(0x4e083c00, 0), (0x4e183c00, 1)] {
        if is(base, r2) {
            return ins(
                "umov",
                vec![x(rd), Op::Text(format!("v{}.d[{}]", rn, index))],
            );
        }
    }
    if is(0x4d40cc00, r2) {
        return ins(
            "ld1r",
            vec![Op::Text(format!("{{v{}.2d}}", rd)), Op::Mem(rn, Off::None)],
        );
    }

    // integer registers
    if is(0xaa0003e0, M_D | M_M) {
        return ins("mov", vec![x(rd), x(rm)]);
    }
    if is(0xea00001f, M_N | M_M) {
        return ins("tst", vec![x(rn), x(rm)]);
    }
    for (base, m) in [
        (0x8a000000, "and"),
        (0xea000000, "ands"),
        (0xaa000000, "orr"),
        (0xaa200000, "orn"),
        (0xca000000, "eor"),
        (0xab000000, "adds"),
    ] {
        if is(base, r3) {
            return ins(m, vec![x(rd), x(rn), x(rm)]);
        }
    }
    if is(0x8b000000, r3 | (0x3f << 10)) {
        let shift = (w >> 10) & 0x3f;
        let mut ops = vec![x(rd), x(rn), x(rm)];
        if shift != 0 {
            ops.push(Op::Text(format!("lsl #{}", shift)));
        }
        return ins("add", ops);
    }

    // add/sub immediate (lsl #12 in bit 22); `mov` to or from sp is `add #0`
    for (base, m) in [
        (0x91000000, "add"),
        (0xd1000000, "sub"),
        (0xf1000000, "subs"),
    ] {
        if is(base, r2 | (0xfff << 10) | (1 << 22)) {
            let value = ((w >> 10) & 0xfff) as i64;
            let lsl12 = w & (1 << 22) != 0;
            if m == "add" && value == 0 && !lsl12 && (rd == 31 || rn == 31) {
                return ins("mov", vec![xs(rd), xs(rn)]);
            }
            let dst = if m == "subs" { x(rd) } else { xs(rd) };
            let mut ops = vec![dst, xs(rn), imm(value)];
            if lsl12 {
                ops.push(t("lsl #12"));
            }
            return ins(m, ops);
        }
    }

    if is(0xd3400000, r2 | (0x3f << 16) | (0x3f << 10)) {
        let immr = (w >> 16) & 0x3f;
        let imms = (w >> 10) & 0x3f;
        if imms == 63 {
            return ins("lsr", vec![x(rd), x(rn), imm(immr as i64)]);
        }
        return ins(
            "ubfm",
            vec![x(rd), x(rn), imm(immr as i64), imm(imms as i64)],
        );
    }

    for (base, m) in [(0xd2800000, "movz"), (0xf2800000, "movk")] {
        if is(base, M_D | (0xffff << 5) | (3 << 21)) {
            let value = (w >> 5) & 0xffff;
            let hw = (w >> 21) & 3;
            let mut ops = vec![x(rd), Op::Text(format!("#0x{:x}", value))];
            if hw != 0 {
                ops.push(Op::Text(format!("lsl #{}", 16 * hw)));
            }
            return ins(m, ops);
        }
    }

    if w & 0x9f000000 == 0x90000000 {
        let immlo = (w >> 29) & 3;
        let immhi = (w >> 5) & 0x7ffff;
        let pages = sext((immhi << 2) | immlo, 21);
        return ins("adrp", vec![x(rd), Op::Page(pages << 12)]);
    }

    // loads and stores: unsigned offset (scaled), register offset, literal, pairs
    for (base, m, reg, scale) in [
        (0xfd400000u32, "ldr", 'd', 8),
        (0xfd000000, "str", 'd', 8),
        (0xf9400000, "ldr", 'x', 8),
        (0xf9000000, "str", 'x', 8),
        (0x3dc00000, "ldr", 'q', 16),
        (0x3d800000, "str", 'q', 16),
    ] {
        if is(base, r2 | (0xfff << 10)) {
            let ofs = ((w >> 10) & 0xfff) as i64 * scale;
            let r = match reg {
                'd' => d(rd),
                'x' => x(rd),
                _ => q(rd),
            };
            return ins(m, vec![r, Op::Mem(rn, Off::Imm(ofs))]);
        }
    }

    for (base, m, reg, shift) in [
        (0xfc607800u32, "ldr", 'd', 3),
        (0xfc207800, "str", 'd', 3),
        (0xf8607800, "ldr", 'x', 3),
        (0x3ce07800, "ldr", 'q', 4),
        (0x3ca07800, "str", 'q', 4),
    ] {
        if is(base, r3) {
            let r = match reg {
                'd' => d(rd),
                'x' => x(rd),
                _ => q(rd),
            };
            return ins(m, vec![r, Op::Mem(rn, Off::Reg(rm, shift))]);
        }
    }

    for (base, reg) in [(0x5c000000u32, 'd'), (0x58000000, 'x'), (0x9c000000, 'q')] {
        if is(base, M_D | (0x7ffff << 5)) {
            let rel = sext((w >> 5) & 0x7ffff, 19) * 4;
            let r = match reg {
                'd' => d(rd),
                'x' => x(rd),
                _ => q(rd),
            };
            return ins("ldr", vec![r, Op::Rel(rel)]);
        }
    }

    for (base, m, reg, scale) in [
        (0x6d400000u32, "ldp", 'd', 8),
        (0x6d000000, "stp", 'd', 8),
        (0xad400000, "ldp", 'q', 16),
        (0xad000000, "stp", 'q', 16),
        (0xa9400000, "ldp", 'x', 8),
        (0xa9000000, "stp", 'x', 8),
    ] {
        if is(base, r2 | M_A | (0x7f << 15)) {
            let ofs = sext((w >> 15) & 0x7f, 7) * scale;
            let (r1, r2) = match reg {
                'd' => (d(rd), d(ra)),
                'x' => (x(rd), x(ra)),
                _ => (q(rd), q(ra)),
            };
            return ins(m, vec![r1, r2, Op::Mem(rn, Off::Imm(ofs))]);
        }
    }

    // branches
    if is(0x14000000, 0x03ffffff) {
        return ins("b", vec![Op::Rel(sext(w & 0x03ffffff, 26) * 4)]);
    }
    if is(0x94000000, 0x03ffffff) {
        return ins("bl", vec![Op::Rel(sext(w & 0x03ffffff, 26) * 4)]);
    }
    if is(0x54000000, 0xf | (0x7ffff << 5)) {
        let cond = COND[(w & 0xf) as usize];
        let rel = sext((w >> 5) & 0x7ffff, 19) * 4;
        return ins(&format!("b.{}", cond), vec![Op::Rel(rel)]);
    }
    for (base, m) in [(0x36000000u32, "tbz"), (0x37000000, "tbnz")] {
        if is(base, M_D | (0x3fff << 5) | (0x1f << 19) | (1 << 31)) {
            let bit = ((w >> 31) << 5) | ((w >> 19) & 0x1f);
            let rel = sext((w >> 5) & 0x3fff, 14) * 4;
            return ins(m, vec![x(rd), imm(bit as i64), Op::Rel(rel)]);
        }
    }
    if is(0xd63f0000, M_N) {
        return ins("blr", vec![x(rn)]);
    }
    if w == 0xd65f03c0 {
        return ins("ret", vec![]);
    }

    None
}

fn mem_text(base: u32, off: &Off) -> String {
    let b = if base == 31 {
        "sp".to_string()
    } else {
        format!("x{}", base)
    };
    match off {
        Off::None => format!("[{}]", b),
        Off::Imm(i) => format!("[{}, #{}]", b, i),
        Off::Reg(r, s) => {
            let r = if *r == 31 {
                "xzr".to_string()
            } else {
                format!("x{}", r)
            };
            format!("[{}, {}, lsl #{}]", b, r, s)
        }
    }
}

/// the instruction as assembly text; targets relative (`b #256`, `adrp x1, #0x5000`), as the
/// encoding tests write them
#[cfg(test)]
pub fn format(ins: &Ins) -> String {
    let ops: Vec<String> = ins
        .ops
        .iter()
        .map(|op| match op {
            Op::Text(s) => s.clone(),
            Op::Rel(rel) => format!("#{}", rel),
            Op::Page(rel) => format!("#0x{:x}", rel),
            Op::Mem(base, off) => mem_text(*base, off),
        })
        .collect();

    if ops.is_empty() {
        ins.mnemonic.clone()
    } else {
        format!("{} {}", ins.mnemonic, ops.join(", "))
    }
}

/// the instruction as pieces at `pc`; `pair` is the target of an `adrp` + `ldr`/`add` pair
fn pieces(ins: &Ins, pc: usize, pair: Option<usize>) -> Vec<Piece> {
    let mut out = vec![Piece::Text(ins.mnemonic.clone())];

    for (k, op) in ins.ops.iter().enumerate() {
        out.push(Piece::Text(if k == 0 { " " } else { ", " }.to_string()));
        match op {
            Op::Text(s) => out.push(Piece::Text(s.clone())),
            Op::Rel(rel) => out.push(Piece::Target((pc as i64 + rel) as usize)),
            Op::Page(rel) => out.push(Piece::Page {
                page: ((pc & !0xfff) as i64 + rel) as usize,
                target: pair,
            }),
            Op::Mem(base, Off::Imm(i)) if pair.is_some() => {
                let b = if *base == 31 {
                    "sp".to_string()
                } else {
                    format!("x{}", base)
                };
                out.push(Piece::Text(format!("[{}, #", b)));
                out.push(Piece::Lo12 {
                    target: pair.unwrap(),
                    value: *i,
                });
                out.push(Piece::Text("]".to_string()));
            }
            Op::Mem(base, off) => out.push(Piece::Text(mem_text(*base, off))),
        }
    }

    // `add xd, xn, #lo12` of a pair
    if let (Some(target), "add") = (pair, ins.mnemonic.as_str()) {
        if let Some(Piece::Text(last)) = out.last().cloned() {
            if let Some(value) = last.strip_prefix('#').and_then(|s| s.parse::<i64>().ok()) {
                out.pop();
                out.push(Piece::Text("#".to_string()));
                out.push(Piece::Lo12 { target, value });
            }
        }
    }

    out
}

/// the base register and the low 12 bits of the second half of an `adrp` pair
fn pair_low(ins: &Ins) -> Option<(u32, i64)> {
    match (ins.mnemonic.as_str(), &ins.ops[..]) {
        ("ldr", [_, Op::Mem(base, Off::Imm(i))]) => Some((*base, *i)),
        ("add", [_, Op::Text(base), Op::Text(value)]) => {
            let base = base.strip_prefix('x')?.parse().ok()?;
            let value = value.strip_prefix('#')?.parse().ok()?;
            Some((base, value))
        }
        _ => None,
    }
}

pub struct ArmDecoder;

impl Decoder for ArmDecoder {
    fn comment(&self) -> &'static str {
        "//"
    }

    // fcmla/fcadd are Armv8.3
    fn directives(&self) -> Vec<&'static str> {
        vec![".arch armv8.3-a", ".p2align 4"]
    }

    fn decode(&self, code: &[u8]) -> Vec<Item> {
        let words: Vec<(usize, u32)> = code
            .chunks_exact(4)
            .enumerate()
            .map(|(k, b)| (4 * k, u32::from_le_bytes(b.try_into().unwrap())))
            .collect();
        let decoded: Vec<Option<Ins>> = words.iter().map(|(_, w)| decode_word(*w)).collect();

        // the targets of adrp pairs, at both halves
        let mut pairs: Vec<Option<usize>> = vec![None; words.len()];
        for k in 0..words.len().saturating_sub(1) {
            if let (Some(a), Some(b)) = (&decoded[k], &decoded[k + 1]) {
                if let ("adrp", [Op::Text(r), Op::Page(rel)]) = (a.mnemonic.as_str(), &a.ops[..]) {
                    let reg: Option<u32> = r.strip_prefix('x').and_then(|s| s.parse().ok());
                    if let (Some(reg), Some((base, lo))) = (reg, pair_low(b)) {
                        if reg == base && (0..4096).contains(&lo) {
                            let pc = words[k].0;
                            let target = ((pc & !0xfff) as i64 + rel + lo) as usize;
                            pairs[k] = Some(target);
                            pairs[k + 1] = Some(target);
                        }
                    }
                }
            }
        }

        let mut items = Vec::new();
        for (k, (pc, w)) in words.iter().enumerate() {
            let raw = format!(".inst 0x{:08x}", w);
            let hex = format!("{:08x}", w);
            match &decoded[k] {
                Some(ins) => {
                    let is_adrp = ins.mnemonic == "adrp";
                    let mut item = Item {
                        pc: *pc,
                        hex,
                        pieces: pieces(ins, *pc, pairs[k]),
                        raw,
                        refers: None,
                        note: None,
                    };
                    // the comment names the label at the second half (and at literals)
                    if !is_adrp {
                        item.refers = pairs[k];
                    }
                    if ins.mnemonic == "ldr" {
                        if let [_, Op::Rel(rel)] = &ins.ops[..] {
                            item.refers = Some((*pc as i64 + rel) as usize);
                        }
                    }
                    items.push(item);
                }
                None => items.push(Item::text(*pc, hex, raw.clone(), raw)),
            }
        }

        let tail = words.len() * 4;
        if tail < code.len() {
            let b = &code[tail..];
            let hex: Vec<String> = b.iter().map(|x| format!("{:02x}", x)).collect();
            let bytes: Vec<String> = b.iter().map(|x| format!("0x{:02x}", x)).collect();
            let text = format!(".byte {}", bytes.join(", "));
            items.push(Item::text(tail, hex.join(""), text.clone(), text));
        }

        items
    }
}
