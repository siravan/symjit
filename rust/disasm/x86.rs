//! x86-64 decoder for the instructions symjit emits (`amd/asm.rs`, `fused.rs`, `f64x8.rs`):
//! legacy integer and SSE instructions, VEX and EVEX. Anything else decodes as `.byte`. The
//! text is GNU `as` Intel syntax (`.intel_syntax noprefix`), with the pseudo-prefixes that make
//! `as` choose symjit's encoding where its own choice would differ: `{load}`/`{store}` (the
//! direction of a register-to-register form), `{disp8}`/`{disp32}` (a displacement longer than
//! needed; jumps with a 32-bit offset), `{vex3}`, `{evex}`, and `rex.w` (a REX.W that changes
//! nothing). An encoding no pseudo-prefix gives (a scalar instruction with VEX.L=1, an
//! immediate longer than needed) is written as `.insn` with its encoding, the instruction in a
//! comment.

use super::{Decoder, Item, Piece};

const R64: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
const R32: [&str; 16] = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d",
    "r13d", "r14d", "r15d",
];
const R16: [&str; 16] = [
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w",
    "r14w", "r15w",
];
const CC: [&str; 16] = [
    "o", "no", "b", "nb", "z", "nz", "be", "nbe", "s", "ns", "pe", "po", "l", "nl", "le", "nle",
];
const PRED: [&str; 32] = [
    "eq", "lt", "le", "unord", "neq", "nlt", "nle", "ord", "eq_uq", "nge", "ngt", "false",
    "neq_oq", "ge", "gt", "true", "eq_os", "lt_oq", "le_oq", "unord_s", "neq_us", "nlt_uq",
    "nle_uq", "ord_s", "eq_us", "nge_uq", "ngt_uq", "false_os", "neq_os", "ge_oq", "gt_oq",
    "true_us",
];

/// a memory operand
#[derive(Debug, Clone, PartialEq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<(u8, u8)>,
    pub disp: i64,
    pub rip: bool,
    /// the displacement's size in the encoding: 0, 8 or 32 (bits)
    pub disp_size: u8,
}

/// an operand
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Text(String),
    /// `ptr` (e.g. "qword ptr " or "") and the address
    Mem(&'static str, Mem),
    /// a branch or call target, relative to the next instruction
    Rel(i64),
}

/// an instruction: pseudo-prefixes, mnemonic and operands; `insn` is the `.insn` encoding
/// (before the operands) when no pseudo-prefix gives symjit's encoding
#[derive(Debug, Clone, PartialEq)]
pub struct Ins {
    pub len: usize,
    pub prefixes: Vec<&'static str>,
    pub mnemonic: String,
    pub ops: Vec<Op>,
    pub insn: Option<String>,
}

fn t(s: &str) -> Op {
    Op::Text(s.to_string())
}

/// the vector register of a size (16, 32, 64 bytes)
fn vreg(n: u8, bytes: u32) -> Op {
    let p = match bytes {
        64 => "zmm",
        32 => "ymm",
        _ => "xmm",
    };
    Op::Text(format!("{}{}", p, n))
}

fn ptr(bytes: u32) -> &'static str {
    match bytes {
        1 => "byte ptr ",
        2 => "word ptr ",
        4 => "dword ptr ",
        8 => "qword ptr ",
        16 => "xmmword ptr ",
        32 => "ymmword ptr ",
        64 => "zmmword ptr ",
        _ => "",
    }
}

/// the prefixes of an encoding
#[derive(Default, Clone, Copy)]
struct Prefix {
    p66: bool,
    pf2: bool,
    pf3: bool,
    rex: Option<u8>,
    // VEX or EVEX
    vex: Option<Vex>,
}

#[derive(Clone, Copy)]
struct Vex {
    evex: bool,
    three: bool,
    map: u8,
    w: bool,
    l: u8,
    pp: u8,
    vvvv: u8,
    r: u8,
    x: u8,
    b: u8,
    // EVEX
    rr: u8,
    vv: u8,
    aaa: u8,
    z: bool,
    bcast: bool,
}

/// a decoded ModRM (and SIB, displacement)
#[derive(Clone)]
struct ModRm {
    md: u8,
    reg: u8,
    rm: u8,
    mem: Option<Mem>,
    len: usize,
}

fn modrm(code: &[u8], at: usize, rex_x: u8, rex_b: u8) -> Option<ModRm> {
    let b = *code.get(at)?;
    let md = b >> 6;
    let reg = (b >> 3) & 7;
    let rm = b & 7;

    if md == 3 {
        return Some(ModRm {
            md,
            reg,
            rm,
            mem: None,
            len: 1,
        });
    }

    let mut len = 1;
    let mut base = Some(rm | (rex_b << 3));
    let mut index = None;
    let mut rip = false;
    let mut disp_size = match md {
        1 => 8,
        2 => 32,
        _ => 0,
    };

    if rm == 4 {
        let s = *code.get(at + 1)?;
        len += 1;
        let idx = ((s >> 3) & 7) | (rex_x << 3);
        if idx != 4 {
            index = Some((idx, 1 << (s >> 6)));
        }
        base = Some((s & 7) | (rex_b << 3));
        if s & 7 == 5 && md == 0 {
            base = None;
            disp_size = 32;
        }
    } else if rm == 5 && md == 0 {
        base = None;
        rip = true;
        disp_size = 32;
    }

    let disp = match disp_size {
        8 => *code.get(at + len)? as i8 as i64,
        32 => i32::from_le_bytes(code.get(at + len..at + len + 4)?.try_into().ok()?) as i64,
        _ => 0,
    };
    len += disp_size as usize / 8;

    Some(ModRm {
        md,
        reg,
        rm,
        mem: Some(Mem {
            base,
            index,
            disp,
            rip,
            disp_size,
        }),
        len,
    })
}

/// the displacement size `as` chooses for an address (`n`: the EVEX disp8 scale, else 1)
fn gas_disp_size(m: &Mem, n: i64) -> u8 {
    if m.rip || m.base.is_none() {
        32
    } else if m.disp == 0 && m.base.map(|b| b & 7) != Some(5) {
        0
    } else if m.disp % n == 0 && (-128..128).contains(&(m.disp / n)) {
        8
    } else {
        32
    }
}

/// `+0x10`, `-0x10`
pub fn signed_hex(x: i64) -> String {
    if x < 0 {
        format!("-{:#x}", -x)
    } else {
        format!("+{:#x}", x)
    }
}

fn mem_text(m: &Mem, target: Option<&str>) -> String {
    let mut s = String::from("[");
    if m.rip {
        s.push_str("rip");
        match target {
            Some(t) => {
                s.push('+');
                s.push_str(t);
            }
            None => s.push_str(&signed_hex(m.disp)),
        }
        s.push(']');
        return s;
    }
    let mut parts = Vec::new();
    if let Some(b) = m.base {
        parts.push(R64[b as usize].to_string());
    }
    if let Some((i, sc)) = m.index {
        parts.push(format!("{}*{}", R64[i as usize], sc));
    }
    s.push_str(&parts.join("+"));
    if m.disp != 0 || parts.is_empty() {
        if m.disp < 0 {
            s.push_str(&format!("-{:#x}", -m.disp));
        } else if parts.is_empty() {
            s.push_str(&format!("{:#x}", m.disp));
        } else {
            s.push_str(&format!("+{:#x}", m.disp));
        }
    }
    s.push(']');
    s
}

/// the instruction text for the tests: relative targets as `L` when `label` is the target
#[cfg(test)]
pub fn format(ins: &Ins, label: Option<i64>) -> String {
    let ops: Vec<String> = ins
        .ops
        .iter()
        .map(|op| match op {
            Op::Text(s) => s.clone(),
            Op::Mem(p, m) => {
                let target = if m.rip && Some(m.disp) == label {
                    Some("L")
                } else {
                    None
                };
                format!("{}{}", p, mem_text(m, target))
            }
            Op::Rel(r) if Some(*r) == label => "L".to_string(),
            Op::Rel(r) => format!("{:+}", r),
        })
        .collect();
    let mut s = String::new();
    for p in &ins.prefixes {
        s.push_str(p);
        s.push(' ');
    }
    s.push_str(&ins.mnemonic);
    if !ops.is_empty() {
        s.push(' ');
        s.push_str(&ops.join(", "));
    }
    s
}

/// the `.insn` form of an instruction (`ins.insn` set), as the tests assemble it
#[cfg(test)]
pub fn format_insn(ins: &Ins, label: Option<i64>) -> String {
    let plain = format(
        &Ins {
            prefixes: vec![],
            ..ins.clone()
        },
        label,
    );
    match &ins.insn {
        Some(spec) => {
            let ops = plain.split_once(' ').map(|(_, o)| o).unwrap_or("");
            let ops = ops.replace(" {:s32}", "{:s32}");
            format!(".insn {}, {}", spec, ops)
        }
        None => format(ins, label),
    }
}

struct Ctx<'a> {
    code: &'a [u8],
    p: Prefix,
    i: usize,
    prefixes: Vec<&'static str>,
    insn: Option<String>,
}

impl<'a> Ctx<'a> {
    fn byte(&self, at: usize) -> Option<u8> {
        self.code.get(at).copied()
    }

    fn rex_bits(&self) -> (u8, u8, u8, u8) {
        match (self.p.vex, self.p.rex) {
            (Some(v), _) => (v.w as u8, v.r, v.x, v.b),
            (None, Some(r)) => ((r >> 3) & 1, (r >> 2) & 1, (r >> 1) & 1, r & 1),
            (None, None) => (0, 0, 0, 0),
        }
    }

    fn modrm(&mut self) -> Option<ModRm> {
        let (_, _, x, b) = self.rex_bits();
        let m = modrm(self.code, self.i, x, b)?;
        self.i += m.len;
        Some(m)
    }

    fn imm8(&mut self) -> Option<i64> {
        let v = self.byte(self.i)? as i64;
        self.i += 1;
        Some(v)
    }

    fn imm32(&mut self) -> Option<i64> {
        let v = i32::from_le_bytes(self.code.get(self.i..self.i + 4)?.try_into().ok()?) as i64;
        self.i += 4;
        Some(v)
    }

    /// the displacement pseudo-prefix if `as` would choose another size
    fn disp_prefix(&mut self, m: &ModRm, n: i64) {
        if let Some(mem) = &m.mem {
            let want = mem.disp_size;
            if want != gas_disp_size(mem, n) {
                self.prefixes
                    .push(if want == 8 { "{disp8}" } else { "{disp32}" });
            }
        }
    }

    fn done(self, mnemonic: &str, ops: Vec<Op>) -> Option<Ins> {
        Some(Ins {
            len: self.i,
            prefixes: self.prefixes,
            mnemonic: mnemonic.to_string(),
            ops,
            insn: self.insn,
        })
    }
}

/// decodes the instruction at the start of `code`; `None` if symjit does not emit it
pub fn decode(code: &[u8]) -> Option<Ins> {
    let mut p = Prefix::default();
    let mut i = 0;

    loop {
        match *code.get(i)? {
            0x66 => p.p66 = true,
            0xf2 => p.pf2 = true,
            0xf3 => p.pf3 = true,
            _ => break,
        }
        i += 1;
    }

    let b0 = *code.get(i)?;
    if (0x40..0x50).contains(&b0) {
        p.rex = Some(b0);
        i += 1;
    }

    let b0 = *code.get(i)?;
    if p.rex.is_none() && !p.p66 && !p.pf2 && !p.pf3 && (b0 == 0xc5 || b0 == 0xc4 || b0 == 0x62) {
        let b1 = *code.get(i + 1)?;
        let mut v = Vex {
            evex: false,
            three: false,
            map: 1,
            w: false,
            l: 0,
            pp: 0,
            vvvv: 0,
            r: (!b1 >> 7) & 1,
            x: 0,
            b: 0,
            rr: 0,
            vv: 0,
            aaa: 0,
            z: false,
            bcast: false,
        };
        match b0 {
            0xc5 => {
                v.vvvv = (!b1 >> 3) & 15;
                v.l = (b1 >> 2) & 1;
                v.pp = b1 & 3;
                i += 2;
            }
            0xc4 => {
                let b2 = *code.get(i + 2)?;
                v.three = true;
                v.x = (!b1 >> 6) & 1;
                v.b = (!b1 >> 5) & 1;
                v.map = b1 & 31;
                v.w = b2 >> 7 != 0;
                v.vvvv = (!b2 >> 3) & 15;
                v.l = (b2 >> 2) & 1;
                v.pp = b2 & 3;
                i += 3;
            }
            _ => {
                let b2 = *code.get(i + 2)?;
                let b3 = *code.get(i + 3)?;
                if b2 & 4 == 0 || b1 & 0x0c != 0 {
                    return None;
                }
                v.evex = true;
                v.x = (!b1 >> 6) & 1;
                v.b = (!b1 >> 5) & 1;
                v.rr = (!b1 >> 4) & 1;
                v.map = b1 & 3;
                v.w = b2 >> 7 != 0;
                v.vvvv = (!b2 >> 3) & 15;
                v.pp = b2 & 3;
                v.z = b3 >> 7 != 0;
                v.l = (b3 >> 5) & 3;
                v.bcast = (b3 >> 4) & 1 != 0;
                v.vv = (!b3 >> 3) & 1;
                v.aaa = b3 & 7;
                i += 4;
            }
        }
        p.vex = Some(v);
        let ctx = Ctx {
            code,
            p,
            i,
            prefixes: Vec::new(),
            insn: None,
        };
        return decode_vex(ctx, v);
    }

    let ctx = Ctx {
        code,
        p,
        i,
        prefixes: Vec::new(),
        insn: None,
    };
    decode_legacy(ctx)
}

/// integer and SSE instructions
fn decode_legacy(mut c: Ctx) -> Option<Ins> {
    let (w, r, _, b) = c.rex_bits();
    let gpr = |n: u8| -> Op {
        if w == 1 {
            t(R64[n as usize])
        } else {
            t(R32[n as usize])
        }
    };
    let op = c.byte(c.i)?;
    c.i += 1;
    let simple = !c.p.p66 && !c.p.pf2 && !c.p.pf3;

    if op != 0x0f {
        if !simple {
            return None;
        }
        return match op {
            0x50..=0x57 if w == 0 => {
                let n = (op & 7) | (b << 3);
                c.done("push", vec![t(R64[n as usize])])
            }
            0x58..=0x5f if w == 0 => {
                let n = (op & 7) | (b << 3);
                c.done("pop", vec![t(R64[n as usize])])
            }
            0x90 if c.p.rex.is_none() => c.done("nop", vec![]),
            0xc3 if c.p.rex.is_none() => c.done("ret", vec![]),
            0xb8..=0xbf => {
                let n = (op & 7) | (b << 3);
                if w == 1 {
                    let v = u64::from_le_bytes(c.code.get(c.i..c.i + 8)?.try_into().ok()?);
                    c.i += 8;
                    c.done("movabs", vec![t(R64[n as usize]), t(&format!("{:#x}", v))])
                } else {
                    let v = c.imm32()? as u32;
                    c.done("mov", vec![t(R32[n as usize]), t(&format!("{:#x}", v))])
                }
            }
            0xe8 => {
                let rel = c.imm32()?;
                c.done("call", vec![Op::Rel(rel)])
            }
            0xe9 => {
                let rel = c.imm32()?;
                c.prefixes.push("{disp32}");
                c.done("jmp", vec![Op::Rel(rel)])
            }
            0xeb => {
                let rel = c.imm8()? as i8 as i64;
                c.done("jmp", vec![Op::Rel(rel)])
            }
            0x70..=0x7f => {
                let rel = c.imm8()? as i8 as i64;
                c.done(&format!("j{}", CC[(op & 15) as usize]), vec![Op::Rel(rel)])
            }
            0x01 | 0x03 | 0x09 | 0x0b | 0x21 | 0x23 | 0x29 | 0x2b | 0x31 | 0x33 | 0x39 | 0x3b
            | 0x89 | 0x8b | 0x85 => {
                let m = c.modrm()?;
                let name = match op {
                    0x89 | 0x8b => "mov",
                    0x85 => "test",
                    _ => {
                        ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"][(op >> 3) as usize]
                    }
                };
                let reg = gpr(m.reg | (r << 3));
                let rm = match &m.mem {
                    Some(mem) => Op::Mem(
                        if w == 1 { "qword ptr " } else { "dword ptr " },
                        mem.clone(),
                    ),
                    None => gpr(m.rm | (b << 3)),
                };
                c.disp_prefix(&m, 1);
                let load = op & 2 != 0 && op != 0x85;
                if m.md == 3 && op != 0x85 {
                    c.prefixes
                        .insert(0, if load { "{load}" } else { "{store}" });
                }
                if load {
                    c.done(name, vec![reg, rm])
                } else {
                    c.done(name, vec![rm, reg])
                }
            }
            0x8d => {
                let m = c.modrm()?;
                let mem = m.mem.clone()?;
                c.disp_prefix(&m, 1);
                c.done("lea", vec![gpr(m.reg | (r << 3)), Op::Mem("", mem)])
            }
            0x81 | 0x83 | 0xc7 | 0xc1 => {
                let m = c.modrm()?;
                let rm = match &m.mem {
                    Some(mem) => Op::Mem(
                        if w == 1 { "qword ptr " } else { "dword ptr " },
                        mem.clone(),
                    ),
                    None => gpr(m.rm | (b << 3)),
                };
                c.disp_prefix(&m, 1);
                let (name, imm, long) = match op {
                    0x81 => (
                        ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"][m.reg as usize],
                        c.imm32()?,
                        true,
                    ),
                    0x83 => (
                        ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"][m.reg as usize],
                        c.imm8()? as i8 as i64,
                        false,
                    ),
                    0xc7 if m.reg == 0 => ("mov", c.imm32()?, false),
                    0xc1 => {
                        let name = match m.reg {
                            4 => "shl",
                            5 => "shr",
                            7 => "sar",
                            _ => return None,
                        };
                        let imm = c.imm8()?;
                        if imm == 1 {
                            // `as` encodes a shift by 1 as D1
                            c.insn = Some(format!("{:#04x}/{}", op, m.reg));
                            return c.done(name, vec![rm, t("1{:u8}")]);
                        }
                        (name, imm, false)
                    }
                    _ => return None,
                };
                let accumulator = m.md == 3 && (m.rm | (b << 3)) == 0;
                if long && ((-128..128).contains(&imm) || accumulator) {
                    // `as` encodes a small immediate in one byte (83), and an immediate to rax
                    // in its short form (05, 0d, 25, ...)
                    c.insn = Some(format!("{:#04x}/{}", op, m.reg));
                    return c.done(name, vec![rm, t(&format!("{}{{:s32}}", imm))]);
                }
                c.done(name, vec![rm, t(&format!("{:#x}", imm))])
            }
            0xff => {
                let m = c.modrm()?;
                c.disp_prefix(&m, 1);
                match m.reg {
                    0 | 1 => {
                        let rm = match &m.mem {
                            Some(mem) => Op::Mem("qword ptr ", mem.clone()),
                            None => gpr(m.rm | (b << 3)),
                        };
                        c.done(if m.reg == 0 { "inc" } else { "dec" }, vec![rm])
                    }
                    2 | 4 if w == 0 => {
                        let rm = match &m.mem {
                            Some(mem) => Op::Mem("qword ptr ", mem.clone()),
                            None => t(R64[(m.rm | (b << 3)) as usize]),
                        };
                        c.done(if m.reg == 2 { "call" } else { "jmp" }, vec![rm])
                    }
                    _ => None,
                }
            }
            _ => None,
        };
    }

    // 0F maps
    let op2 = c.byte(c.i)?;
    c.i += 1;

    if op2 == 0x3a {
        // SSE4.1 round
        let op3 = c.byte(c.i)?;
        c.i += 1;
        if !c.p.p66 || c.p.pf2 || c.p.pf3 {
            return None;
        }
        let name = match op3 {
            0x0b => "roundsd",
            0x09 => "roundpd",
            _ => return None,
        };
        let m = c.modrm()?;
        if w == 1 {
            c.prefixes.push("rex.w");
        }
        c.disp_prefix(&m, 1);
        let src = match &m.mem {
            Some(mem) => Op::Mem(ptr(if op3 == 0x0b { 8 } else { 16 }), mem.clone()),
            None => vreg(m.rm | (b << 3), 16),
        };
        let imm = c.imm8()?;
        return c.done(
            name,
            vec![vreg(m.reg | (r << 3), 16), src, t(&imm.to_string())],
        );
    }

    if simple {
        match op2 {
            0x80..=0x8f => {
                let rel = c.imm32()?;
                c.prefixes.push("{disp32}");
                return c.done(&format!("j{}", CC[(op2 & 15) as usize]), vec![Op::Rel(rel)]);
            }
            0x40..=0x4f => {
                let m = c.modrm()?;
                c.disp_prefix(&m, 1);
                let rm = match &m.mem {
                    Some(mem) => Op::Mem(
                        if w == 1 { "qword ptr " } else { "dword ptr " },
                        mem.clone(),
                    ),
                    None => gpr(m.rm | (b << 3)),
                };
                return c.done(
                    &format!("cmov{}", CC[(op2 & 15) as usize]),
                    vec![gpr(m.reg | (r << 3)), rm],
                );
            }
            0xb6 | 0xb7 | 0xbe | 0xbf => {
                let m = c.modrm()?;
                c.disp_prefix(&m, 1);
                let word = op2 & 1 == 1;
                let rm = match &m.mem {
                    Some(mem) => Op::Mem(if word { "word ptr " } else { "byte ptr " }, mem.clone()),
                    None if word => t(R16[(m.rm | (b << 3)) as usize]),
                    None => return None,
                };
                let name = if op2 < 0xbe { "movzx" } else { "movsx" };
                return c.done(name, vec![gpr(m.reg | (r << 3)), rm]);
            }
            0x18 => {
                let m = c.modrm()?;
                let mem = m.mem.clone()?;
                c.disp_prefix(&m, 1);
                let name = ["prefetchnta", "prefetcht0", "prefetcht1", "prefetcht2"]
                    .get(m.reg as usize)?;
                return c.done(name, vec![Op::Mem("", mem)]);
            }
            _ => {}
        }
    }

    // SSE: the mandatory prefix selects the form
    let kind = match (c.p.p66, c.p.pf2, c.p.pf3) {
        (false, false, false) => "ps",
        (true, false, false) => "pd",
        (false, true, false) => "sd",
        (false, false, true) => "ss",
        _ => return None,
    };
    let m = c.modrm()?;
    c.disp_prefix(&m, 1);
    let scalar = kind == "sd" || kind == "ss";
    let msize = match kind {
        "sd" => 8,
        "ss" => 4,
        _ => 16,
    };
    let xr = vreg(m.reg | (r << 3), 16);
    let xm = match &m.mem {
        Some(mem) => Op::Mem(ptr(msize), mem.clone()),
        None => vreg(m.rm | (b << 3), 16),
    };
    // REX.W changes only the integer forms
    let w_matters = matches!(op2, 0x6e | 0x7e | 0x2c | 0x2d);
    if w == 1 && !w_matters {
        c.prefixes.push("rex.w");
    }

    match op2 {
        0x10 | 0x11 | 0x28 | 0x29 => {
            let name = match (op2, kind) {
                (0x10 | 0x11, "sd") => "movsd",
                (0x10 | 0x11, "ss") => "movss",
                (0x10 | 0x11, "pd") => "movupd",
                (0x10 | 0x11, _) => "movups",
                (_, "pd") => "movapd",
                (_, "ps") => "movaps",
                _ => return None,
            };
            let load = op2 & 1 == 0;
            if m.md == 3 {
                c.prefixes
                    .insert(0, if load { "{load}" } else { "{store}" });
            }
            if load {
                c.done(name, vec![xr, xm])
            } else {
                c.done(name, vec![xm, xr])
            }
        }
        0x51 | 0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f => {
            let base = match op2 {
                0x51 => "sqrt",
                0x58 => "add",
                0x59 => "mul",
                0x5c => "sub",
                0x5d => "min",
                0x5e => "div",
                _ => "max",
            };
            c.done(&format!("{}{}", base, kind), vec![xr, xm])
        }
        0x54..=0x57 if !scalar => {
            let base = ["and", "andn", "or", "xor"][(op2 - 0x54) as usize];
            c.done(&format!("{}{}", base, kind), vec![xr, xm])
        }
        0x2e | 0x2f if kind == "pd" || kind == "ps" => {
            let name = if op2 == 0x2e { "ucomi" } else { "comi" };
            let s = if kind == "pd" { "sd" } else { "ss" };
            let xm = match &m.mem {
                Some(mem) => Op::Mem(ptr(if kind == "pd" { 8 } else { 4 }), mem.clone()),
                None => xm,
            };
            c.done(&format!("{}{}", name, s), vec![xr, xm])
        }
        0x2c if scalar => {
            let name = if kind == "sd" {
                "cvttsd2si"
            } else {
                "cvttss2si"
            };
            c.done(name, vec![gpr(m.reg | (r << 3)), xm])
        }
        0xc2 => {
            let pred = c.imm8()? as usize;
            if pred > 7 {
                return None;
            }
            c.done(&format!("cmp{}{}", PRED[pred], kind), vec![xr, xm])
        }
        0x6e | 0x7e if kind == "pd" && w == 1 => {
            let g = match &m.mem {
                Some(mem) => Op::Mem("qword ptr ", mem.clone()),
                None => t(R64[(m.rm | (b << 3)) as usize]),
            };
            if op2 == 0x6e {
                c.done("movq", vec![xr, g])
            } else {
                c.done("movq", vec![g, xr])
            }
        }
        _ => None,
    }
}

/// the FMA mnemonic of a 0F38 opcode (W selects double)
fn fma_name(op: u8, w: bool) -> Option<String> {
    let order = match op & 0xf0 {
        0x90 => "132",
        0xa0 => "213",
        0xb0 => "231",
        _ => return None,
    };
    let kind = match op & 0x0e {
        0x8 => "fmadd",
        0xa => "fmsub",
        0xc => "fnmadd",
        0xe => "fnmsub",
        _ => return None,
    };
    let suffix = match (op & 1, w) {
        (0, true) => "pd",
        (0, false) => "ps",
        (1, true) => "sd",
        (1, false) => "ss",
        _ => unreachable!(),
    };
    Some(format!("v{}{}{}", kind, order, suffix))
}

/// VEX and EVEX instructions
fn decode_vex(mut c: Ctx, v: Vex) -> Option<Ins> {
    let op = c.byte(c.i)?;
    c.i += 1;

    // no ModRM
    if v.map == 1 && op == 0x77 && !v.evex && v.vvvv == 0 && v.pp == 0 {
        if v.three && !v.w && v.x == 0 && v.b == 0 {
            c.prefixes.push("{vex3}");
        }
        let name = if v.l == 0 { "vzeroupper" } else { "vzeroall" };
        return c.done(name, vec![]);
    }

    let m = c.modrm()?;

    let reg_n = m.reg | (v.r << 3) | (v.rr << 4);
    let rm_n = m.rm | (v.b << 3) | if v.evex { v.x << 4 } else { 0 };
    let vvvv_n = v.vvvv | (v.vv << 4);

    // the vector size: VEX.L (128, 256), EVEX.L'L (128, 256, 512)
    let vbytes: u32 = match v.l {
        0 => 16,
        1 => 32,
        2 => 64,
        _ => return None,
    };
    let kind = match v.pp {
        0 => "ps",
        1 => "pd",
        2 => "ss",
        _ => "sd",
    };
    let scalar = v.pp >= 2;

    let mask = |op: Op| -> Op {
        if !v.evex || (v.aaa == 0 && !v.z) {
            return op;
        }
        match op {
            Op::Text(s) => {
                let mut s = format!("{}{{k{}}}", s, v.aaa);
                if v.z {
                    s.push_str("{z}");
                }
                Op::Text(s)
            }
            other => other,
        }
    };

    // the operands of the common forms
    let vr = |bytes: u32| vreg(reg_n, bytes);
    let hv = |bytes: u32| vreg(vvvv_n, bytes);
    let wm = |bytes: u32, msize: u32| -> Op {
        match &m.mem {
            Some(mem) => Op::Mem(ptr(msize), mem.clone()),
            None => vreg(rm_n, bytes),
        }
    };

    // the EVEX disp8 scale and whether the instruction is LIG/WIG, filled below
    let mut n: i64 = 1;
    let mut lig = false;
    let mut wig = false;
    let mut has_vex_form = true;
    let mut reg_reg_dir: Option<bool> = None; // Some(load)

    let result: (String, Vec<Op>) = match (v.map, op) {
        // moves
        (1, 0x10 | 0x11) => {
            let load = op == 0x10;
            wig = true;
            if scalar {
                lig = true;
                let size = if kind == "sd" { 8 } else { 4 };
                n = size as i64;
                let name = if kind == "sd" { "vmovsd" } else { "vmovss" };
                match &m.mem {
                    Some(mem) => {
                        let mo = Op::Mem(ptr(size), mem.clone());
                        if load {
                            (name.into(), vec![mask(vr(16)), mo])
                        } else {
                            (name.into(), vec![mo, vr(16)])
                        }
                    }
                    None => {
                        reg_reg_dir = Some(load);
                        if load {
                            (name.into(), vec![mask(vr(16)), hv(16), vreg(rm_n, 16)])
                        } else {
                            (name.into(), vec![mask(vreg(rm_n, 16)), hv(16), vr(16)])
                        }
                    }
                }
            } else {
                n = vbytes as i64;
                let name = format!("vmovu{}", kind);
                if m.mem.is_none() {
                    reg_reg_dir = Some(load);
                }
                if load {
                    (name, vec![mask(vr(vbytes)), wm(vbytes, vbytes)])
                } else {
                    (name, vec![mask(wm(vbytes, vbytes)), vr(vbytes)])
                }
            }
        }
        (1, 0x28 | 0x29) if !scalar => {
            let load = op == 0x28;
            wig = true;
            n = vbytes as i64;
            let name = format!("vmova{}", kind);
            if m.mem.is_none() {
                reg_reg_dir = Some(load);
            }
            if load {
                (name, vec![mask(vr(vbytes)), wm(vbytes, vbytes)])
            } else {
                (name, vec![mask(wm(vbytes, vbytes)), vr(vbytes)])
            }
        }
        (1, 0x2e | 0x2f) if !scalar => {
            lig = true;
            wig = true;
            let (s, size) = if kind == "pd" { ("sd", 8) } else { ("ss", 4) };
            n = size as i64;
            let name = format!("v{}{}", if op == 0x2e { "ucomi" } else { "comi" }, s);
            (name, vec![vr(16), wm(16, size)])
        }
        (1, 0x2c) if scalar => {
            lig = true;
            let size = if kind == "sd" { 8 } else { 4 };
            n = size as i64;
            let g = if v.w {
                R64[reg_n as usize & 15]
            } else {
                R32[reg_n as usize & 15]
            };
            let name = if kind == "sd" {
                "vcvttsd2si"
            } else {
                "vcvttss2si"
            };
            (name.into(), vec![t(g), wm(16, size)])
        }
        (1, 0x50) if !scalar && m.mem.is_none() => {
            wig = true;
            let name = format!("vmovmsk{}", kind);
            (name, vec![t(R32[reg_n as usize & 15]), vreg(rm_n, vbytes)])
        }
        (1, 0x51) => {
            wig = true;
            if scalar {
                lig = true;
                let size = if kind == "sd" { 8 } else { 4 };
                n = size as i64;
                (
                    format!("vsqrt{}", kind),
                    vec![mask(vr(16)), hv(16), wm(16, size)],
                )
            } else {
                n = vbytes as i64;
                (
                    format!("vsqrt{}", kind),
                    vec![mask(vr(vbytes)), wm(vbytes, vbytes)],
                )
            }
        }
        (1, 0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f) => {
            wig = true;
            let base = match op {
                0x58 => "add",
                0x59 => "mul",
                0x5c => "sub",
                0x5d => "min",
                0x5e => "div",
                _ => "max",
            };
            if scalar {
                lig = true;
                let size = if kind == "sd" { 8 } else { 4 };
                n = size as i64;
                (
                    format!("v{}{}", base, kind),
                    vec![mask(vr(16)), hv(16), wm(16, size)],
                )
            } else {
                n = vbytes as i64;
                (
                    format!("v{}{}", base, kind),
                    vec![mask(vr(vbytes)), hv(vbytes), wm(vbytes, vbytes)],
                )
            }
        }
        (1, 0x54..=0x57 | 0x14 | 0x15 | 0x7c | 0xd0) if !scalar => {
            wig = true;
            n = vbytes as i64;
            let base = match op {
                0x54 => "and",
                0x55 => "andn",
                0x56 => "or",
                0x57 => "xor",
                0x14 => "unpckl",
                0x15 => "unpckh",
                0x7c if kind == "pd" => "hadd",
                0xd0 if kind == "pd" => "addsub",
                _ => return None,
            };
            if v.evex && matches!(op, 0x7c | 0xd0) {
                return None;
            }
            (
                format!("v{}{}", base, kind),
                vec![mask(vr(vbytes)), hv(vbytes), wm(vbytes, vbytes)],
            )
        }
        (1, 0xc6) if !scalar => {
            wig = true;
            n = vbytes as i64;
            let imm = c.imm8()?;
            (
                format!("vshuf{}", kind),
                vec![
                    mask(vr(vbytes)),
                    hv(vbytes),
                    wm(vbytes, vbytes),
                    t(&imm.to_string()),
                ],
            )
        }
        (1, 0xc2) => {
            wig = true;
            let pred = c.imm8()? as usize;
            let pred = PRED.get(pred)?;
            let (dst, bytes, size) = if scalar {
                lig = true;
                let size = if kind == "sd" { 8 } else { 4 };
                (vr(16), 16, size)
            } else if v.evex {
                (t(&format!("k{}", m.reg)), vbytes, vbytes)
            } else {
                (vr(vbytes), vbytes, vbytes)
            };
            n = size as i64;
            let dst = if v.evex { mask(dst) } else { dst };
            (
                format!("vcmp{}{}", pred, kind),
                vec![dst, hv(bytes), wm(bytes, size)],
            )
        }
        (1, 0x6e | 0x7e) if kind == "pd" && v.w && !v.evex => {
            let g = match &m.mem {
                Some(mem) => Op::Mem("qword ptr ", mem.clone()),
                None => t(R64[rm_n as usize & 15]),
            };
            if op == 0x6e {
                ("vmovq".into(), vec![vr(16), g])
            } else {
                ("vmovq".into(), vec![g, vr(16)])
            }
        }
        (1, 0x92 | 0x93 | 0x44) if !v.evex && v.pp == 0 && !v.w && v.l == 0 && m.md == 3 => {
            match op {
                0x92 => (
                    "kmovw".into(),
                    vec![t(&format!("k{}", m.reg)), t(R32[rm_n as usize & 15])],
                ),
                0x93 => (
                    "kmovw".into(),
                    vec![t(R32[reg_n as usize & 15]), t(&format!("k{}", m.rm))],
                ),
                _ => (
                    "knotw".into(),
                    vec![t(&format!("k{}", m.reg)), t(&format!("k{}", m.rm))],
                ),
            }
        }
        // 0F38
        (2, 0x18 | 0x19) if kind == "pd" => {
            if op == 0x19 && (v.l == 0 || v.w != v.evex) {
                return None;
            }
            n = if op == 0x19 { 8 } else { 4 };
            let size = n as u32;
            let name = if op == 0x19 {
                "vbroadcastsd"
            } else {
                "vbroadcastss"
            };
            (name.into(), vec![mask(vr(vbytes)), wm(16, size)])
        }
        (2, 0x98..=0x9f | 0xa8..=0xaf | 0xb8..=0xbf) if kind == "pd" => {
            let name = fma_name(op, v.w)?;
            if op & 1 == 1 {
                lig = true;
                let size = if v.w { 8 } else { 4 };
                n = size as i64;
                (name, vec![mask(vr(16)), hv(16), wm(16, size)])
            } else {
                n = vbytes as i64;
                (name, vec![mask(vr(vbytes)), hv(vbytes), wm(vbytes, vbytes)])
            }
        }
        (2, 0x38) if v.evex && v.pp == 2 && v.w && m.md == 3 => {
            has_vex_form = false;
            (
                "vpmovm2q".into(),
                vec![vr(vbytes), t(&format!("k{}", m.rm))],
            )
        }
        (2, 0x39) if v.evex && v.pp == 2 && v.w && m.md == 3 => {
            has_vex_form = false;
            (
                "vpmovq2m".into(),
                vec![t(&format!("k{}", m.reg)), vreg(rm_n, vbytes)],
            )
        }
        // 0F3A
        (3, 0x08 | 0x09) if kind == "pd" && !v.evex => {
            wig = true;
            let imm = c.imm8()?;
            let name = if op == 0x09 { "vroundpd" } else { "vroundps" };
            (
                name.into(),
                vec![vr(vbytes), wm(vbytes, vbytes), t(&imm.to_string())],
            )
        }
        (3, 0x09) if kind == "pd" && v.evex && v.w => {
            has_vex_form = false;
            n = vbytes as i64;
            let imm = c.imm8()?;
            (
                "vrndscalepd".into(),
                vec![mask(vr(vbytes)), wm(vbytes, vbytes), t(&imm.to_string())],
            )
        }
        (3, 0x0a | 0x0b) if kind == "pd" && !v.evex => {
            wig = true;
            lig = true;
            let size = if op == 0x0b { 8 } else { 4 };
            let imm = c.imm8()?;
            let name = if op == 0x0b { "vroundsd" } else { "vroundss" };
            (
                name.into(),
                vec![vr(16), hv(16), wm(16, size), t(&imm.to_string())],
            )
        }
        (3, 0x18) if kind == "pd" && !v.evex && !v.w && v.l == 1 => {
            let imm = c.imm8()?;
            (
                "vinsertf128".into(),
                vec![vr(32), hv(32), wm(16, 16), t(&imm.to_string())],
            )
        }
        (3, 0x19) if kind == "pd" && !v.evex && !v.w && v.l == 1 && v.vvvv == 0 => {
            let imm = c.imm8()?;
            (
                "vextractf128".into(),
                vec![wm(16, 16), vr(32), t(&imm.to_string())],
            )
        }
        _ => return None,
    };

    if v.bcast {
        return None;
    }

    // EVEX scales an 8-bit displacement by the access size (disp8*N)
    let mut m = m;
    let (name, mut ops) = result;
    if v.evex {
        if let Some(mem) = m.mem.as_mut() {
            if mem.disp_size == 8 {
                mem.disp *= n;
            }
        }
        for op in ops.iter_mut() {
            if let Op::Mem(_, mem) = op {
                if mem.disp_size == 8 {
                    mem.disp *= n;
                }
            }
        }
    }

    c.disp_prefix(&m, if v.evex { n } else { 1 });

    if let Some(load) = reg_reg_dir {
        c.prefixes
            .insert(0, if load { "{load}" } else { "{store}" });
    }

    if v.evex {
        // `as` uses VEX when it can: an EVEX encoding VEX can give needs {evex}
        let high = reg_n >= 16 || rm_n >= 16 || vvvv_n >= 16;
        if has_vex_form && !high && v.l < 2 && v.aaa == 0 && !v.z {
            c.prefixes.push("{evex}");
        }
        if lig && v.l != 0 {
            return None;
        }
    } else {
        // VEX2 where `as` can use it
        let vex2_possible = v.map == 1 && !v.w && v.x == 0 && v.b == 0;
        let exact_l = lig && v.l != 0;
        let exact_w = wig && v.w;
        if exact_l || exact_w {
            let pp = ["", "66.", "F3.", "F2."][v.pp as usize];
            let map = ["", "0F", "0F38", "0F3A"].get(v.map as usize)?;
            c.insn = Some(format!(
                "VEX.L{}.{}{}.W{} {:#04x}",
                v.l, pp, map, v.w as u8, op
            ));
            c.prefixes.retain(|p| *p != "{load}" && *p != "{store}");
            if v.three && vex2_possible && !exact_w {
                // .insn uses VEX2 when it can, as symjit does except here
                return None;
            }
        } else if v.three && vex2_possible {
            c.prefixes.push("{vex3}");
        }
    }

    c.done(&name, ops)
}

/// the instruction as pieces of text at `pc` (targets absolute)
fn pieces(ins: &Ins, pc: usize) -> (Vec<Piece>, Option<usize>) {
    let next = (pc + ins.len) as i64;
    let mut out = Vec::new();
    let mut refers = None;

    match &ins.insn {
        Some(spec) => out.push(Piece::Text(format!(".insn {}", spec))),
        None => {
            for p in &ins.prefixes {
                out.push(Piece::Text(format!("{} ", p)));
            }
            out.push(Piece::Text(ins.mnemonic.clone()));
        }
    }

    for (k, op) in ins.ops.iter().enumerate() {
        let sep = if k > 0 || ins.insn.is_some() {
            ", "
        } else {
            " "
        };
        out.push(Piece::Text(sep.to_string()));
        match op {
            Op::Text(s) => out.push(Piece::Text(s.clone())),
            Op::Rel(rel) => out.push(Piece::Target((next + rel) as usize)),
            Op::Mem(p, m) if m.rip => {
                let target = (next + m.disp) as usize;
                refers = Some(target);
                out.push(Piece::Text(format!("{}[rip", p)));
                out.push(Piece::Rel {
                    target,
                    value: m.disp,
                });
                out.push(Piece::Text("]".to_string()));
            }
            Op::Mem(p, m) => out.push(Piece::Text(format!("{}{}", p, mem_text(m, None)))),
        }
    }

    (out, refers)
}

pub struct X86Decoder;

impl Decoder for X86Decoder {
    fn comment(&self) -> &'static str {
        "#"
    }

    fn directives(&self) -> Vec<&'static str> {
        vec![".intel_syntax noprefix", ".p2align 5"]
    }

    fn decode(&self, code: &[u8]) -> Vec<Item> {
        let mut items = Vec::new();
        let mut pc = 0;

        while pc < code.len() {
            let ins = decode(&code[pc..]).filter(|ins| pc + ins.len <= code.len());
            match ins {
                Some(ins) => {
                    let b = &code[pc..pc + ins.len];
                    let hex: Vec<String> = b.iter().map(|x| format!("{:02x}", x)).collect();
                    let bytes: Vec<String> = b.iter().map(|x| format!("0x{:02x}", x)).collect();
                    let (pieces, refers) = pieces(&ins, pc);
                    let mut item = Item {
                        pc,
                        hex: hex.join(""),
                        pieces,
                        raw: format!(".byte {}", bytes.join(", ")),
                        refers,
                        note: None,
                    };
                    if ins.insn.is_some() {
                        let ops: Vec<String> = ins
                            .ops
                            .iter()
                            .map(|op| match op {
                                Op::Text(s) => s.replace("{:s32}", "").replace("{:u8}", ""),
                                Op::Mem(p, m) => format!("{}{}", p, mem_text(m, None)),
                                Op::Rel(r) => format!("{:+}", r),
                            })
                            .collect();
                        item.note = Some(format!("{} {}", ins.mnemonic, ops.join(", ")));
                    }
                    items.push(item);
                    pc += ins.len;
                }
                None => {
                    let x = code[pc];
                    items.push(Item::text(
                        pc,
                        format!("{:02x}", x),
                        format!(".byte 0x{:02x}", x),
                        format!(".byte 0x{:02x}", x),
                    ));
                    pc += 1;
                }
            }
        }

        items
    }
}
