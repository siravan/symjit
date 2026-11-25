//! x86-64 f64 kernels rewritten into f32 kernels (see `crate::downgrade`), instruction for
//! instruction: every double instruction symjit's AVX generators emit has a single twin of the same
//! length and operand form (vaddpd -> vaddps, vfmadd231pd -> vfmadd231ps, ...). A ymm holds 4
//! doubles or 8 floats and a zmm 8 or 16, both 32 or 64 bytes, so the stack slots, the offsets and
//! the compress-mode slot arithmetic are the same bytes; only the lanes double.
//!
//! The rewrite, per instruction (anything else of the vector instructions is refused):
//! - VEX: the packed double ops of map 0F (pp = 66: vmovupd, vmovapd, vaddpd, vsubpd, vmulpd,
//!   vdivpd, vsqrtpd, vminpd, vmaxpd, vandpd, vandnpd, vorpd, vxorpd, vcmppd, vucomisd, vcomisd) ->
//!   pp = none (ps, ss); the scalar double ops (pp = F2) -> pp = F3 (ss); the FMA family (map 0F38,
//!   0x96-0xBF) -> W = 0; vbroadcastsd (map 0F38 0x19) -> vbroadcastss (0x18); vroundpd / vroundsd
//!   (map 0F3A 0x09 / 0x0B) -> vroundps / vroundss (0x08 / 0x0A), and their EVEX twins
//!   vrndscalepd / vrndscalesd (W = 1) -> vrndscaleps / vrndscaless (W = 0).
//! - EVEX (AVX-512): the same, with W = 1 -> 0 where the double form has it; vpmovm2q / vpmovq2m
//!   (map 0F38, F3, 0x38 / 0x39, W = 1) -> vpmovm2d / vpmovd2m (W = 0). A broadcast load with a
//!   compressed 8-bit displacement (disp8 x 8 for sd, x 4 for ss) gets its displacement doubled.
//!   Embedded broadcasts ({1toN}) are refused.
//! - Mask (k) and integer instructions: kept. A relative call (`call rel32`, the compress-mode
//!   subroutines) is kept; an indirect call (`call [rip + ..]`, a math function) is refused, and so
//!   is vmovmskpd (a SIMD kernel's branch on its lanes: the f32 mask has twice the bits).
//!
//! The decoder knows the instruction forms symjit's x86-64 generators emit (`amd/asm.rs`); an
//! instruction it does not know is an error.

use anyhow::{anyhow, Result};

use crate::downgrade::Kernel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Legacy,
    Vex,
    Evex,
}

#[derive(Clone, Debug)]
struct Ins {
    pc: usize,
    length: usize,
    kind: Kind,
    /// 1: 0F, 2: 0F38, 3: 0F3A
    map: u8,
    opcode: u16,
    /// 0 none, 1: 66, 2: F3, 3: F2
    pp: u8,
    w: u8,
    modrm: Option<u8>,
    /// the address a RIP-relative operand reads
    rip: Option<usize>,
    disp8_at: Option<usize>,
    /// EVEX.b
    broadcast: bool,
}

impl Ins {
    fn new(pc: usize, kind: Kind) -> Ins {
        Ins {
            pc,
            length: 0,
            kind,
            map: 0,
            opcode: 0,
            pp: 0,
            w: 0,
            modrm: None,
            rip: None,
            disp8_at: None,
            broadcast: false,
        }
    }
}

fn byte(code: &[u8], i: usize) -> Result<u8> {
    code.get(i)
        .copied()
        .ok_or_else(|| anyhow!("0x{i:x}: the code ends inside an instruction"))
}

/// The length of ModRM + SIB + displacement at `i`; the RIP-relative displacement's offset
/// (relative to `i`), if any; the 8-bit displacement's offset (relative to `i`), if any.
fn modrm(code: &[u8], i: usize) -> Result<(usize, Option<usize>, Option<usize>)> {
    let m = byte(code, i)?;
    let (md, rm) = (m >> 6, m & 7);
    let mut n = 1;
    let mut rip = None;
    let mut disp8 = None;

    if md == 3 {
        return Ok((n, None, None));
    }

    if rm == 4 {
        let sib = byte(code, i + 1)?;
        n += 1;
        if md == 0 && (sib & 7) == 5 {
            n += 4;
        }
    } else if md == 0 && rm == 5 {
        rip = Some(n);
        n += 4;
    }

    if md == 1 {
        disp8 = Some(n);
        n += 1;
    } else if md == 2 {
        n += 4;
    }

    Ok((n, rip, disp8))
}

/// legacy opcodes: (has ModRM, immediate bytes); 0xB8-0xBF, the pushes and pops and the jumps apart
fn legacy(op: u8) -> Option<(bool, usize)> {
    Some(match op {
        0x01 | 0x03 | 0x09 | 0x0B | 0x21 | 0x23 | 0x29 | 0x2B | 0x31 | 0x33 | 0x39 | 0x3B
        | 0x85 | 0x89 | 0x8B | 0x8D => (true, 0),
        0x81 | 0xC7 => (true, 4),
        0x83 | 0xC1 => (true, 1),
        0xD1 | 0xF7 | 0xFF => (true, 0),
        0x90 | 0xC3 | 0xCC => (false, 0),
        _ => return None,
    })
}

fn legacy_0f(op: u8) -> Option<(bool, usize)> {
    match op {
        0xB6 | 0xB7 | 0x1F | 0xAF => Some((true, 0)),
        _ => None,
    }
}

fn vex_imm(map: u8, opcode: u16) -> usize {
    if map == 3 || (map == 1 && matches!(opcode, 0xC2 | 0xC6 | 0x70 | 0x71 | 0x72 | 0x73)) {
        1
    } else {
        0
    }
}

fn unknown(code: &[u8], pc: usize) -> anyhow::Error {
    let end = (pc + 8).min(code.len());
    let hex: String = code[pc..end].iter().map(|b| format!("{b:02x}")).collect();
    anyhow!("0x{pc:x}: unknown instruction {hex}")
}

fn decode_one(code: &[u8], pc: usize) -> Result<Ins> {
    let mut i = pc;
    let mut rex_w = false;

    while matches!(byte(code, i)?, 0x66 | 0xF2 | 0xF3) {
        i += 1;
    }

    if (0x40..=0x4F).contains(&byte(code, i)?) {
        rex_w = byte(code, i)? & 8 != 0;
        i += 1;
    }

    let b = byte(code, i)?;

    if matches!(b, 0xC4 | 0xC5 | 0x62) && i == pc {
        let (mut ins, op_at) = if b == 0xC5 {
            let b1 = byte(code, i + 1)?;
            let mut ins = Ins::new(pc, Kind::Vex);
            ins.map = 1;
            ins.pp = b1 & 3;
            (ins, i + 2)
        } else if b == 0xC4 {
            let (b1, b2) = (byte(code, i + 1)?, byte(code, i + 2)?);
            let mut ins = Ins::new(pc, Kind::Vex);
            ins.map = b1 & 0x1F;
            ins.pp = b2 & 3;
            ins.w = b2 >> 7;
            (ins, i + 3)
        } else {
            let (p0, p1, p2) = (byte(code, i + 1)?, byte(code, i + 2)?, byte(code, i + 3)?);
            if p1 & 4 == 0 {
                return Err(anyhow!("0x{pc:x}: not an EVEX prefix"));
            }
            let mut ins = Ins::new(pc, Kind::Evex);
            ins.map = p0 & 7;
            ins.pp = p1 & 3;
            ins.w = p1 >> 7;
            ins.broadcast = p2 & 0x10 != 0;
            (ins, i + 4)
        };

        ins.opcode = byte(code, op_at)? as u16;

        if ins.map == 1 && ins.opcode == 0x77 {
            // vzeroupper / vzeroall: no operands
            ins.length = op_at + 1 - pc;
            return Ok(ins);
        }

        ins.modrm = Some(byte(code, op_at + 1)?);
        let (n, rip, d8) = modrm(code, op_at + 1)?;
        let end = op_at + 1 + n + vex_imm(ins.map, ins.opcode);
        ins.length = end - pc;

        if let Some(r) = rip {
            let at = op_at + 1 + r;
            let disp = i32::from_le_bytes(
                code.get(at..at + 4)
                    .ok_or_else(|| anyhow!("0x{pc:x}: the code ends inside an instruction"))?
                    .try_into()
                    .unwrap(),
            );
            ins.rip = Some((end as i64 + disp as i64) as usize);
        }

        if let Some(d) = d8 {
            ins.disp8_at = Some(op_at + 1 + d);
        }

        return Ok(ins);
    }

    let mut ins = Ins::new(pc, Kind::Legacy);
    ins.opcode = b as u16;

    if (0xB8..=0xBF).contains(&b) {
        ins.length = i + 1 + if rex_w { 8 } else { 4 } - pc;
        return Ok(ins);
    }

    if b == 0xE8 || b == 0xE9 {
        ins.length = i + 5 - pc;
        return Ok(ins);
    }

    if b == 0xEB || (0x70..=0x7F).contains(&b) {
        ins.length = i + 2 - pc;
        return Ok(ins);
    }

    if (0x50..=0x5F).contains(&b) {
        ins.length = i + 1 - pc;
        return Ok(ins);
    }

    let (rip, end) = if b == 0x0F {
        let b2 = byte(code, i + 1)?;
        if (0x80..=0x8F).contains(&b2) {
            ins.length = i + 6 - pc;
            return Ok(ins);
        }
        let (_, imm) = legacy_0f(b2).ok_or_else(|| unknown(code, pc))?;
        ins.opcode = 0x0F00 | b2 as u16;
        let (n, rip, _) = modrm(code, i + 2)?;
        (rip, i + 2 + n + imm)
    } else {
        let (has, mut imm) = legacy(b).ok_or_else(|| unknown(code, pc))?;
        if !has {
            ins.length = i + 1 + imm - pc;
            return Ok(ins);
        }
        let reg = (byte(code, i + 1)? >> 3) & 7;
        if b == 0xF7 {
            imm = if reg == 0 { 4 } else { 0 };
        }
        if b == 0xFF && (reg == 2 || reg == 3) {
            return Err(anyhow!(
                "0x{pc:x}: a function call (math functions take and return doubles)"
            ));
        }
        let (n, rip, _) = modrm(code, i + 1)?;
        (rip, i + 1 + n + imm)
    };

    ins.length = end - pc;

    if rip.is_some() {
        return Err(anyhow!(
            "0x{pc:x}: an integer instruction reads RIP-relative data"
        ));
    }

    Ok(ins)
}

fn decode(code: &[u8], end: usize) -> Result<Vec<Ins>> {
    let mut out = Vec::new();
    let mut pc = 0;

    while pc < end {
        let ins = decode_one(code, pc)?;
        pc += ins.length;
        out.push(ins);
    }

    if pc != end {
        return Err(anyhow!(
            "the code does not end at the data area (0x{pc:x}, data at 0x{end:x})"
        ));
    }

    Ok(out)
}

/// the packed double ops of map 0F with a packed single twin (pp 66 -> none)
fn packed_0f(op: u16) -> bool {
    matches!(
        op,
        0x10 | 0x11
            | 0x28
            | 0x29
            | 0x2E
            | 0x2F
            | 0x51
            | 0x54
            | 0x55
            | 0x56
            | 0x57
            | 0x58
            | 0x59
            | 0x5C
            | 0x5D
            | 0x5E
            | 0x5F
            | 0xC2
    )
}

/// the scalar double ops of map 0F (pp F2 -> F3)
fn scalar_0f(op: u16) -> bool {
    matches!(
        op,
        0x10 | 0x11 | 0x51 | 0x58 | 0x59 | 0x5C | 0x5D | 0x5E | 0x5F | 0xC2
    )
}

/// mask register ops (knotw, kmovw, kortestw, ...)
fn k_op(op: u16) -> bool {
    matches!(
        op,
        0x41 | 0x42
            | 0x44
            | 0x45
            | 0x46
            | 0x47
            | 0x4A
            | 0x4B
            | 0x90
            | 0x91
            | 0x92
            | 0x93
            | 0x98
            | 0x99
    )
}

fn set_vex(
    code: &mut [u8],
    ins: &Ins,
    pp: Option<u8>,
    w: Option<u8>,
    opcode: Option<u8>,
) -> Result<()> {
    let pc = ins.pc;

    let op_at = match ins.kind {
        Kind::Vex if code[pc] == 0xC5 => {
            if w == Some(1) {
                return Err(anyhow!("0x{pc:x}: a two-byte VEX cannot hold W"));
            }
            if let Some(pp) = pp {
                code[pc + 1] = (code[pc + 1] & !3) | pp;
            }
            pc + 2
        }
        Kind::Vex | Kind::Evex => {
            let mut b2 = code[pc + 2];
            if let Some(pp) = pp {
                b2 = (b2 & !3) | pp;
            }
            if let Some(w) = w {
                b2 = (b2 & 0x7F) | (w << 7);
            }
            code[pc + 2] = b2;
            if ins.kind == Kind::Vex {
                pc + 3
            } else {
                pc + 4
            }
        }
        Kind::Legacy => unreachable!(),
    };

    if let Some(op) = opcode {
        code[op_at] = op;
    }

    Ok(())
}

/// Rewrites one instruction in place.
fn rewrite(code: &mut [u8], ins: &Ins, kernel: Kernel) -> Result<()> {
    let Some(modrm) = ins.modrm else {
        return Ok(()); // legacy, vzeroupper
    };

    if ins.kind == Kind::Legacy {
        return Ok(());
    }

    let evex = ins.kind == Kind::Evex;
    let (m, op, pp, pc) = (ins.map, ins.opcode, ins.pp, ins.pc);
    let w0 = if evex { Some(0) } else { None };

    if evex && ins.broadcast && (modrm >> 6) != 3 {
        return Err(anyhow!("0x{pc:x}: an embedded broadcast"));
    }

    if m == 1 && pp == 0 && k_op(op) && !evex {
        return Ok(());
    }

    if m == 1 && pp == 1 && op == 0x50 {
        return Err(anyhow!(
            "0x{pc:x}: vmovmskpd: {}",
            match kernel {
                Kernel::Simd => "a branch on the lanes of a SIMD kernel",
                Kernel::Scalar => "a mask of doubles",
            }
        ));
    }

    if m == 1 && pp == 1 && packed_0f(op) {
        return set_vex(code, ins, Some(0), w0, None);
    }

    if m == 1 && pp == 3 && scalar_0f(op) {
        return set_vex(code, ins, Some(2), w0, None);
    }

    if m == 2 && pp == 1 && (0x96..=0xBF).contains(&op) {
        // FMA: W selects the double form
        if ins.w != 1 {
            return Err(anyhow!("0x{pc:x}: a single-precision FMA in an f64 kernel"));
        }
        return set_vex(code, ins, None, Some(0), None);
    }

    if m == 2 && pp == 1 && op == 0x19 {
        // vbroadcastsd -> vbroadcastss
        set_vex(code, ins, None, Some(0), Some(0x18))?;
        if let (true, Some(at)) = (evex, ins.disp8_at) {
            // disp8 x 8 -> x 4
            let d = (code[at] as i8) as i32 * 2;
            if !(-128..=127).contains(&d) {
                return Err(anyhow!(
                    "0x{pc:x}: a broadcast displacement out of disp8 range"
                ));
            }
            code[at] = d as i8 as u8;
        }
        return Ok(());
    }

    if m == 3 && pp == 1 && (op == 0x09 || op == 0x0B) {
        // vroundpd -> vroundps, vroundsd -> vroundss; EVEX vrndscalepd / sd (W = 1) -> ps / ss (W = 0)
        return set_vex(code, ins, None, w0, Some(op as u8 - 1));
    }

    if evex && m == 2 && pp == 2 && (op == 0x38 || op == 0x39) && ins.w == 1 {
        // vpmovm2q / vpmovq2m
        return set_vex(code, ins, None, Some(0), None);
    }

    Err(anyhow!(
        "0x{pc:x}: no f32 rewrite for {:?} map {m} pp {pp} opcode 0x{op:02x} (W {})",
        ins.kind,
        ins.w
    ))
}

/// The machine code `code` with its instructions (up to `end`, the data area) rewritten to f32; the
/// bytes from `end` on as they are.
pub fn rewrite_code(code: &[u8], end: usize, kernel: Kernel) -> Result<Vec<u8>> {
    let mut out = code.to_vec();

    for ins in decode(code, end)? {
        if matches!(ins.rip, Some(r) if r < end) {
            return Err(anyhow!("0x{:x}: a RIP-relative read of the code", ins.pc));
        }
        rewrite(&mut out, &ins, kernel)?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    //! The f32 twins against GNU as: each pair is an f64 instruction and its f32 twin as the
    //! assembler encodes them (generated, each instruction assembled on its own, so a RIP-relative
    //! operand points past the code).

    use super::*;

    fn twin(code: &[u8]) -> Result<Vec<u8>> {
        rewrite_code(code, code.len(), Kernel::Simd)
    }

    fn h(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // generated by GNU as (Intel syntax, objcopy -O binary); (f64 code, f32 code)
    const PAIRS: &[(&str, &str, &str)] = &[
        ("c4a17d1004e0", "c4a17c1004e0", "vmovupd ymm0, ymmword ptr [rax+r12*8] -> vmovups ymm0, ymmword ptr [rax+r12*8]"),
        ("c4c17d114520", "c4c17c114520", "vmovupd ymmword ptr [r13+0x20], ymm0 -> vmovups ymmword ptr [r13+0x20], ymm0"),
        ("c57d29cb", "c57c29cb", "vmovapd ymm3, ymm9 -> vmovaps ymm3, ymm9"),
        ("c5ed58cb", "c5ec58cb", "vaddpd ymm1, ymm2, ymm3 -> vaddps ymm1, ymm2, ymm3"),
        ("c4412d5ccb", "c4412c5ccb", "vsubpd ymm9, ymm10, ymm11 -> vsubps ymm9, ymm10, ymm11"),
        ("c5ed598b00380000", "c5ec598b00380000", "vmulpd ymm1, ymm2, ymmword ptr [rbx+0x3800] -> vmulps ymm1, ymm2, ymmword ptr [rbx+0x3800]"),
        ("c5ed5ecb", "c5ec5ecb", "vdivpd ymm1, ymm2, ymm3 -> vdivps ymm1, ymm2, ymm3"),
        ("c5fd51ca", "c5fc51ca", "vsqrtpd ymm1, ymm2 -> vsqrtps ymm1, ymm2"),
        ("c5ed54cb", "c5ec54cb", "vandpd ymm1, ymm2, ymm3 -> vandps ymm1, ymm2, ymm3"),
        ("c5ed55cb", "c5ec55cb", "vandnpd ymm1, ymm2, ymm3 -> vandnps ymm1, ymm2, ymm3"),
        ("c5ed56cb", "c5ec56cb", "vorpd ymm1, ymm2, ymm3 -> vorps ymm1, ymm2, ymm3"),
        ("c5b5572d00000000", "c5b4572d00000000", "vxorpd ymm5, ymm9, ymmword ptr [rip+0] -> vxorps ymm5, ymm9, ymmword ptr [rip+0]"),
        ("c5edc2cb02", "c5ecc2cb02", "vcmppd ymm1, ymm2, ymm3, 2 -> vcmpps ymm1, ymm2, ymm3, 2"),
        ("c5ed5dcb", "c5ec5dcb", "vminpd ymm1, ymm2, ymm3 -> vminps ymm1, ymm2, ymm3"),
        ("c5ed5fcb", "c5ec5fcb", "vmaxpd ymm1, ymm2, ymm3 -> vmaxps ymm1, ymm2, ymm3"),
        ("c4e37d09ca09", "c4e37d08ca09", "vroundpd ymm1, ymm2, 9 -> vroundps ymm1, ymm2, 9"),
        ("c4e2edb8cb", "c4e26db8cb", "vfmadd231pd ymm1, ymm2, ymm3 -> vfmadd231ps ymm1, ymm2, ymm3"),
        ("c4429d98dd", "c4421d98dd", "vfmadd132pd ymm11, ymm12, ymm13 -> vfmadd132ps ymm11, ymm12, ymm13"),
        ("c4e2edbacb", "c4e26dbacb", "vfmsub231pd ymm1, ymm2, ymm3 -> vfmsub231ps ymm1, ymm2, ymm3"),
        ("c4e2edaccb", "c4e26daccb", "vfnmadd213pd ymm1, ymm2, ymm3 -> vfnmadd213ps ymm1, ymm2, ymm3"),
        ("c4e27d191500000000", "c4e27d181500000000", "vbroadcastsd ymm2, qword ptr [rip+0] -> vbroadcastss ymm2, dword ptr [rip+0]"),
        ("c4c27d195748", "c4c27d185748", "vbroadcastsd ymm2, qword ptr [r15+0x48] -> vbroadcastss ymm2, dword ptr [r15+0x48]"),
        ("c5eb58cb", "c5ea58cb", "vaddsd xmm1, xmm2, xmm3 -> vaddss xmm1, xmm2, xmm3"),
        ("c4412b5ccb", "c4412a5ccb", "vsubsd xmm9, xmm10, xmm11 -> vsubss xmm9, xmm10, xmm11"),
        ("c5eb590d00000000", "c5ea590d00000000", "vmulsd xmm1, xmm2, qword ptr [rip+0] -> vmulss xmm1, xmm2, dword ptr [rip+0]"),
        ("c5eb5ecb", "c5ea5ecb", "vdivsd xmm1, xmm2, xmm3 -> vdivss xmm1, xmm2, xmm3"),
        ("c5eb51cb", "c5ea51cb", "vsqrtsd xmm1, xmm2, xmm3 -> vsqrtss xmm1, xmm2, xmm3"),
        ("c5fb104b18", "c5fa104b18", "vmovsd xmm1, qword ptr [rbx+0x18] -> vmovss xmm1, dword ptr [rbx+0x18]"),
        ("c5fb114b18", "c5fa114b18", "vmovsd qword ptr [rbx+0x18], xmm1 -> vmovss dword ptr [rbx+0x18], xmm1"),
        ("c5ebc2cb01", "c5eac2cb01", "vcmpsd xmm1, xmm2, xmm3, 1 -> vcmpss xmm1, xmm2, xmm3, 1"),
        ("c4e2e9b9cb", "c4e269b9cb", "vfmadd231sd xmm1, xmm2, xmm3 -> vfmadd231ss xmm1, xmm2, xmm3"),
        ("c4e3690bcb09", "c4e3690acb09", "vroundsd xmm1, xmm2, xmm3, 9 -> vroundss xmm1, xmm2, xmm3, 9"),
        ("c5f92ec9", "c5f82ec9", "vucomisd xmm1, xmm1 -> vucomiss xmm1, xmm1"),
        ("c5f92fca", "c5f82fca", "vcomisd xmm1, xmm2 -> vcomiss xmm1, xmm2"),
        ("62b1fd481004e0", "62b17c481004e0", "vmovupd zmm0, zmmword ptr [rax+r12*8] -> vmovups zmm0, zmmword ptr [rax+r12*8]"),
        ("62f1ed4858cb", "62f16c4858cb", "vaddpd zmm1, zmm2, zmm3 -> vaddps zmm1, zmm2, zmm3"),
        ("62e1ed40594b01", "62e16c40594b01", "vmulpd zmm17, zmm18, zmmword ptr [rbx+0x40] -> vmulps zmm17, zmm18, zmmword ptr [rbx+0x40]"),
        ("62f1b548572d00000000", "62f13448572d00000000", "vxorpd zmm5, zmm9, zmmword ptr [rip+0] -> vxorps zmm5, zmm9, zmmword ptr [rip+0]"),
        ("62f1ed48c2cb01", "62f16c48c2cb01", "vcmppd k1, zmm2, zmm3, 1 -> vcmpps k1, zmm2, zmm3, 1"),
        ("62f2ed48b8cb", "62f26d48b8cb", "vfmadd231pd zmm1, zmm2, zmm3 -> vfmadd231ps zmm1, zmm2, zmm3"),
        ("62f2fd48191500000000", "62f27d48181500000000", "vbroadcastsd zmm2, qword ptr [rip+0] -> vbroadcastss zmm2, dword ptr [rip+0]"),
        ("62f2fd48195001", "62f27d48185002", "vbroadcastsd zmm2, qword ptr [rax+8] -> vbroadcastss zmm2, dword ptr [rax+8]"),
        ("62f2fe4838d9", "62f27e4838d9", "vpmovm2q zmm3, k1 -> vpmovm2d zmm3, k1"),
        ("62f2fe4839c9", "62f27e4839c9", "vpmovq2m k1, zmm1 -> vpmovd2m k1, zmm1"),
        ("62f1fd4851ca", "62f17c4851ca", "vsqrtpd zmm1, zmm2 -> vsqrtps zmm1, zmm2"),
        ("c4e37d09ca0a", "c4e37d08ca0a", "vroundpd ymm1, ymm2, 10 -> vroundps ymm1, ymm2, 10"),
        ("62f3fd4809f601", "62f37d4808f601", "vrndscalepd zmm6, zmm6, 1 -> vrndscaleps zmm6, zmm6, 1"),
        ("62b3fd4809fa02", "62b37d4808fa02", "vrndscalepd zmm7, zmm18, 2 -> vrndscaleps zmm7, zmm18, 2"),
        ("62a3ed000bcb03", "62a36d000acb03", "vrndscalesd xmm17, xmm18, xmm19, 3 -> vrndscaless xmm17, xmm18, xmm19, 3"),
    ];

    const KEPT: &[(&str, &str)] = &[
        ("c5f844ca", "knotw k1, k2"),
        ("48b94701480149014a01", "movabs rcx, 0x14a014901480147"),
        ("48c1e810", "shr rax, 0x10"),
        ("480fb7c0", "movzx rax, ax"),
        ("e8fb000000", "call .+0x100"),
        ("c5f877", "vzeroupper"),
        ("55", "push rbp"),
        ("5d", "pop rbp"),
        ("c3", "ret"),
        ("488b4308", "mov rax, qword ptr [rbx+8]"),
        ("4883c440", "add rsp, 0x40"),
        ("751e", "jne .+0x20"),
        ("e9fb0f0000", "jmp .+0x1000"),
        ("488d5c2440", "lea rbx, [rsp+0x40]"),
        ("4885c0", "test rax, rax"),
        ("c5f898c9", "kortestw k1, k1"),
    ];

    // (code, the error says, what)
    const REFUSED: &[(&str, &str, &str)] = &[
        ("c4e37d05ca05", "no f32 rewrite", "vpermilpd ymm1, ymm2, 5"),
        (
            "62f1ed585808",
            "embedded broadcast",
            "vaddpd zmm1, zmm2, qword ptr [rax]{1to8}",
        ),
        ("ff1500000000", "function call", "call qword ptr [rip+0]"),
        ("ffd0", "function call", "call rax"),
        ("c5fd50c1", "branch on the lanes", "vmovmskpd eax, ymm1"),
        (
            "488b0500000000",
            "RIP-relative data",
            "mov rax, qword ptr [rip+0]",
        ),
        ("c4e1f97ec8", "no f32 rewrite", "vmovq rax, xmm1"),
        ("c5ec58cb", "no f32 rewrite", "vaddps ymm1, ymm2, ymm3"),
        (
            "c4e36d4bcb40",
            "no f32 rewrite",
            "vblendvpd ymm1, ymm2, ymm3, ymm4",
        ),
        ("f2480f2ac8", "unknown instruction", "cvtsi2sd xmm1, rax"),
    ];

    #[test]
    fn each_double_becomes_its_single_twin() {
        for &(double, single, what) in PAIRS {
            assert_eq!(twin(&h(double)).unwrap(), h(single), "{what}");
            // the scalar kernel's rules are the same
            assert_eq!(
                rewrite_code(&h(double), double.len() / 2, Kernel::Scalar).unwrap(),
                h(single),
                "{what}"
            );
        }
    }

    #[test]
    fn integer_and_mask_instructions_are_kept() {
        for &(code, what) in KEPT {
            assert_eq!(twin(&h(code)).unwrap(), h(code), "{what}");
        }
    }

    #[test]
    fn refusals() {
        for &(code, why, what) in REFUSED {
            let e = twin(&h(code)).unwrap_err().to_string();
            assert!(e.contains(why), "{what}: {e}");
        }
        // downgraded twice
        let once = twin(&h(PAIRS
            .iter()
            .find(|p| p.2.starts_with("vfmadd231pd ymm1"))
            .unwrap()
            .0))
        .unwrap();
        assert!(twin(&once)
            .unwrap_err()
            .to_string()
            .contains("single-precision FMA"));
    }

    #[test]
    fn code_and_data() {
        // vaddpd ymm1, ymm2, ymm3; ret; then data
        let mut code = h("c5ed58cbc3");
        code.extend([0u8; 3]);
        assert!(rewrite_code(&code, 5, Kernel::Simd).is_ok());
        // the data area cut inside an instruction
        assert!(rewrite_code(&code, 3, Kernel::Simd).is_err());
    }
}
