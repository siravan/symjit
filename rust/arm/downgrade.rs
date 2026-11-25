//! aarch64 f64 kernels rewritten into f32 kernels (see `crate::downgrade`): word by word up to the
//! data area. Every word of the "data processing: scalar floating-point and Advanced SIMD" class
//! must match a rule of `RULES` (a template from the `arm!` macro and a mask of the bits that make
//! the instruction, the operand fields cleared), which rewrites it or keeps it; anything else of that
//! class is refused, as are function calls (`blr`). Loads, stores and the integer code are kept.
//!
//! The double and single forms of the floating-point instructions differ in one size bit, bit 22
//! (`ftype` = 01 or 00 in the scalar encodings, `sz`/`size<0>` in the Advanced SIMD ones), except
//! the vector `fmov #imm`, whose `op` (bit 29) selects the double form. The lane moves depend on the
//! kernel (`Kernel`):
//!
//! - scalar (the value in the low half of its slot; a fast complex number's (re, im) in the low
//!   eight bytes of its 16-byte register): `zip1`/`zip2`/`uzp1`/`uzp2` .2d become .2s, `ext` .16b
//!   #8 becomes `ext` .8b #4 (swaps re and im), `dup` and the by-element ops (`fmla`, `fmls`,
//!   `fmul` by d[i]) take s[i], `faddp` d becomes `faddp` s (re + im), `umov` x of d[0] (a branch
//!   on a mask) is kept;
//! - SIMD (twice the lanes): only `dup` of d[0] has a twin (`dup` of s[0]: a parameter
//!   broadcast); the other lane moves and `umov` (the branch on the lanes) are refused.
//!
//! `fmov` between d and x (an f64 bit pattern built in an integer register, the stable math of the
//! complex kernel) is refused.

use anyhow::{anyhow, Result};

use crate::downgrade::Kernel;

const MASK_2R: u32 = 0xfffffc00; // Rn, Rd
const MASK_3R: u32 = 0xffe0fc00; // Rm, Rn, Rd
const MASK_4R: u32 = 0xffe08000; // Rm, Ra, Rn, Rd
const MASK_FCMLA: u32 = 0xffe0e400; // Rm, rot (12:11), Rn, Rd
const MASK_FCADD: u32 = 0xffe0ec00; // Rm, rot (12), Rn, Rd
const MASK_ELEM: u32 = 0xffc0f400; // L, M:Rm, H, Rn, Rd (by element)
const MASK_FCMP: u32 = 0xfffffc1f; // Rn
const MASK_FCSEL: u32 = 0xffe00c00; // Rm, cond, Rn, Rd
const MASK_FMOV_IMM: u32 = 0xffe01fe0; // imm8, Rd
const MASK_VFMOV_IMM: u32 = 0xfff8fc00; // abc, defgh, Rd
const MASK_ELEM_D: u32 = 0xffeffc00; // dup / umov of a d element: the index (bit 20), Rn, Rd
const MASK_RD: u32 = 0xffffffe0;

const FLIP22: u32 = 1 << 22;
const FLIP29: u32 = 1 << 29;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    /// flips these bits (the size bit)
    Flip(u32),
    /// kept as it is
    Keep,
    /// zip/uzp .2d -> .2s (Q = 0, size = 10)
    Pair2S,
    /// ext .16b #8 -> ext .8b #4
    Ext8B4,
    /// dup v.2d, v.d[i] -> dup v.4s, v.s[i]
    DupS,
    /// dup v.2d, v.d[0] -> dup v.4s, v.s[0]; d[1] refused
    DupS0,
    /// fmla/fmls/fmul by element d[i] -> s[i]
    ElemS,
    /// umov x, v.d[0] kept; d[1] refused
    UmovD0,
    Refuse(&'static str),
}

use Act::*;

struct Rule {
    bits: u32,
    mask: u32,
    scalar: Act,
    simd: Act,
}

const fn r(bits: u32, mask: u32, scalar: Act, simd: Act) -> Rule {
    Rule {
        bits,
        mask,
        scalar,
        simd,
    }
}

const fn both(bits: u32, mask: u32, act: Act) -> Rule {
    r(bits, mask, act, act)
}

const LANE_MOVE: Act = Refuse("a lane move with no twin in a SIMD kernel of twice the lanes");
const BRANCH: Act = Refuse("a branch on the lanes of a SIMD kernel");

static RULES: [Rule; 64] = [
    // scalar floating point (ftype, bit 22)
    both(arm! {fadd d(0), d(0), d(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fsub d(0), d(0), d(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fmul d(0), d(0), d(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fdiv d(0), d(0), d(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fsqrt d(0), d(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {fneg d(0), d(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {fabs d(0), d(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frinti d(0), d(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frintm d(0), d(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frintp d(0), d(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frintz d(0), d(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {fmadd d(0), d(0), d(0), d(0)}, MASK_4R, Flip(FLIP22)),
    both(arm! {fnmadd d(0), d(0), d(0), d(0)}, MASK_4R, Flip(FLIP22)),
    both(arm! {fmsub d(0), d(0), d(0), d(0)}, MASK_4R, Flip(FLIP22)),
    both(arm! {fnmsub d(0), d(0), d(0), d(0)}, MASK_4R, Flip(FLIP22)),
    both(arm! {fcmp d(0), #0.0}, MASK_FCMP, Flip(FLIP22)),
    both(arm! {fcsel d(0), d(0), d(0), eq}, MASK_FCSEL, Flip(FLIP22)),
    both(arm! {fmov d(0), #0.5}, MASK_FMOV_IMM, Flip(FLIP22)),
    // a 64-bit register move: the float and the zeros above it, or two floats
    both(arm! {fmov d(0), d(0)}, MASK_2R, Keep),
    both(
        arm! {fmov d(0), x(0)},
        MASK_2R,
        Refuse("an f64 bit pattern moved from an integer register"),
    ),
    both(
        arm! {fmov x(0), d(0)},
        MASK_2R,
        Refuse("an f64 bit pattern moved to an integer register"),
    ),
    // Advanced SIMD scalar (sz, bit 22)
    both(arm! {fcmeq d(0), d(0), d(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fcmgt d(0), d(0), d(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fcmge d(0), d(0), d(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fcmeq d(0), d(0), #0.0}, MASK_2R, Flip(FLIP22)),
    both(arm! {fcmge d(0), d(0), #0.0}, MASK_2R, Flip(FLIP22)),
    both(arm! {fcmlt d(0), d(0), #0.0}, MASK_2R, Flip(FLIP22)),
    r(arm! {faddp d(0), q(0)}, MASK_2R, Flip(FLIP22), LANE_MOVE),
    // Advanced SIMD vector (sz, bit 22)
    both(arm! {fadd q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fsub q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fmul q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fdiv q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fmla q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fmls q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fsqrt q(0), q(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {fneg q(0), q(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {fabs q(0), q(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frinti q(0), q(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frintm q(0), q(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frintp q(0), q(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {frintz q(0), q(0)}, MASK_2R, Flip(FLIP22)),
    both(arm! {fcmeq q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fcmgt q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fcmge q(0), q(0), q(0)}, MASK_3R, Flip(FLIP22)),
    both(arm! {fcmeq q(0), q(0), #0.0}, MASK_2R, Flip(FLIP22)),
    // vector fmov #imm: op (bit 29); fmov q, #0.0 is movi (below)
    both(arm! {fmov q(0), #0.5}, MASK_VFMOV_IMM, Flip(FLIP29)),
    // complex arithmetic (size 11 -> 10: bit 22)
    r(
        arm! {fcmla q(0), q(0), q(0), #0},
        MASK_FCMLA,
        Flip(FLIP22),
        LANE_MOVE,
    ),
    r(
        arm! {fcadd q(0), q(0), q(0), #90},
        MASK_FCADD,
        Flip(FLIP22),
        LANE_MOVE,
    ),
    // by element
    r(arm! {fmla q(0), q(0), d(0)[0]}, MASK_ELEM, ElemS, LANE_MOVE),
    r(arm! {fmls q(0), q(0), d(0)[0]}, MASK_ELEM, ElemS, LANE_MOVE),
    r(arm! {fmul q(0), q(0), d(0)[0]}, MASK_ELEM, ElemS, LANE_MOVE),
    // lane moves
    r(arm! {zip1 q(0), q(0), q(0)}, MASK_3R, Pair2S, LANE_MOVE),
    r(arm! {zip2 q(0), q(0), q(0)}, MASK_3R, Pair2S, LANE_MOVE),
    r(arm! {uzp1 q(0), q(0), q(0)}, MASK_3R, Pair2S, LANE_MOVE),
    r(arm! {uzp2 q(0), q(0), q(0)}, MASK_3R, Pair2S, LANE_MOVE),
    r(arm! {ext q(0), q(0), q(0), #8}, MASK_3R, Ext8B4, LANE_MOVE),
    r(arm! {dup q(0), q(0)[0]}, MASK_ELEM_D, DupS, DupS0),
    r(arm! {umov x(0), v(0).d[0]}, MASK_ELEM_D, UmovD0, BRANCH),
    // bit operations: any size
    both(arm! {movi v(0).16b, #0}, MASK_RD, Keep),
    both(arm! {movi d(0), #0}, MASK_RD, Keep),
    both(arm! {movi q(0), #0}, MASK_RD, Keep),
    // and, bic, orr (mov), orn, eor, bsl, bit, bif: .8b and .16b
    both(0x0e201c00, 0x9f20fc00, Keep),
    // not (mvn): .8b and .16b
    both(0x2e205800, 0xbffffc00, Keep),
    both(
        arm! {blr x(0)},
        0xfffffc1f,
        Refuse("a function call (math functions take and return doubles)"),
    ),
];

/// Is `w` in the "data processing: scalar floating-point and Advanced SIMD" class (op0 = x111)?
fn is_fp_simd(w: u32) -> bool {
    (w >> 25) & 7 == 7
}

/// The f32 twin of the word `w` at `pc` in a `kernel`.
fn rewrite(w: u32, pc: usize, kernel: Kernel) -> Result<u32> {
    let rule = RULES.iter().find(|r| w & r.mask == r.bits & r.mask);

    let act = match rule {
        Some(r) => match kernel {
            Kernel::Scalar => r.scalar,
            Kernel::Simd => r.simd,
        },
        None if is_fp_simd(w) => {
            return Err(anyhow!(
                "0x{pc:x}: no f32 rule for the instruction 0x{w:08x}"
            ))
        }
        None => Keep,
    };

    let index_d = (w >> 20) & 1; // dup / umov: imm5 = i:1000

    Ok(match act {
        Flip(bits) => w ^ bits,
        Keep => w,
        Pair2S => w & !(1 << 30) & !FLIP22,
        Ext8B4 => (w & !(1 << 30) & !(0xf << 11)) | (4 << 11),
        DupS => (w & !(0x1f << 16)) | (((index_d << 3) | 4) << 16),
        DupS0 if index_d == 0 => (w & !(0x1f << 16)) | (4 << 16),
        DupS0 => {
            return Err(anyhow!(
                "0x{pc:x}: dup of d[1]: a lane move with no twin in a SIMD kernel (0x{w:08x})"
            ))
        }
        ElemS => {
            // sz = 1: the index is H (bit 11), L (bit 21) is 0; sz = 0: H:L
            let i = (w >> 11) & 1;
            (w & !FLIP22 & !(1 << 11) & !(1 << 21)) | (i << 21)
        }
        UmovD0 if index_d == 0 => w,
        UmovD0 => {
            return Err(anyhow!(
                "0x{pc:x}: umov of d[1] in a scalar kernel (0x{w:08x})"
            ))
        }
        Refuse(why) => return Err(anyhow!("0x{pc:x}: {why} (0x{w:08x})")),
    })
}

/// The machine code `code` with its instructions (up to `end`, the data area) rewritten to f32; the
/// bytes from `end` on as they are.
pub fn rewrite_code(code: &[u8], end: usize, kernel: Kernel) -> Result<Vec<u8>> {
    if end % 4 != 0 {
        return Err(anyhow!("the code (0x{end:x} bytes) is not in 4-byte words"));
    }

    let mut out = code.to_vec();

    for pc in (0..end).step_by(4) {
        let w = u32::from_le_bytes(code[pc..pc + 4].try_into().unwrap());
        out[pc..pc + 4].copy_from_slice(&rewrite(w, pc, kernel)?.to_le_bytes());
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    //! The f32 twins against GNU as: each pair is an f64 instruction and its f32 twin as the
    //! assembler encodes them (operands v29, v25, v28, v22: every register field set).

    use super::*;

    fn twin(w: u32, kernel: Kernel) -> u32 {
        rewrite(w, 0, kernel).unwrap()
    }

    fn refused(w: u32, kernel: Kernel) -> String {
        rewrite(w, 0, kernel).unwrap_err().to_string()
    }

    // generated by GNU as (aarch64-linux-gnu-as -march=armv8.3-a, objdump -d): (f64 word, f32 word);
    // COMMON: both kernels (a kept instruction: its own twin), SCALAR: scalar kernels only
    const COMMON: &[(u32, u32, &str)] = &[
        (
            0x1e7c2b3d,
            0x1e3c2b3d,
            "fadd d29, d25, d28 -> fadd s29, s25, s28",
        ),
        (
            0x1e7c3b3d,
            0x1e3c3b3d,
            "fsub d29, d25, d28 -> fsub s29, s25, s28",
        ),
        (
            0x1e7c0b3d,
            0x1e3c0b3d,
            "fmul d29, d25, d28 -> fmul s29, s25, s28",
        ),
        (
            0x1e7c1b3d,
            0x1e3c1b3d,
            "fdiv d29, d25, d28 -> fdiv s29, s25, s28",
        ),
        (0x1e61c33d, 0x1e21c33d, "fsqrt d29, d25 -> fsqrt s29, s25"),
        (0x1e61433d, 0x1e21433d, "fneg d29, d25 -> fneg s29, s25"),
        (0x1e60c33d, 0x1e20c33d, "fabs d29, d25 -> fabs s29, s25"),
        (0x1e67c33d, 0x1e27c33d, "frinti d29, d25 -> frinti s29, s25"),
        (0x1e65433d, 0x1e25433d, "frintm d29, d25 -> frintm s29, s25"),
        (0x1e64c33d, 0x1e24c33d, "frintp d29, d25 -> frintp s29, s25"),
        (0x1e65c33d, 0x1e25c33d, "frintz d29, d25 -> frintz s29, s25"),
        (
            0x1f5c5b3d,
            0x1f1c5b3d,
            "fmadd d29, d25, d28, d22 -> fmadd s29, s25, s28, s22",
        ),
        (
            0x1f7c5b3d,
            0x1f3c5b3d,
            "fnmadd d29, d25, d28, d22 -> fnmadd s29, s25, s28, s22",
        ),
        (
            0x1f5cdb3d,
            0x1f1cdb3d,
            "fmsub d29, d25, d28, d22 -> fmsub s29, s25, s28, s22",
        ),
        (
            0x1f7cdb3d,
            0x1f3cdb3d,
            "fnmsub d29, d25, d28, d22 -> fnmsub s29, s25, s28, s22",
        ),
        (0x1e602328, 0x1e202328, "fcmp d25, #0.0 -> fcmp s25, #0.0"),
        (
            0x1e7caf3d,
            0x1e3caf3d,
            "fcsel d29, d25, d28, ge -> fcsel s29, s25, s28, ge",
        ),
        (
            0x1e7cbf3d,
            0x1e3cbf3d,
            "fcsel d29, d25, d28, lt -> fcsel s29, s25, s28, lt",
        ),
        (0x1e6c101d, 0x1e2c101d, "fmov d29, #0.5 -> fmov s29, #0.5"),
        (0x1e6e101d, 0x1e2e101d, "fmov d29, #1.0 -> fmov s29, #1.0"),
        (0x1e7e101d, 0x1e3e101d, "fmov d29, #-1.0 -> fmov s29, #-1.0"),
        (0x1e60101d, 0x1e20101d, "fmov d29, #2.0 -> fmov s29, #2.0"),
        (
            0x5e7ce73d,
            0x5e3ce73d,
            "fcmeq d29, d25, d28 -> fcmeq s29, s25, s28",
        ),
        (
            0x7efce73d,
            0x7ebce73d,
            "fcmgt d29, d25, d28 -> fcmgt s29, s25, s28",
        ),
        (
            0x7e7ce73d,
            0x7e3ce73d,
            "fcmge d29, d25, d28 -> fcmge s29, s25, s28",
        ),
        (
            0x5ee0db3d,
            0x5ea0db3d,
            "fcmeq d29, d25, #0.0 -> fcmeq s29, s25, #0.0",
        ),
        (
            0x7ee0cb3d,
            0x7ea0cb3d,
            "fcmge d29, d25, #0.0 -> fcmge s29, s25, #0.0",
        ),
        (
            0x5ee0eb3d,
            0x5ea0eb3d,
            "fcmlt d29, d25, #0.0 -> fcmlt s29, s25, #0.0",
        ),
        (
            0x4e7cd73d,
            0x4e3cd73d,
            "fadd v29.2d, v25.2d, v28.2d -> fadd v29.4s, v25.4s, v28.4s",
        ),
        (
            0x4efcd73d,
            0x4ebcd73d,
            "fsub v29.2d, v25.2d, v28.2d -> fsub v29.4s, v25.4s, v28.4s",
        ),
        (
            0x6e7cdf3d,
            0x6e3cdf3d,
            "fmul v29.2d, v25.2d, v28.2d -> fmul v29.4s, v25.4s, v28.4s",
        ),
        (
            0x6e7cff3d,
            0x6e3cff3d,
            "fdiv v29.2d, v25.2d, v28.2d -> fdiv v29.4s, v25.4s, v28.4s",
        ),
        (
            0x4e7ccf3d,
            0x4e3ccf3d,
            "fmla v29.2d, v25.2d, v28.2d -> fmla v29.4s, v25.4s, v28.4s",
        ),
        (
            0x4efccf3d,
            0x4ebccf3d,
            "fmls v29.2d, v25.2d, v28.2d -> fmls v29.4s, v25.4s, v28.4s",
        ),
        (
            0x6ee1fb3d,
            0x6ea1fb3d,
            "fsqrt v29.2d, v25.2d -> fsqrt v29.4s, v25.4s",
        ),
        (
            0x6ee0fb3d,
            0x6ea0fb3d,
            "fneg v29.2d, v25.2d -> fneg v29.4s, v25.4s",
        ),
        (
            0x4ee0fb3d,
            0x4ea0fb3d,
            "fabs v29.2d, v25.2d -> fabs v29.4s, v25.4s",
        ),
        (
            0x6ee19b3d,
            0x6ea19b3d,
            "frinti v29.2d, v25.2d -> frinti v29.4s, v25.4s",
        ),
        (
            0x4e619b3d,
            0x4e219b3d,
            "frintm v29.2d, v25.2d -> frintm v29.4s, v25.4s",
        ),
        (
            0x4ee18b3d,
            0x4ea18b3d,
            "frintp v29.2d, v25.2d -> frintp v29.4s, v25.4s",
        ),
        (
            0x4ee19b3d,
            0x4ea19b3d,
            "frintz v29.2d, v25.2d -> frintz v29.4s, v25.4s",
        ),
        (
            0x4e7ce73d,
            0x4e3ce73d,
            "fcmeq v29.2d, v25.2d, v28.2d -> fcmeq v29.4s, v25.4s, v28.4s",
        ),
        (
            0x6efce73d,
            0x6ebce73d,
            "fcmgt v29.2d, v25.2d, v28.2d -> fcmgt v29.4s, v25.4s, v28.4s",
        ),
        (
            0x6e7ce73d,
            0x6e3ce73d,
            "fcmge v29.2d, v25.2d, v28.2d -> fcmge v29.4s, v25.4s, v28.4s",
        ),
        (
            0x4ee0db3d,
            0x4ea0db3d,
            "fcmeq v29.2d, v25.2d, #0.0 -> fcmeq v29.4s, v25.4s, #0.0",
        ),
        (
            0x6f03f41d,
            0x4f03f41d,
            "fmov v29.2d, #0.5 -> fmov v29.4s, #0.5",
        ),
        (
            0x6f03f61d,
            0x4f03f61d,
            "fmov v29.2d, #1.0 -> fmov v29.4s, #1.0",
        ),
        (
            0x6f00f41d,
            0x4f00f41d,
            "fmov v29.2d, #2.0 -> fmov v29.4s, #2.0",
        ),
        (0x1e60433d, 0x1e60433d, "fmov d29, d25"),
        (0x4f00e41d, 0x4f00e41d, "movi v29.16b, #0"),
        (0x2f00e41d, 0x2f00e41d, "movi d29, #0"),
        (0x6f00e41d, 0x6f00e41d, "movi v29.2d, #0"),
        (0x0e3c1f3d, 0x0e3c1f3d, "and v29.8b, v25.8b, v28.8b"),
        (0x4e3c1f3d, 0x4e3c1f3d, "and v29.16b, v25.16b, v28.16b"),
        (0x4ebc1f3d, 0x4ebc1f3d, "orr v29.16b, v25.16b, v28.16b"),
        (0x4efc1f3d, 0x4efc1f3d, "orn v29.16b, v25.16b, v28.16b"),
        (0x6e3c1f3d, 0x6e3c1f3d, "eor v29.16b, v25.16b, v28.16b"),
        (0x4e7c1f3d, 0x4e7c1f3d, "bic v29.16b, v25.16b, v28.16b"),
        (0x6e7c1f3d, 0x6e7c1f3d, "bsl v29.16b, v25.16b, v28.16b"),
        (0x6ebc1f3d, 0x6ebc1f3d, "bit v29.16b, v25.16b, v28.16b"),
        (0x6efc1f3d, 0x6efc1f3d, "bif v29.16b, v25.16b, v28.16b"),
        (0x6e205b3d, 0x6e205b3d, "not v29.16b, v25.16b"),
        (0x2e205b3d, 0x2e205b3d, "not v29.8b, v25.8b"),
        (0x4d40cf3d, 0x4d40cf3d, "ld1r {v29.2d}, [x25]"),
        (0xfd400b3d, 0xfd400b3d, "ldr d29, [x25, #16]"),
        (0x3dc0073d, 0x3dc0073d, "ldr q29, [x25, #16]"),
        (0x3d80073d, 0x3d80073d, "str q29, [x25, #16]"),
        (0x6d40673d, 0x6d40673d, "ldp d29, d25, [x25]"),
        (0x8b1c033d, 0x8b1c033d, "add x29, x25, x28"),
        (0x94000000, 0x94000000, "bl ."),
    ];
    const SCALAR: &[(u32, u32, &str)] = &[
        (
            0x7e70db3d,
            0x7e30db3d,
            "faddp d29, v25.2d -> faddp s29, v25.2s",
        ),
        (
            0x6edcc73d,
            0x6e9cc73d,
            "fcmla v29.2d, v25.2d, v28.2d, #0 -> fcmla v29.4s, v25.4s, v28.4s, #0",
        ),
        (
            0x6edccf3d,
            0x6e9ccf3d,
            "fcmla v29.2d, v25.2d, v28.2d, #90 -> fcmla v29.4s, v25.4s, v28.4s, #90",
        ),
        (
            0x6edcd73d,
            0x6e9cd73d,
            "fcmla v29.2d, v25.2d, v28.2d, #180 -> fcmla v29.4s, v25.4s, v28.4s, #180",
        ),
        (
            0x6edcdf3d,
            0x6e9cdf3d,
            "fcmla v29.2d, v25.2d, v28.2d, #270 -> fcmla v29.4s, v25.4s, v28.4s, #270",
        ),
        (
            0x6edce73d,
            0x6e9ce73d,
            "fcadd v29.2d, v25.2d, v28.2d, #90 -> fcadd v29.4s, v25.4s, v28.4s, #90",
        ),
        (
            0x6edcf73d,
            0x6e9cf73d,
            "fcadd v29.2d, v25.2d, v28.2d, #270 -> fcadd v29.4s, v25.4s, v28.4s, #270",
        ),
        (
            0x4fdc133d,
            0x4f9c133d,
            "fmla v29.2d, v25.2d, v28.d[0] -> fmla v29.4s, v25.4s, v28.s[0]",
        ),
        (
            0x4fdc1b3d,
            0x4fbc133d,
            "fmla v29.2d, v25.2d, v28.d[1] -> fmla v29.4s, v25.4s, v28.s[1]",
        ),
        (
            0x4fdc533d,
            0x4f9c533d,
            "fmls v29.2d, v25.2d, v28.d[0] -> fmls v29.4s, v25.4s, v28.s[0]",
        ),
        (
            0x4fdc5b3d,
            0x4fbc533d,
            "fmls v29.2d, v25.2d, v28.d[1] -> fmls v29.4s, v25.4s, v28.s[1]",
        ),
        (
            0x4fdc933d,
            0x4f9c933d,
            "fmul v29.2d, v25.2d, v28.d[0] -> fmul v29.4s, v25.4s, v28.s[0]",
        ),
        (
            0x4fdc9b3d,
            0x4fbc933d,
            "fmul v29.2d, v25.2d, v28.d[1] -> fmul v29.4s, v25.4s, v28.s[1]",
        ),
        (
            0x4edc3b3d,
            0x0e9c3b3d,
            "zip1 v29.2d, v25.2d, v28.2d -> zip1 v29.2s, v25.2s, v28.2s",
        ),
        (
            0x4edc7b3d,
            0x0e9c7b3d,
            "zip2 v29.2d, v25.2d, v28.2d -> zip2 v29.2s, v25.2s, v28.2s",
        ),
        (
            0x4edc1b3d,
            0x0e9c1b3d,
            "uzp1 v29.2d, v25.2d, v28.2d -> uzp1 v29.2s, v25.2s, v28.2s",
        ),
        (
            0x4edc5b3d,
            0x0e9c5b3d,
            "uzp2 v29.2d, v25.2d, v28.2d -> uzp2 v29.2s, v25.2s, v28.2s",
        ),
        (
            0x6e1c433d,
            0x2e1c233d,
            "ext v29.16b, v25.16b, v28.16b, #8 -> ext v29.8b, v25.8b, v28.8b, #4",
        ),
        (
            0x4e08073d,
            0x4e04073d,
            "dup v29.2d, v25.d[0] -> dup v29.4s, v25.s[0]",
        ),
        (
            0x4e18073d,
            0x4e0c073d,
            "dup v29.2d, v25.d[1] -> dup v29.4s, v25.s[1]",
        ),
        (0x4e083f3d, 0x4e083f3d, "umov x29, v25.d[0]"),
    ];
    // refused in both kernels (umov of d[1]: in scalar kernels; in SIMD kernels every umov)
    const REFUSED: &[(u32, &str)] = &[
        (0xd63f0120, "blr x9"),
        (0x9e67001d, "fmov d29, x0"),
        (0x9e6603a0, "fmov x0, d29"),
        (0x1e7c4b3d, "fmax d29, d25, d28"),
        (0x1e3c2b3d, "fadd s29, s25, s28"),
        (0x4e3cd73d, "fadd v29.4s, v25.4s, v28.4s"),
        (0x4e183f3d, "umov x29, v25.d[1]"),
        (0x9e780320, "fcvtzs x0, d25"),
        (0x9e62001d, "scvtf d29, x0"),
    ];

    #[test]
    fn twins_of_both_kernels() {
        for &(f64w, f32w, what) in COMMON {
            assert_eq!(twin(f64w, Kernel::Scalar), f32w, "scalar: {what}");
            assert_eq!(twin(f64w, Kernel::Simd), f32w, "simd: {what}");
        }
    }

    #[test]
    fn twins_of_scalar_kernels() {
        for &(f64w, f32w, what) in SCALAR {
            assert_eq!(twin(f64w, Kernel::Scalar), f32w, "{what}");
        }
    }

    #[test]
    fn simd_kernels_refuse_lane_moves() {
        for &(f64w, f32w, what) in SCALAR {
            if f64w == 0x4e08073d {
                // dup of d[0] (a parameter broadcast) has a twin
                assert_eq!(twin(f64w, Kernel::Simd), f32w, "{what}");
            } else {
                let e = refused(f64w, Kernel::Simd);
                assert!(e.contains("lane") || e.contains("branch"), "{what}: {e}");
            }
        }
    }

    /// the word GNU as gives `what` in the tables
    fn gnu(what: &str) -> u32 {
        COMMON
            .iter()
            .chain(SCALAR.iter())
            .find(|p| p.2.split(" -> ").next() == Some(what))
            .unwrap_or_else(|| panic!("{what} is not in the tables"))
            .0
    }

    #[test]
    fn templates_are_the_generators_encodings() {
        // the generators' encodings (the rules' templates) with every register field set
        let cases = [
            (
                arm! {fcmla q(29), q(25), q(28), #90},
                "fcmla v29.2d, v25.2d, v28.2d, #90",
            ),
            (
                arm! {fcadd q(29), q(25), q(28), #270},
                "fcadd v29.2d, v25.2d, v28.2d, #270",
            ),
            (arm! {fcmeq d(29), d(25), #0.0}, "fcmeq d29, d25, #0.0"),
            (arm! {fcmlt d(29), d(25), #0.0}, "fcmlt d29, d25, #0.0"),
            (
                arm! {fmla q(29), q(25), d(28)[1]},
                "fmla v29.2d, v25.2d, v28.d[1]",
            ),
            (
                arm! {fmls q(29), q(25), d(28)[0]},
                "fmls v29.2d, v25.2d, v28.d[0]",
            ),
            (
                arm! {fmul q(29), q(25), d(28)[1]},
                "fmul v29.2d, v25.2d, v28.d[1]",
            ),
            (
                arm! {ext q(29), q(25), q(28), #8},
                "ext v29.16b, v25.16b, v28.16b, #8",
            ),
            (arm! {dup q(29), q(25)[1]}, "dup v29.2d, v25.d[1]"),
            (arm! {umov x(29), v(25).d[0]}, "umov x29, v25.d[0]"),
            (arm! {fmov q(29), #0.5}, "fmov v29.2d, #0.5"),
            (arm! {fmov q(29), #2.0}, "fmov v29.2d, #2.0"),
            (arm! {fmov d(29), #0.5}, "fmov d29, #0.5"),
            (arm! {fmov d(29), #-1.0}, "fmov d29, #-1.0"),
            (
                arm! {fcsel d(29), d(25), d(28), ge},
                "fcsel d29, d25, d28, ge",
            ),
            (
                arm! {fcsel d(29), d(25), d(28), lt},
                "fcsel d29, d25, d28, lt",
            ),
            (arm! {fcmp d(25), #0.0}, "fcmp d25, #0.0"),
            (
                arm! {zip2 q(29), q(25), q(28)},
                "zip2 v29.2d, v25.2d, v28.2d",
            ),
            (
                arm! {uzp1 q(29), q(25), q(28)},
                "uzp1 v29.2d, v25.2d, v28.2d",
            ),
            (arm! {faddp d(29), q(25)}, "faddp d29, v25.2d"),
            (arm! {ld1r {q(29)}, [x(25)]}, "ld1r {v29.2d}, [x25]"),
            (
                arm! {fmadd d(29), d(25), d(28), d(22)},
                "fmadd d29, d25, d28, d22",
            ),
            (arm! {frinti q(29), q(25)}, "frinti v29.2d, v25.2d"),
            (arm! {fmov d(29), d(25)}, "fmov d29, d25"),
        ];
        for (w, what) in cases {
            assert_eq!(w, gnu(what), "{what}: 0x{w:08x}");
        }
    }

    #[test]
    fn refusals() {
        for k in [Kernel::Scalar, Kernel::Simd] {
            for &(w, what) in REFUSED {
                let e = refused(w, k);
                let why = if what.starts_with("blr") {
                    "function call"
                } else if what.starts_with("fmov") {
                    "integer register"
                } else if what.starts_with("umov") {
                    if k == Kernel::Scalar {
                        "d[1]"
                    } else {
                        "branch"
                    }
                } else {
                    // fmax, the conversions, and single forms (a kernel downgraded twice)
                    "no f32 rule"
                };
                assert!(e.contains(why), "{k:?}, {what}: {e}");
            }
        }
        assert!(
            refused(0x4e083f3d, Kernel::Simd).contains("branch"),
            "umov x29, v25.d[0]"
        );
    }

    #[test]
    fn code_and_data() {
        let mut code = Vec::new();
        code.extend(0x1e7c2b3du32.to_le_bytes()); // fadd d29, d25, d28
        code.extend(0xd65f03c0u32.to_le_bytes()); // ret
        code.extend(1.0f64.to_le_bytes());
        let out = rewrite_code(&code, 8, Kernel::Scalar).unwrap();
        assert_eq!(&out[..4], &0x1e3c2b3du32.to_le_bytes());
        assert_eq!(&out[4..], &code[4..]);
        assert!(rewrite_code(&code, 6, Kernel::Scalar).is_err());
    }
}
