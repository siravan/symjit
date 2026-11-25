//! f32 kernels from f64 kernels: the machine code of a compiled f64 kernel rewritten instruction by
//! instruction into the same kernel in f32 (`Application::downgrade_f32`, `dump(.., "scalar-f32")`
//! and `dump(.., "simd-f32")`). The rewrite keeps every instruction's length and every load, store
//! and offset, so the f32 kernel reads and writes the f64 kernel's bytes:
//!
//! - a scalar kernel (`Kernel::Scalar`, the `compiled` kernel) computes in the low half of each
//!   8-byte slot: its states, observables and parameters are f32 values in the low four bytes of
//!   8-byte slots (a fast complex kernel: (re, im) in the low eight bytes of a 16-byte slot);
//! - a SIMD kernel (`Kernel::Simd`, the `compiled_simd` kernel) has twice the lanes (a register of
//!   2, 4 or 8 doubles holds 4, 8 or 16 floats): its state and observable columns are packed f32
//!   arrays, and it is called with half the index (element `idx` of a column is read at byte
//!   8 x idx); its parameters are f32 values in the low half of 8-byte slots.
//!
//! The data area after the code (the constant pool and the generators' tables) is converted:
//! each 8-byte double becomes its float in both halves, and a NaN pattern (the masks: all ones,
//! 0x7fff_ffff_ffff_ffff) keeps its high 32 bits in both halves, so -0.0 (0x8000_0000_0000_0000)
//! becomes the sign mask 0x8000_0000_8000_0000. A double out of the float range is an error.
//!
//! The architectures' rules are in `amd/downgrade.rs` (x86-64: AVX, AVX2, AVX-512),
//! `arm/downgrade.rs` (aarch64) and `riscv64/downgrade.rs` (riscv64: the scalar kernel; a complex
//! number's re and im each in the low half of its own slot). An instruction with no rule is an error, and so are function
//! calls (math functions take and return doubles) and, in a SIMD kernel, branches on the lanes.

use anyhow::{anyhow, Result};

/// The kind of kernel: the scalar (`compiled`) or the SIMD (`compiled_simd`) kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Simd,
}

/// Where the data area starts: the assembler's `@data` label.
pub fn data_start(labels: &[(usize, String)]) -> Result<usize> {
    labels
        .iter()
        .find(|(_, s)| s == "@data")
        .map(|(k, _)| *k)
        .ok_or_else(|| anyhow!("the machine code has no @data label (loaded, not compiled here?)"))
}

/// An 8-byte double as two copies of its float (a NaN pattern: of its high 32 bits).
pub fn float_bits(q: u64) -> Result<u64> {
    let exp = (q >> 52) & 0x7ff;
    let h = if exp == 0x7ff && q & ((1 << 52) - 1) != 0 {
        q >> 32
    } else {
        let x = f64::from_bits(q);
        if x.is_finite() && x.abs() > f32::MAX as f64 {
            return Err(anyhow!("the constant {x:e} has no float"));
        }
        (x as f32).to_bits() as u64
    };
    Ok(h | (h << 32))
}

/// Converts the data area `code[start..]` in place (see the module).
pub fn convert_data(code: &mut [u8], start: usize) -> Result<()> {
    // the constants are 8-byte quads from `start` (on riscv64 it may be 4 mod 8)
    if start > code.len() || (code.len() - start) % 8 != 0 {
        return Err(anyhow!(
            "the data area (0x{:x}..0x{:x}) is not in 8-byte units",
            start,
            code.len()
        ));
    }

    for at in (start..code.len()).step_by(8) {
        let q = u64::from_le_bytes(code[at..at + 8].try_into().unwrap());
        code[at..at + 8].copy_from_slice(&float_bits(q)?.to_le_bytes());
    }

    Ok(())
}

/// The f32 twin of the machine code `code` of a `kernel` for `arch` ("x86_64", "aarch64" or
/// "riscv64": the scalar kernel only), whose
/// assembler labels are `labels` (`MachineCode::labels`).
pub fn downgrade_f32(
    arch: &str,
    kernel: Kernel,
    code: &[u8],
    labels: &[(usize, String)],
) -> Result<Vec<u8>> {
    let start = data_start(labels)?;
    if start > code.len() {
        return Err(anyhow!("@data (0x{start:x}) is past the code's end"));
    }

    let mut out = match arch {
        "x86_64" => crate::amd::downgrade::rewrite_code(code, start, kernel)?,
        "aarch64" => crate::arm::downgrade::rewrite_code(code, start, kernel)?,
        "riscv64" => crate::riscv64::downgrade::rewrite_code(code, start, kernel)?,
        _ => return Err(anyhow!("no f32 kernels for {arch}")),
    };

    convert_data(&mut out, start)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_of_constants() {
        let f = |x: f64| float_bits(x.to_bits()).unwrap();
        assert_eq!(f(1.0), 0x3f80_0000_3f80_0000);
        assert_eq!(f(0.5), 0x3f00_0000_3f00_0000);
        assert_eq!(f(0.0), 0);
        // -0.0 is the sign mask of both floats
        assert_eq!(f(-0.0), 0x8000_0000_8000_0000);
        assert_eq!(f(f64::INFINITY), 0x7f80_0000_7f80_0000);
        assert_eq!(f(f64::NEG_INFINITY), 0xff80_0000_ff80_0000);
        // rounded to nearest
        assert_eq!(
            f(0.1),
            ((0.1f32.to_bits() as u64) << 32) | 0.1f32.to_bits() as u64
        );
        // underflow to zero, not an error
        assert_eq!(f(1e-300), 0);
        // NaN patterns keep their high bits: the masks
        assert_eq!(float_bits(u64::MAX).unwrap(), u64::MAX);
        assert_eq!(
            float_bits(0x7fff_ffff_ffff_ffff).unwrap(),
            0x7fff_ffff_7fff_ffff
        );
        assert!(float_bits(1e300f64.to_bits()).is_err());
    }

    #[test]
    fn data_area() {
        let mut code = vec![0u8; 8];
        code.extend(2.0f64.to_le_bytes());
        code.extend((-0.0f64).to_le_bytes());
        convert_data(&mut code, 8).unwrap();
        assert_eq!(&code[..8], &[0; 8]);
        assert_eq!(&code[8..16], &0x4000_0000_4000_0000u64.to_le_bytes());
        assert_eq!(&code[16..24], &0x8000_0000_8000_0000u64.to_le_bytes());
        assert!(convert_data(&mut code, 4).is_err());
        // a data area at 4 mod 8 (riscv64): its quads from there
        let mut code4 = vec![0u8; 4];
        code4.extend(1.0f64.to_le_bytes());
        convert_data(&mut code4, 4).unwrap();
        assert_eq!(&code4[4..], &0x3f80_0000_3f80_0000u64.to_le_bytes());
        let labels = vec![(0, "@main".to_string()), (8, "@data".to_string())];
        assert_eq!(data_start(&labels).unwrap(), 8);
        assert!(data_start(&labels[..1]).is_err());
    }
}
