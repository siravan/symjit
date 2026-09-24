// Encoding tests for the x86-64 assembler (`asm.rs`, `fused.rs`, `f64x8.rs`).
//
// Every test emits one instruction through `Amd` and compares the bytes with the
// encoding produced by GNU `as` (Intel syntax) for the equivalent instruction.
// The expected bytes were generated once from the assembler, never from the code
// under test.  The `// comment` in each test is the assembly line that was used.
//
// Notes on the oracle:
//  * scalar (LIG) VEX/FMA forms: `Amd` always sets VEX.L=1; the CPU ignores L for
//    scalar instructions, so the expected bytes have L set as well.
//  * label forms place the label 200 bytes after the instruction, so the rel32
//    displacement is 200 (0xc8).
//  * `#[ignore]`d tests document known defects; run them with
//    `cargo test amd::tests -- --ignored`.
//
// Not covered because no valid encoding exists:
//  * `vaddsubqd`: there is no EVEX form of VADDSUBPD (it is VEX-only).
//  * `vbroadcastdd`/`vbroadcastdd_label`: VBROADCASTSD has no 128-bit form (use
//    VMOVDDUP); the emitted bytes are #UD.

use super::asm::{Amd, RoundingMode};
use crate::utils::DataType;

#[track_caller]
fn check<F: FnOnce(&mut Amd)>(dtype: DataType, expected: &[u8], f: F) {
    let mut a = Amd::new(dtype);
    f(&mut a);
    let got = a.bytes();
    assert_eq!(
        got, expected,
        "\n got:      {:02x?}\n expected: {:02x?}",
        got, expected
    );
}

// The label is placed 200 bytes after the end of the instruction.
#[track_caller]
fn check_label<F: FnOnce(&mut Amd)>(dtype: DataType, expected: &[u8], f: F) {
    let mut a = Amd::new(dtype);
    f(&mut a);
    let n = a.bytes().len();
    a.append_bytes(&[0x90; 200]);
    a.a.set_label("L");
    a.a.apply_jumps();
    let got = a.bytes()[..n].to_vec();
    assert_eq!(
        got, expected,
        "\n got:      {:02x?}\n expected: {:02x?}",
        got, expected
    );
}

#[test]
fn vaddsd_0() {
    // vaddsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0x58, 0xcb], |a| { a.vaddsd(1, 2, 3); });
}

#[test]
fn vaddsd_1() {
    // vaddsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0x58, 0xcb], |a| { a.vaddsd(9, 10, 11); });
}

#[test]
fn vaddsd_2() {
    // vaddsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0x58, 0xd7], |a| { a.vaddsd(2, 8, 15); });
}

#[test]
fn vsubsd_0() {
    // vsubsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0x5c, 0xcb], |a| { a.vsubsd(1, 2, 3); });
}

#[test]
fn vsubsd_1() {
    // vsubsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0x5c, 0xcb], |a| { a.vsubsd(9, 10, 11); });
}

#[test]
fn vsubsd_2() {
    // vsubsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0x5c, 0xd7], |a| { a.vsubsd(2, 8, 15); });
}

#[test]
fn vmulsd_0() {
    // vmulsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0x59, 0xcb], |a| { a.vmulsd(1, 2, 3); });
}

#[test]
fn vmulsd_1() {
    // vmulsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0x59, 0xcb], |a| { a.vmulsd(9, 10, 11); });
}

#[test]
fn vmulsd_2() {
    // vmulsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0x59, 0xd7], |a| { a.vmulsd(2, 8, 15); });
}

#[test]
fn vdivsd_0() {
    // vdivsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0x5e, 0xcb], |a| { a.vdivsd(1, 2, 3); });
}

#[test]
fn vdivsd_1() {
    // vdivsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0x5e, 0xcb], |a| { a.vdivsd(9, 10, 11); });
}

#[test]
fn vdivsd_2() {
    // vdivsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0x5e, 0xd7], |a| { a.vdivsd(2, 8, 15); });
}

#[test]
fn vsqrtsd_0() {
    // vsqrtsd xmm1, xmm0, xmm2
    check(DataType::F64, &[0xc5, 0xfb, 0x51, 0xca], |a| { a.vsqrtsd(1, 2); });
}

#[test]
fn vsqrtsd_1() {
    // vsqrtsd xmm9, xmm0, xmm10
    check(DataType::F64, &[0xc4, 0x41, 0x7b, 0x51, 0xca], |a| { a.vsqrtsd(9, 10); });
}

#[test]
fn vsqrtsd_2() {
    // vsqrtsd xmm2, xmm0, xmm8
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x51, 0xd0], |a| { a.vsqrtsd(2, 8); });
}

#[test]
fn vroundsd_floor_0() {
    // vroundsd xmm1, xmm1, xmm2, 1
    check(DataType::F64, &[0xc4, 0xe3, 0x75, 0x0b, 0xca, 0x01], |a| { a.vroundsd(1, 2, RoundingMode::Floor); });
}

#[test]
fn vroundsd_floor_1() {
    // vroundsd xmm9, xmm9, xmm10, 1
    check(DataType::F64, &[0xc4, 0x43, 0x35, 0x0b, 0xca, 0x01], |a| { a.vroundsd(9, 10, RoundingMode::Floor); });
}

#[test]
fn vroundsd_floor_2() {
    // vroundsd xmm2, xmm2, xmm8, 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x0b, 0xd0, 0x01], |a| { a.vroundsd(2, 8, RoundingMode::Floor); });
}

#[test]
fn vroundsd_trunc_0() {
    // vroundsd xmm1, xmm1, xmm2, 3
    check(DataType::F64, &[0xc4, 0xe3, 0x75, 0x0b, 0xca, 0x03], |a| { a.vroundsd(1, 2, RoundingMode::Trunc); });
}

#[test]
fn vroundsd_trunc_1() {
    // vroundsd xmm9, xmm9, xmm10, 3
    check(DataType::F64, &[0xc4, 0x43, 0x35, 0x0b, 0xca, 0x03], |a| { a.vroundsd(9, 10, RoundingMode::Trunc); });
}

#[test]
fn vroundsd_trunc_2() {
    // vroundsd xmm2, xmm2, xmm8, 3
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x0b, 0xd0, 0x03], |a| { a.vroundsd(2, 8, RoundingMode::Trunc); });
}

#[test]
fn vcmpeqsd_0() {
    // vcmpeqsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqsd(1, 2, 3); });
}

#[test]
fn vcmpeqsd_1() {
    // vcmpeqsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqsd(9, 10, 11); });
}

#[test]
fn vcmpeqsd_2() {
    // vcmpeqsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x00], |a| { a.vcmpeqsd(2, 8, 15); });
}

#[test]
fn vcmpltsd_0() {
    // vcmpltsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x01], |a| { a.vcmpltsd(1, 2, 3); });
}

#[test]
fn vcmpltsd_1() {
    // vcmpltsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x01], |a| { a.vcmpltsd(9, 10, 11); });
}

#[test]
fn vcmpltsd_2() {
    // vcmpltsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x01], |a| { a.vcmpltsd(2, 8, 15); });
}

#[test]
fn vcmplesd_0() {
    // vcmplesd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x02], |a| { a.vcmplesd(1, 2, 3); });
}

#[test]
fn vcmplesd_1() {
    // vcmplesd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x02], |a| { a.vcmplesd(9, 10, 11); });
}

#[test]
fn vcmplesd_2() {
    // vcmplesd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x02], |a| { a.vcmplesd(2, 8, 15); });
}

#[test]
fn vcmpunordsd_0() {
    // vcmpunordsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x03], |a| { a.vcmpunordsd(1, 2, 3); });
}

#[test]
fn vcmpunordsd_1() {
    // vcmpunordsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x03], |a| { a.vcmpunordsd(9, 10, 11); });
}

#[test]
fn vcmpunordsd_2() {
    // vcmpunordsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x03], |a| { a.vcmpunordsd(2, 8, 15); });
}

#[test]
fn vcmpneqsd_0() {
    // vcmpneqsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqsd(1, 2, 3); });
}

#[test]
fn vcmpneqsd_1() {
    // vcmpneqsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqsd(9, 10, 11); });
}

#[test]
fn vcmpneqsd_2() {
    // vcmpneqsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x04], |a| { a.vcmpneqsd(2, 8, 15); });
}

#[test]
fn vcmpnltsd_0() {
    // vcmpnltsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltsd(1, 2, 3); });
}

#[test]
fn vcmpnltsd_1() {
    // vcmpnltsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltsd(9, 10, 11); });
}

#[test]
fn vcmpnltsd_2() {
    // vcmpnltsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x05], |a| { a.vcmpnltsd(2, 8, 15); });
}

#[test]
fn vcmpnlesd_0() {
    // vcmpnlesd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x06], |a| { a.vcmpnlesd(1, 2, 3); });
}

#[test]
fn vcmpnlesd_1() {
    // vcmpnlesd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x06], |a| { a.vcmpnlesd(9, 10, 11); });
}

#[test]
fn vcmpnlesd_2() {
    // vcmpnlesd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x06], |a| { a.vcmpnlesd(2, 8, 15); });
}

#[test]
fn vcmpordsd_0() {
    // vcmpordsd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xeb, 0xc2, 0xcb, 0x07], |a| { a.vcmpordsd(1, 2, 3); });
}

#[test]
fn vcmpordsd_1() {
    // vcmpordsd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x2b, 0xc2, 0xcb, 0x07], |a| { a.vcmpordsd(9, 10, 11); });
}

#[test]
fn vcmpordsd_2() {
    // vcmpordsd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3b, 0xc2, 0xd7, 0x07], |a| { a.vcmpordsd(2, 8, 15); });
}

#[test]
fn vucomisd_0() {
    // vucomisd xmm1, xmm2
    check(DataType::F64, &[0xc5, 0xfd, 0x2e, 0xca], |a| { a.vucomisd(1, 2); });
}

#[test]
fn vucomisd_1() {
    // vucomisd xmm9, xmm10
    check(DataType::F64, &[0xc4, 0x41, 0x7d, 0x2e, 0xca], |a| { a.vucomisd(9, 10); });
}

#[test]
fn vucomisd_2() {
    // vucomisd xmm2, xmm8
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x2e, 0xd0], |a| { a.vucomisd(2, 8); });
}

#[test]
fn vmovapd_0() {
    // {load} vmovapd ymm1, ymm2
    check(DataType::F64, &[0xc5, 0xfd, 0x28, 0xca], |a| { a.vmovapd(1, 2); });
}

#[test]
fn vmovapd_1() {
    // {load} vmovapd ymm9, ymm10
    check(DataType::F64, &[0xc4, 0x41, 0x7d, 0x28, 0xca], |a| { a.vmovapd(9, 10); });
}

#[test]
fn vmovapd_2() {
    // {load} vmovapd ymm2, ymm8
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x28, 0xd0], |a| { a.vmovapd(2, 8); });
}

#[test]
fn vaddpd_0() {
    // vaddpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x58, 0xcb], |a| { a.vaddpd(1, 2, 3); });
}

#[test]
fn vaddpd_1() {
    // vaddpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x58, 0xcb], |a| { a.vaddpd(9, 10, 11); });
}

#[test]
fn vaddpd_2() {
    // vaddpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x58, 0xd7], |a| { a.vaddpd(2, 8, 15); });
}

#[test]
fn vsubpd_0() {
    // vsubpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x5c, 0xcb], |a| { a.vsubpd(1, 2, 3); });
}

#[test]
fn vsubpd_1() {
    // vsubpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x5c, 0xcb], |a| { a.vsubpd(9, 10, 11); });
}

#[test]
fn vsubpd_2() {
    // vsubpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x5c, 0xd7], |a| { a.vsubpd(2, 8, 15); });
}

#[test]
fn vmulpd_0() {
    // vmulpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x59, 0xcb], |a| { a.vmulpd(1, 2, 3); });
}

#[test]
fn vmulpd_1() {
    // vmulpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x59, 0xcb], |a| { a.vmulpd(9, 10, 11); });
}

#[test]
fn vmulpd_2() {
    // vmulpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x59, 0xd7], |a| { a.vmulpd(2, 8, 15); });
}

#[test]
fn vdivpd_0() {
    // vdivpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x5e, 0xcb], |a| { a.vdivpd(1, 2, 3); });
}

#[test]
fn vdivpd_1() {
    // vdivpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x5e, 0xcb], |a| { a.vdivpd(9, 10, 11); });
}

#[test]
fn vdivpd_2() {
    // vdivpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x5e, 0xd7], |a| { a.vdivpd(2, 8, 15); });
}

#[test]
fn vsqrtpd_0() {
    // vsqrtpd ymm1, ymm2
    check(DataType::F64, &[0xc5, 0xfd, 0x51, 0xca], |a| { a.vsqrtpd(1, 2); });
}

#[test]
fn vsqrtpd_1() {
    // vsqrtpd ymm9, ymm10
    check(DataType::F64, &[0xc4, 0x41, 0x7d, 0x51, 0xca], |a| { a.vsqrtpd(9, 10); });
}

#[test]
fn vsqrtpd_2() {
    // vsqrtpd ymm2, ymm8
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x51, 0xd0], |a| { a.vsqrtpd(2, 8); });
}

#[test]
fn vroundpd_floor_0() {
    // vroundpd ymm1, ymm2, 1
    check(DataType::F64, &[0xc4, 0xe3, 0x7d, 0x09, 0xca, 0x01], |a| { a.vroundpd(1, 2, RoundingMode::Floor); });
}

#[test]
fn vroundpd_floor_1() {
    // vroundpd ymm9, ymm10, 1
    check(DataType::F64, &[0xc4, 0x43, 0x7d, 0x09, 0xca, 0x01], |a| { a.vroundpd(9, 10, RoundingMode::Floor); });
}

#[test]
fn vroundpd_floor_2() {
    // vroundpd ymm2, ymm8, 1
    check(DataType::F64, &[0xc4, 0xc3, 0x7d, 0x09, 0xd0, 0x01], |a| { a.vroundpd(2, 8, RoundingMode::Floor); });
}

#[test]
fn vroundpd_trunc_0() {
    // vroundpd ymm1, ymm2, 3
    check(DataType::F64, &[0xc4, 0xe3, 0x7d, 0x09, 0xca, 0x03], |a| { a.vroundpd(1, 2, RoundingMode::Trunc); });
}

#[test]
fn vroundpd_trunc_1() {
    // vroundpd ymm9, ymm10, 3
    check(DataType::F64, &[0xc4, 0x43, 0x7d, 0x09, 0xca, 0x03], |a| { a.vroundpd(9, 10, RoundingMode::Trunc); });
}

#[test]
fn vroundpd_trunc_2() {
    // vroundpd ymm2, ymm8, 3
    check(DataType::F64, &[0xc4, 0xc3, 0x7d, 0x09, 0xd0, 0x03], |a| { a.vroundpd(2, 8, RoundingMode::Trunc); });
}

#[test]
fn vandpd_0() {
    // vandpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x54, 0xcb], |a| { a.vandpd(1, 2, 3); });
}

#[test]
fn vandpd_1() {
    // vandpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x54, 0xcb], |a| { a.vandpd(9, 10, 11); });
}

#[test]
fn vandpd_2() {
    // vandpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x54, 0xd7], |a| { a.vandpd(2, 8, 15); });
}

#[test]
fn vandnpd_0() {
    // vandnpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x55, 0xcb], |a| { a.vandnpd(1, 2, 3); });
}

#[test]
fn vandnpd_1() {
    // vandnpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x55, 0xcb], |a| { a.vandnpd(9, 10, 11); });
}

#[test]
fn vandnpd_2() {
    // vandnpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x55, 0xd7], |a| { a.vandnpd(2, 8, 15); });
}

#[test]
fn vorpd_0() {
    // vorpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x56, 0xcb], |a| { a.vorpd(1, 2, 3); });
}

#[test]
fn vorpd_1() {
    // vorpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x56, 0xcb], |a| { a.vorpd(9, 10, 11); });
}

#[test]
fn vorpd_2() {
    // vorpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x56, 0xd7], |a| { a.vorpd(2, 8, 15); });
}

#[test]
fn vxorpd_0() {
    // vxorpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x57, 0xcb], |a| { a.vxorpd(1, 2, 3); });
}

#[test]
fn vxorpd_1() {
    // vxorpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x57, 0xcb], |a| { a.vxorpd(9, 10, 11); });
}

#[test]
fn vxorpd_2() {
    // vxorpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x57, 0xd7], |a| { a.vxorpd(2, 8, 15); });
}

#[test]
fn vcmpeqpd_0() {
    // vcmpeqpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqpd(1, 2, 3); });
}

#[test]
fn vcmpeqpd_1() {
    // vcmpeqpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqpd(9, 10, 11); });
}

#[test]
fn vcmpeqpd_2() {
    // vcmpeqpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x00], |a| { a.vcmpeqpd(2, 8, 15); });
}

#[test]
fn vcmpltpd_0() {
    // vcmpltpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x01], |a| { a.vcmpltpd(1, 2, 3); });
}

#[test]
fn vcmpltpd_1() {
    // vcmpltpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x01], |a| { a.vcmpltpd(9, 10, 11); });
}

#[test]
fn vcmpltpd_2() {
    // vcmpltpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x01], |a| { a.vcmpltpd(2, 8, 15); });
}

#[test]
fn vcmplepd_0() {
    // vcmplepd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x02], |a| { a.vcmplepd(1, 2, 3); });
}

#[test]
fn vcmplepd_1() {
    // vcmplepd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x02], |a| { a.vcmplepd(9, 10, 11); });
}

#[test]
fn vcmplepd_2() {
    // vcmplepd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x02], |a| { a.vcmplepd(2, 8, 15); });
}

#[test]
fn vcmpunordpd_0() {
    // vcmpunordpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x03], |a| { a.vcmpunordpd(1, 2, 3); });
}

#[test]
fn vcmpunordpd_1() {
    // vcmpunordpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x03], |a| { a.vcmpunordpd(9, 10, 11); });
}

#[test]
fn vcmpunordpd_2() {
    // vcmpunordpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x03], |a| { a.vcmpunordpd(2, 8, 15); });
}

#[test]
fn vcmpneqpd_0() {
    // vcmpneqpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqpd(1, 2, 3); });
}

#[test]
fn vcmpneqpd_1() {
    // vcmpneqpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqpd(9, 10, 11); });
}

#[test]
fn vcmpneqpd_2() {
    // vcmpneqpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x04], |a| { a.vcmpneqpd(2, 8, 15); });
}

#[test]
fn vcmpnltpd_0() {
    // vcmpnltpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltpd(1, 2, 3); });
}

#[test]
fn vcmpnltpd_1() {
    // vcmpnltpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltpd(9, 10, 11); });
}

#[test]
fn vcmpnltpd_2() {
    // vcmpnltpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x05], |a| { a.vcmpnltpd(2, 8, 15); });
}

#[test]
fn vcmpnlepd_0() {
    // vcmpnlepd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x06], |a| { a.vcmpnlepd(1, 2, 3); });
}

#[test]
fn vcmpnlepd_1() {
    // vcmpnlepd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x06], |a| { a.vcmpnlepd(9, 10, 11); });
}

#[test]
fn vcmpnlepd_2() {
    // vcmpnlepd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x06], |a| { a.vcmpnlepd(2, 8, 15); });
}

#[test]
fn vcmpordpd_0() {
    // vcmpordpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xc2, 0xcb, 0x07], |a| { a.vcmpordpd(1, 2, 3); });
}

#[test]
fn vcmpordpd_1() {
    // vcmpordpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc2, 0xcb, 0x07], |a| { a.vcmpordpd(9, 10, 11); });
}

#[test]
fn vcmpordpd_2() {
    // vcmpordpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc2, 0xd7, 0x07], |a| { a.vcmpordpd(2, 8, 15); });
}

#[test]
fn vmovmskpd_0() {
    // vmovmskpd ecx, ymm2
    check(DataType::F64, &[0xc5, 0xfd, 0x50, 0xca], |a| { a.vmovmskpd(1, 2); });
}

#[test]
fn vmovmskpd_1() {
    // vmovmskpd r9d, ymm10
    check(DataType::F64, &[0xc4, 0x41, 0x7d, 0x50, 0xca], |a| { a.vmovmskpd(9, 10); });
}

#[test]
fn vmovmskpd_2() {
    // vmovmskpd edx, ymm8
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x50, 0xd0], |a| { a.vmovmskpd(2, 8); });
}

#[test]
fn vadddd_0() {
    // vaddpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x58, 0xcb], |a| { a.vadddd(1, 2, 3); });
}

#[test]
fn vadddd_1() {
    // vaddpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x58, 0xcb], |a| { a.vadddd(9, 10, 11); });
}

#[test]
fn vadddd_2() {
    // vaddpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x58, 0xd7], |a| { a.vadddd(2, 8, 15); });
}

#[test]
fn vhadddd_0() {
    // vhaddpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x7c, 0xcb], |a| { a.vhadddd(1, 2, 3); });
}

#[test]
fn vhadddd_1() {
    // vhaddpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x7c, 0xcb], |a| { a.vhadddd(9, 10, 11); });
}

#[test]
fn vhadddd_2() {
    // vhaddpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x7c, 0xd7], |a| { a.vhadddd(2, 8, 15); });
}

#[test]
fn vsubdd_0() {
    // vsubpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x5c, 0xcb], |a| { a.vsubdd(1, 2, 3); });
}

#[test]
fn vsubdd_1() {
    // vsubpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x5c, 0xcb], |a| { a.vsubdd(9, 10, 11); });
}

#[test]
fn vsubdd_2() {
    // vsubpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x5c, 0xd7], |a| { a.vsubdd(2, 8, 15); });
}

#[test]
fn vmuldd_0() {
    // vmulpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x59, 0xcb], |a| { a.vmuldd(1, 2, 3); });
}

#[test]
fn vmuldd_1() {
    // vmulpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x59, 0xcb], |a| { a.vmuldd(9, 10, 11); });
}

#[test]
fn vmuldd_2() {
    // vmulpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x59, 0xd7], |a| { a.vmuldd(2, 8, 15); });
}

#[test]
fn vdivdd_0() {
    // vdivpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x5e, 0xcb], |a| { a.vdivdd(1, 2, 3); });
}

#[test]
fn vdivdd_1() {
    // vdivpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x5e, 0xcb], |a| { a.vdivdd(9, 10, 11); });
}

#[test]
fn vdivdd_2() {
    // vdivpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x5e, 0xd7], |a| { a.vdivdd(2, 8, 15); });
}

#[test]
fn vsqrtdd_0() {
    // vsqrtpd xmm1, xmm2
    check(DataType::F64, &[0xc5, 0xf9, 0x51, 0xca], |a| { a.vsqrtdd(1, 2); });
}

#[test]
fn vsqrtdd_1() {
    // vsqrtpd xmm9, xmm10
    check(DataType::F64, &[0xc4, 0x41, 0x79, 0x51, 0xca], |a| { a.vsqrtdd(9, 10); });
}

#[test]
fn vsqrtdd_2() {
    // vsqrtpd xmm2, xmm8
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x51, 0xd0], |a| { a.vsqrtdd(2, 8); });
}

#[test]
fn vrounddd_floor_0() {
    // vroundpd xmm1, xmm2, 1
    check(DataType::F64, &[0xc4, 0xe3, 0x79, 0x09, 0xca, 0x01], |a| { a.vrounddd(1, 2, RoundingMode::Floor); });
}

#[test]
fn vrounddd_floor_1() {
    // vroundpd xmm9, xmm10, 1
    check(DataType::F64, &[0xc4, 0x43, 0x79, 0x09, 0xca, 0x01], |a| { a.vrounddd(9, 10, RoundingMode::Floor); });
}

#[test]
fn vrounddd_floor_2() {
    // vroundpd xmm2, xmm8, 1
    check(DataType::F64, &[0xc4, 0xc3, 0x79, 0x09, 0xd0, 0x01], |a| { a.vrounddd(2, 8, RoundingMode::Floor); });
}

#[test]
fn vrounddd_trunc_0() {
    // vroundpd xmm1, xmm2, 3
    check(DataType::F64, &[0xc4, 0xe3, 0x79, 0x09, 0xca, 0x03], |a| { a.vrounddd(1, 2, RoundingMode::Trunc); });
}

#[test]
fn vrounddd_trunc_1() {
    // vroundpd xmm9, xmm10, 3
    check(DataType::F64, &[0xc4, 0x43, 0x79, 0x09, 0xca, 0x03], |a| { a.vrounddd(9, 10, RoundingMode::Trunc); });
}

#[test]
fn vrounddd_trunc_2() {
    // vroundpd xmm2, xmm8, 3
    check(DataType::F64, &[0xc4, 0xc3, 0x79, 0x09, 0xd0, 0x03], |a| { a.vrounddd(2, 8, RoundingMode::Trunc); });
}

#[test]
fn vanddd_0() {
    // vandpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x54, 0xcb], |a| { a.vanddd(1, 2, 3); });
}

#[test]
fn vanddd_1() {
    // vandpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x54, 0xcb], |a| { a.vanddd(9, 10, 11); });
}

#[test]
fn vanddd_2() {
    // vandpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x54, 0xd7], |a| { a.vanddd(2, 8, 15); });
}

#[test]
fn vandndd_0() {
    // vandnpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x55, 0xcb], |a| { a.vandndd(1, 2, 3); });
}

#[test]
fn vandndd_1() {
    // vandnpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x55, 0xcb], |a| { a.vandndd(9, 10, 11); });
}

#[test]
fn vandndd_2() {
    // vandnpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x55, 0xd7], |a| { a.vandndd(2, 8, 15); });
}

#[test]
fn vordd_0() {
    // vorpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x56, 0xcb], |a| { a.vordd(1, 2, 3); });
}

#[test]
fn vordd_1() {
    // vorpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x56, 0xcb], |a| { a.vordd(9, 10, 11); });
}

#[test]
fn vordd_2() {
    // vorpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x56, 0xd7], |a| { a.vordd(2, 8, 15); });
}

#[test]
fn vxordd_0() {
    // vxorpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x57, 0xcb], |a| { a.vxordd(1, 2, 3); });
}

#[test]
fn vxordd_1() {
    // vxorpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x57, 0xcb], |a| { a.vxordd(9, 10, 11); });
}

#[test]
fn vxordd_2() {
    // vxorpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x57, 0xd7], |a| { a.vxordd(2, 8, 15); });
}

#[test]
fn vcmpeqdd_0() {
    // vcmpeqpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqdd(1, 2, 3); });
}

#[test]
fn vcmpeqdd_1() {
    // vcmpeqpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqdd(9, 10, 11); });
}

#[test]
fn vcmpeqdd_2() {
    // vcmpeqpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x00], |a| { a.vcmpeqdd(2, 8, 15); });
}

#[test]
fn vcmpltdd_0() {
    // vcmpltpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x01], |a| { a.vcmpltdd(1, 2, 3); });
}

#[test]
fn vcmpltdd_1() {
    // vcmpltpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x01], |a| { a.vcmpltdd(9, 10, 11); });
}

#[test]
fn vcmpltdd_2() {
    // vcmpltpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x01], |a| { a.vcmpltdd(2, 8, 15); });
}

#[test]
fn vcmpledd_0() {
    // vcmplepd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x02], |a| { a.vcmpledd(1, 2, 3); });
}

#[test]
fn vcmpledd_1() {
    // vcmplepd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x02], |a| { a.vcmpledd(9, 10, 11); });
}

#[test]
fn vcmpledd_2() {
    // vcmplepd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x02], |a| { a.vcmpledd(2, 8, 15); });
}

#[test]
fn vcmpunorddd_0() {
    // vcmpunordpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x03], |a| { a.vcmpunorddd(1, 2, 3); });
}

#[test]
fn vcmpunorddd_1() {
    // vcmpunordpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x03], |a| { a.vcmpunorddd(9, 10, 11); });
}

#[test]
fn vcmpunorddd_2() {
    // vcmpunordpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x03], |a| { a.vcmpunorddd(2, 8, 15); });
}

#[test]
fn vcmpneqdd_0() {
    // vcmpneqpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqdd(1, 2, 3); });
}

#[test]
fn vcmpneqdd_1() {
    // vcmpneqpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqdd(9, 10, 11); });
}

#[test]
fn vcmpneqdd_2() {
    // vcmpneqpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x04], |a| { a.vcmpneqdd(2, 8, 15); });
}

#[test]
fn vcmpnltdd_0() {
    // vcmpnltpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltdd(1, 2, 3); });
}

#[test]
fn vcmpnltdd_1() {
    // vcmpnltpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltdd(9, 10, 11); });
}

#[test]
fn vcmpnltdd_2() {
    // vcmpnltpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x05], |a| { a.vcmpnltdd(2, 8, 15); });
}

#[test]
fn vcmpnledd_0() {
    // vcmpnlepd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x06], |a| { a.vcmpnledd(1, 2, 3); });
}

#[test]
fn vcmpnledd_1() {
    // vcmpnlepd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x06], |a| { a.vcmpnledd(9, 10, 11); });
}

#[test]
fn vcmpnledd_2() {
    // vcmpnlepd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x06], |a| { a.vcmpnledd(2, 8, 15); });
}

#[test]
fn vcmporddd_0() {
    // vcmpordpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xc2, 0xcb, 0x07], |a| { a.vcmporddd(1, 2, 3); });
}

#[test]
fn vcmporddd_1() {
    // vcmpordpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc2, 0xcb, 0x07], |a| { a.vcmporddd(9, 10, 11); });
}

#[test]
fn vcmporddd_2() {
    // vcmpordpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xc2, 0xd7, 0x07], |a| { a.vcmporddd(2, 8, 15); });
}

#[test]
fn vshufdd_0_0() {
    // vshufpd xmm1, xmm2, xmm3, 0
    check(DataType::F64, &[0xc5, 0xe9, 0xc6, 0xcb, 0x00], |a| { a.vshufdd(1, 2, 3, 0); });
}

#[test]
fn vshufdd_0_1() {
    // vshufpd xmm9, xmm10, xmm11, 0
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc6, 0xcb, 0x00], |a| { a.vshufdd(9, 10, 11, 0); });
}

#[test]
fn vshufdd_1_0() {
    // vshufpd xmm1, xmm2, xmm3, 1
    check(DataType::F64, &[0xc5, 0xe9, 0xc6, 0xcb, 0x01], |a| { a.vshufdd(1, 2, 3, 1); });
}

#[test]
fn vshufdd_1_1() {
    // vshufpd xmm9, xmm10, xmm11, 1
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc6, 0xcb, 0x01], |a| { a.vshufdd(9, 10, 11, 1); });
}

#[test]
fn vshufdd_2_0() {
    // vshufpd xmm1, xmm2, xmm3, 2
    check(DataType::F64, &[0xc5, 0xe9, 0xc6, 0xcb, 0x02], |a| { a.vshufdd(1, 2, 3, 2); });
}

#[test]
fn vshufdd_2_1() {
    // vshufpd xmm9, xmm10, xmm11, 2
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc6, 0xcb, 0x02], |a| { a.vshufdd(9, 10, 11, 2); });
}

#[test]
fn vshufdd_3_0() {
    // vshufpd xmm1, xmm2, xmm3, 3
    check(DataType::F64, &[0xc5, 0xe9, 0xc6, 0xcb, 0x03], |a| { a.vshufdd(1, 2, 3, 3); });
}

#[test]
fn vshufdd_3_1() {
    // vshufpd xmm9, xmm10, xmm11, 3
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xc6, 0xcb, 0x03], |a| { a.vshufdd(9, 10, 11, 3); });
}

#[test]
fn vunpckhdd_0() {
    // vunpckhpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x15, 0xcb], |a| { a.vunpckhdd(1, 2, 3); });
}

#[test]
fn vunpckhdd_1() {
    // vunpckhpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x15, 0xcb], |a| { a.vunpckhdd(9, 10, 11); });
}

#[test]
fn vunpckhdd_2() {
    // vunpckhpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x15, 0xd7], |a| { a.vunpckhdd(2, 8, 15); });
}

#[test]
fn vunpckldd_0() {
    // vunpcklpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0x14, 0xcb], |a| { a.vunpckldd(1, 2, 3); });
}

#[test]
fn vunpckldd_1() {
    // vunpcklpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0x14, 0xcb], |a| { a.vunpckldd(9, 10, 11); });
}

#[test]
fn vunpckldd_2() {
    // vunpcklpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0x14, 0xd7], |a| { a.vunpckldd(2, 8, 15); });
}

#[test]
fn vaddsubdd_0() {
    // vaddsubpd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc5, 0xe9, 0xd0, 0xcb], |a| { a.vaddsubdd(1, 2, 3); });
}

#[test]
fn vaddsubdd_1() {
    // vaddsubpd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x41, 0x29, 0xd0, 0xcb], |a| { a.vaddsubdd(9, 10, 11); });
}

#[test]
fn vaddsubdd_2() {
    // vaddsubpd xmm2, xmm8, xmm15
    check(DataType::F64, &[0xc4, 0xc1, 0x39, 0xd0, 0xd7], |a| { a.vaddsubdd(2, 8, 15); });
}

#[test]
fn vshufpd_0() {
    // vshufpd ymm1, ymm2, ymm3, 5
    check(DataType::F64, &[0xc5, 0xed, 0xc6, 0xcb, 0x05], |a| { a.vshufpd(1, 2, 3, 5); });
}

#[test]
fn vshufpd_1() {
    // vshufpd ymm9, ymm10, ymm11, 5
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xc6, 0xcb, 0x05], |a| { a.vshufpd(9, 10, 11, 5); });
}

#[test]
fn vshufpd_2() {
    // vshufpd ymm2, ymm8, ymm15, 5
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xc6, 0xd7, 0x05], |a| { a.vshufpd(2, 8, 15, 5); });
}

#[test]
fn vunpckhpd_0() {
    // vunpckhpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x15, 0xcb], |a| { a.vunpckhpd(1, 2, 3); });
}

#[test]
fn vunpckhpd_1() {
    // vunpckhpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x15, 0xcb], |a| { a.vunpckhpd(9, 10, 11); });
}

#[test]
fn vunpckhpd_2() {
    // vunpckhpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x15, 0xd7], |a| { a.vunpckhpd(2, 8, 15); });
}

#[test]
fn vunpcklpd_0() {
    // vunpcklpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0x14, 0xcb], |a| { a.vunpcklpd(1, 2, 3); });
}

#[test]
fn vunpcklpd_1() {
    // vunpcklpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0x14, 0xcb], |a| { a.vunpcklpd(9, 10, 11); });
}

#[test]
fn vunpcklpd_2() {
    // vunpcklpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0x14, 0xd7], |a| { a.vunpcklpd(2, 8, 15); });
}

#[test]
fn vaddsubpd_0() {
    // vaddsubpd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc5, 0xed, 0xd0, 0xcb], |a| { a.vaddsubpd(1, 2, 3); });
}

#[test]
fn vaddsubpd_1() {
    // vaddsubpd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x41, 0x2d, 0xd0, 0xcb], |a| { a.vaddsubpd(9, 10, 11); });
}

#[test]
fn vaddsubpd_2() {
    // vaddsubpd ymm2, ymm8, ymm15
    check(DataType::F64, &[0xc4, 0xc1, 0x3d, 0xd0, 0xd7], |a| { a.vaddsubpd(2, 8, 15); });
}

#[test]
fn vinsertf128_0() {
    // vinsertf128 ymm1, ymm2, xmm3, 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0xcb, 0x01], |a| { a.vinsertf128(1, 2, 3, 1); });
}

#[test]
fn vinsertf128_1() {
    // vinsertf128 ymm9, ymm10, xmm11, 1
    check(DataType::F64, &[0xc4, 0x43, 0x2d, 0x18, 0xcb, 0x01], |a| { a.vinsertf128(9, 10, 11, 1); });
}

#[test]
fn vinsertf128_2() {
    // vinsertf128 ymm2, ymm8, xmm15, 1
    check(DataType::F64, &[0xc4, 0xc3, 0x3d, 0x18, 0xd7, 0x01], |a| { a.vinsertf128(2, 8, 15, 1); });
}

#[test]
fn vextractf128_0() {
    // vextractf128 xmm1, ymm2, 1
    check(DataType::F64, &[0xc4, 0xe3, 0x7d, 0x19, 0xd1, 0x01], |a| { a.vextractf128(1, 2, 1); });
}

#[test]
fn vextractf128_1() {
    // vextractf128 xmm9, ymm10, 1
    check(DataType::F64, &[0xc4, 0x43, 0x7d, 0x19, 0xd1, 0x01], |a| { a.vextractf128(9, 10, 1); });
}

#[test]
fn vextractf128_2() {
    // vextractf128 xmm2, ymm8, 1
    check(DataType::F64, &[0xc4, 0x63, 0x7d, 0x19, 0xc2, 0x01], |a| { a.vextractf128(2, 8, 1); });
}

#[test]
fn vzeroupper_0() {
    // vzeroupper
    check(DataType::F64, &[0xc5, 0xf8, 0x77], |a| { a.vzeroupper(); });
}

#[test]
fn movapd_0() {
    // movapd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x28, 0xca], |a| { a.movapd(1, 2); });
}

#[test]
fn movapd_1() {
    // movapd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x28, 0xca], |a| { a.movapd(9, 10); });
}

#[test]
fn movapd_2() {
    // movapd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x28, 0xd0], |a| { a.movapd(2, 8); });
}

#[test]
fn addsd_0() {
    // addsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x58, 0xca], |a| { a.addsd(1, 2); });
}

#[test]
fn addsd_1() {
    // addsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0x58, 0xca], |a| { a.addsd(9, 10); });
}

#[test]
fn addsd_2() {
    // addsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x58, 0xd0], |a| { a.addsd(2, 8); });
}

#[test]
fn subsd_0() {
    // subsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x5c, 0xca], |a| { a.subsd(1, 2); });
}

#[test]
fn subsd_1() {
    // subsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0x5c, 0xca], |a| { a.subsd(9, 10); });
}

#[test]
fn subsd_2() {
    // subsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x5c, 0xd0], |a| { a.subsd(2, 8); });
}

#[test]
fn mulsd_0() {
    // mulsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x59, 0xca], |a| { a.mulsd(1, 2); });
}

#[test]
fn mulsd_1() {
    // mulsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0x59, 0xca], |a| { a.mulsd(9, 10); });
}

#[test]
fn mulsd_2() {
    // mulsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x59, 0xd0], |a| { a.mulsd(2, 8); });
}

#[test]
fn divsd_0() {
    // divsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x5e, 0xca], |a| { a.divsd(1, 2); });
}

#[test]
fn divsd_1() {
    // divsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0x5e, 0xca], |a| { a.divsd(9, 10); });
}

#[test]
fn divsd_2() {
    // divsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x5e, 0xd0], |a| { a.divsd(2, 8); });
}

#[test]
fn sqrtsd_0() {
    // sqrtsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x51, 0xca], |a| { a.sqrtsd(1, 2); });
}

#[test]
fn sqrtsd_1() {
    // sqrtsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0x51, 0xca], |a| { a.sqrtsd(9, 10); });
}

#[test]
fn sqrtsd_2() {
    // sqrtsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x51, 0xd0], |a| { a.sqrtsd(2, 8); });
}

#[test]
fn roundsd_floor_0() {
    // roundsd xmm1, xmm2, 1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x3a, 0x0b, 0xca, 0x01], |a| { a.roundsd(1, 2, RoundingMode::Floor); });
}

#[test]
fn roundsd_floor_1() {
    // roundsd xmm9, xmm10, 1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x3a, 0x0b, 0xca, 0x01], |a| { a.roundsd(9, 10, RoundingMode::Floor); });
}

#[test]
fn roundsd_floor_2() {
    // roundsd xmm2, xmm8, 1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x3a, 0x0b, 0xd0, 0x01], |a| { a.roundsd(2, 8, RoundingMode::Floor); });
}

#[test]
fn roundsd_trunc_0() {
    // roundsd xmm1, xmm2, 3
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x3a, 0x0b, 0xca, 0x03], |a| { a.roundsd(1, 2, RoundingMode::Trunc); });
}

#[test]
fn roundsd_trunc_1() {
    // roundsd xmm9, xmm10, 3
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x3a, 0x0b, 0xca, 0x03], |a| { a.roundsd(9, 10, RoundingMode::Trunc); });
}

#[test]
fn roundsd_trunc_2() {
    // roundsd xmm2, xmm8, 3
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x3a, 0x0b, 0xd0, 0x03], |a| { a.roundsd(2, 8, RoundingMode::Trunc); });
}

#[test]
fn cmpeqsd_0() {
    // cmpeqsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x00], |a| { a.cmpeqsd(1, 2); });
}

#[test]
fn cmpeqsd_1() {
    // cmpeqsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x00], |a| { a.cmpeqsd(9, 10); });
}

#[test]
fn cmpeqsd_2() {
    // cmpeqsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x00], |a| { a.cmpeqsd(2, 8); });
}

#[test]
fn cmpltsd_0() {
    // cmpltsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x01], |a| { a.cmpltsd(1, 2); });
}

#[test]
fn cmpltsd_1() {
    // cmpltsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x01], |a| { a.cmpltsd(9, 10); });
}

#[test]
fn cmpltsd_2() {
    // cmpltsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x01], |a| { a.cmpltsd(2, 8); });
}

#[test]
fn cmplesd_0() {
    // cmplesd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x02], |a| { a.cmplesd(1, 2); });
}

#[test]
fn cmplesd_1() {
    // cmplesd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x02], |a| { a.cmplesd(9, 10); });
}

#[test]
fn cmplesd_2() {
    // cmplesd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x02], |a| { a.cmplesd(2, 8); });
}

#[test]
fn cmpunordsd_0() {
    // cmpunordsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x03], |a| { a.cmpunordsd(1, 2); });
}

#[test]
fn cmpunordsd_1() {
    // cmpunordsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x03], |a| { a.cmpunordsd(9, 10); });
}

#[test]
fn cmpunordsd_2() {
    // cmpunordsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x03], |a| { a.cmpunordsd(2, 8); });
}

#[test]
fn cmpneqsd_0() {
    // cmpneqsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x04], |a| { a.cmpneqsd(1, 2); });
}

#[test]
fn cmpneqsd_1() {
    // cmpneqsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x04], |a| { a.cmpneqsd(9, 10); });
}

#[test]
fn cmpneqsd_2() {
    // cmpneqsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x04], |a| { a.cmpneqsd(2, 8); });
}

#[test]
fn cmpnltsd_0() {
    // cmpnltsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x05], |a| { a.cmpnltsd(1, 2); });
}

#[test]
fn cmpnltsd_1() {
    // cmpnltsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x05], |a| { a.cmpnltsd(9, 10); });
}

#[test]
fn cmpnltsd_2() {
    // cmpnltsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x05], |a| { a.cmpnltsd(2, 8); });
}

#[test]
fn cmpnlesd_0() {
    // cmpnlesd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x06], |a| { a.cmpnlesd(1, 2); });
}

#[test]
fn cmpnlesd_1() {
    // cmpnlesd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x06], |a| { a.cmpnlesd(9, 10); });
}

#[test]
fn cmpnlesd_2() {
    // cmpnlesd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x06], |a| { a.cmpnlesd(2, 8); });
}

#[test]
fn cmpordsd_0() {
    // cmpordsd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0xc2, 0xca, 0x07], |a| { a.cmpordsd(1, 2); });
}

#[test]
fn cmpordsd_1() {
    // cmpordsd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4d, 0x0f, 0xc2, 0xca, 0x07], |a| { a.cmpordsd(9, 10); });
}

#[test]
fn cmpordsd_2() {
    // cmpordsd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0xc2, 0xd0, 0x07], |a| { a.cmpordsd(2, 8); });
}

#[test]
fn ucomisd_0() {
    // ucomisd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x2e, 0xca], |a| { a.ucomisd(1, 2); });
}

#[test]
fn ucomisd_1() {
    // ucomisd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x2e, 0xca], |a| { a.ucomisd(9, 10); });
}

#[test]
fn ucomisd_2() {
    // ucomisd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x2e, 0xd0], |a| { a.ucomisd(2, 8); });
}

#[test]
fn andpd_0() {
    // andpd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x54, 0xca], |a| { a.andpd(1, 2); });
}

#[test]
fn andpd_1() {
    // andpd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x54, 0xca], |a| { a.andpd(9, 10); });
}

#[test]
fn andpd_2() {
    // andpd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x54, 0xd0], |a| { a.andpd(2, 8); });
}

#[test]
fn andnpd_0() {
    // andnpd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x55, 0xca], |a| { a.andnpd(1, 2); });
}

#[test]
fn andnpd_1() {
    // andnpd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x55, 0xca], |a| { a.andnpd(9, 10); });
}

#[test]
fn andnpd_2() {
    // andnpd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x55, 0xd0], |a| { a.andnpd(2, 8); });
}

#[test]
fn orpd_0() {
    // orpd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x56, 0xca], |a| { a.orpd(1, 2); });
}

#[test]
fn orpd_1() {
    // orpd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x56, 0xca], |a| { a.orpd(9, 10); });
}

#[test]
fn orpd_2() {
    // orpd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x56, 0xd0], |a| { a.orpd(2, 8); });
}

#[test]
fn xorpd_0() {
    // xorpd xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x57, 0xca], |a| { a.xorpd(1, 2); });
}

#[test]
fn xorpd_1() {
    // xorpd xmm9, xmm10
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x57, 0xca], |a| { a.xorpd(9, 10); });
}

#[test]
fn xorpd_2() {
    // xorpd xmm2, xmm8
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x57, 0xd0], |a| { a.xorpd(2, 8); });
}

#[test]
fn movq_reg_xmm_0() {
    // movq rcx, xmm2
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x7e, 0xd1], |a| { a.movq_reg_xmm(1, 2); });
}

#[test]
fn movq_reg_xmm_1() {
    // movq r9, xmm10
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x7e, 0xd1], |a| { a.movq_reg_xmm(9, 10); });
}

#[test]
fn movq_reg_xmm_2() {
    // movq rdx, xmm8
    check(DataType::F64, &[0x66, 0x4c, 0x0f, 0x7e, 0xc2], |a| { a.movq_reg_xmm(2, 8); });
}

#[test]
fn movq_xmm_reg_0() {
    // movq xmm1, rdx
    check(DataType::F64, &[0x66, 0x48, 0x0f, 0x6e, 0xca], |a| { a.movq_xmm_reg(1, 2); });
}

#[test]
fn movq_xmm_reg_1() {
    // movq xmm9, r10
    check(DataType::F64, &[0x66, 0x4d, 0x0f, 0x6e, 0xca], |a| { a.movq_xmm_reg(9, 10); });
}

#[test]
fn movq_xmm_reg_2() {
    // movq xmm2, r8
    check(DataType::F64, &[0x66, 0x49, 0x0f, 0x6e, 0xd0], |a| { a.movq_xmm_reg(2, 8); });
}

#[test]
fn vmovq_reg_xmm_0() {
    // vmovq rcx, xmm2
    check(DataType::F64, &[0xc4, 0xe1, 0xf9, 0x7e, 0xd1], |a| { a.vmovq_reg_xmm(1, 2); });
}

#[test]
fn vmovq_reg_xmm_1() {
    // vmovq r9, xmm10
    check(DataType::F64, &[0xc4, 0x41, 0xf9, 0x7e, 0xd1], |a| { a.vmovq_reg_xmm(9, 10); });
}

#[test]
fn vmovq_reg_xmm_2() {
    // vmovq rdx, xmm8
    check(DataType::F64, &[0xc4, 0x61, 0xf9, 0x7e, 0xc2], |a| { a.vmovq_reg_xmm(2, 8); });
}

#[test]
fn vmovq_xmm_reg_0() {
    // vmovq xmm1, rdx
    check(DataType::F64, &[0xc4, 0xe1, 0xf9, 0x6e, 0xca], |a| { a.vmovq_xmm_reg(1, 2); });
}

#[test]
fn vmovq_xmm_reg_1() {
    // vmovq xmm9, r10
    check(DataType::F64, &[0xc4, 0x41, 0xf9, 0x6e, 0xca], |a| { a.vmovq_xmm_reg(9, 10); });
}

#[test]
fn vmovq_xmm_reg_2() {
    // vmovq xmm2, r8
    check(DataType::F64, &[0xc4, 0xc1, 0xf9, 0x6e, 0xd0], |a| { a.vmovq_xmm_reg(2, 8); });
}

#[test]
fn vmovaqd_0() {
    // vmovapd zmm1, zmm2
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x28, 0xca], |a| { a.vmovaqd(1, 2); });
}

#[test]
fn vmovaqd_1() {
    // vmovapd zmm9, zmm10
    check(DataType::F64, &[0x62, 0x51, 0xfd, 0x48, 0x28, 0xca], |a| { a.vmovaqd(9, 10); });
}

#[test]
fn vmovaqd_2() {
    // vmovapd zmm17, zmm18
    check(DataType::F64, &[0x62, 0xa1, 0xfd, 0x48, 0x28, 0xca], |a| { a.vmovaqd(17, 18); });
}

#[test]
fn vmovaqd_3() {
    // vmovapd zmm30, zmm25
    check(DataType::F64, &[0x62, 0x01, 0xfd, 0x48, 0x28, 0xf1], |a| { a.vmovaqd(30, 25); });
}

#[test]
fn vmovaqd_masked_copy() {
    // vmovapd zmm1{k3}, zmm2
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x4b, 0x28, 0xca], |a| { a.vmovaqd_masked(1, 2, 3, false); });
}

#[test]
fn vmovaqd_masked_zero() {
    // vmovapd zmm1{k3}{z}, zmm2
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0xcb, 0x28, 0xca], |a| { a.vmovaqd_masked(1, 2, 3, true); });
}

#[test]
fn vaddqd_0() {
    // vaddpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x58, 0xcb], |a| { a.vaddqd(1, 2, 3); });
}

#[test]
fn vaddqd_1() {
    // vaddpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x58, 0xcb], |a| { a.vaddqd(9, 10, 11); });
}

#[test]
fn vaddqd_2() {
    // vaddpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x58, 0xcb], |a| { a.vaddqd(17, 18, 19); });
}

#[test]
fn vaddqd_3() {
    // vaddpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x58, 0xf5], |a| { a.vaddqd(30, 25, 21); });
}

#[test]
fn vsubqd_0() {
    // vsubpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x5c, 0xcb], |a| { a.vsubqd(1, 2, 3); });
}

#[test]
fn vsubqd_1() {
    // vsubpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x5c, 0xcb], |a| { a.vsubqd(9, 10, 11); });
}

#[test]
fn vsubqd_2() {
    // vsubpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x5c, 0xcb], |a| { a.vsubqd(17, 18, 19); });
}

#[test]
fn vsubqd_3() {
    // vsubpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x5c, 0xf5], |a| { a.vsubqd(30, 25, 21); });
}

#[test]
fn vmulqd_0() {
    // vmulpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x59, 0xcb], |a| { a.vmulqd(1, 2, 3); });
}

#[test]
fn vmulqd_1() {
    // vmulpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x59, 0xcb], |a| { a.vmulqd(9, 10, 11); });
}

#[test]
fn vmulqd_2() {
    // vmulpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x59, 0xcb], |a| { a.vmulqd(17, 18, 19); });
}

#[test]
fn vmulqd_3() {
    // vmulpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x59, 0xf5], |a| { a.vmulqd(30, 25, 21); });
}

#[test]
fn vdivqd_0() {
    // vdivpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x5e, 0xcb], |a| { a.vdivqd(1, 2, 3); });
}

#[test]
fn vdivqd_1() {
    // vdivpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x5e, 0xcb], |a| { a.vdivqd(9, 10, 11); });
}

#[test]
fn vdivqd_2() {
    // vdivpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x5e, 0xcb], |a| { a.vdivqd(17, 18, 19); });
}

#[test]
fn vdivqd_3() {
    // vdivpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x5e, 0xf5], |a| { a.vdivqd(30, 25, 21); });
}

#[test]
fn vsqrtqd_0() {
    // vsqrtpd zmm1, zmm2
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x51, 0xca], |a| { a.vsqrtqd(1, 2); });
}

#[test]
fn vsqrtqd_1() {
    // vsqrtpd zmm9, zmm10
    check(DataType::F64, &[0x62, 0x51, 0xfd, 0x48, 0x51, 0xca], |a| { a.vsqrtqd(9, 10); });
}

#[test]
fn vsqrtqd_2() {
    // vsqrtpd zmm17, zmm18
    check(DataType::F64, &[0x62, 0xa1, 0xfd, 0x48, 0x51, 0xca], |a| { a.vsqrtqd(17, 18); });
}

#[test]
fn vsqrtqd_3() {
    // vsqrtpd zmm30, zmm25
    check(DataType::F64, &[0x62, 0x01, 0xfd, 0x48, 0x51, 0xf1], |a| { a.vsqrtqd(30, 25); });
}

#[test]
fn vroundqd_floor_0() {
    // vrndscalepd zmm1, zmm2, 1
    check(DataType::F64, &[0x62, 0xf3, 0xfd, 0x48, 0x09, 0xca, 0x01], |a| { a.vroundqd(1, 2, RoundingMode::Floor); });
}

#[test]
fn vroundqd_floor_1() {
    // vrndscalepd zmm9, zmm10, 1
    check(DataType::F64, &[0x62, 0x53, 0xfd, 0x48, 0x09, 0xca, 0x01], |a| { a.vroundqd(9, 10, RoundingMode::Floor); });
}

#[test]
fn vroundqd_floor_2() {
    // vrndscalepd zmm17, zmm18, 1
    check(DataType::F64, &[0x62, 0xa3, 0xfd, 0x48, 0x09, 0xca, 0x01], |a| { a.vroundqd(17, 18, RoundingMode::Floor); });
}

#[test]
fn vroundqd_floor_3() {
    // vrndscalepd zmm30, zmm25, 1
    check(DataType::F64, &[0x62, 0x03, 0xfd, 0x48, 0x09, 0xf1, 0x01], |a| { a.vroundqd(30, 25, RoundingMode::Floor); });
}

#[test]
fn vroundqd_trunc_0() {
    // vrndscalepd zmm1, zmm2, 3
    check(DataType::F64, &[0x62, 0xf3, 0xfd, 0x48, 0x09, 0xca, 0x03], |a| { a.vroundqd(1, 2, RoundingMode::Trunc); });
}

#[test]
fn vroundqd_trunc_1() {
    // vrndscalepd zmm9, zmm10, 3
    check(DataType::F64, &[0x62, 0x53, 0xfd, 0x48, 0x09, 0xca, 0x03], |a| { a.vroundqd(9, 10, RoundingMode::Trunc); });
}

#[test]
fn vroundqd_trunc_2() {
    // vrndscalepd zmm17, zmm18, 3
    check(DataType::F64, &[0x62, 0xa3, 0xfd, 0x48, 0x09, 0xca, 0x03], |a| { a.vroundqd(17, 18, RoundingMode::Trunc); });
}

#[test]
fn vroundqd_trunc_3() {
    // vrndscalepd zmm30, zmm25, 3
    check(DataType::F64, &[0x62, 0x03, 0xfd, 0x48, 0x09, 0xf1, 0x03], |a| { a.vroundqd(30, 25, RoundingMode::Trunc); });
}

#[test]
fn vandqd_0() {
    // vandpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x54, 0xcb], |a| { a.vandqd(1, 2, 3); });
}

#[test]
fn vandqd_1() {
    // vandpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x54, 0xcb], |a| { a.vandqd(9, 10, 11); });
}

#[test]
fn vandqd_2() {
    // vandpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x54, 0xcb], |a| { a.vandqd(17, 18, 19); });
}

#[test]
fn vandqd_3() {
    // vandpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x54, 0xf5], |a| { a.vandqd(30, 25, 21); });
}

#[test]
fn vandnqd_0() {
    // vandnpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x55, 0xcb], |a| { a.vandnqd(1, 2, 3); });
}

#[test]
fn vandnqd_1() {
    // vandnpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x55, 0xcb], |a| { a.vandnqd(9, 10, 11); });
}

#[test]
fn vandnqd_2() {
    // vandnpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x55, 0xcb], |a| { a.vandnqd(17, 18, 19); });
}

#[test]
fn vandnqd_3() {
    // vandnpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x55, 0xf5], |a| { a.vandnqd(30, 25, 21); });
}

#[test]
fn vorqd_0() {
    // vorpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x56, 0xcb], |a| { a.vorqd(1, 2, 3); });
}

#[test]
fn vorqd_1() {
    // vorpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x56, 0xcb], |a| { a.vorqd(9, 10, 11); });
}

#[test]
fn vorqd_2() {
    // vorpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x56, 0xcb], |a| { a.vorqd(17, 18, 19); });
}

#[test]
fn vorqd_3() {
    // vorpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x56, 0xf5], |a| { a.vorqd(30, 25, 21); });
}

#[test]
fn vxorqd_0() {
    // vxorpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x57, 0xcb], |a| { a.vxorqd(1, 2, 3); });
}

#[test]
fn vxorqd_1() {
    // vxorpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x57, 0xcb], |a| { a.vxorqd(9, 10, 11); });
}

#[test]
fn vxorqd_2() {
    // vxorpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x57, 0xcb], |a| { a.vxorqd(17, 18, 19); });
}

#[test]
fn vxorqd_3() {
    // vxorpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x57, 0xf5], |a| { a.vxorqd(30, 25, 21); });
}

#[test]
fn vcmpeqqd_0() {
    // vcmpeqpd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqqd(1, 2, 3); });
}

#[test]
fn vcmpeqqd_1() {
    // vcmpeqpd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqqd(1, 10, 11); });
}

#[test]
fn vcmpeqqd_2() {
    // vcmpeqpd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x00], |a| { a.vcmpeqqd(1, 18, 19); });
}

#[test]
fn vcmpeqqd_3() {
    // vcmpeqpd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x00], |a| { a.vcmpeqqd(6, 25, 21); });
}

#[test]
fn vcmpltqd_0() {
    // vcmpltpd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x01], |a| { a.vcmpltqd(1, 2, 3); });
}

#[test]
fn vcmpltqd_1() {
    // vcmpltpd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x01], |a| { a.vcmpltqd(1, 10, 11); });
}

#[test]
fn vcmpltqd_2() {
    // vcmpltpd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x01], |a| { a.vcmpltqd(1, 18, 19); });
}

#[test]
fn vcmpltqd_3() {
    // vcmpltpd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x01], |a| { a.vcmpltqd(6, 25, 21); });
}

#[test]
fn vcmpleqd_0() {
    // vcmplepd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x02], |a| { a.vcmpleqd(1, 2, 3); });
}

#[test]
fn vcmpleqd_1() {
    // vcmplepd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x02], |a| { a.vcmpleqd(1, 10, 11); });
}

#[test]
fn vcmpleqd_2() {
    // vcmplepd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x02], |a| { a.vcmpleqd(1, 18, 19); });
}

#[test]
fn vcmpleqd_3() {
    // vcmplepd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x02], |a| { a.vcmpleqd(6, 25, 21); });
}

#[test]
fn vcmpunordqd_0() {
    // vcmpunordpd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x03], |a| { a.vcmpunordqd(1, 2, 3); });
}

#[test]
fn vcmpunordqd_1() {
    // vcmpunordpd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x03], |a| { a.vcmpunordqd(1, 10, 11); });
}

#[test]
fn vcmpunordqd_2() {
    // vcmpunordpd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x03], |a| { a.vcmpunordqd(1, 18, 19); });
}

#[test]
fn vcmpunordqd_3() {
    // vcmpunordpd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x03], |a| { a.vcmpunordqd(6, 25, 21); });
}

#[test]
fn vcmpneqqd_0() {
    // vcmpneqpd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqqd(1, 2, 3); });
}

#[test]
fn vcmpneqqd_1() {
    // vcmpneqpd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqqd(1, 10, 11); });
}

#[test]
fn vcmpneqqd_2() {
    // vcmpneqpd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x04], |a| { a.vcmpneqqd(1, 18, 19); });
}

#[test]
fn vcmpneqqd_3() {
    // vcmpneqpd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x04], |a| { a.vcmpneqqd(6, 25, 21); });
}

#[test]
fn vcmpnltqd_0() {
    // vcmpnltpd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltqd(1, 2, 3); });
}

#[test]
fn vcmpnltqd_1() {
    // vcmpnltpd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltqd(1, 10, 11); });
}

#[test]
fn vcmpnltqd_2() {
    // vcmpnltpd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x05], |a| { a.vcmpnltqd(1, 18, 19); });
}

#[test]
fn vcmpnltqd_3() {
    // vcmpnltpd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x05], |a| { a.vcmpnltqd(6, 25, 21); });
}

#[test]
fn vcmpnleqd_0() {
    // vcmpnlepd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x06], |a| { a.vcmpnleqd(1, 2, 3); });
}

#[test]
fn vcmpnleqd_1() {
    // vcmpnlepd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x06], |a| { a.vcmpnleqd(1, 10, 11); });
}

#[test]
fn vcmpnleqd_2() {
    // vcmpnlepd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x06], |a| { a.vcmpnleqd(1, 18, 19); });
}

#[test]
fn vcmpnleqd_3() {
    // vcmpnlepd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x06], |a| { a.vcmpnleqd(6, 25, 21); });
}

#[test]
fn vcmpordqd_0() {
    // vcmpordpd k1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc2, 0xcb, 0x07], |a| { a.vcmpordqd(1, 2, 3); });
}

#[test]
fn vcmpordqd_1() {
    // vcmpordpd k1, zmm10, zmm11
    check(DataType::F64, &[0x62, 0xd1, 0xad, 0x48, 0xc2, 0xcb, 0x07], |a| { a.vcmpordqd(1, 10, 11); });
}

#[test]
fn vcmpordqd_2() {
    // vcmpordpd k1, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xb1, 0xed, 0x40, 0xc2, 0xcb, 0x07], |a| { a.vcmpordqd(1, 18, 19); });
}

#[test]
fn vcmpordqd_3() {
    // vcmpordpd k6, zmm25, zmm21
    check(DataType::F64, &[0x62, 0xb1, 0xb5, 0x40, 0xc2, 0xf5, 0x07], |a| { a.vcmpordqd(6, 25, 21); });
}

#[test]
fn vshufqd_0() {
    // vshufpd zmm1, zmm2, zmm3, 5
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0xc6, 0xcb, 0x05], |a| { a.vshufqd(1, 2, 3, 5); });
}

#[test]
fn vshufqd_1() {
    // vshufpd zmm9, zmm10, zmm11, 5
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0xc6, 0xcb, 0x05], |a| { a.vshufqd(9, 10, 11, 5); });
}

#[test]
fn vshufqd_2() {
    // vshufpd zmm17, zmm18, zmm19, 5
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0xc6, 0xcb, 0x05], |a| { a.vshufqd(17, 18, 19, 5); });
}

#[test]
fn vshufqd_3() {
    // vshufpd zmm30, zmm25, zmm21, 5
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0xc6, 0xf5, 0x05], |a| { a.vshufqd(30, 25, 21, 5); });
}

#[test]
fn vunpckhqd_0() {
    // vunpckhpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x15, 0xcb], |a| { a.vunpckhqd(1, 2, 3); });
}

#[test]
fn vunpckhqd_1() {
    // vunpckhpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x15, 0xcb], |a| { a.vunpckhqd(9, 10, 11); });
}

#[test]
fn vunpckhqd_2() {
    // vunpckhpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x15, 0xcb], |a| { a.vunpckhqd(17, 18, 19); });
}

#[test]
fn vunpckhqd_3() {
    // vunpckhpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x15, 0xf5], |a| { a.vunpckhqd(30, 25, 21); });
}

#[test]
fn vunpcklqd_0() {
    // vunpcklpd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x14, 0xcb], |a| { a.vunpcklqd(1, 2, 3); });
}

#[test]
fn vunpcklqd_1() {
    // vunpcklpd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x51, 0xad, 0x48, 0x14, 0xcb], |a| { a.vunpcklqd(9, 10, 11); });
}

#[test]
fn vunpcklqd_2() {
    // vunpcklpd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa1, 0xed, 0x40, 0x14, 0xcb], |a| { a.vunpcklqd(17, 18, 19); });
}

#[test]
fn vunpcklqd_3() {
    // vunpcklpd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x21, 0xb5, 0x40, 0x14, 0xf5], |a| { a.vunpcklqd(30, 25, 21); });
}

#[test]
fn kmovw_reg_k_0() {
    // kmovw ecx, k2
    // NOTE: the 3-byte VEX form is used where the 2-byte form would do (valid, not minimal)
    check(DataType::F64, &[0xc4, 0xe1, 0x78, 0x93, 0xca], |a| { a.kmovw_reg_k(1, 2); });
}

#[test]
fn kmovw_reg_k_1() {
    // kmovw r9d, k3
    // NOTE: the 3-byte VEX form is used where the 2-byte form would do (valid, not minimal)
    check(DataType::F64, &[0xc4, 0x61, 0x78, 0x93, 0xcb], |a| { a.kmovw_reg_k(9, 3); });
}

#[test]
fn kmovw_reg_k_2() {
    // kmovw r14d, k7
    // NOTE: the 3-byte VEX form is used where the 2-byte form would do (valid, not minimal)
    check(DataType::F64, &[0xc4, 0x61, 0x78, 0x93, 0xf7], |a| { a.kmovw_reg_k(14, 7); });
}

#[test]
fn kmovw_k_reg_0() {
    // kmovw k1, edx
    // NOTE: the 3-byte VEX form is used where the 2-byte form would do (valid, not minimal)
    check(DataType::F64, &[0xc4, 0xe1, 0x78, 0x92, 0xca], |a| { a.kmovw_k_reg(1, 2); });
}

#[test]
fn kmovw_k_reg_1() {
    // kmovw k2, r9d
    check(DataType::F64, &[0xc4, 0xc1, 0x78, 0x92, 0xd1], |a| { a.kmovw_k_reg(2, 9); });
}

#[test]
fn kmovw_k_reg_2() {
    // kmovw k7, r14d
    check(DataType::F64, &[0xc4, 0xc1, 0x78, 0x92, 0xfe], |a| { a.kmovw_k_reg(7, 14); });
}

#[test]
fn knotw_0() {
    // knotw k1, k2
    // NOTE: the 3-byte VEX form is used where the 2-byte form would do (valid, not minimal)
    check(DataType::F64, &[0xc4, 0xe1, 0x78, 0x44, 0xca], |a| { a.knotw(1, 2); });
}

#[test]
fn knotw_1() {
    // knotw k7, k6
    // NOTE: the 3-byte VEX form is used where the 2-byte form would do (valid, not minimal)
    check(DataType::F64, &[0xc4, 0xe1, 0x78, 0x44, 0xfe], |a| { a.knotw(7, 6); });
}

#[test]
fn vpmovq2m_qd_0() {
    // vpmovq2m k1, zmm2
    check(DataType::F64, &[0x62, 0xf2, 0xfe, 0x48, 0x39, 0xca], |a| { a.vpmovq2m_qd(1, 2); });
}

#[test]
fn vpmovq2m_qd_1() {
    // vpmovq2m k2, zmm9
    check(DataType::F64, &[0x62, 0xd2, 0xfe, 0x48, 0x39, 0xd1], |a| { a.vpmovq2m_qd(2, 9); });
}

#[test]
fn vpmovq2m_qd_2() {
    // vpmovq2m k3, zmm30
    check(DataType::F64, &[0x62, 0x92, 0xfe, 0x48, 0x39, 0xde], |a| { a.vpmovq2m_qd(3, 30); });
}

#[test]
fn vpmovm2q_qd_0() {
    // vpmovm2q zmm1, k2
    check(DataType::F64, &[0x62, 0xf2, 0xfe, 0x48, 0x38, 0xca], |a| { a.vpmovm2q_qd(1, 2); });
}

#[test]
fn vpmovm2q_qd_1() {
    // vpmovm2q zmm9, k3
    check(DataType::F64, &[0x62, 0x72, 0xfe, 0x48, 0x38, 0xcb], |a| { a.vpmovm2q_qd(9, 3); });
}

#[test]
fn vpmovm2q_qd_2() {
    // vpmovm2q zmm30, k7
    check(DataType::F64, &[0x62, 0x62, 0xfe, 0x48, 0x38, 0xf7], |a| { a.vpmovm2q_qd(30, 7); });
}

#[test]
fn vfmadd132sd_0() {
    // vfmadd132sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x99, 0xcb], |a| { a.vfmadd132sd(1, 2, 3); });
}

#[test]
fn vfmadd132sd_1() {
    // vfmadd132sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x99, 0xcb], |a| { a.vfmadd132sd(9, 10, 11); });
}

#[test]
fn vfmadd132pd_0() {
    // vfmadd132pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x98, 0xcb], |a| { a.vfmadd132pd(1, 2, 3); });
}

#[test]
fn vfmadd132pd_1() {
    // vfmadd132pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x98, 0xcb], |a| { a.vfmadd132pd(9, 10, 11); });
}

#[test]
fn vfmadd132dd_0() {
    // vfmadd132pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0x98, 0xcb], |a| { a.vfmadd132dd(1, 2, 3); });
}

#[test]
fn vfmadd132dd_1() {
    // vfmadd132pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0x98, 0xcb], |a| { a.vfmadd132dd(9, 10, 11); });
}

#[test]
fn vfmadd132qd_0() {
    // vfmadd132pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0x98, 0xcb], |a| { a.vfmadd132qd(1, 2, 3); });
}

#[test]
fn vfmadd132qd_1() {
    // vfmadd132pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0x98, 0xcb], |a| { a.vfmadd132qd(9, 10, 11); });
}

#[test]
fn vfmadd132qd_2() {
    // vfmadd132pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0x98, 0xcb], |a| { a.vfmadd132qd(17, 18, 19); });
}

#[test]
fn vfmadd132qd_3() {
    // vfmadd132pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0x98, 0xf5], |a| { a.vfmadd132qd(30, 25, 21); });
}

#[test]
fn vfmadd213sd_0() {
    // vfmadd213sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xa9, 0xcb], |a| { a.vfmadd213sd(1, 2, 3); });
}

#[test]
fn vfmadd213sd_1() {
    // vfmadd213sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xa9, 0xcb], |a| { a.vfmadd213sd(9, 10, 11); });
}

#[test]
fn vfmadd213pd_0() {
    // vfmadd213pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xa8, 0xcb], |a| { a.vfmadd213pd(1, 2, 3); });
}

#[test]
fn vfmadd213pd_1() {
    // vfmadd213pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xa8, 0xcb], |a| { a.vfmadd213pd(9, 10, 11); });
}

#[test]
fn vfmadd213dd_0() {
    // vfmadd213pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xa8, 0xcb], |a| { a.vfmadd213dd(1, 2, 3); });
}

#[test]
fn vfmadd213dd_1() {
    // vfmadd213pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xa8, 0xcb], |a| { a.vfmadd213dd(9, 10, 11); });
}

#[test]
fn vfmadd213qd_0() {
    // vfmadd213pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xa8, 0xcb], |a| { a.vfmadd213qd(1, 2, 3); });
}

#[test]
fn vfmadd213qd_1() {
    // vfmadd213pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xa8, 0xcb], |a| { a.vfmadd213qd(9, 10, 11); });
}

#[test]
fn vfmadd213qd_2() {
    // vfmadd213pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xa8, 0xcb], |a| { a.vfmadd213qd(17, 18, 19); });
}

#[test]
fn vfmadd213qd_3() {
    // vfmadd213pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xa8, 0xf5], |a| { a.vfmadd213qd(30, 25, 21); });
}

#[test]
fn vfmadd231sd_0() {
    // vfmadd231sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xb9, 0xcb], |a| { a.vfmadd231sd(1, 2, 3); });
}

#[test]
fn vfmadd231sd_1() {
    // vfmadd231sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xb9, 0xcb], |a| { a.vfmadd231sd(9, 10, 11); });
}

#[test]
fn vfmadd231pd_0() {
    // vfmadd231pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xb8, 0xcb], |a| { a.vfmadd231pd(1, 2, 3); });
}

#[test]
fn vfmadd231pd_1() {
    // vfmadd231pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xb8, 0xcb], |a| { a.vfmadd231pd(9, 10, 11); });
}

#[test]
fn vfmadd231dd_0() {
    // vfmadd231pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xb8, 0xcb], |a| { a.vfmadd231dd(1, 2, 3); });
}

#[test]
fn vfmadd231dd_1() {
    // vfmadd231pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xb8, 0xcb], |a| { a.vfmadd231dd(9, 10, 11); });
}

#[test]
fn vfmadd231qd_0() {
    // vfmadd231pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xb8, 0xcb], |a| { a.vfmadd231qd(1, 2, 3); });
}

#[test]
fn vfmadd231qd_1() {
    // vfmadd231pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xb8, 0xcb], |a| { a.vfmadd231qd(9, 10, 11); });
}

#[test]
fn vfmadd231qd_2() {
    // vfmadd231pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xb8, 0xcb], |a| { a.vfmadd231qd(17, 18, 19); });
}

#[test]
fn vfmadd231qd_3() {
    // vfmadd231pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xb8, 0xf5], |a| { a.vfmadd231qd(30, 25, 21); });
}

#[test]
fn vfmsub132sd_0() {
    // vfmsub132sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x9b, 0xcb], |a| { a.vfmsub132sd(1, 2, 3); });
}

#[test]
fn vfmsub132sd_1() {
    // vfmsub132sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x9b, 0xcb], |a| { a.vfmsub132sd(9, 10, 11); });
}

#[test]
fn vfmsub132pd_0() {
    // vfmsub132pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x9a, 0xcb], |a| { a.vfmsub132pd(1, 2, 3); });
}

#[test]
fn vfmsub132pd_1() {
    // vfmsub132pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x9a, 0xcb], |a| { a.vfmsub132pd(9, 10, 11); });
}

#[test]
fn vfmsub132dd_0() {
    // vfmsub132pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0x9a, 0xcb], |a| { a.vfmsub132dd(1, 2, 3); });
}

#[test]
fn vfmsub132dd_1() {
    // vfmsub132pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0x9a, 0xcb], |a| { a.vfmsub132dd(9, 10, 11); });
}

#[test]
fn vfmsub132qd_0() {
    // vfmsub132pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0x9a, 0xcb], |a| { a.vfmsub132qd(1, 2, 3); });
}

#[test]
fn vfmsub132qd_1() {
    // vfmsub132pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0x9a, 0xcb], |a| { a.vfmsub132qd(9, 10, 11); });
}

#[test]
fn vfmsub132qd_2() {
    // vfmsub132pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0x9a, 0xcb], |a| { a.vfmsub132qd(17, 18, 19); });
}

#[test]
fn vfmsub132qd_3() {
    // vfmsub132pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0x9a, 0xf5], |a| { a.vfmsub132qd(30, 25, 21); });
}

#[test]
fn vfmsub213sd_0() {
    // vfmsub213sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xab, 0xcb], |a| { a.vfmsub213sd(1, 2, 3); });
}

#[test]
fn vfmsub213sd_1() {
    // vfmsub213sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xab, 0xcb], |a| { a.vfmsub213sd(9, 10, 11); });
}

#[test]
fn vfmsub213pd_0() {
    // vfmsub213pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xaa, 0xcb], |a| { a.vfmsub213pd(1, 2, 3); });
}

#[test]
fn vfmsub213pd_1() {
    // vfmsub213pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xaa, 0xcb], |a| { a.vfmsub213pd(9, 10, 11); });
}

#[test]
fn vfmsub213dd_0() {
    // vfmsub213pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xaa, 0xcb], |a| { a.vfmsub213dd(1, 2, 3); });
}

#[test]
fn vfmsub213dd_1() {
    // vfmsub213pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xaa, 0xcb], |a| { a.vfmsub213dd(9, 10, 11); });
}

#[test]
fn vfmsub213qd_0() {
    // vfmsub213pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xaa, 0xcb], |a| { a.vfmsub213qd(1, 2, 3); });
}

#[test]
fn vfmsub213qd_1() {
    // vfmsub213pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xaa, 0xcb], |a| { a.vfmsub213qd(9, 10, 11); });
}

#[test]
fn vfmsub213qd_2() {
    // vfmsub213pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xaa, 0xcb], |a| { a.vfmsub213qd(17, 18, 19); });
}

#[test]
fn vfmsub213qd_3() {
    // vfmsub213pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xaa, 0xf5], |a| { a.vfmsub213qd(30, 25, 21); });
}

#[test]
fn vfmsub231sd_0() {
    // vfmsub231sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xbb, 0xcb], |a| { a.vfmsub231sd(1, 2, 3); });
}

#[test]
fn vfmsub231sd_1() {
    // vfmsub231sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xbb, 0xcb], |a| { a.vfmsub231sd(9, 10, 11); });
}

#[test]
fn vfmsub231pd_0() {
    // vfmsub231pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xba, 0xcb], |a| { a.vfmsub231pd(1, 2, 3); });
}

#[test]
fn vfmsub231pd_1() {
    // vfmsub231pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xba, 0xcb], |a| { a.vfmsub231pd(9, 10, 11); });
}

#[test]
fn vfmsub231dd_0() {
    // vfmsub231pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xba, 0xcb], |a| { a.vfmsub231dd(1, 2, 3); });
}

#[test]
fn vfmsub231dd_1() {
    // vfmsub231pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xba, 0xcb], |a| { a.vfmsub231dd(9, 10, 11); });
}

#[test]
fn vfmsub231qd_0() {
    // vfmsub231pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xba, 0xcb], |a| { a.vfmsub231qd(1, 2, 3); });
}

#[test]
fn vfmsub231qd_1() {
    // vfmsub231pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xba, 0xcb], |a| { a.vfmsub231qd(9, 10, 11); });
}

#[test]
fn vfmsub231qd_2() {
    // vfmsub231pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xba, 0xcb], |a| { a.vfmsub231qd(17, 18, 19); });
}

#[test]
fn vfmsub231qd_3() {
    // vfmsub231pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xba, 0xf5], |a| { a.vfmsub231qd(30, 25, 21); });
}

#[test]
fn vfnmadd132sd_0() {
    // vfnmadd132sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x9d, 0xcb], |a| { a.vfnmadd132sd(1, 2, 3); });
}

#[test]
fn vfnmadd132sd_1() {
    // vfnmadd132sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x9d, 0xcb], |a| { a.vfnmadd132sd(9, 10, 11); });
}

#[test]
fn vfnmadd132pd_0() {
    // vfnmadd132pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x9c, 0xcb], |a| { a.vfnmadd132pd(1, 2, 3); });
}

#[test]
fn vfnmadd132pd_1() {
    // vfnmadd132pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x9c, 0xcb], |a| { a.vfnmadd132pd(9, 10, 11); });
}

#[test]
fn vfnmadd132dd_0() {
    // vfnmadd132pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0x9c, 0xcb], |a| { a.vfnmadd132dd(1, 2, 3); });
}

#[test]
fn vfnmadd132dd_1() {
    // vfnmadd132pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0x9c, 0xcb], |a| { a.vfnmadd132dd(9, 10, 11); });
}

#[test]
fn vfnmadd132qd_0() {
    // vfnmadd132pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0x9c, 0xcb], |a| { a.vfnmadd132qd(1, 2, 3); });
}

#[test]
fn vfnmadd132qd_1() {
    // vfnmadd132pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0x9c, 0xcb], |a| { a.vfnmadd132qd(9, 10, 11); });
}

#[test]
fn vfnmadd132qd_2() {
    // vfnmadd132pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0x9c, 0xcb], |a| { a.vfnmadd132qd(17, 18, 19); });
}

#[test]
fn vfnmadd132qd_3() {
    // vfnmadd132pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0x9c, 0xf5], |a| { a.vfnmadd132qd(30, 25, 21); });
}

#[test]
fn vfnmadd213sd_0() {
    // vfnmadd213sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xad, 0xcb], |a| { a.vfnmadd213sd(1, 2, 3); });
}

#[test]
fn vfnmadd213sd_1() {
    // vfnmadd213sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xad, 0xcb], |a| { a.vfnmadd213sd(9, 10, 11); });
}

#[test]
fn vfnmadd213pd_0() {
    // vfnmadd213pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xac, 0xcb], |a| { a.vfnmadd213pd(1, 2, 3); });
}

#[test]
fn vfnmadd213pd_1() {
    // vfnmadd213pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xac, 0xcb], |a| { a.vfnmadd213pd(9, 10, 11); });
}

#[test]
fn vfnmadd213dd_0() {
    // vfnmadd213pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xac, 0xcb], |a| { a.vfnmadd213dd(1, 2, 3); });
}

#[test]
fn vfnmadd213dd_1() {
    // vfnmadd213pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xac, 0xcb], |a| { a.vfnmadd213dd(9, 10, 11); });
}

#[test]
fn vfnmadd213qd_0() {
    // vfnmadd213pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xac, 0xcb], |a| { a.vfnmadd213qd(1, 2, 3); });
}

#[test]
fn vfnmadd213qd_1() {
    // vfnmadd213pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xac, 0xcb], |a| { a.vfnmadd213qd(9, 10, 11); });
}

#[test]
fn vfnmadd213qd_2() {
    // vfnmadd213pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xac, 0xcb], |a| { a.vfnmadd213qd(17, 18, 19); });
}

#[test]
fn vfnmadd213qd_3() {
    // vfnmadd213pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xac, 0xf5], |a| { a.vfnmadd213qd(30, 25, 21); });
}

#[test]
fn vfnmadd231sd_0() {
    // vfnmadd231sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xbd, 0xcb], |a| { a.vfnmadd231sd(1, 2, 3); });
}

#[test]
fn vfnmadd231sd_1() {
    // vfnmadd231sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xbd, 0xcb], |a| { a.vfnmadd231sd(9, 10, 11); });
}

#[test]
fn vfnmadd231pd_0() {
    // vfnmadd231pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xbc, 0xcb], |a| { a.vfnmadd231pd(1, 2, 3); });
}

#[test]
fn vfnmadd231pd_1() {
    // vfnmadd231pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xbc, 0xcb], |a| { a.vfnmadd231pd(9, 10, 11); });
}

#[test]
fn vfnmadd231dd_0() {
    // vfnmadd231pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xbc, 0xcb], |a| { a.vfnmadd231dd(1, 2, 3); });
}

#[test]
fn vfnmadd231dd_1() {
    // vfnmadd231pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xbc, 0xcb], |a| { a.vfnmadd231dd(9, 10, 11); });
}

#[test]
fn vfnmadd231qd_0() {
    // vfnmadd231pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xbc, 0xcb], |a| { a.vfnmadd231qd(1, 2, 3); });
}

#[test]
fn vfnmadd231qd_1() {
    // vfnmadd231pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xbc, 0xcb], |a| { a.vfnmadd231qd(9, 10, 11); });
}

#[test]
fn vfnmadd231qd_2() {
    // vfnmadd231pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xbc, 0xcb], |a| { a.vfnmadd231qd(17, 18, 19); });
}

#[test]
fn vfnmadd231qd_3() {
    // vfnmadd231pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xbc, 0xf5], |a| { a.vfnmadd231qd(30, 25, 21); });
}

#[test]
fn vfnmsub132sd_0() {
    // vfnmsub132sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x9f, 0xcb], |a| { a.vfnmsub132sd(1, 2, 3); });
}

#[test]
fn vfnmsub132sd_1() {
    // vfnmsub132sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x9f, 0xcb], |a| { a.vfnmsub132sd(9, 10, 11); });
}

#[test]
fn vfnmsub132pd_0() {
    // vfnmsub132pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0x9e, 0xcb], |a| { a.vfnmsub132pd(1, 2, 3); });
}

#[test]
fn vfnmsub132pd_1() {
    // vfnmsub132pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0x9e, 0xcb], |a| { a.vfnmsub132pd(9, 10, 11); });
}

#[test]
fn vfnmsub132dd_0() {
    // vfnmsub132pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0x9e, 0xcb], |a| { a.vfnmsub132dd(1, 2, 3); });
}

#[test]
fn vfnmsub132dd_1() {
    // vfnmsub132pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0x9e, 0xcb], |a| { a.vfnmsub132dd(9, 10, 11); });
}

#[test]
fn vfnmsub132qd_0() {
    // vfnmsub132pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0x9e, 0xcb], |a| { a.vfnmsub132qd(1, 2, 3); });
}

#[test]
fn vfnmsub132qd_1() {
    // vfnmsub132pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0x9e, 0xcb], |a| { a.vfnmsub132qd(9, 10, 11); });
}

#[test]
fn vfnmsub132qd_2() {
    // vfnmsub132pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0x9e, 0xcb], |a| { a.vfnmsub132qd(17, 18, 19); });
}

#[test]
fn vfnmsub132qd_3() {
    // vfnmsub132pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0x9e, 0xf5], |a| { a.vfnmsub132qd(30, 25, 21); });
}

#[test]
fn vfnmsub213sd_0() {
    // vfnmsub213sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xaf, 0xcb], |a| { a.vfnmsub213sd(1, 2, 3); });
}

#[test]
fn vfnmsub213sd_1() {
    // vfnmsub213sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xaf, 0xcb], |a| { a.vfnmsub213sd(9, 10, 11); });
}

#[test]
fn vfnmsub213pd_0() {
    // vfnmsub213pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xae, 0xcb], |a| { a.vfnmsub213pd(1, 2, 3); });
}

#[test]
fn vfnmsub213pd_1() {
    // vfnmsub213pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xae, 0xcb], |a| { a.vfnmsub213pd(9, 10, 11); });
}

#[test]
fn vfnmsub213dd_0() {
    // vfnmsub213pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xae, 0xcb], |a| { a.vfnmsub213dd(1, 2, 3); });
}

#[test]
fn vfnmsub213dd_1() {
    // vfnmsub213pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xae, 0xcb], |a| { a.vfnmsub213dd(9, 10, 11); });
}

#[test]
fn vfnmsub213qd_0() {
    // vfnmsub213pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xae, 0xcb], |a| { a.vfnmsub213qd(1, 2, 3); });
}

#[test]
fn vfnmsub213qd_1() {
    // vfnmsub213pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xae, 0xcb], |a| { a.vfnmsub213qd(9, 10, 11); });
}

#[test]
fn vfnmsub213qd_2() {
    // vfnmsub213pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xae, 0xcb], |a| { a.vfnmsub213qd(17, 18, 19); });
}

#[test]
fn vfnmsub213qd_3() {
    // vfnmsub213pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xae, 0xf5], |a| { a.vfnmsub213qd(30, 25, 21); });
}

#[test]
fn vfnmsub231sd_0() {
    // vfnmsub231sd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xbf, 0xcb], |a| { a.vfnmsub231sd(1, 2, 3); });
}

#[test]
fn vfnmsub231sd_1() {
    // vfnmsub231sd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xbf, 0xcb], |a| { a.vfnmsub231sd(9, 10, 11); });
}

#[test]
fn vfnmsub231pd_0() {
    // vfnmsub231pd ymm1, ymm2, ymm3
    check(DataType::F64, &[0xc4, 0xe2, 0xed, 0xbe, 0xcb], |a| { a.vfnmsub231pd(1, 2, 3); });
}

#[test]
fn vfnmsub231pd_1() {
    // vfnmsub231pd ymm9, ymm10, ymm11
    check(DataType::F64, &[0xc4, 0x42, 0xad, 0xbe, 0xcb], |a| { a.vfnmsub231pd(9, 10, 11); });
}

#[test]
fn vfnmsub231dd_0() {
    // vfnmsub231pd xmm1, xmm2, xmm3
    check(DataType::F64, &[0xc4, 0xe2, 0xe9, 0xbe, 0xcb], |a| { a.vfnmsub231dd(1, 2, 3); });
}

#[test]
fn vfnmsub231dd_1() {
    // vfnmsub231pd xmm9, xmm10, xmm11
    check(DataType::F64, &[0xc4, 0x42, 0xa9, 0xbe, 0xcb], |a| { a.vfnmsub231dd(9, 10, 11); });
}

#[test]
fn vfnmsub231qd_0() {
    // vfnmsub231pd zmm1, zmm2, zmm3
    check(DataType::F64, &[0x62, 0xf2, 0xed, 0x48, 0xbe, 0xcb], |a| { a.vfnmsub231qd(1, 2, 3); });
}

#[test]
fn vfnmsub231qd_1() {
    // vfnmsub231pd zmm9, zmm10, zmm11
    check(DataType::F64, &[0x62, 0x52, 0xad, 0x48, 0xbe, 0xcb], |a| { a.vfnmsub231qd(9, 10, 11); });
}

#[test]
fn vfnmsub231qd_2() {
    // vfnmsub231pd zmm17, zmm18, zmm19
    check(DataType::F64, &[0x62, 0xa2, 0xed, 0x40, 0xbe, 0xcb], |a| { a.vfnmsub231qd(17, 18, 19); });
}

#[test]
fn vfnmsub231qd_3() {
    // vfnmsub231pd zmm30, zmm25, zmm21
    check(DataType::F64, &[0x62, 0x22, 0xb5, 0x40, 0xbe, 0xf5], |a| { a.vfnmsub231qd(30, 25, 21); });
}

#[test]
fn mov_0() {
    // {load} mov rcx, rdx
    check(DataType::F64, &[0x48, 0x8b, 0xca], |a| { a.mov(1, 2); });
}

#[test]
fn mov_1() {
    // {load} mov r9, rdx
    check(DataType::F64, &[0x4c, 0x8b, 0xca], |a| { a.mov(9, 2); });
}

#[test]
fn mov_2() {
    // {load} mov rdx, r12
    check(DataType::F64, &[0x49, 0x8b, 0xd4], |a| { a.mov(2, 12); });
}

#[test]
fn or_0() {
    // {load} or rcx, rdx
    check(DataType::F64, &[0x48, 0x0b, 0xca], |a| { a.or(1, 2); });
}

#[test]
fn or_1() {
    // {load} or r9, rdx
    check(DataType::F64, &[0x4c, 0x0b, 0xca], |a| { a.or(9, 2); });
}

#[test]
fn xor_0() {
    // {load} xor rcx, rdx
    check(DataType::F64, &[0x48, 0x33, 0xca], |a| { a.xor(1, 2); });
}

#[test]
fn xor_1() {
    // {load} xor r9, rdx
    check(DataType::F64, &[0x4c, 0x33, 0xca], |a| { a.xor(9, 2); });
}

#[test]
fn add_0() {
    // {load} add rcx, rdx
    check(DataType::F64, &[0x48, 0x03, 0xca], |a| { a.add(1, 2); });
}

#[test]
fn add_1() {
    // {load} add r9, rdx
    check(DataType::F64, &[0x4c, 0x03, 0xca], |a| { a.add(9, 2); });
}

#[test]
fn cmovz_0() {
    // cmovz rcx, rdx
    check(DataType::F64, &[0x48, 0x0f, 0x44, 0xca], |a| { a.cmovz(1, 2); });
}

#[test]
fn cmovz_1() {
    // cmovz r9, rdx
    check(DataType::F64, &[0x4c, 0x0f, 0x44, 0xca], |a| { a.cmovz(9, 2); });
}

#[test]
fn cmovz_2() {
    // cmovz rdx, r12
    check(DataType::F64, &[0x49, 0x0f, 0x44, 0xd4], |a| { a.cmovz(2, 12); });
}

#[test]
fn cmovnz_0() {
    // cmovnz rcx, rdx
    check(DataType::F64, &[0x48, 0x0f, 0x45, 0xca], |a| { a.cmovnz(1, 2); });
}

#[test]
fn cmovnz_1() {
    // cmovnz r9, rdx
    check(DataType::F64, &[0x4c, 0x0f, 0x45, 0xca], |a| { a.cmovnz(9, 2); });
}

#[test]
fn cmovnz_2() {
    // cmovnz rdx, r12
    check(DataType::F64, &[0x49, 0x0f, 0x45, 0xd4], |a| { a.cmovnz(2, 12); });
}

#[test]
fn movsx_0() {
    // movsx rcx, dx
    check(DataType::F64, &[0x48, 0x0f, 0xbf, 0xca], |a| { a.movsx(1, 2); });
}

#[test]
fn movsx_1() {
    // movsx r9, r10w
    check(DataType::F64, &[0x4d, 0x0f, 0xbf, 0xca], |a| { a.movsx(9, 10); });
}

#[test]
fn movzx_0() {
    // movzx rcx, dx
    check(DataType::F64, &[0x48, 0x0f, 0xb7, 0xca], |a| { a.movzx(1, 2); });
}

#[test]
fn movzx_1() {
    // movzx r9, r10w
    check(DataType::F64, &[0x4d, 0x0f, 0xb7, 0xca], |a| { a.movzx(9, 10); });
}

#[test]
fn call_0() {
    // call rax
    check(DataType::F64, &[0xff, 0xd0], |a| { a.call(0); });
}

#[test]
fn call_1() {
    // call rcx
    check(DataType::F64, &[0xff, 0xd1], |a| { a.call(1); });
}

#[test]
fn call_2() {
    // call r9
    check(DataType::F64, &[0x41, 0xff, 0xd1], |a| { a.call(9); });
}

#[test]
fn call_3() {
    // call r15
    check(DataType::F64, &[0x41, 0xff, 0xd7], |a| { a.call(15); });
}

#[test]
fn push_0() {
    // push rax
    check(DataType::F64, &[0x50], |a| { a.push(0); });
}

#[test]
fn push_1() {
    // push rbx
    check(DataType::F64, &[0x53], |a| { a.push(3); });
}

#[test]
fn push_2() {
    // push r9
    check(DataType::F64, &[0x41, 0x51], |a| { a.push(9); });
}

#[test]
fn push_3() {
    // push r15
    check(DataType::F64, &[0x41, 0x57], |a| { a.push(15); });
}

#[test]
fn pop_0() {
    // pop rax
    check(DataType::F64, &[0x58], |a| { a.pop(0); });
}

#[test]
fn pop_1() {
    // pop rbx
    check(DataType::F64, &[0x5b], |a| { a.pop(3); });
}

#[test]
fn pop_2() {
    // pop r9
    check(DataType::F64, &[0x41, 0x59], |a| { a.pop(9); });
}

#[test]
fn pop_3() {
    // pop r15
    check(DataType::F64, &[0x41, 0x5f], |a| { a.pop(15); });
}

#[test]
fn ret_0() {
    // ret
    check(DataType::F64, &[0xc3], |a| { a.ret(); });
}

#[test]
fn nop_0() {
    // nop
    check(DataType::F64, &[0x90], |a| { a.nop(); });
}

#[test]
fn add_imm_0() {
    // add rcx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xc1, 0x78, 0x56, 0x34, 0x12], |a| { a.add_imm(1, 0x12345678); });
}

#[test]
fn add_imm_1() {
    // add r9, 0x12345678
    check(DataType::F64, &[0x49, 0x81, 0xc1, 0x78, 0x56, 0x34, 0x12], |a| { a.add_imm(9, 0x12345678); });
}

#[test]
fn add_imm_2() {
    // add rbx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xc3, 0x78, 0x56, 0x34, 0x12], |a| { a.add_imm(3, 0x12345678); });
}

#[test]
fn or_imm_0() {
    // or rcx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xc9, 0x78, 0x56, 0x34, 0x12], |a| { a.or_imm(1, 0x12345678); });
}

#[test]
fn or_imm_1() {
    // or r9, 0x12345678
    check(DataType::F64, &[0x49, 0x81, 0xc9, 0x78, 0x56, 0x34, 0x12], |a| { a.or_imm(9, 0x12345678); });
}

#[test]
fn or_imm_2() {
    // or rbx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xcb, 0x78, 0x56, 0x34, 0x12], |a| { a.or_imm(3, 0x12345678); });
}

#[test]
fn and_imm_0() {
    // and rcx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xe1, 0x78, 0x56, 0x34, 0x12], |a| { a.and_imm(1, 0x12345678); });
}

#[test]
fn and_imm_1() {
    // and r9, 0x12345678
    check(DataType::F64, &[0x49, 0x81, 0xe1, 0x78, 0x56, 0x34, 0x12], |a| { a.and_imm(9, 0x12345678); });
}

#[test]
fn and_imm_2() {
    // and rbx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xe3, 0x78, 0x56, 0x34, 0x12], |a| { a.and_imm(3, 0x12345678); });
}

#[test]
fn sub_imm_0() {
    // sub rcx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xe9, 0x78, 0x56, 0x34, 0x12], |a| { a.sub_imm(1, 0x12345678); });
}

#[test]
fn sub_imm_1() {
    // sub r9, 0x12345678
    check(DataType::F64, &[0x49, 0x81, 0xe9, 0x78, 0x56, 0x34, 0x12], |a| { a.sub_imm(9, 0x12345678); });
}

#[test]
fn sub_imm_2() {
    // sub rbx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xeb, 0x78, 0x56, 0x34, 0x12], |a| { a.sub_imm(3, 0x12345678); });
}

#[test]
fn xor_imm_0() {
    // xor rcx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xf1, 0x78, 0x56, 0x34, 0x12], |a| { a.xor_imm(1, 0x12345678); });
}

#[test]
fn xor_imm_1() {
    // xor r9, 0x12345678
    check(DataType::F64, &[0x49, 0x81, 0xf1, 0x78, 0x56, 0x34, 0x12], |a| { a.xor_imm(9, 0x12345678); });
}

#[test]
fn xor_imm_2() {
    // xor rbx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xf3, 0x78, 0x56, 0x34, 0x12], |a| { a.xor_imm(3, 0x12345678); });
}

#[test]
fn cmp_imm_0() {
    // cmp rcx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xf9, 0x78, 0x56, 0x34, 0x12], |a| { a.cmp_imm(1, 0x12345678); });
}

#[test]
fn cmp_imm_1() {
    // cmp r9, 0x12345678
    check(DataType::F64, &[0x49, 0x81, 0xf9, 0x78, 0x56, 0x34, 0x12], |a| { a.cmp_imm(9, 0x12345678); });
}

#[test]
fn cmp_imm_2() {
    // cmp rbx, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xfb, 0x78, 0x56, 0x34, 0x12], |a| { a.cmp_imm(3, 0x12345678); });
}

#[test]
fn mov_imm_0() {
    // mov rcx, 0x12345678
    check(DataType::F64, &[0x48, 0xc7, 0xc1, 0x78, 0x56, 0x34, 0x12], |a| { a.mov_imm(1, 0x12345678); });
}

#[test]
fn mov_imm_1() {
    // mov r9, 0x12345678
    check(DataType::F64, &[0x49, 0xc7, 0xc1, 0x78, 0x56, 0x34, 0x12], |a| { a.mov_imm(9, 0x12345678); });
}

#[test]
fn mov_reg_imm32_0() {
    // mov ecx, 0x12345678
    check(DataType::F64, &[0xb9, 0x78, 0x56, 0x34, 0x12], |a| { a.mov_reg_imm32(1, 0x12345678); });
}

#[test]
fn mov_reg_imm32_1() {
    // mov r9d, 0x12345678
    check(DataType::F64, &[0x41, 0xb9, 0x78, 0x56, 0x34, 0x12], |a| { a.mov_reg_imm32(9, 0x12345678); });
}

#[test]
fn movabs_0() {
    // movabs rcx, 0x123456789abcdef0
    check(DataType::F64, &[0x48, 0xb9, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12], |a| { a.movabs(1, 0x123456789abcdef0); });
}

#[test]
fn movabs_1() {
    // movabs r9, 0x123456789abcdef0
    check(DataType::F64, &[0x49, 0xb9, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12], |a| { a.movabs(9, 0x123456789abcdef0); });
}

#[test]
fn shl_imm_0() {
    // shl rcx, 3
    check(DataType::F64, &[0x48, 0xc1, 0xe1, 0x03], |a| { a.shl_imm(1, 3); });
}

#[test]
fn shl_imm_1() {
    // shl r9, 3
    check(DataType::F64, &[0x49, 0xc1, 0xe1, 0x03], |a| { a.shl_imm(9, 3); });
}

#[test]
fn shr_imm_0() {
    // shr rcx, 3
    check(DataType::F64, &[0x48, 0xc1, 0xe9, 0x03], |a| { a.shr_imm(1, 3); });
}

#[test]
fn shr_imm_1() {
    // shr r9, 3
    check(DataType::F64, &[0x49, 0xc1, 0xe9, 0x03], |a| { a.shr_imm(9, 3); });
}

#[test]
fn inc_0() {
    // inc rcx
    check(DataType::F64, &[0x48, 0xff, 0xc1], |a| { a.inc(1); });
}

#[test]
fn inc_1() {
    // inc r9
    check(DataType::F64, &[0x49, 0xff, 0xc1], |a| { a.inc(9); });
}

#[test]
fn dec_0() {
    // dec rcx
    check(DataType::F64, &[0x48, 0xff, 0xc9], |a| { a.dec(1); });
}

#[test]
fn dec_1() {
    // dec r9
    check(DataType::F64, &[0x49, 0xff, 0xc9], |a| { a.dec(9); });
}

#[test]
fn add_rsp_0() {
    // add rsp, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xc4, 0x78, 0x56, 0x34, 0x12], |a| { a.add_rsp(0x12345678); });
}

#[test]
fn sub_rsp_0() {
    // sub rsp, 0x12345678
    check(DataType::F64, &[0x48, 0x81, 0xec, 0x78, 0x56, 0x34, 0x12], |a| { a.sub_rsp(0x12345678); });
}

#[test]
fn prefetcht0_ip_0() {
    // prefetcht0 [rip + 0x1234]
    check(DataType::F64, &[0x0f, 0x18, 0x0d, 0x34, 0x12, 0x00, 0x00], |a| { a.prefetcht0_ip(0x1234); });
}

#[test]
fn prefetcht1_ip_0() {
    // prefetcht1 [rip + 0x1234]
    check(DataType::F64, &[0x0f, 0x18, 0x15, 0x34, 0x12, 0x00, 0x00], |a| { a.prefetcht1_ip(0x1234); });
}

#[test]
fn prefetcht2_ip_0() {
    // prefetcht2 [rip + 0x1234]
    check(DataType::F64, &[0x0f, 0x18, 0x1d, 0x34, 0x12, 0x00, 0x00], |a| { a.prefetcht2_ip(0x1234); });
}

#[test]
fn prefetchnta_ip_0() {
    // prefetchnta [rip + 0x1234]
    check(DataType::F64, &[0x0f, 0x18, 0x05, 0x34, 0x12, 0x00, 0x00], |a| { a.prefetchnta_ip(0x1234); });
}

#[test]
fn mov_reg_mem_rbx_8() {
    // mov rcx, qword ptr [rbx+0x8]
    check(DataType::F64, &[0x48, 0x8b, 0x4b, 0x08], |a| { a.mov_reg_mem(1, 3, 8); });
}

#[test]
fn mov_reg_mem_rbx_1234() {
    // mov rcx, qword ptr [rbx+0x1234]
    check(DataType::F64, &[0x48, 0x8b, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_reg_mem(1, 3, 4660); });
}

#[test]
fn mov_reg_mem_rbx_m8() {
    // mov rcx, qword ptr [rbx-0x8]
    check(DataType::F64, &[0x48, 0x8b, 0x4b, 0xf8], |a| { a.mov_reg_mem(1, 3, -8); });
}

#[test]
fn mov_reg_mem_rsp_8() {
    // mov rcx, qword ptr [rsp+0x8]
    check(DataType::F64, &[0x48, 0x8b, 0x4c, 0x24, 0x08], |a| { a.mov_reg_mem(1, 4, 8); });
}

#[test]
fn mov_reg_mem_rsp_1234() {
    // mov rcx, qword ptr [rsp+0x1234]
    check(DataType::F64, &[0x48, 0x8b, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_reg_mem(1, 4, 4660); });
}

#[test]
fn mov_reg_mem_rsp_m8() {
    // mov rcx, qword ptr [rsp-0x8]
    check(DataType::F64, &[0x48, 0x8b, 0x4c, 0x24, 0xf8], |a| { a.mov_reg_mem(1, 4, -8); });
}

#[test]
fn mov_reg_mem_rbp_8() {
    // mov rcx, qword ptr [rbp+0x8]
    check(DataType::F64, &[0x48, 0x8b, 0x4d, 0x08], |a| { a.mov_reg_mem(1, 5, 8); });
}

#[test]
fn mov_reg_mem_rbp_1234() {
    // mov rcx, qword ptr [rbp+0x1234]
    check(DataType::F64, &[0x48, 0x8b, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_reg_mem(1, 5, 4660); });
}

#[test]
fn mov_reg_mem_rbp_m8() {
    // mov rcx, qword ptr [rbp-0x8]
    check(DataType::F64, &[0x48, 0x8b, 0x4d, 0xf8], |a| { a.mov_reg_mem(1, 5, -8); });
}

#[test]
fn mov_reg_mem_r9_8() {
    // mov rcx, qword ptr [r9+0x8]
    check(DataType::F64, &[0x49, 0x8b, 0x49, 0x08], |a| { a.mov_reg_mem(1, 9, 8); });
}

#[test]
fn mov_reg_mem_r9_1234() {
    // mov rcx, qword ptr [r9+0x1234]
    check(DataType::F64, &[0x49, 0x8b, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_reg_mem(1, 9, 4660); });
}

#[test]
fn mov_reg_mem_r9_m8() {
    // mov rcx, qword ptr [r9-0x8]
    check(DataType::F64, &[0x49, 0x8b, 0x49, 0xf8], |a| { a.mov_reg_mem(1, 9, -8); });
}

#[test]
fn mov_reg_mem_r12_8() {
    // mov rcx, qword ptr [r12+0x8]
    check(DataType::F64, &[0x49, 0x8b, 0x4c, 0x24, 0x08], |a| { a.mov_reg_mem(1, 12, 8); });
}

#[test]
fn mov_reg_mem_r12_1234() {
    // mov rcx, qword ptr [r12+0x1234]
    check(DataType::F64, &[0x49, 0x8b, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_reg_mem(1, 12, 4660); });
}

#[test]
fn mov_reg_mem_r12_m8() {
    // mov rcx, qword ptr [r12-0x8]
    check(DataType::F64, &[0x49, 0x8b, 0x4c, 0x24, 0xf8], |a| { a.mov_reg_mem(1, 12, -8); });
}

#[test]
fn mov_reg_mem_r13_8() {
    // mov rcx, qword ptr [r13+0x8]
    check(DataType::F64, &[0x49, 0x8b, 0x4d, 0x08], |a| { a.mov_reg_mem(1, 13, 8); });
}

#[test]
fn mov_reg_mem_r13_1234() {
    // mov rcx, qword ptr [r13+0x1234]
    check(DataType::F64, &[0x49, 0x8b, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_reg_mem(1, 13, 4660); });
}

#[test]
fn mov_reg_mem_r13_m8() {
    // mov rcx, qword ptr [r13-0x8]
    check(DataType::F64, &[0x49, 0x8b, 0x4d, 0xf8], |a| { a.mov_reg_mem(1, 13, -8); });
}

#[test]
fn mov_mem_reg_rbx_8() {
    // mov qword ptr [rbx+0x8], rcx
    check(DataType::F64, &[0x48, 0x89, 0x4b, 0x08], |a| { a.mov_mem_reg(3, 8, 1); });
}

#[test]
fn mov_mem_reg_rbx_1234() {
    // mov qword ptr [rbx+0x1234], rcx
    check(DataType::F64, &[0x48, 0x89, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_mem_reg(3, 4660, 1); });
}

#[test]
fn mov_mem_reg_rbx_m8() {
    // mov qword ptr [rbx-0x8], rcx
    check(DataType::F64, &[0x48, 0x89, 0x4b, 0xf8], |a| { a.mov_mem_reg(3, -8, 1); });
}

#[test]
fn mov_mem_reg_rsp_8() {
    // mov qword ptr [rsp+0x8], rcx
    check(DataType::F64, &[0x48, 0x89, 0x4c, 0x24, 0x08], |a| { a.mov_mem_reg(4, 8, 1); });
}

#[test]
fn mov_mem_reg_rsp_1234() {
    // mov qword ptr [rsp+0x1234], rcx
    check(DataType::F64, &[0x48, 0x89, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_mem_reg(4, 4660, 1); });
}

#[test]
fn mov_mem_reg_rsp_m8() {
    // mov qword ptr [rsp-0x8], rcx
    check(DataType::F64, &[0x48, 0x89, 0x4c, 0x24, 0xf8], |a| { a.mov_mem_reg(4, -8, 1); });
}

#[test]
fn mov_mem_reg_rbp_8() {
    // mov qword ptr [rbp+0x8], rcx
    check(DataType::F64, &[0x48, 0x89, 0x4d, 0x08], |a| { a.mov_mem_reg(5, 8, 1); });
}

#[test]
fn mov_mem_reg_rbp_1234() {
    // mov qword ptr [rbp+0x1234], rcx
    check(DataType::F64, &[0x48, 0x89, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_mem_reg(5, 4660, 1); });
}

#[test]
fn mov_mem_reg_rbp_m8() {
    // mov qword ptr [rbp-0x8], rcx
    check(DataType::F64, &[0x48, 0x89, 0x4d, 0xf8], |a| { a.mov_mem_reg(5, -8, 1); });
}

#[test]
fn mov_mem_reg_r9_8() {
    // mov qword ptr [r9+0x8], rcx
    check(DataType::F64, &[0x49, 0x89, 0x49, 0x08], |a| { a.mov_mem_reg(9, 8, 1); });
}

#[test]
fn mov_mem_reg_r9_1234() {
    // mov qword ptr [r9+0x1234], rcx
    check(DataType::F64, &[0x49, 0x89, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_mem_reg(9, 4660, 1); });
}

#[test]
fn mov_mem_reg_r9_m8() {
    // mov qword ptr [r9-0x8], rcx
    check(DataType::F64, &[0x49, 0x89, 0x49, 0xf8], |a| { a.mov_mem_reg(9, -8, 1); });
}

#[test]
fn mov_mem_reg_r12_8() {
    // mov qword ptr [r12+0x8], rcx
    check(DataType::F64, &[0x49, 0x89, 0x4c, 0x24, 0x08], |a| { a.mov_mem_reg(12, 8, 1); });
}

#[test]
fn mov_mem_reg_r12_1234() {
    // mov qword ptr [r12+0x1234], rcx
    check(DataType::F64, &[0x49, 0x89, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_mem_reg(12, 4660, 1); });
}

#[test]
fn mov_mem_reg_r12_m8() {
    // mov qword ptr [r12-0x8], rcx
    check(DataType::F64, &[0x49, 0x89, 0x4c, 0x24, 0xf8], |a| { a.mov_mem_reg(12, -8, 1); });
}

#[test]
fn mov_mem_reg_r13_8() {
    // mov qword ptr [r13+0x8], rcx
    check(DataType::F64, &[0x49, 0x89, 0x4d, 0x08], |a| { a.mov_mem_reg(13, 8, 1); });
}

#[test]
fn mov_mem_reg_r13_1234() {
    // mov qword ptr [r13+0x1234], rcx
    check(DataType::F64, &[0x49, 0x89, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.mov_mem_reg(13, 4660, 1); });
}

#[test]
fn mov_mem_reg_r13_m8() {
    // mov qword ptr [r13-0x8], rcx
    check(DataType::F64, &[0x49, 0x89, 0x4d, 0xf8], |a| { a.mov_mem_reg(13, -8, 1); });
}

#[test]
fn lea_mem_rbx_8() {
    // lea rcx, [rbx+0x8]
    check(DataType::F64, &[0x48, 0x8d, 0x4b, 0x08], |a| { a.lea_mem(1, 3, 8); });
}

#[test]
fn lea_mem_rbx_1234() {
    // lea rcx, [rbx+0x1234]
    check(DataType::F64, &[0x48, 0x8d, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.lea_mem(1, 3, 4660); });
}

#[test]
fn lea_mem_rbx_m8() {
    // lea rcx, [rbx-0x8]
    check(DataType::F64, &[0x48, 0x8d, 0x4b, 0xf8], |a| { a.lea_mem(1, 3, -8); });
}

#[test]
fn lea_mem_rsp_8() {
    // lea rcx, [rsp+0x8]
    check(DataType::F64, &[0x48, 0x8d, 0x4c, 0x24, 0x08], |a| { a.lea_mem(1, 4, 8); });
}

#[test]
fn lea_mem_rsp_1234() {
    // lea rcx, [rsp+0x1234]
    check(DataType::F64, &[0x48, 0x8d, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.lea_mem(1, 4, 4660); });
}

#[test]
fn lea_mem_rsp_m8() {
    // lea rcx, [rsp-0x8]
    check(DataType::F64, &[0x48, 0x8d, 0x4c, 0x24, 0xf8], |a| { a.lea_mem(1, 4, -8); });
}

#[test]
fn lea_mem_rbp_8() {
    // lea rcx, [rbp+0x8]
    check(DataType::F64, &[0x48, 0x8d, 0x4d, 0x08], |a| { a.lea_mem(1, 5, 8); });
}

#[test]
fn lea_mem_rbp_1234() {
    // lea rcx, [rbp+0x1234]
    check(DataType::F64, &[0x48, 0x8d, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.lea_mem(1, 5, 4660); });
}

#[test]
fn lea_mem_rbp_m8() {
    // lea rcx, [rbp-0x8]
    check(DataType::F64, &[0x48, 0x8d, 0x4d, 0xf8], |a| { a.lea_mem(1, 5, -8); });
}

#[test]
fn lea_mem_r9_8() {
    // lea rcx, [r9+0x8]
    check(DataType::F64, &[0x49, 0x8d, 0x49, 0x08], |a| { a.lea_mem(1, 9, 8); });
}

#[test]
fn lea_mem_r9_1234() {
    // lea rcx, [r9+0x1234]
    check(DataType::F64, &[0x49, 0x8d, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.lea_mem(1, 9, 4660); });
}

#[test]
fn lea_mem_r9_m8() {
    // lea rcx, [r9-0x8]
    check(DataType::F64, &[0x49, 0x8d, 0x49, 0xf8], |a| { a.lea_mem(1, 9, -8); });
}

#[test]
fn lea_mem_r12_8() {
    // lea rcx, [r12+0x8]
    check(DataType::F64, &[0x49, 0x8d, 0x4c, 0x24, 0x08], |a| { a.lea_mem(1, 12, 8); });
}

#[test]
fn lea_mem_r12_1234() {
    // lea rcx, [r12+0x1234]
    check(DataType::F64, &[0x49, 0x8d, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.lea_mem(1, 12, 4660); });
}

#[test]
fn lea_mem_r12_m8() {
    // lea rcx, [r12-0x8]
    check(DataType::F64, &[0x49, 0x8d, 0x4c, 0x24, 0xf8], |a| { a.lea_mem(1, 12, -8); });
}

#[test]
fn lea_mem_r13_8() {
    // lea rcx, [r13+0x8]
    check(DataType::F64, &[0x49, 0x8d, 0x4d, 0x08], |a| { a.lea_mem(1, 13, 8); });
}

#[test]
fn lea_mem_r13_1234() {
    // lea rcx, [r13+0x1234]
    check(DataType::F64, &[0x49, 0x8d, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.lea_mem(1, 13, 4660); });
}

#[test]
fn lea_mem_r13_m8() {
    // lea rcx, [r13-0x8]
    check(DataType::F64, &[0x49, 0x8d, 0x4d, 0xf8], |a| { a.lea_mem(1, 13, -8); });
}

#[test]
fn vmovsd_xmm_mem_rbx_8() {
    // vmovsd xmm1, qword ptr [rbx+0x8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4b, 0x08], |a| { a.vmovsd_xmm_mem(1, 3, 8); });
}

#[test]
fn vmovsd_xmm_mem_rbx_1234() {
    // vmovsd xmm1, qword ptr [rbx+0x1234]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_xmm_mem(1, 3, 4660); });
}

#[test]
fn vmovsd_xmm_mem_rbx_m8() {
    // vmovsd xmm1, qword ptr [rbx-0x8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4b, 0xf8], |a| { a.vmovsd_xmm_mem(1, 3, -8); });
}

#[test]
fn vmovsd_xmm_mem_rsp_8() {
    // vmovsd xmm1, qword ptr [rsp+0x8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4c, 0x24, 0x08], |a| { a.vmovsd_xmm_mem(1, 4, 8); });
}

#[test]
fn vmovsd_xmm_mem_rsp_1234() {
    // vmovsd xmm1, qword ptr [rsp+0x1234]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_xmm_mem(1, 4, 4660); });
}

#[test]
fn vmovsd_xmm_mem_rsp_m8() {
    // vmovsd xmm1, qword ptr [rsp-0x8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4c, 0x24, 0xf8], |a| { a.vmovsd_xmm_mem(1, 4, -8); });
}

#[test]
fn vmovsd_xmm_mem_rbp_8() {
    // vmovsd xmm1, qword ptr [rbp+0x8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4d, 0x08], |a| { a.vmovsd_xmm_mem(1, 5, 8); });
}

#[test]
fn vmovsd_xmm_mem_rbp_1234() {
    // vmovsd xmm1, qword ptr [rbp+0x1234]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_xmm_mem(1, 5, 4660); });
}

#[test]
fn vmovsd_xmm_mem_rbp_m8() {
    // vmovsd xmm1, qword ptr [rbp-0x8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4d, 0xf8], |a| { a.vmovsd_xmm_mem(1, 5, -8); });
}

#[test]
fn vmovsd_xmm_mem_r9_8() {
    // vmovsd xmm1, qword ptr [r9+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x49, 0x08], |a| { a.vmovsd_xmm_mem(1, 9, 8); });
}

#[test]
fn vmovsd_xmm_mem_r9_1234() {
    // vmovsd xmm1, qword ptr [r9+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_xmm_mem(1, 9, 4660); });
}

#[test]
fn vmovsd_xmm_mem_r9_m8() {
    // vmovsd xmm1, qword ptr [r9-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x49, 0xf8], |a| { a.vmovsd_xmm_mem(1, 9, -8); });
}

#[test]
fn vmovsd_xmm_mem_r12_8() {
    // vmovsd xmm1, qword ptr [r12+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x4c, 0x24, 0x08], |a| { a.vmovsd_xmm_mem(1, 12, 8); });
}

#[test]
fn vmovsd_xmm_mem_r12_1234() {
    // vmovsd xmm1, qword ptr [r12+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_xmm_mem(1, 12, 4660); });
}

#[test]
fn vmovsd_xmm_mem_r12_m8() {
    // vmovsd xmm1, qword ptr [r12-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x4c, 0x24, 0xf8], |a| { a.vmovsd_xmm_mem(1, 12, -8); });
}

#[test]
fn vmovsd_xmm_mem_r13_8() {
    // vmovsd xmm1, qword ptr [r13+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x4d, 0x08], |a| { a.vmovsd_xmm_mem(1, 13, 8); });
}

#[test]
fn vmovsd_xmm_mem_r13_1234() {
    // vmovsd xmm1, qword ptr [r13+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_xmm_mem(1, 13, 4660); });
}

#[test]
fn vmovsd_xmm_mem_r13_m8() {
    // vmovsd xmm1, qword ptr [r13-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x10, 0x4d, 0xf8], |a| { a.vmovsd_xmm_mem(1, 13, -8); });
}

#[test]
fn vmovsd_mem_xmm_rbx_8() {
    // vmovsd qword ptr [rbx+0x8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4b, 0x08], |a| { a.vmovsd_mem_xmm(3, 8, 1); });
}

#[test]
fn vmovsd_mem_xmm_rbx_1234() {
    // vmovsd qword ptr [rbx+0x1234], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_mem_xmm(3, 4660, 1); });
}

#[test]
fn vmovsd_mem_xmm_rbx_m8() {
    // vmovsd qword ptr [rbx-0x8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4b, 0xf8], |a| { a.vmovsd_mem_xmm(3, -8, 1); });
}

#[test]
fn vmovsd_mem_xmm_rsp_8() {
    // vmovsd qword ptr [rsp+0x8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4c, 0x24, 0x08], |a| { a.vmovsd_mem_xmm(4, 8, 1); });
}

#[test]
fn vmovsd_mem_xmm_rsp_1234() {
    // vmovsd qword ptr [rsp+0x1234], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_mem_xmm(4, 4660, 1); });
}

#[test]
fn vmovsd_mem_xmm_rsp_m8() {
    // vmovsd qword ptr [rsp-0x8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4c, 0x24, 0xf8], |a| { a.vmovsd_mem_xmm(4, -8, 1); });
}

#[test]
fn vmovsd_mem_xmm_rbp_8() {
    // vmovsd qword ptr [rbp+0x8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4d, 0x08], |a| { a.vmovsd_mem_xmm(5, 8, 1); });
}

#[test]
fn vmovsd_mem_xmm_rbp_1234() {
    // vmovsd qword ptr [rbp+0x1234], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_mem_xmm(5, 4660, 1); });
}

#[test]
fn vmovsd_mem_xmm_rbp_m8() {
    // vmovsd qword ptr [rbp-0x8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4d, 0xf8], |a| { a.vmovsd_mem_xmm(5, -8, 1); });
}

#[test]
fn vmovsd_mem_xmm_r9_8() {
    // vmovsd qword ptr [r9+0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x49, 0x08], |a| { a.vmovsd_mem_xmm(9, 8, 1); });
}

#[test]
fn vmovsd_mem_xmm_r9_1234() {
    // vmovsd qword ptr [r9+0x1234], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_mem_xmm(9, 4660, 1); });
}

#[test]
fn vmovsd_mem_xmm_r9_m8() {
    // vmovsd qword ptr [r9-0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x49, 0xf8], |a| { a.vmovsd_mem_xmm(9, -8, 1); });
}

#[test]
fn vmovsd_mem_xmm_r12_8() {
    // vmovsd qword ptr [r12+0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x4c, 0x24, 0x08], |a| { a.vmovsd_mem_xmm(12, 8, 1); });
}

#[test]
fn vmovsd_mem_xmm_r12_1234() {
    // vmovsd qword ptr [r12+0x1234], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_mem_xmm(12, 4660, 1); });
}

#[test]
fn vmovsd_mem_xmm_r12_m8() {
    // vmovsd qword ptr [r12-0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x4c, 0x24, 0xf8], |a| { a.vmovsd_mem_xmm(12, -8, 1); });
}

#[test]
fn vmovsd_mem_xmm_r13_8() {
    // vmovsd qword ptr [r13+0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x4d, 0x08], |a| { a.vmovsd_mem_xmm(13, 8, 1); });
}

#[test]
fn vmovsd_mem_xmm_r13_1234() {
    // vmovsd qword ptr [r13+0x1234], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovsd_mem_xmm(13, 4660, 1); });
}

#[test]
fn vmovsd_mem_xmm_r13_m8() {
    // vmovsd qword ptr [r13-0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7b, 0x11, 0x4d, 0xf8], |a| { a.vmovsd_mem_xmm(13, -8, 1); });
}

#[test]
fn vbroadcastsd_rbx_8() {
    // vbroadcastsd ymm1, qword ptr [rbx+0x8]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x4b, 0x08], |a| { a.vbroadcastsd(1, 3, 8); });
}

#[test]
fn vbroadcastsd_rbx_1234() {
    // vbroadcastsd ymm1, qword ptr [rbx+0x1234]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd(1, 3, 4660); });
}

#[test]
fn vbroadcastsd_rbx_m8() {
    // vbroadcastsd ymm1, qword ptr [rbx-0x8]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x4b, 0xf8], |a| { a.vbroadcastsd(1, 3, -8); });
}

#[test]
fn vbroadcastsd_rsp_8() {
    // vbroadcastsd ymm1, qword ptr [rsp+0x8]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x4c, 0x24, 0x08], |a| { a.vbroadcastsd(1, 4, 8); });
}

#[test]
fn vbroadcastsd_rsp_1234() {
    // vbroadcastsd ymm1, qword ptr [rsp+0x1234]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd(1, 4, 4660); });
}

#[test]
fn vbroadcastsd_rsp_m8() {
    // vbroadcastsd ymm1, qword ptr [rsp-0x8]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x4c, 0x24, 0xf8], |a| { a.vbroadcastsd(1, 4, -8); });
}

#[test]
fn vbroadcastsd_rbp_8() {
    // vbroadcastsd ymm1, qword ptr [rbp+0x8]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x4d, 0x08], |a| { a.vbroadcastsd(1, 5, 8); });
}

#[test]
fn vbroadcastsd_rbp_1234() {
    // vbroadcastsd ymm1, qword ptr [rbp+0x1234]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd(1, 5, 4660); });
}

#[test]
fn vbroadcastsd_rbp_m8() {
    // vbroadcastsd ymm1, qword ptr [rbp-0x8]
    check(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x4d, 0xf8], |a| { a.vbroadcastsd(1, 5, -8); });
}

#[test]
fn vbroadcastsd_r9_8() {
    // vbroadcastsd ymm1, qword ptr [r9+0x8]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x49, 0x08], |a| { a.vbroadcastsd(1, 9, 8); });
}

#[test]
fn vbroadcastsd_r9_1234() {
    // vbroadcastsd ymm1, qword ptr [r9+0x1234]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd(1, 9, 4660); });
}

#[test]
fn vbroadcastsd_r9_m8() {
    // vbroadcastsd ymm1, qword ptr [r9-0x8]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x49, 0xf8], |a| { a.vbroadcastsd(1, 9, -8); });
}

#[test]
fn vbroadcastsd_r12_8() {
    // vbroadcastsd ymm1, qword ptr [r12+0x8]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x4c, 0x24, 0x08], |a| { a.vbroadcastsd(1, 12, 8); });
}

#[test]
fn vbroadcastsd_r12_1234() {
    // vbroadcastsd ymm1, qword ptr [r12+0x1234]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd(1, 12, 4660); });
}

#[test]
fn vbroadcastsd_r12_m8() {
    // vbroadcastsd ymm1, qword ptr [r12-0x8]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x4c, 0x24, 0xf8], |a| { a.vbroadcastsd(1, 12, -8); });
}

#[test]
fn vbroadcastsd_r13_8() {
    // vbroadcastsd ymm1, qword ptr [r13+0x8]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x4d, 0x08], |a| { a.vbroadcastsd(1, 13, 8); });
}

#[test]
fn vbroadcastsd_r13_1234() {
    // vbroadcastsd ymm1, qword ptr [r13+0x1234]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd(1, 13, 4660); });
}

#[test]
fn vbroadcastsd_r13_m8() {
    // vbroadcastsd ymm1, qword ptr [r13-0x8]
    check(DataType::F64, &[0xc4, 0xc2, 0x7d, 0x19, 0x4d, 0xf8], |a| { a.vbroadcastsd(1, 13, -8); });
}

#[test]
fn vmovpd_ymm_mem_rbx_8() {
    // vmovupd ymm1, ymmword ptr [rbx+0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4b, 0x08], |a| { a.vmovpd_ymm_mem(1, 3, 8); });
}

#[test]
fn vmovpd_ymm_mem_rbx_1234() {
    // vmovupd ymm1, ymmword ptr [rbx+0x1234]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_mem(1, 3, 4660); });
}

#[test]
fn vmovpd_ymm_mem_rbx_m8() {
    // vmovupd ymm1, ymmword ptr [rbx-0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4b, 0xf8], |a| { a.vmovpd_ymm_mem(1, 3, -8); });
}

#[test]
fn vmovpd_ymm_mem_rsp_8() {
    // vmovupd ymm1, ymmword ptr [rsp+0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4c, 0x24, 0x08], |a| { a.vmovpd_ymm_mem(1, 4, 8); });
}

#[test]
fn vmovpd_ymm_mem_rsp_1234() {
    // vmovupd ymm1, ymmword ptr [rsp+0x1234]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_mem(1, 4, 4660); });
}

#[test]
fn vmovpd_ymm_mem_rsp_m8() {
    // vmovupd ymm1, ymmword ptr [rsp-0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4c, 0x24, 0xf8], |a| { a.vmovpd_ymm_mem(1, 4, -8); });
}

#[test]
fn vmovpd_ymm_mem_rbp_8() {
    // vmovupd ymm1, ymmword ptr [rbp+0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4d, 0x08], |a| { a.vmovpd_ymm_mem(1, 5, 8); });
}

#[test]
fn vmovpd_ymm_mem_rbp_1234() {
    // vmovupd ymm1, ymmword ptr [rbp+0x1234]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_mem(1, 5, 4660); });
}

#[test]
fn vmovpd_ymm_mem_rbp_m8() {
    // vmovupd ymm1, ymmword ptr [rbp-0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4d, 0xf8], |a| { a.vmovpd_ymm_mem(1, 5, -8); });
}

#[test]
fn vmovpd_ymm_mem_r9_8() {
    // vmovupd ymm1, ymmword ptr [r9+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x49, 0x08], |a| { a.vmovpd_ymm_mem(1, 9, 8); });
}

#[test]
fn vmovpd_ymm_mem_r9_1234() {
    // vmovupd ymm1, ymmword ptr [r9+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_mem(1, 9, 4660); });
}

#[test]
fn vmovpd_ymm_mem_r9_m8() {
    // vmovupd ymm1, ymmword ptr [r9-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x49, 0xf8], |a| { a.vmovpd_ymm_mem(1, 9, -8); });
}

#[test]
fn vmovpd_ymm_mem_r12_8() {
    // vmovupd ymm1, ymmword ptr [r12+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x4c, 0x24, 0x08], |a| { a.vmovpd_ymm_mem(1, 12, 8); });
}

#[test]
fn vmovpd_ymm_mem_r12_1234() {
    // vmovupd ymm1, ymmword ptr [r12+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_mem(1, 12, 4660); });
}

#[test]
fn vmovpd_ymm_mem_r12_m8() {
    // vmovupd ymm1, ymmword ptr [r12-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x4c, 0x24, 0xf8], |a| { a.vmovpd_ymm_mem(1, 12, -8); });
}

#[test]
fn vmovpd_ymm_mem_r13_8() {
    // vmovupd ymm1, ymmword ptr [r13+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x4d, 0x08], |a| { a.vmovpd_ymm_mem(1, 13, 8); });
}

#[test]
fn vmovpd_ymm_mem_r13_1234() {
    // vmovupd ymm1, ymmword ptr [r13+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_mem(1, 13, 4660); });
}

#[test]
fn vmovpd_ymm_mem_r13_m8() {
    // vmovupd ymm1, ymmword ptr [r13-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x10, 0x4d, 0xf8], |a| { a.vmovpd_ymm_mem(1, 13, -8); });
}

#[test]
fn vmovpd_mem_ymm_rbx_8() {
    // vmovupd ymmword ptr [rbx+0x8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4b, 0x08], |a| { a.vmovpd_mem_ymm(3, 8, 1); });
}

#[test]
fn vmovpd_mem_ymm_rbx_1234() {
    // vmovupd ymmword ptr [rbx+0x1234], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_mem_ymm(3, 4660, 1); });
}

#[test]
fn vmovpd_mem_ymm_rbx_m8() {
    // vmovupd ymmword ptr [rbx-0x8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4b, 0xf8], |a| { a.vmovpd_mem_ymm(3, -8, 1); });
}

#[test]
fn vmovpd_mem_ymm_rsp_8() {
    // vmovupd ymmword ptr [rsp+0x8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4c, 0x24, 0x08], |a| { a.vmovpd_mem_ymm(4, 8, 1); });
}

#[test]
fn vmovpd_mem_ymm_rsp_1234() {
    // vmovupd ymmword ptr [rsp+0x1234], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_mem_ymm(4, 4660, 1); });
}

#[test]
fn vmovpd_mem_ymm_rsp_m8() {
    // vmovupd ymmword ptr [rsp-0x8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4c, 0x24, 0xf8], |a| { a.vmovpd_mem_ymm(4, -8, 1); });
}

#[test]
fn vmovpd_mem_ymm_rbp_8() {
    // vmovupd ymmword ptr [rbp+0x8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4d, 0x08], |a| { a.vmovpd_mem_ymm(5, 8, 1); });
}

#[test]
fn vmovpd_mem_ymm_rbp_1234() {
    // vmovupd ymmword ptr [rbp+0x1234], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_mem_ymm(5, 4660, 1); });
}

#[test]
fn vmovpd_mem_ymm_rbp_m8() {
    // vmovupd ymmword ptr [rbp-0x8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4d, 0xf8], |a| { a.vmovpd_mem_ymm(5, -8, 1); });
}

#[test]
fn vmovpd_mem_ymm_r9_8() {
    // vmovupd ymmword ptr [r9+0x8], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x49, 0x08], |a| { a.vmovpd_mem_ymm(9, 8, 1); });
}

#[test]
fn vmovpd_mem_ymm_r9_1234() {
    // vmovupd ymmword ptr [r9+0x1234], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_mem_ymm(9, 4660, 1); });
}

#[test]
fn vmovpd_mem_ymm_r9_m8() {
    // vmovupd ymmword ptr [r9-0x8], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x49, 0xf8], |a| { a.vmovpd_mem_ymm(9, -8, 1); });
}

#[test]
fn vmovpd_mem_ymm_r12_8() {
    // vmovupd ymmword ptr [r12+0x8], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x4c, 0x24, 0x08], |a| { a.vmovpd_mem_ymm(12, 8, 1); });
}

#[test]
fn vmovpd_mem_ymm_r12_1234() {
    // vmovupd ymmword ptr [r12+0x1234], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_mem_ymm(12, 4660, 1); });
}

#[test]
fn vmovpd_mem_ymm_r12_m8() {
    // vmovupd ymmword ptr [r12-0x8], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x4c, 0x24, 0xf8], |a| { a.vmovpd_mem_ymm(12, -8, 1); });
}

#[test]
fn vmovpd_mem_ymm_r13_8() {
    // vmovupd ymmword ptr [r13+0x8], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x4d, 0x08], |a| { a.vmovpd_mem_ymm(13, 8, 1); });
}

#[test]
fn vmovpd_mem_ymm_r13_1234() {
    // vmovupd ymmword ptr [r13+0x1234], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_mem_ymm(13, 4660, 1); });
}

#[test]
fn vmovpd_mem_ymm_r13_m8() {
    // vmovupd ymmword ptr [r13-0x8], ymm1
    check(DataType::F64, &[0xc4, 0xc1, 0x7d, 0x11, 0x4d, 0xf8], |a| { a.vmovpd_mem_ymm(13, -8, 1); });
}

#[test]
fn vmovdd_xmm_mem_rbx_8() {
    // vmovupd xmm1, xmmword ptr [rbx+0x8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4b, 0x08], |a| { a.vmovdd_xmm_mem(1, 3, 8); });
}

#[test]
fn vmovdd_xmm_mem_rbx_1234() {
    // vmovupd xmm1, xmmword ptr [rbx+0x1234]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_xmm_mem(1, 3, 4660); });
}

#[test]
fn vmovdd_xmm_mem_rbx_m8() {
    // vmovupd xmm1, xmmword ptr [rbx-0x8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4b, 0xf8], |a| { a.vmovdd_xmm_mem(1, 3, -8); });
}

#[test]
fn vmovdd_xmm_mem_rsp_8() {
    // vmovupd xmm1, xmmword ptr [rsp+0x8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4c, 0x24, 0x08], |a| { a.vmovdd_xmm_mem(1, 4, 8); });
}

#[test]
fn vmovdd_xmm_mem_rsp_1234() {
    // vmovupd xmm1, xmmword ptr [rsp+0x1234]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_xmm_mem(1, 4, 4660); });
}

#[test]
fn vmovdd_xmm_mem_rsp_m8() {
    // vmovupd xmm1, xmmword ptr [rsp-0x8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4c, 0x24, 0xf8], |a| { a.vmovdd_xmm_mem(1, 4, -8); });
}

#[test]
fn vmovdd_xmm_mem_rbp_8() {
    // vmovupd xmm1, xmmword ptr [rbp+0x8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4d, 0x08], |a| { a.vmovdd_xmm_mem(1, 5, 8); });
}

#[test]
fn vmovdd_xmm_mem_rbp_1234() {
    // vmovupd xmm1, xmmword ptr [rbp+0x1234]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_xmm_mem(1, 5, 4660); });
}

#[test]
fn vmovdd_xmm_mem_rbp_m8() {
    // vmovupd xmm1, xmmword ptr [rbp-0x8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4d, 0xf8], |a| { a.vmovdd_xmm_mem(1, 5, -8); });
}

#[test]
fn vmovdd_xmm_mem_r9_8() {
    // vmovupd xmm1, xmmword ptr [r9+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x49, 0x08], |a| { a.vmovdd_xmm_mem(1, 9, 8); });
}

#[test]
fn vmovdd_xmm_mem_r9_1234() {
    // vmovupd xmm1, xmmword ptr [r9+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_xmm_mem(1, 9, 4660); });
}

#[test]
fn vmovdd_xmm_mem_r9_m8() {
    // vmovupd xmm1, xmmword ptr [r9-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x49, 0xf8], |a| { a.vmovdd_xmm_mem(1, 9, -8); });
}

#[test]
fn vmovdd_xmm_mem_r12_8() {
    // vmovupd xmm1, xmmword ptr [r12+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x4c, 0x24, 0x08], |a| { a.vmovdd_xmm_mem(1, 12, 8); });
}

#[test]
fn vmovdd_xmm_mem_r12_1234() {
    // vmovupd xmm1, xmmword ptr [r12+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_xmm_mem(1, 12, 4660); });
}

#[test]
fn vmovdd_xmm_mem_r12_m8() {
    // vmovupd xmm1, xmmword ptr [r12-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x4c, 0x24, 0xf8], |a| { a.vmovdd_xmm_mem(1, 12, -8); });
}

#[test]
fn vmovdd_xmm_mem_r13_8() {
    // vmovupd xmm1, xmmword ptr [r13+0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x4d, 0x08], |a| { a.vmovdd_xmm_mem(1, 13, 8); });
}

#[test]
fn vmovdd_xmm_mem_r13_1234() {
    // vmovupd xmm1, xmmword ptr [r13+0x1234]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_xmm_mem(1, 13, 4660); });
}

#[test]
fn vmovdd_xmm_mem_r13_m8() {
    // vmovupd xmm1, xmmword ptr [r13-0x8]
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x10, 0x4d, 0xf8], |a| { a.vmovdd_xmm_mem(1, 13, -8); });
}

#[test]
fn vmovdd_mem_xmm_rbx_8() {
    // vmovupd xmmword ptr [rbx+0x8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4b, 0x08], |a| { a.vmovdd_mem_xmm(3, 8, 1); });
}

#[test]
fn vmovdd_mem_xmm_rbx_1234() {
    // vmovupd xmmword ptr [rbx+0x1234], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_mem_xmm(3, 4660, 1); });
}

#[test]
fn vmovdd_mem_xmm_rbx_m8() {
    // vmovupd xmmword ptr [rbx-0x8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4b, 0xf8], |a| { a.vmovdd_mem_xmm(3, -8, 1); });
}

#[test]
fn vmovdd_mem_xmm_rsp_8() {
    // vmovupd xmmword ptr [rsp+0x8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4c, 0x24, 0x08], |a| { a.vmovdd_mem_xmm(4, 8, 1); });
}

#[test]
fn vmovdd_mem_xmm_rsp_1234() {
    // vmovupd xmmword ptr [rsp+0x1234], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_mem_xmm(4, 4660, 1); });
}

#[test]
fn vmovdd_mem_xmm_rsp_m8() {
    // vmovupd xmmword ptr [rsp-0x8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4c, 0x24, 0xf8], |a| { a.vmovdd_mem_xmm(4, -8, 1); });
}

#[test]
fn vmovdd_mem_xmm_rbp_8() {
    // vmovupd xmmword ptr [rbp+0x8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4d, 0x08], |a| { a.vmovdd_mem_xmm(5, 8, 1); });
}

#[test]
fn vmovdd_mem_xmm_rbp_1234() {
    // vmovupd xmmword ptr [rbp+0x1234], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_mem_xmm(5, 4660, 1); });
}

#[test]
fn vmovdd_mem_xmm_rbp_m8() {
    // vmovupd xmmword ptr [rbp-0x8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4d, 0xf8], |a| { a.vmovdd_mem_xmm(5, -8, 1); });
}

#[test]
fn vmovdd_mem_xmm_r9_8() {
    // vmovupd xmmword ptr [r9+0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x49, 0x08], |a| { a.vmovdd_mem_xmm(9, 8, 1); });
}

#[test]
fn vmovdd_mem_xmm_r9_1234() {
    // vmovupd xmmword ptr [r9+0x1234], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_mem_xmm(9, 4660, 1); });
}

#[test]
fn vmovdd_mem_xmm_r9_m8() {
    // vmovupd xmmword ptr [r9-0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x49, 0xf8], |a| { a.vmovdd_mem_xmm(9, -8, 1); });
}

#[test]
fn vmovdd_mem_xmm_r12_8() {
    // vmovupd xmmword ptr [r12+0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x4c, 0x24, 0x08], |a| { a.vmovdd_mem_xmm(12, 8, 1); });
}

#[test]
fn vmovdd_mem_xmm_r12_1234() {
    // vmovupd xmmword ptr [r12+0x1234], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_mem_xmm(12, 4660, 1); });
}

#[test]
fn vmovdd_mem_xmm_r12_m8() {
    // vmovupd xmmword ptr [r12-0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x4c, 0x24, 0xf8], |a| { a.vmovdd_mem_xmm(12, -8, 1); });
}

#[test]
fn vmovdd_mem_xmm_r13_8() {
    // vmovupd xmmword ptr [r13+0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x4d, 0x08], |a| { a.vmovdd_mem_xmm(13, 8, 1); });
}

#[test]
fn vmovdd_mem_xmm_r13_1234() {
    // vmovupd xmmword ptr [r13+0x1234], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovdd_mem_xmm(13, 4660, 1); });
}

#[test]
fn vmovdd_mem_xmm_r13_m8() {
    // vmovupd xmmword ptr [r13-0x8], xmm1
    check(DataType::F64, &[0xc4, 0xc1, 0x79, 0x11, 0x4d, 0xf8], |a| { a.vmovdd_mem_xmm(13, -8, 1); });
}

#[test]
fn movsd_xmm_mem_rbx_8() {
    // movsd xmm1, qword ptr [rbx+0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4b, 0x08], |a| { a.movsd_xmm_mem(1, 3, 8); });
}

#[test]
fn movsd_xmm_mem_rbx_1234() {
    // movsd xmm1, qword ptr [rbx+0x1234]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_xmm_mem(1, 3, 4660); });
}

#[test]
fn movsd_xmm_mem_rbx_m8() {
    // movsd xmm1, qword ptr [rbx-0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4b, 0xf8], |a| { a.movsd_xmm_mem(1, 3, -8); });
}

#[test]
fn movsd_xmm_mem_rsp_8() {
    // movsd xmm1, qword ptr [rsp+0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4c, 0x24, 0x08], |a| { a.movsd_xmm_mem(1, 4, 8); });
}

#[test]
fn movsd_xmm_mem_rsp_1234() {
    // movsd xmm1, qword ptr [rsp+0x1234]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_xmm_mem(1, 4, 4660); });
}

#[test]
fn movsd_xmm_mem_rsp_m8() {
    // movsd xmm1, qword ptr [rsp-0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4c, 0x24, 0xf8], |a| { a.movsd_xmm_mem(1, 4, -8); });
}

#[test]
fn movsd_xmm_mem_rbp_8() {
    // movsd xmm1, qword ptr [rbp+0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4d, 0x08], |a| { a.movsd_xmm_mem(1, 5, 8); });
}

#[test]
fn movsd_xmm_mem_rbp_1234() {
    // movsd xmm1, qword ptr [rbp+0x1234]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_xmm_mem(1, 5, 4660); });
}

#[test]
fn movsd_xmm_mem_rbp_m8() {
    // movsd xmm1, qword ptr [rbp-0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4d, 0xf8], |a| { a.movsd_xmm_mem(1, 5, -8); });
}

#[test]
fn movsd_xmm_mem_r9_8() {
    // movsd xmm1, qword ptr [r9+0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x49, 0x08], |a| { a.movsd_xmm_mem(1, 9, 8); });
}

#[test]
fn movsd_xmm_mem_r9_1234() {
    // movsd xmm1, qword ptr [r9+0x1234]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_xmm_mem(1, 9, 4660); });
}

#[test]
fn movsd_xmm_mem_r9_m8() {
    // movsd xmm1, qword ptr [r9-0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x49, 0xf8], |a| { a.movsd_xmm_mem(1, 9, -8); });
}

#[test]
fn movsd_xmm_mem_r12_8() {
    // movsd xmm1, qword ptr [r12+0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x4c, 0x24, 0x08], |a| { a.movsd_xmm_mem(1, 12, 8); });
}

#[test]
fn movsd_xmm_mem_r12_1234() {
    // movsd xmm1, qword ptr [r12+0x1234]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_xmm_mem(1, 12, 4660); });
}

#[test]
fn movsd_xmm_mem_r12_m8() {
    // movsd xmm1, qword ptr [r12-0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x4c, 0x24, 0xf8], |a| { a.movsd_xmm_mem(1, 12, -8); });
}

#[test]
fn movsd_xmm_mem_r13_8() {
    // movsd xmm1, qword ptr [r13+0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x4d, 0x08], |a| { a.movsd_xmm_mem(1, 13, 8); });
}

#[test]
fn movsd_xmm_mem_r13_1234() {
    // movsd xmm1, qword ptr [r13+0x1234]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_xmm_mem(1, 13, 4660); });
}

#[test]
fn movsd_xmm_mem_r13_m8() {
    // movsd xmm1, qword ptr [r13-0x8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x10, 0x4d, 0xf8], |a| { a.movsd_xmm_mem(1, 13, -8); });
}

#[test]
fn movsd_mem_xmm_rbx_8() {
    // movsd qword ptr [rbx+0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4b, 0x08], |a| { a.movsd_mem_xmm(3, 8, 1); });
}

#[test]
fn movsd_mem_xmm_rbx_1234() {
    // movsd qword ptr [rbx+0x1234], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_mem_xmm(3, 4660, 1); });
}

#[test]
fn movsd_mem_xmm_rbx_m8() {
    // movsd qword ptr [rbx-0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4b, 0xf8], |a| { a.movsd_mem_xmm(3, -8, 1); });
}

#[test]
fn movsd_mem_xmm_rsp_8() {
    // movsd qword ptr [rsp+0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4c, 0x24, 0x08], |a| { a.movsd_mem_xmm(4, 8, 1); });
}

#[test]
fn movsd_mem_xmm_rsp_1234() {
    // movsd qword ptr [rsp+0x1234], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_mem_xmm(4, 4660, 1); });
}

#[test]
fn movsd_mem_xmm_rsp_m8() {
    // movsd qword ptr [rsp-0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4c, 0x24, 0xf8], |a| { a.movsd_mem_xmm(4, -8, 1); });
}

#[test]
fn movsd_mem_xmm_rbp_8() {
    // movsd qword ptr [rbp+0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4d, 0x08], |a| { a.movsd_mem_xmm(5, 8, 1); });
}

#[test]
fn movsd_mem_xmm_rbp_1234() {
    // movsd qword ptr [rbp+0x1234], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_mem_xmm(5, 4660, 1); });
}

#[test]
fn movsd_mem_xmm_rbp_m8() {
    // movsd qword ptr [rbp-0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4d, 0xf8], |a| { a.movsd_mem_xmm(5, -8, 1); });
}

#[test]
fn movsd_mem_xmm_r9_8() {
    // movsd qword ptr [r9+0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x49, 0x08], |a| { a.movsd_mem_xmm(9, 8, 1); });
}

#[test]
fn movsd_mem_xmm_r9_1234() {
    // movsd qword ptr [r9+0x1234], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_mem_xmm(9, 4660, 1); });
}

#[test]
fn movsd_mem_xmm_r9_m8() {
    // movsd qword ptr [r9-0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x49, 0xf8], |a| { a.movsd_mem_xmm(9, -8, 1); });
}

#[test]
fn movsd_mem_xmm_r12_8() {
    // movsd qword ptr [r12+0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x4c, 0x24, 0x08], |a| { a.movsd_mem_xmm(12, 8, 1); });
}

#[test]
fn movsd_mem_xmm_r12_1234() {
    // movsd qword ptr [r12+0x1234], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_mem_xmm(12, 4660, 1); });
}

#[test]
fn movsd_mem_xmm_r12_m8() {
    // movsd qword ptr [r12-0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x4c, 0x24, 0xf8], |a| { a.movsd_mem_xmm(12, -8, 1); });
}

#[test]
fn movsd_mem_xmm_r13_8() {
    // movsd qword ptr [r13+0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x4d, 0x08], |a| { a.movsd_mem_xmm(13, 8, 1); });
}

#[test]
fn movsd_mem_xmm_r13_1234() {
    // movsd qword ptr [r13+0x1234], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.movsd_mem_xmm(13, 4660, 1); });
}

#[test]
fn movsd_mem_xmm_r13_m8() {
    // movsd qword ptr [r13-0x8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x49, 0x0f, 0x11, 0x4d, 0xf8], |a| { a.movsd_mem_xmm(13, -8, 1); });
}

#[test]
fn vinsertf128_mem_rbx_8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rbx+0x8], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x4b, 0x08, 0x01], |a| { a.vinsertf128_mem(1, 2, 3, 8, 1); });
}

#[test]
fn vinsertf128_mem_rbx_1234() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rbx+0x1234], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x8b, 0x34, 0x12, 0x00, 0x00, 0x01], |a| { a.vinsertf128_mem(1, 2, 3, 4660, 1); });
}

#[test]
fn vinsertf128_mem_rbx_m8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rbx-0x8], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x4b, 0xf8, 0x01], |a| { a.vinsertf128_mem(1, 2, 3, -8, 1); });
}

#[test]
fn vinsertf128_mem_rsp_8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rsp+0x8], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x4c, 0x24, 0x08, 0x01], |a| { a.vinsertf128_mem(1, 2, 4, 8, 1); });
}

#[test]
fn vinsertf128_mem_rsp_1234() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rsp+0x1234], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00, 0x01], |a| { a.vinsertf128_mem(1, 2, 4, 4660, 1); });
}

#[test]
fn vinsertf128_mem_rsp_m8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rsp-0x8], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x4c, 0x24, 0xf8, 0x01], |a| { a.vinsertf128_mem(1, 2, 4, -8, 1); });
}

#[test]
fn vinsertf128_mem_rbp_8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rbp+0x8], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x4d, 0x08, 0x01], |a| { a.vinsertf128_mem(1, 2, 5, 8, 1); });
}

#[test]
fn vinsertf128_mem_rbp_1234() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rbp+0x1234], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x8d, 0x34, 0x12, 0x00, 0x00, 0x01], |a| { a.vinsertf128_mem(1, 2, 5, 4660, 1); });
}

#[test]
fn vinsertf128_mem_rbp_m8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [rbp-0x8], 1
    check(DataType::F64, &[0xc4, 0xe3, 0x6d, 0x18, 0x4d, 0xf8, 0x01], |a| { a.vinsertf128_mem(1, 2, 5, -8, 1); });
}

#[test]
fn vinsertf128_mem_r9_8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r9+0x8], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x49, 0x08, 0x01], |a| { a.vinsertf128_mem(1, 2, 9, 8, 1); });
}

#[test]
fn vinsertf128_mem_r9_1234() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r9+0x1234], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x89, 0x34, 0x12, 0x00, 0x00, 0x01], |a| { a.vinsertf128_mem(1, 2, 9, 4660, 1); });
}

#[test]
fn vinsertf128_mem_r9_m8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r9-0x8], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x49, 0xf8, 0x01], |a| { a.vinsertf128_mem(1, 2, 9, -8, 1); });
}

#[test]
fn vinsertf128_mem_r12_8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r12+0x8], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x4c, 0x24, 0x08, 0x01], |a| { a.vinsertf128_mem(1, 2, 12, 8, 1); });
}

#[test]
fn vinsertf128_mem_r12_1234() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r12+0x1234], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00, 0x01], |a| { a.vinsertf128_mem(1, 2, 12, 4660, 1); });
}

#[test]
fn vinsertf128_mem_r12_m8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r12-0x8], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x4c, 0x24, 0xf8, 0x01], |a| { a.vinsertf128_mem(1, 2, 12, -8, 1); });
}

#[test]
fn vinsertf128_mem_r13_8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r13+0x8], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x4d, 0x08, 0x01], |a| { a.vinsertf128_mem(1, 2, 13, 8, 1); });
}

#[test]
fn vinsertf128_mem_r13_1234() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r13+0x1234], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x8d, 0x34, 0x12, 0x00, 0x00, 0x01], |a| { a.vinsertf128_mem(1, 2, 13, 4660, 1); });
}

#[test]
fn vinsertf128_mem_r13_m8() {
    // vinsertf128 ymm1, ymm2, xmmword ptr [r13-0x8], 1
    check(DataType::F64, &[0xc4, 0xc3, 0x6d, 0x18, 0x4d, 0xf8, 0x01], |a| { a.vinsertf128_mem(1, 2, 13, -8, 1); });
}

#[test]
fn vmovqd_zmm_mem_rbx_8() {
    // vmovupd zmm1, zmmword ptr [rbx+0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8b, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 3, 8); });
}

#[test]
fn vmovqd_zmm_mem_rbx_80() {
    // vmovupd zmm1, zmmword ptr [rbx+0x80]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4b, 0x02], |a| { a.vmovqd_zmm_mem(1, 3, 128); });
}

#[test]
fn vmovqd_zmm_mem_rbx_1234() {
    // vmovupd zmm1, zmmword ptr [rbx+0x1234]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 3, 4660); });
}

#[test]
fn vmovqd_zmm_mem_rbx_m8() {
    // vmovupd zmm1, zmmword ptr [rbx-0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8b, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_mem(1, 3, -8); });
}

#[test]
fn vmovqd_zmm_mem_rbx_m80() {
    // vmovupd zmm1, zmmword ptr [rbx-0x80]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4b, 0xfe], |a| { a.vmovqd_zmm_mem(1, 3, -128); });
}

#[test]
fn vmovqd_zmm_mem_rsp_8() {
    // vmovupd zmm1, zmmword ptr [rsp+0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8c, 0x24, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 4, 8); });
}

#[test]
fn vmovqd_zmm_mem_rsp_80() {
    // vmovupd zmm1, zmmword ptr [rsp+0x80]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4c, 0x24, 0x02], |a| { a.vmovqd_zmm_mem(1, 4, 128); });
}

#[test]
fn vmovqd_zmm_mem_rsp_1234() {
    // vmovupd zmm1, zmmword ptr [rsp+0x1234]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 4, 4660); });
}

#[test]
fn vmovqd_zmm_mem_rsp_m8() {
    // vmovupd zmm1, zmmword ptr [rsp-0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8c, 0x24, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_mem(1, 4, -8); });
}

#[test]
fn vmovqd_zmm_mem_rsp_m80() {
    // vmovupd zmm1, zmmword ptr [rsp-0x80]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4c, 0x24, 0xfe], |a| { a.vmovqd_zmm_mem(1, 4, -128); });
}

#[test]
fn vmovqd_zmm_mem_rbp_8() {
    // vmovupd zmm1, zmmword ptr [rbp+0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8d, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 5, 8); });
}

#[test]
fn vmovqd_zmm_mem_rbp_80() {
    // vmovupd zmm1, zmmword ptr [rbp+0x80]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4d, 0x02], |a| { a.vmovqd_zmm_mem(1, 5, 128); });
}

#[test]
fn vmovqd_zmm_mem_rbp_1234() {
    // vmovupd zmm1, zmmword ptr [rbp+0x1234]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 5, 4660); });
}

#[test]
fn vmovqd_zmm_mem_rbp_m8() {
    // vmovupd zmm1, zmmword ptr [rbp-0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8d, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_mem(1, 5, -8); });
}

#[test]
fn vmovqd_zmm_mem_rbp_m80() {
    // vmovupd zmm1, zmmword ptr [rbp-0x80]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4d, 0xfe], |a| { a.vmovqd_zmm_mem(1, 5, -128); });
}

#[test]
fn vmovqd_zmm_mem_r9_8() {
    // vmovupd zmm1, zmmword ptr [r9+0x8]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x89, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 9, 8); });
}

#[test]
fn vmovqd_zmm_mem_r9_80() {
    // vmovupd zmm1, zmmword ptr [r9+0x80]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x49, 0x02], |a| { a.vmovqd_zmm_mem(1, 9, 128); });
}

#[test]
fn vmovqd_zmm_mem_r9_1234() {
    // vmovupd zmm1, zmmword ptr [r9+0x1234]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 9, 4660); });
}

#[test]
fn vmovqd_zmm_mem_r9_m8() {
    // vmovupd zmm1, zmmword ptr [r9-0x8]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x89, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_mem(1, 9, -8); });
}

#[test]
fn vmovqd_zmm_mem_r9_m80() {
    // vmovupd zmm1, zmmword ptr [r9-0x80]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x49, 0xfe], |a| { a.vmovqd_zmm_mem(1, 9, -128); });
}

#[test]
fn vmovqd_zmm_mem_r12_8() {
    // vmovupd zmm1, zmmword ptr [r12+0x8]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x8c, 0x24, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 12, 8); });
}

#[test]
fn vmovqd_zmm_mem_r12_80() {
    // vmovupd zmm1, zmmword ptr [r12+0x80]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x4c, 0x24, 0x02], |a| { a.vmovqd_zmm_mem(1, 12, 128); });
}

#[test]
fn vmovqd_zmm_mem_r12_1234() {
    // vmovupd zmm1, zmmword ptr [r12+0x1234]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 12, 4660); });
}

#[test]
fn vmovqd_zmm_mem_r12_m8() {
    // vmovupd zmm1, zmmword ptr [r12-0x8]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x8c, 0x24, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_mem(1, 12, -8); });
}

#[test]
fn vmovqd_zmm_mem_r12_m80() {
    // vmovupd zmm1, zmmword ptr [r12-0x80]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x4c, 0x24, 0xfe], |a| { a.vmovqd_zmm_mem(1, 12, -128); });
}

#[test]
fn vmovqd_zmm_mem_r13_8() {
    // vmovupd zmm1, zmmword ptr [r13+0x8]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x8d, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 13, 8); });
}

#[test]
fn vmovqd_zmm_mem_r13_80() {
    // vmovupd zmm1, zmmword ptr [r13+0x80]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x4d, 0x02], |a| { a.vmovqd_zmm_mem(1, 13, 128); });
}

#[test]
fn vmovqd_zmm_mem_r13_1234() {
    // vmovupd zmm1, zmmword ptr [r13+0x1234]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 13, 4660); });
}

#[test]
fn vmovqd_zmm_mem_r13_m8() {
    // vmovupd zmm1, zmmword ptr [r13-0x8]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x8d, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_mem(1, 13, -8); });
}

#[test]
fn vmovqd_zmm_mem_r13_m80() {
    // vmovupd zmm1, zmmword ptr [r13-0x80]
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x10, 0x4d, 0xfe], |a| { a.vmovqd_zmm_mem(1, 13, -128); });
}

#[test]
fn vmovqd_mem_zmm_rbx_8() {
    // vmovupd zmmword ptr [rbx+0x8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8b, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(3, 8, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbx_80() {
    // vmovupd zmmword ptr [rbx+0x80], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4b, 0x02], |a| { a.vmovqd_mem_zmm(3, 128, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbx_1234() {
    // vmovupd zmmword ptr [rbx+0x1234], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(3, 4660, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbx_m8() {
    // vmovupd zmmword ptr [rbx-0x8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8b, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_mem_zmm(3, -8, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbx_m80() {
    // vmovupd zmmword ptr [rbx-0x80], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4b, 0xfe], |a| { a.vmovqd_mem_zmm(3, -128, 1); });
}

#[test]
fn vmovqd_mem_zmm_rsp_8() {
    // vmovupd zmmword ptr [rsp+0x8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8c, 0x24, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(4, 8, 1); });
}

#[test]
fn vmovqd_mem_zmm_rsp_80() {
    // vmovupd zmmword ptr [rsp+0x80], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4c, 0x24, 0x02], |a| { a.vmovqd_mem_zmm(4, 128, 1); });
}

#[test]
fn vmovqd_mem_zmm_rsp_1234() {
    // vmovupd zmmword ptr [rsp+0x1234], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(4, 4660, 1); });
}

#[test]
fn vmovqd_mem_zmm_rsp_m8() {
    // vmovupd zmmword ptr [rsp-0x8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8c, 0x24, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_mem_zmm(4, -8, 1); });
}

#[test]
fn vmovqd_mem_zmm_rsp_m80() {
    // vmovupd zmmword ptr [rsp-0x80], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4c, 0x24, 0xfe], |a| { a.vmovqd_mem_zmm(4, -128, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbp_8() {
    // vmovupd zmmword ptr [rbp+0x8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8d, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(5, 8, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbp_80() {
    // vmovupd zmmword ptr [rbp+0x80], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4d, 0x02], |a| { a.vmovqd_mem_zmm(5, 128, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbp_1234() {
    // vmovupd zmmword ptr [rbp+0x1234], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(5, 4660, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbp_m8() {
    // vmovupd zmmword ptr [rbp-0x8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x8d, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_mem_zmm(5, -8, 1); });
}

#[test]
fn vmovqd_mem_zmm_rbp_m80() {
    // vmovupd zmmword ptr [rbp-0x80], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4d, 0xfe], |a| { a.vmovqd_mem_zmm(5, -128, 1); });
}

#[test]
fn vmovqd_mem_zmm_r9_8() {
    // vmovupd zmmword ptr [r9+0x8], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x89, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(9, 8, 1); });
}

#[test]
fn vmovqd_mem_zmm_r9_80() {
    // vmovupd zmmword ptr [r9+0x80], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x49, 0x02], |a| { a.vmovqd_mem_zmm(9, 128, 1); });
}

#[test]
fn vmovqd_mem_zmm_r9_1234() {
    // vmovupd zmmword ptr [r9+0x1234], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(9, 4660, 1); });
}

#[test]
fn vmovqd_mem_zmm_r9_m8() {
    // vmovupd zmmword ptr [r9-0x8], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x89, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_mem_zmm(9, -8, 1); });
}

#[test]
fn vmovqd_mem_zmm_r9_m80() {
    // vmovupd zmmword ptr [r9-0x80], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x49, 0xfe], |a| { a.vmovqd_mem_zmm(9, -128, 1); });
}

#[test]
fn vmovqd_mem_zmm_r12_8() {
    // vmovupd zmmword ptr [r12+0x8], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x8c, 0x24, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(12, 8, 1); });
}

#[test]
fn vmovqd_mem_zmm_r12_80() {
    // vmovupd zmmword ptr [r12+0x80], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x4c, 0x24, 0x02], |a| { a.vmovqd_mem_zmm(12, 128, 1); });
}

#[test]
fn vmovqd_mem_zmm_r12_1234() {
    // vmovupd zmmword ptr [r12+0x1234], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(12, 4660, 1); });
}

#[test]
fn vmovqd_mem_zmm_r12_m8() {
    // vmovupd zmmword ptr [r12-0x8], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x8c, 0x24, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_mem_zmm(12, -8, 1); });
}

#[test]
fn vmovqd_mem_zmm_r12_m80() {
    // vmovupd zmmword ptr [r12-0x80], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x4c, 0x24, 0xfe], |a| { a.vmovqd_mem_zmm(12, -128, 1); });
}

#[test]
fn vmovqd_mem_zmm_r13_8() {
    // vmovupd zmmword ptr [r13+0x8], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x8d, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(13, 8, 1); });
}

#[test]
fn vmovqd_mem_zmm_r13_80() {
    // vmovupd zmmword ptr [r13+0x80], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x4d, 0x02], |a| { a.vmovqd_mem_zmm(13, 128, 1); });
}

#[test]
fn vmovqd_mem_zmm_r13_1234() {
    // vmovupd zmmword ptr [r13+0x1234], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_mem_zmm(13, 4660, 1); });
}

#[test]
fn vmovqd_mem_zmm_r13_m8() {
    // vmovupd zmmword ptr [r13-0x8], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x8d, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_mem_zmm(13, -8, 1); });
}

#[test]
fn vmovqd_mem_zmm_r13_m80() {
    // vmovupd zmmword ptr [r13-0x80], zmm1
    check(DataType::F64, &[0x62, 0xd1, 0xfd, 0x48, 0x11, 0x4d, 0xfe], |a| { a.vmovqd_mem_zmm(13, -128, 1); });
}

#[test]
fn vbroadcastsd_zmm_rbx_8() {
    // vbroadcastsd zmm1, qword ptr [rbx+0x8]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4b, 0x01], |a| { a.vbroadcastsd_zmm(1, 3, 8); });
}

#[test]
fn vbroadcastsd_zmm_rbx_1234() {
    // vbroadcastsd zmm1, qword ptr [rbx+0x1234]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x8b, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd_zmm(1, 3, 4660); });
}

#[test]
fn vbroadcastsd_zmm_rbx_m8() {
    // vbroadcastsd zmm1, qword ptr [rbx-0x8]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4b, 0xff], |a| { a.vbroadcastsd_zmm(1, 3, -8); });
}

#[test]
fn vbroadcastsd_zmm_rsp_8() {
    // vbroadcastsd zmm1, qword ptr [rsp+0x8]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4c, 0x24, 0x01], |a| { a.vbroadcastsd_zmm(1, 4, 8); });
}

#[test]
fn vbroadcastsd_zmm_rsp_1234() {
    // vbroadcastsd zmm1, qword ptr [rsp+0x1234]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd_zmm(1, 4, 4660); });
}

#[test]
fn vbroadcastsd_zmm_rsp_m8() {
    // vbroadcastsd zmm1, qword ptr [rsp-0x8]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4c, 0x24, 0xff], |a| { a.vbroadcastsd_zmm(1, 4, -8); });
}

#[test]
fn vbroadcastsd_zmm_rbp_8() {
    // vbroadcastsd zmm1, qword ptr [rbp+0x8]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4d, 0x01], |a| { a.vbroadcastsd_zmm(1, 5, 8); });
}

#[test]
fn vbroadcastsd_zmm_rbp_1234() {
    // vbroadcastsd zmm1, qword ptr [rbp+0x1234]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd_zmm(1, 5, 4660); });
}

#[test]
fn vbroadcastsd_zmm_rbp_m8() {
    // vbroadcastsd zmm1, qword ptr [rbp-0x8]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4d, 0xff], |a| { a.vbroadcastsd_zmm(1, 5, -8); });
}

#[test]
fn vbroadcastsd_zmm_r9_8() {
    // vbroadcastsd zmm1, qword ptr [r9+0x8]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x49, 0x01], |a| { a.vbroadcastsd_zmm(1, 9, 8); });
}

#[test]
fn vbroadcastsd_zmm_r9_1234() {
    // vbroadcastsd zmm1, qword ptr [r9+0x1234]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x89, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd_zmm(1, 9, 4660); });
}

#[test]
fn vbroadcastsd_zmm_r9_m8() {
    // vbroadcastsd zmm1, qword ptr [r9-0x8]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x49, 0xff], |a| { a.vbroadcastsd_zmm(1, 9, -8); });
}

#[test]
fn vbroadcastsd_zmm_r12_8() {
    // vbroadcastsd zmm1, qword ptr [r12+0x8]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x4c, 0x24, 0x01], |a| { a.vbroadcastsd_zmm(1, 12, 8); });
}

#[test]
fn vbroadcastsd_zmm_r12_1234() {
    // vbroadcastsd zmm1, qword ptr [r12+0x1234]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x8c, 0x24, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd_zmm(1, 12, 4660); });
}

#[test]
fn vbroadcastsd_zmm_r12_m8() {
    // vbroadcastsd zmm1, qword ptr [r12-0x8]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x4c, 0x24, 0xff], |a| { a.vbroadcastsd_zmm(1, 12, -8); });
}

#[test]
fn vbroadcastsd_zmm_r13_8() {
    // vbroadcastsd zmm1, qword ptr [r13+0x8]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x4d, 0x01], |a| { a.vbroadcastsd_zmm(1, 13, 8); });
}

#[test]
fn vbroadcastsd_zmm_r13_1234() {
    // vbroadcastsd zmm1, qword ptr [r13+0x1234]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vbroadcastsd_zmm(1, 13, 4660); });
}

#[test]
fn vbroadcastsd_zmm_r13_m8() {
    // vbroadcastsd zmm1, qword ptr [r13-0x8]
    check(DataType::F64, &[0x62, 0xd2, 0xfd, 0x48, 0x19, 0x4d, 0xff], |a| { a.vbroadcastsd_zmm(1, 13, -8); });
}

#[test]
fn vmovqd_zmm_mem_big_rbx_m2000() {
    // vmovupd zmm1, zmmword ptr [rbx-0x2000]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4b, 0x80], |a| { a.vmovqd_zmm_mem(1, 3, -8192); });
}

#[test]
fn vmovqd_zmm_mem_big_rbx_m4000() {
    // vmovupd zmm1, zmmword ptr [rbx-0x4000]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8b, 0x00, 0xc0, 0xff, 0xff], |a| { a.vmovqd_zmm_mem(1, 3, -16384); });
}

#[test]
fn vmovqd_zmm_mem_big_rbx_1fc0() {
    // vmovupd zmm1, zmmword ptr [rbx+0x1fc0]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4b, 0x7f], |a| { a.vmovqd_zmm_mem(1, 3, 8128); });
}

#[test]
fn vmovqd_zmm_mem_big_rbx_2000() {
    // vmovupd zmm1, zmmword ptr [rbx+0x2000]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8b, 0x00, 0x20, 0x00, 0x00], |a| { a.vmovqd_zmm_mem(1, 3, 8192); });
}

#[test]
fn vbroadcastsd_zmm_scaled_rbx_40() {
    // vbroadcastsd zmm1, qword ptr [rbx+0x40]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4b, 0x08], |a| { a.vbroadcastsd_zmm(1, 3, 64); });
}

#[test]
fn vbroadcastsd_zmm_scaled_rbx_80() {
    // vbroadcastsd zmm1, qword ptr [rbx+0x80]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4b, 0x10], |a| { a.vbroadcastsd_zmm(1, 3, 128); });
}

#[test]
fn vbroadcastsd_zmm_scaled_rbx_3f8() {
    // vbroadcastsd zmm1, qword ptr [rbx+0x3f8]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4b, 0x7f], |a| { a.vbroadcastsd_zmm(1, 3, 1016); });
}

#[test]
fn vbroadcastsd_zmm_scaled_rbx_m400() {
    // vbroadcastsd zmm1, qword ptr [rbx-0x400]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x4b, 0x80], |a| { a.vbroadcastsd_zmm(1, 3, -1024); });
}

#[test]
fn vbroadcastsd_zmm_scaled_rbx_m408() {
    // vbroadcastsd zmm1, qword ptr [rbx-0x408]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x8b, 0xf8, 0xfb, 0xff, 0xff], |a| { a.vbroadcastsd_zmm(1, 3, -1032); });
}

#[test]
fn vbroadcastsd_zmm_scaled_rbx_m800() {
    // vbroadcastsd zmm1, qword ptr [rbx-0x800]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x8b, 0x00, 0xf8, 0xff, 0xff], |a| { a.vbroadcastsd_zmm(1, 3, -2048); });
}

#[test]
fn vbroadcastsd_zmm_scaled_rbx_400() {
    // vbroadcastsd zmm1, qword ptr [rbx+0x400]
    check(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x8b, 0x00, 0x04, 0x00, 0x00], |a| { a.vbroadcastsd_zmm(1, 3, 1024); });
}

#[test]
fn lea_indexed_rbx_rdx_1() {
    // lea rcx, [rbx+rdx*1]
    check(DataType::F64, &[0x48, 0x8d, 0x0c, 0x13], |a| { a.lea_indexed(1, 3, 2, 1); });
}

#[test]
fn lea_indexed_rbx_rdx_8() {
    // lea rcx, [rbx+rdx*8]
    check(DataType::F64, &[0x48, 0x8d, 0x0c, 0xd3], |a| { a.lea_indexed(1, 3, 2, 8); });
}

#[test]
fn lea_indexed_rbp_rdx_1() {
    // lea rcx, [rbp+rdx*1]
    check(DataType::F64, &[0x48, 0x8d, 0x4c, 0x15, 0x00], |a| { a.lea_indexed(1, 5, 2, 1); });
}

#[test]
fn lea_indexed_rbp_rdx_8() {
    // lea rcx, [rbp+rdx*8]
    check(DataType::F64, &[0x48, 0x8d, 0x4c, 0xd5, 0x00], |a| { a.lea_indexed(1, 5, 2, 8); });
}

#[test]
fn lea_indexed_r13_r9_1() {
    // lea rcx, [r13+r9*1]
    check(DataType::F64, &[0x4b, 0x8d, 0x4c, 0x0d, 0x00], |a| { a.lea_indexed(1, 13, 9, 1); });
}

#[test]
fn lea_indexed_r13_r9_8() {
    // lea rcx, [r13+r9*8]
    check(DataType::F64, &[0x4b, 0x8d, 0x4c, 0xcd, 0x00], |a| { a.lea_indexed(1, 13, 9, 8); });
}

#[test]
fn lea_indexed_r12_r12_1() {
    // lea rcx, [r12+r12*1]
    check(DataType::F64, &[0x4b, 0x8d, 0x0c, 0x24], |a| { a.lea_indexed(1, 12, 12, 1); });
}

#[test]
fn lea_indexed_r12_r12_8() {
    // lea rcx, [r12+r12*8]
    check(DataType::F64, &[0x4b, 0x8d, 0x0c, 0xe4], |a| { a.lea_indexed(1, 12, 12, 8); });
}

#[test]
fn lea_indexed_rsp_rdx_1() {
    // lea rcx, [rsp+rdx*1]
    check(DataType::F64, &[0x48, 0x8d, 0x0c, 0x14], |a| { a.lea_indexed(1, 4, 2, 1); });
}

#[test]
fn lea_indexed_rsp_rdx_8() {
    // lea rcx, [rsp+rdx*8]
    check(DataType::F64, &[0x48, 0x8d, 0x0c, 0xd4], |a| { a.lea_indexed(1, 4, 2, 8); });
}

#[test]
fn lea_indexed_r9_r12_1() {
    // lea rcx, [r9+r12*1]
    check(DataType::F64, &[0x4b, 0x8d, 0x0c, 0x21], |a| { a.lea_indexed(1, 9, 12, 1); });
}

#[test]
fn lea_indexed_r9_r12_8() {
    // lea rcx, [r9+r12*8]
    check(DataType::F64, &[0x4b, 0x8d, 0x0c, 0xe1], |a| { a.lea_indexed(1, 9, 12, 8); });
}

#[test]
fn vmovsd_xmm_indexed_rbx_rdx_1() {
    // vmovsd xmm1, qword ptr [rbx+rdx*1]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x0c, 0x13], |a| { a.vmovsd_xmm_indexed(1, 3, 2, 1); });
}

#[test]
fn vmovsd_xmm_indexed_rbx_rdx_8() {
    // vmovsd xmm1, qword ptr [rbx+rdx*8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x0c, 0xd3], |a| { a.vmovsd_xmm_indexed(1, 3, 2, 8); });
}

#[test]
fn vmovsd_xmm_indexed_rbp_rdx_1() {
    // vmovsd xmm1, qword ptr [rbp+rdx*1]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4c, 0x15, 0x00], |a| { a.vmovsd_xmm_indexed(1, 5, 2, 1); });
}

#[test]
fn vmovsd_xmm_indexed_rbp_rdx_8() {
    // vmovsd xmm1, qword ptr [rbp+rdx*8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x4c, 0xd5, 0x00], |a| { a.vmovsd_xmm_indexed(1, 5, 2, 8); });
}

#[test]
fn vmovsd_xmm_indexed_r13_r9_1() {
    // vmovsd xmm1, qword ptr [r13+r9*1]
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x10, 0x4c, 0x0d, 0x00], |a| { a.vmovsd_xmm_indexed(1, 13, 9, 1); });
}

#[test]
fn vmovsd_xmm_indexed_r13_r9_8() {
    // vmovsd xmm1, qword ptr [r13+r9*8]
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x10, 0x4c, 0xcd, 0x00], |a| { a.vmovsd_xmm_indexed(1, 13, 9, 8); });
}

#[test]
fn vmovsd_xmm_indexed_r12_r12_1() {
    // vmovsd xmm1, qword ptr [r12+r12*1]
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x10, 0x0c, 0x24], |a| { a.vmovsd_xmm_indexed(1, 12, 12, 1); });
}

#[test]
fn vmovsd_xmm_indexed_r12_r12_8() {
    // vmovsd xmm1, qword ptr [r12+r12*8]
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x10, 0x0c, 0xe4], |a| { a.vmovsd_xmm_indexed(1, 12, 12, 8); });
}

#[test]
fn vmovsd_xmm_indexed_rsp_rdx_1() {
    // vmovsd xmm1, qword ptr [rsp+rdx*1]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x0c, 0x14], |a| { a.vmovsd_xmm_indexed(1, 4, 2, 1); });
}

#[test]
fn vmovsd_xmm_indexed_rsp_rdx_8() {
    // vmovsd xmm1, qword ptr [rsp+rdx*8]
    check(DataType::F64, &[0xc5, 0xfb, 0x10, 0x0c, 0xd4], |a| { a.vmovsd_xmm_indexed(1, 4, 2, 8); });
}

#[test]
fn vmovsd_xmm_indexed_r9_r12_1() {
    // vmovsd xmm1, qword ptr [r9+r12*1]
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x10, 0x0c, 0x21], |a| { a.vmovsd_xmm_indexed(1, 9, 12, 1); });
}

#[test]
fn vmovsd_xmm_indexed_r9_r12_8() {
    // vmovsd xmm1, qword ptr [r9+r12*8]
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x10, 0x0c, 0xe1], |a| { a.vmovsd_xmm_indexed(1, 9, 12, 8); });
}

#[test]
fn vmovsd_indexed_xmm_rbx_rdx_1() {
    // vmovsd qword ptr [rbx+rdx*1], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x0c, 0x13], |a| { a.vmovsd_indexed_xmm(3, 2, 1, 1); });
}

#[test]
fn vmovsd_indexed_xmm_rbx_rdx_8() {
    // vmovsd qword ptr [rbx+rdx*8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x0c, 0xd3], |a| { a.vmovsd_indexed_xmm(3, 2, 8, 1); });
}

#[test]
fn vmovsd_indexed_xmm_rbp_rdx_1() {
    // vmovsd qword ptr [rbp+rdx*1], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4c, 0x15, 0x00], |a| { a.vmovsd_indexed_xmm(5, 2, 1, 1); });
}

#[test]
fn vmovsd_indexed_xmm_rbp_rdx_8() {
    // vmovsd qword ptr [rbp+rdx*8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x4c, 0xd5, 0x00], |a| { a.vmovsd_indexed_xmm(5, 2, 8, 1); });
}

#[test]
fn vmovsd_indexed_xmm_r13_r9_1() {
    // vmovsd qword ptr [r13+r9*1], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x11, 0x4c, 0x0d, 0x00], |a| { a.vmovsd_indexed_xmm(13, 9, 1, 1); });
}

#[test]
fn vmovsd_indexed_xmm_r13_r9_8() {
    // vmovsd qword ptr [r13+r9*8], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x11, 0x4c, 0xcd, 0x00], |a| { a.vmovsd_indexed_xmm(13, 9, 8, 1); });
}

#[test]
fn vmovsd_indexed_xmm_r12_r12_1() {
    // vmovsd qword ptr [r12+r12*1], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x11, 0x0c, 0x24], |a| { a.vmovsd_indexed_xmm(12, 12, 1, 1); });
}

#[test]
fn vmovsd_indexed_xmm_r12_r12_8() {
    // vmovsd qword ptr [r12+r12*8], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x11, 0x0c, 0xe4], |a| { a.vmovsd_indexed_xmm(12, 12, 8, 1); });
}

#[test]
fn vmovsd_indexed_xmm_rsp_rdx_1() {
    // vmovsd qword ptr [rsp+rdx*1], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x0c, 0x14], |a| { a.vmovsd_indexed_xmm(4, 2, 1, 1); });
}

#[test]
fn vmovsd_indexed_xmm_rsp_rdx_8() {
    // vmovsd qword ptr [rsp+rdx*8], xmm1
    check(DataType::F64, &[0xc5, 0xfb, 0x11, 0x0c, 0xd4], |a| { a.vmovsd_indexed_xmm(4, 2, 8, 1); });
}

#[test]
fn vmovsd_indexed_xmm_r9_r12_1() {
    // vmovsd qword ptr [r9+r12*1], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x11, 0x0c, 0x21], |a| { a.vmovsd_indexed_xmm(9, 12, 1, 1); });
}

#[test]
fn vmovsd_indexed_xmm_r9_r12_8() {
    // vmovsd qword ptr [r9+r12*8], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x7b, 0x11, 0x0c, 0xe1], |a| { a.vmovsd_indexed_xmm(9, 12, 8, 1); });
}

#[test]
fn vmovpd_ymm_indexed_rbx_rdx_1() {
    // vmovupd ymm1, ymmword ptr [rbx+rdx*1]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x0c, 0x13], |a| { a.vmovpd_ymm_indexed(1, 3, 2, 1); });
}

#[test]
fn vmovpd_ymm_indexed_rbx_rdx_8() {
    // vmovupd ymm1, ymmword ptr [rbx+rdx*8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x0c, 0xd3], |a| { a.vmovpd_ymm_indexed(1, 3, 2, 8); });
}

#[test]
fn vmovpd_ymm_indexed_rbp_rdx_1() {
    // vmovupd ymm1, ymmword ptr [rbp+rdx*1]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4c, 0x15, 0x00], |a| { a.vmovpd_ymm_indexed(1, 5, 2, 1); });
}

#[test]
fn vmovpd_ymm_indexed_rbp_rdx_8() {
    // vmovupd ymm1, ymmword ptr [rbp+rdx*8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4c, 0xd5, 0x00], |a| { a.vmovpd_ymm_indexed(1, 5, 2, 8); });
}

#[test]
fn vmovpd_ymm_indexed_r13_r9_1() {
    // vmovupd ymm1, ymmword ptr [r13+r9*1]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x4c, 0x0d, 0x00], |a| { a.vmovpd_ymm_indexed(1, 13, 9, 1); });
}

#[test]
fn vmovpd_ymm_indexed_r13_r9_8() {
    // vmovupd ymm1, ymmword ptr [r13+r9*8]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x4c, 0xcd, 0x00], |a| { a.vmovpd_ymm_indexed(1, 13, 9, 8); });
}

#[test]
fn vmovpd_ymm_indexed_r12_r12_1() {
    // vmovupd ymm1, ymmword ptr [r12+r12*1]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x0c, 0x24], |a| { a.vmovpd_ymm_indexed(1, 12, 12, 1); });
}

#[test]
fn vmovpd_ymm_indexed_r12_r12_8() {
    // vmovupd ymm1, ymmword ptr [r12+r12*8]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x0c, 0xe4], |a| { a.vmovpd_ymm_indexed(1, 12, 12, 8); });
}

#[test]
fn vmovpd_ymm_indexed_rsp_rdx_1() {
    // vmovupd ymm1, ymmword ptr [rsp+rdx*1]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x0c, 0x14], |a| { a.vmovpd_ymm_indexed(1, 4, 2, 1); });
}

#[test]
fn vmovpd_ymm_indexed_rsp_rdx_8() {
    // vmovupd ymm1, ymmword ptr [rsp+rdx*8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x0c, 0xd4], |a| { a.vmovpd_ymm_indexed(1, 4, 2, 8); });
}

#[test]
fn vmovpd_ymm_indexed_r9_r12_1() {
    // vmovupd ymm1, ymmword ptr [r9+r12*1]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x0c, 0x21], |a| { a.vmovpd_ymm_indexed(1, 9, 12, 1); });
}

#[test]
fn vmovpd_ymm_indexed_r9_r12_8() {
    // vmovupd ymm1, ymmword ptr [r9+r12*8]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x0c, 0xe1], |a| { a.vmovpd_ymm_indexed(1, 9, 12, 8); });
}

#[test]
fn vmovpd_indexed_ymm_rbx_rdx_1() {
    // vmovupd ymmword ptr [rbx+rdx*1], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x0c, 0x13], |a| { a.vmovpd_indexed_ymm(3, 2, 1, 1); });
}

#[test]
fn vmovpd_indexed_ymm_rbx_rdx_8() {
    // vmovupd ymmword ptr [rbx+rdx*8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x0c, 0xd3], |a| { a.vmovpd_indexed_ymm(3, 2, 8, 1); });
}

#[test]
fn vmovpd_indexed_ymm_rbp_rdx_1() {
    // vmovupd ymmword ptr [rbp+rdx*1], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4c, 0x15, 0x00], |a| { a.vmovpd_indexed_ymm(5, 2, 1, 1); });
}

#[test]
fn vmovpd_indexed_ymm_rbp_rdx_8() {
    // vmovupd ymmword ptr [rbp+rdx*8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x4c, 0xd5, 0x00], |a| { a.vmovpd_indexed_ymm(5, 2, 8, 1); });
}

#[test]
fn vmovpd_indexed_ymm_r13_r9_1() {
    // vmovupd ymmword ptr [r13+r9*1], ymm1
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x11, 0x4c, 0x0d, 0x00], |a| { a.vmovpd_indexed_ymm(13, 9, 1, 1); });
}

#[test]
fn vmovpd_indexed_ymm_r13_r9_8() {
    // vmovupd ymmword ptr [r13+r9*8], ymm1
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x11, 0x4c, 0xcd, 0x00], |a| { a.vmovpd_indexed_ymm(13, 9, 8, 1); });
}

#[test]
fn vmovpd_indexed_ymm_r12_r12_1() {
    // vmovupd ymmword ptr [r12+r12*1], ymm1
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x11, 0x0c, 0x24], |a| { a.vmovpd_indexed_ymm(12, 12, 1, 1); });
}

#[test]
fn vmovpd_indexed_ymm_r12_r12_8() {
    // vmovupd ymmword ptr [r12+r12*8], ymm1
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x11, 0x0c, 0xe4], |a| { a.vmovpd_indexed_ymm(12, 12, 8, 1); });
}

#[test]
fn vmovpd_indexed_ymm_rsp_rdx_1() {
    // vmovupd ymmword ptr [rsp+rdx*1], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x0c, 0x14], |a| { a.vmovpd_indexed_ymm(4, 2, 1, 1); });
}

#[test]
fn vmovpd_indexed_ymm_rsp_rdx_8() {
    // vmovupd ymmword ptr [rsp+rdx*8], ymm1
    check(DataType::F64, &[0xc5, 0xfd, 0x11, 0x0c, 0xd4], |a| { a.vmovpd_indexed_ymm(4, 2, 8, 1); });
}

#[test]
fn vmovpd_indexed_ymm_r9_r12_1() {
    // vmovupd ymmword ptr [r9+r12*1], ymm1
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x11, 0x0c, 0x21], |a| { a.vmovpd_indexed_ymm(9, 12, 1, 1); });
}

#[test]
fn vmovpd_indexed_ymm_r9_r12_8() {
    // vmovupd ymmword ptr [r9+r12*8], ymm1
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x11, 0x0c, 0xe1], |a| { a.vmovpd_indexed_ymm(9, 12, 8, 1); });
}

#[test]
fn vmovdd_xmm_indexed_rbx_rdx_1() {
    // vmovupd xmm1, xmmword ptr [rbx+rdx*1]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x0c, 0x13], |a| { a.vmovdd_xmm_indexed(1, 3, 2, 1); });
}

#[test]
fn vmovdd_xmm_indexed_rbx_rdx_8() {
    // vmovupd xmm1, xmmword ptr [rbx+rdx*8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x0c, 0xd3], |a| { a.vmovdd_xmm_indexed(1, 3, 2, 8); });
}

#[test]
fn vmovdd_xmm_indexed_rbp_rdx_1() {
    // vmovupd xmm1, xmmword ptr [rbp+rdx*1]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4c, 0x15, 0x00], |a| { a.vmovdd_xmm_indexed(1, 5, 2, 1); });
}

#[test]
fn vmovdd_xmm_indexed_rbp_rdx_8() {
    // vmovupd xmm1, xmmword ptr [rbp+rdx*8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x4c, 0xd5, 0x00], |a| { a.vmovdd_xmm_indexed(1, 5, 2, 8); });
}

#[test]
fn vmovdd_xmm_indexed_r13_r9_1() {
    // vmovupd xmm1, xmmword ptr [r13+r9*1]
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x10, 0x4c, 0x0d, 0x00], |a| { a.vmovdd_xmm_indexed(1, 13, 9, 1); });
}

#[test]
fn vmovdd_xmm_indexed_r13_r9_8() {
    // vmovupd xmm1, xmmword ptr [r13+r9*8]
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x10, 0x4c, 0xcd, 0x00], |a| { a.vmovdd_xmm_indexed(1, 13, 9, 8); });
}

#[test]
fn vmovdd_xmm_indexed_r12_r12_1() {
    // vmovupd xmm1, xmmword ptr [r12+r12*1]
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x10, 0x0c, 0x24], |a| { a.vmovdd_xmm_indexed(1, 12, 12, 1); });
}

#[test]
fn vmovdd_xmm_indexed_r12_r12_8() {
    // vmovupd xmm1, xmmword ptr [r12+r12*8]
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x10, 0x0c, 0xe4], |a| { a.vmovdd_xmm_indexed(1, 12, 12, 8); });
}

#[test]
fn vmovdd_xmm_indexed_rsp_rdx_1() {
    // vmovupd xmm1, xmmword ptr [rsp+rdx*1]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x0c, 0x14], |a| { a.vmovdd_xmm_indexed(1, 4, 2, 1); });
}

#[test]
fn vmovdd_xmm_indexed_rsp_rdx_8() {
    // vmovupd xmm1, xmmword ptr [rsp+rdx*8]
    check(DataType::F64, &[0xc5, 0xf9, 0x10, 0x0c, 0xd4], |a| { a.vmovdd_xmm_indexed(1, 4, 2, 8); });
}

#[test]
fn vmovdd_xmm_indexed_r9_r12_1() {
    // vmovupd xmm1, xmmword ptr [r9+r12*1]
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x10, 0x0c, 0x21], |a| { a.vmovdd_xmm_indexed(1, 9, 12, 1); });
}

#[test]
fn vmovdd_xmm_indexed_r9_r12_8() {
    // vmovupd xmm1, xmmword ptr [r9+r12*8]
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x10, 0x0c, 0xe1], |a| { a.vmovdd_xmm_indexed(1, 9, 12, 8); });
}

#[test]
fn vmovdd_indexed_xmm_rbx_rdx_1() {
    // vmovupd xmmword ptr [rbx+rdx*1], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x0c, 0x13], |a| { a.vmovdd_indexed_xmm(3, 2, 1, 1); });
}

#[test]
fn vmovdd_indexed_xmm_rbx_rdx_8() {
    // vmovupd xmmword ptr [rbx+rdx*8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x0c, 0xd3], |a| { a.vmovdd_indexed_xmm(3, 2, 8, 1); });
}

#[test]
fn vmovdd_indexed_xmm_rbp_rdx_1() {
    // vmovupd xmmword ptr [rbp+rdx*1], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4c, 0x15, 0x00], |a| { a.vmovdd_indexed_xmm(5, 2, 1, 1); });
}

#[test]
fn vmovdd_indexed_xmm_rbp_rdx_8() {
    // vmovupd xmmword ptr [rbp+rdx*8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x4c, 0xd5, 0x00], |a| { a.vmovdd_indexed_xmm(5, 2, 8, 1); });
}

#[test]
fn vmovdd_indexed_xmm_r13_r9_1() {
    // vmovupd xmmword ptr [r13+r9*1], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x11, 0x4c, 0x0d, 0x00], |a| { a.vmovdd_indexed_xmm(13, 9, 1, 1); });
}

#[test]
fn vmovdd_indexed_xmm_r13_r9_8() {
    // vmovupd xmmword ptr [r13+r9*8], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x11, 0x4c, 0xcd, 0x00], |a| { a.vmovdd_indexed_xmm(13, 9, 8, 1); });
}

#[test]
fn vmovdd_indexed_xmm_r12_r12_1() {
    // vmovupd xmmword ptr [r12+r12*1], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x11, 0x0c, 0x24], |a| { a.vmovdd_indexed_xmm(12, 12, 1, 1); });
}

#[test]
fn vmovdd_indexed_xmm_r12_r12_8() {
    // vmovupd xmmword ptr [r12+r12*8], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x11, 0x0c, 0xe4], |a| { a.vmovdd_indexed_xmm(12, 12, 8, 1); });
}

#[test]
fn vmovdd_indexed_xmm_rsp_rdx_1() {
    // vmovupd xmmword ptr [rsp+rdx*1], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x0c, 0x14], |a| { a.vmovdd_indexed_xmm(4, 2, 1, 1); });
}

#[test]
fn vmovdd_indexed_xmm_rsp_rdx_8() {
    // vmovupd xmmword ptr [rsp+rdx*8], xmm1
    check(DataType::F64, &[0xc5, 0xf9, 0x11, 0x0c, 0xd4], |a| { a.vmovdd_indexed_xmm(4, 2, 8, 1); });
}

#[test]
fn vmovdd_indexed_xmm_r9_r12_1() {
    // vmovupd xmmword ptr [r9+r12*1], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x11, 0x0c, 0x21], |a| { a.vmovdd_indexed_xmm(9, 12, 1, 1); });
}

#[test]
fn vmovdd_indexed_xmm_r9_r12_8() {
    // vmovupd xmmword ptr [r9+r12*8], xmm1
    check(DataType::F64, &[0xc4, 0x81, 0x79, 0x11, 0x0c, 0xe1], |a| { a.vmovdd_indexed_xmm(9, 12, 8, 1); });
}

#[test]
fn movsd_xmm_indexed_rbx_rdx_1() {
    // movsd xmm1, qword ptr [rbx+rdx*1]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x0c, 0x13], |a| { a.movsd_xmm_indexed(1, 3, 2, 1); });
}

#[test]
fn movsd_xmm_indexed_rbx_rdx_8() {
    // movsd xmm1, qword ptr [rbx+rdx*8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x0c, 0xd3], |a| { a.movsd_xmm_indexed(1, 3, 2, 8); });
}

#[test]
fn movsd_xmm_indexed_rbp_rdx_1() {
    // movsd xmm1, qword ptr [rbp+rdx*1]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4c, 0x15, 0x00], |a| { a.movsd_xmm_indexed(1, 5, 2, 1); });
}

#[test]
fn movsd_xmm_indexed_rbp_rdx_8() {
    // movsd xmm1, qword ptr [rbp+rdx*8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x4c, 0xd5, 0x00], |a| { a.movsd_xmm_indexed(1, 5, 2, 8); });
}

#[test]
fn movsd_xmm_indexed_r13_r9_1() {
    // movsd xmm1, qword ptr [r13+r9*1]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x10, 0x4c, 0x0d, 0x00], |a| { a.movsd_xmm_indexed(1, 13, 9, 1); });
}

#[test]
fn movsd_xmm_indexed_r13_r9_8() {
    // movsd xmm1, qword ptr [r13+r9*8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x10, 0x4c, 0xcd, 0x00], |a| { a.movsd_xmm_indexed(1, 13, 9, 8); });
}

#[test]
fn movsd_xmm_indexed_r12_r12_1() {
    // movsd xmm1, qword ptr [r12+r12*1]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x10, 0x0c, 0x24], |a| { a.movsd_xmm_indexed(1, 12, 12, 1); });
}

#[test]
fn movsd_xmm_indexed_r12_r12_8() {
    // movsd xmm1, qword ptr [r12+r12*8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x10, 0x0c, 0xe4], |a| { a.movsd_xmm_indexed(1, 12, 12, 8); });
}

#[test]
fn movsd_xmm_indexed_rsp_rdx_1() {
    // movsd xmm1, qword ptr [rsp+rdx*1]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x0c, 0x14], |a| { a.movsd_xmm_indexed(1, 4, 2, 1); });
}

#[test]
fn movsd_xmm_indexed_rsp_rdx_8() {
    // movsd xmm1, qword ptr [rsp+rdx*8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x0c, 0xd4], |a| { a.movsd_xmm_indexed(1, 4, 2, 8); });
}

#[test]
fn movsd_xmm_indexed_r9_r12_1() {
    // movsd xmm1, qword ptr [r9+r12*1]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x10, 0x0c, 0x21], |a| { a.movsd_xmm_indexed(1, 9, 12, 1); });
}

#[test]
fn movsd_xmm_indexed_r9_r12_8() {
    // movsd xmm1, qword ptr [r9+r12*8]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x10, 0x0c, 0xe1], |a| { a.movsd_xmm_indexed(1, 9, 12, 8); });
}

#[test]
fn movsd_indexed_xmm_rbx_rdx_1() {
    // movsd qword ptr [rbx+rdx*1], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x0c, 0x13], |a| { a.movsd_indexed_xmm(3, 2, 1, 1); });
}

#[test]
fn movsd_indexed_xmm_rbx_rdx_8() {
    // movsd qword ptr [rbx+rdx*8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x0c, 0xd3], |a| { a.movsd_indexed_xmm(3, 2, 8, 1); });
}

#[test]
fn movsd_indexed_xmm_rbp_rdx_1() {
    // movsd qword ptr [rbp+rdx*1], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4c, 0x15, 0x00], |a| { a.movsd_indexed_xmm(5, 2, 1, 1); });
}

#[test]
fn movsd_indexed_xmm_rbp_rdx_8() {
    // movsd qword ptr [rbp+rdx*8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x4c, 0xd5, 0x00], |a| { a.movsd_indexed_xmm(5, 2, 8, 1); });
}

#[test]
fn movsd_indexed_xmm_r13_r9_1() {
    // movsd qword ptr [r13+r9*1], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x11, 0x4c, 0x0d, 0x00], |a| { a.movsd_indexed_xmm(13, 9, 1, 1); });
}

#[test]
fn movsd_indexed_xmm_r13_r9_8() {
    // movsd qword ptr [r13+r9*8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x11, 0x4c, 0xcd, 0x00], |a| { a.movsd_indexed_xmm(13, 9, 8, 1); });
}

#[test]
fn movsd_indexed_xmm_r12_r12_1() {
    // movsd qword ptr [r12+r12*1], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x11, 0x0c, 0x24], |a| { a.movsd_indexed_xmm(12, 12, 1, 1); });
}

#[test]
fn movsd_indexed_xmm_r12_r12_8() {
    // movsd qword ptr [r12+r12*8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x11, 0x0c, 0xe4], |a| { a.movsd_indexed_xmm(12, 12, 8, 1); });
}

#[test]
fn movsd_indexed_xmm_rsp_rdx_1() {
    // movsd qword ptr [rsp+rdx*1], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x0c, 0x14], |a| { a.movsd_indexed_xmm(4, 2, 1, 1); });
}

#[test]
fn movsd_indexed_xmm_rsp_rdx_8() {
    // movsd qword ptr [rsp+rdx*8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x48, 0x0f, 0x11, 0x0c, 0xd4], |a| { a.movsd_indexed_xmm(4, 2, 8, 1); });
}

#[test]
fn movsd_indexed_xmm_r9_r12_1() {
    // movsd qword ptr [r9+r12*1], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x11, 0x0c, 0x21], |a| { a.movsd_indexed_xmm(9, 12, 1, 1); });
}

#[test]
fn movsd_indexed_xmm_r9_r12_8() {
    // movsd qword ptr [r9+r12*8], xmm1
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F64, &[0xf2, 0x4b, 0x0f, 0x11, 0x0c, 0xe1], |a| { a.movsd_indexed_xmm(9, 12, 8, 1); });
}

#[test]
fn vmovqd_zmm_indexed_rbx_rdx_1() {
    // vmovupd zmm1, zmmword ptr [rbx+rdx*1]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x0c, 0x13], |a| { a.vmovqd_zmm_indexed(1, 3, 2, 1); });
}

#[test]
fn vmovqd_zmm_indexed_rbx_rdx_8() {
    // vmovupd zmm1, zmmword ptr [rbx+rdx*8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x0c, 0xd3], |a| { a.vmovqd_zmm_indexed(1, 3, 2, 8); });
}

#[test]
fn vmovqd_zmm_indexed_rbp_rdx_1() {
    // vmovupd zmm1, zmmword ptr [rbp+rdx*1]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4c, 0x15, 0x00], |a| { a.vmovqd_zmm_indexed(1, 5, 2, 1); });
}

#[test]
fn vmovqd_zmm_indexed_rbp_rdx_8() {
    // vmovupd zmm1, zmmword ptr [rbp+rdx*8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4c, 0xd5, 0x00], |a| { a.vmovqd_zmm_indexed(1, 5, 2, 8); });
}

#[test]
fn vmovqd_zmm_indexed_r13_r9_1() {
    // vmovupd zmm1, zmmword ptr [r13+r9*1]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x4c, 0x0d, 0x00], |a| { a.vmovqd_zmm_indexed(1, 13, 9, 1); });
}

#[test]
fn vmovqd_zmm_indexed_r13_r9_8() {
    // vmovupd zmm1, zmmword ptr [r13+r9*8]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x4c, 0xcd, 0x00], |a| { a.vmovqd_zmm_indexed(1, 13, 9, 8); });
}

#[test]
fn vmovqd_zmm_indexed_r12_r12_1() {
    // vmovupd zmm1, zmmword ptr [r12+r12*1]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x0c, 0x24], |a| { a.vmovqd_zmm_indexed(1, 12, 12, 1); });
}

#[test]
fn vmovqd_zmm_indexed_r12_r12_8() {
    // vmovupd zmm1, zmmword ptr [r12+r12*8]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x0c, 0xe4], |a| { a.vmovqd_zmm_indexed(1, 12, 12, 8); });
}

#[test]
fn vmovqd_zmm_indexed_rsp_rdx_1() {
    // vmovupd zmm1, zmmword ptr [rsp+rdx*1]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x0c, 0x14], |a| { a.vmovqd_zmm_indexed(1, 4, 2, 1); });
}

#[test]
fn vmovqd_zmm_indexed_rsp_rdx_8() {
    // vmovupd zmm1, zmmword ptr [rsp+rdx*8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x0c, 0xd4], |a| { a.vmovqd_zmm_indexed(1, 4, 2, 8); });
}

#[test]
fn vmovqd_zmm_indexed_r9_r12_1() {
    // vmovupd zmm1, zmmword ptr [r9+r12*1]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x0c, 0x21], |a| { a.vmovqd_zmm_indexed(1, 9, 12, 1); });
}

#[test]
fn vmovqd_zmm_indexed_r9_r12_8() {
    // vmovupd zmm1, zmmword ptr [r9+r12*8]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x0c, 0xe1], |a| { a.vmovqd_zmm_indexed(1, 9, 12, 8); });
}

#[test]
fn vmovqd_indexed_zmm_rbx_rdx_1() {
    // vmovupd zmmword ptr [rbx+rdx*1], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x0c, 0x13], |a| { a.vmovqd_indexed_zmm(3, 2, 1, 1); });
}

#[test]
fn vmovqd_indexed_zmm_rbx_rdx_8() {
    // vmovupd zmmword ptr [rbx+rdx*8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x0c, 0xd3], |a| { a.vmovqd_indexed_zmm(3, 2, 8, 1); });
}

#[test]
fn vmovqd_indexed_zmm_rbp_rdx_1() {
    // vmovupd zmmword ptr [rbp+rdx*1], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4c, 0x15, 0x00], |a| { a.vmovqd_indexed_zmm(5, 2, 1, 1); });
}

#[test]
fn vmovqd_indexed_zmm_rbp_rdx_8() {
    // vmovupd zmmword ptr [rbp+rdx*8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x4c, 0xd5, 0x00], |a| { a.vmovqd_indexed_zmm(5, 2, 8, 1); });
}

#[test]
fn vmovqd_indexed_zmm_r13_r9_1() {
    // vmovupd zmmword ptr [r13+r9*1], zmm1
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x11, 0x4c, 0x0d, 0x00], |a| { a.vmovqd_indexed_zmm(13, 9, 1, 1); });
}

#[test]
fn vmovqd_indexed_zmm_r13_r9_8() {
    // vmovupd zmmword ptr [r13+r9*8], zmm1
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x11, 0x4c, 0xcd, 0x00], |a| { a.vmovqd_indexed_zmm(13, 9, 8, 1); });
}

#[test]
fn vmovqd_indexed_zmm_r12_r12_1() {
    // vmovupd zmmword ptr [r12+r12*1], zmm1
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x11, 0x0c, 0x24], |a| { a.vmovqd_indexed_zmm(12, 12, 1, 1); });
}

#[test]
fn vmovqd_indexed_zmm_r12_r12_8() {
    // vmovupd zmmword ptr [r12+r12*8], zmm1
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x11, 0x0c, 0xe4], |a| { a.vmovqd_indexed_zmm(12, 12, 8, 1); });
}

#[test]
fn vmovqd_indexed_zmm_rsp_rdx_1() {
    // vmovupd zmmword ptr [rsp+rdx*1], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x0c, 0x14], |a| { a.vmovqd_indexed_zmm(4, 2, 1, 1); });
}

#[test]
fn vmovqd_indexed_zmm_rsp_rdx_8() {
    // vmovupd zmmword ptr [rsp+rdx*8], zmm1
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x11, 0x0c, 0xd4], |a| { a.vmovqd_indexed_zmm(4, 2, 8, 1); });
}

#[test]
fn vmovqd_indexed_zmm_r9_r12_1() {
    // vmovupd zmmword ptr [r9+r12*1], zmm1
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x11, 0x0c, 0x21], |a| { a.vmovqd_indexed_zmm(9, 12, 1, 1); });
}

#[test]
fn vmovqd_indexed_zmm_r9_r12_8() {
    // vmovupd zmmword ptr [r9+r12*8], zmm1
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x11, 0x0c, 0xe1], |a| { a.vmovqd_indexed_zmm(9, 12, 8, 1); });
}

#[test]
fn vmovpd_ymm_indexed_mem_rbx_rdx_8() {
    // vmovupd ymm1, ymmword ptr [rbx+rdx*4+0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4c, 0x93, 0x08], |a| { a.vmovpd_ymm_indexed_mem(1, 3, 2, 4, 8); });
}

#[test]
fn vmovpd_ymm_indexed_mem_rbx_rdx_1234() {
    // vmovupd ymm1, ymmword ptr [rbx+rdx*4+0x1234]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x8c, 0x93, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_indexed_mem(1, 3, 2, 4, 4660); });
}

#[test]
fn vmovpd_ymm_indexed_mem_rbx_rdx_m8() {
    // vmovupd ymm1, ymmword ptr [rbx+rdx*4-0x8]
    check(DataType::F64, &[0xc5, 0xfd, 0x10, 0x4c, 0x93, 0xf8], |a| { a.vmovpd_ymm_indexed_mem(1, 3, 2, 4, -8); });
}

#[test]
fn vmovpd_ymm_indexed_mem_r13_r9_8() {
    // vmovupd ymm1, ymmword ptr [r13+r9*4+0x8]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x4c, 0x8d, 0x08], |a| { a.vmovpd_ymm_indexed_mem(1, 13, 9, 4, 8); });
}

#[test]
fn vmovpd_ymm_indexed_mem_r13_r9_1234() {
    // vmovupd ymm1, ymmword ptr [r13+r9*4+0x1234]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x8c, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_indexed_mem(1, 13, 9, 4, 4660); });
}

#[test]
fn vmovpd_ymm_indexed_mem_r13_r9_m8() {
    // vmovupd ymm1, ymmword ptr [r13+r9*4-0x8]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x4c, 0x8d, 0xf8], |a| { a.vmovpd_ymm_indexed_mem(1, 13, 9, 4, -8); });
}

#[test]
fn vmovpd_ymm_indexed_mem_r12_r12_8() {
    // vmovupd ymm1, ymmword ptr [r12+r12*4+0x8]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x4c, 0xa4, 0x08], |a| { a.vmovpd_ymm_indexed_mem(1, 12, 12, 4, 8); });
}

#[test]
fn vmovpd_ymm_indexed_mem_r12_r12_1234() {
    // vmovupd ymm1, ymmword ptr [r12+r12*4+0x1234]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x8c, 0xa4, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovpd_ymm_indexed_mem(1, 12, 12, 4, 4660); });
}

#[test]
fn vmovpd_ymm_indexed_mem_r12_r12_m8() {
    // vmovupd ymm1, ymmword ptr [r12+r12*4-0x8]
    check(DataType::F64, &[0xc4, 0x81, 0x7d, 0x10, 0x4c, 0xa4, 0xf8], |a| { a.vmovpd_ymm_indexed_mem(1, 12, 12, 4, -8); });
}

#[test]
fn vmovqd_zmm_indexed_mem_rbx_rdx_8() {
    // vmovupd zmm1, zmmword ptr [rbx+rdx*4+0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8c, 0x93, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_indexed_mem(1, 3, 2, 4, 8); });
}

#[test]
fn vmovqd_zmm_indexed_mem_rbx_rdx_80() {
    // vmovupd zmm1, zmmword ptr [rbx+rdx*4+0x80]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x4c, 0x93, 0x02], |a| { a.vmovqd_zmm_indexed_mem(1, 3, 2, 4, 128); });
}

#[test]
fn vmovqd_zmm_indexed_mem_rbx_rdx_1234() {
    // vmovupd zmm1, zmmword ptr [rbx+rdx*4+0x1234]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8c, 0x93, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_indexed_mem(1, 3, 2, 4, 4660); });
}

#[test]
fn vmovqd_zmm_indexed_mem_rbx_rdx_m8() {
    // vmovupd zmm1, zmmword ptr [rbx+rdx*4-0x8]
    check(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x8c, 0x93, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_indexed_mem(1, 3, 2, 4, -8); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r13_r9_8() {
    // vmovupd zmm1, zmmword ptr [r13+r9*4+0x8]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x8c, 0x8d, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_indexed_mem(1, 13, 9, 4, 8); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r13_r9_80() {
    // vmovupd zmm1, zmmword ptr [r13+r9*4+0x80]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x4c, 0x8d, 0x02], |a| { a.vmovqd_zmm_indexed_mem(1, 13, 9, 4, 128); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r13_r9_1234() {
    // vmovupd zmm1, zmmword ptr [r13+r9*4+0x1234]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x8c, 0x8d, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_indexed_mem(1, 13, 9, 4, 4660); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r13_r9_m8() {
    // vmovupd zmm1, zmmword ptr [r13+r9*4-0x8]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x8c, 0x8d, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_indexed_mem(1, 13, 9, 4, -8); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r12_r12_8() {
    // vmovupd zmm1, zmmword ptr [r12+r12*4+0x8]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x8c, 0xa4, 0x08, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_indexed_mem(1, 12, 12, 4, 8); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r12_r12_80() {
    // vmovupd zmm1, zmmword ptr [r12+r12*4+0x80]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x4c, 0xa4, 0x02], |a| { a.vmovqd_zmm_indexed_mem(1, 12, 12, 4, 128); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r12_r12_1234() {
    // vmovupd zmm1, zmmword ptr [r12+r12*4+0x1234]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x8c, 0xa4, 0x34, 0x12, 0x00, 0x00], |a| { a.vmovqd_zmm_indexed_mem(1, 12, 12, 4, 4660); });
}

#[test]
fn vmovqd_zmm_indexed_mem_r12_r12_m8() {
    // vmovupd zmm1, zmmword ptr [r12+r12*4-0x8]
    check(DataType::F64, &[0x62, 0x91, 0xfd, 0x48, 0x10, 0x8c, 0xa4, 0xf8, 0xff, 0xff, 0xff], |a| { a.vmovqd_zmm_indexed_mem(1, 12, 12, 4, -8); });
}

#[test]
fn vmovsd_xmm_label_0() {
    // vmovsd xmm1, qword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xfb, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovsd_xmm_label(1, "L"); });
}

#[test]
fn vmovsd_xmm_label_1() {
    // vmovsd xmm9, qword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x7b, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovsd_xmm_label(9, "L"); });
}

#[test]
fn vbroadcastsd_label_0() {
    // vbroadcastsd ymm1, qword ptr [rip+L]
    check_label(DataType::F64, &[0xc4, 0xe2, 0x7d, 0x19, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vbroadcastsd_label(1, "L"); });
}

#[test]
fn vbroadcastsd_label_1() {
    // vbroadcastsd ymm9, qword ptr [rip+L]
    check_label(DataType::F64, &[0xc4, 0x62, 0x7d, 0x19, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vbroadcastsd_label(9, "L"); });
}

#[test]
fn vmovpd_ymm_label_0() {
    // vmovupd ymm1, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xfd, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovpd_ymm_label(1, "L"); });
}

#[test]
fn vmovpd_ymm_label_1() {
    // vmovupd ymm9, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x7d, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovpd_ymm_label(9, "L"); });
}

#[test]
fn vmovdd_xmm_label_0() {
    // vmovupd xmm1, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xf9, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovdd_xmm_label(1, "L"); });
}

#[test]
fn vmovdd_xmm_label_1() {
    // vmovupd xmm9, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x79, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovdd_xmm_label(9, "L"); });
}

#[test]
fn movsd_xmm_label_0() {
    // movsd xmm1, qword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0xf2, 0x48, 0x0f, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.movsd_xmm_label(1, "L"); });
}

#[test]
fn movsd_xmm_label_1() {
    // movsd xmm9, qword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0xf2, 0x4c, 0x0f, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.movsd_xmm_label(9, "L"); });
}

#[test]
fn mov_reg_label_0() {
    // mov rcx, qword ptr [rip+L]
    check_label(DataType::F64, &[0x48, 0x8b, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.mov_reg_label(1, "L"); });
}

#[test]
fn mov_reg_label_1() {
    // mov r9, qword ptr [rip+L]
    check_label(DataType::F64, &[0x4c, 0x8b, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.mov_reg_label(9, "L"); });
}

#[test]
fn vbroadcastsd_zmm_label_0() {
    // vbroadcastsd zmm1, qword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0xf2, 0xfd, 0x48, 0x19, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vbroadcastsd_zmm_label(1, "L"); });
}

#[test]
fn vbroadcastsd_zmm_label_1() {
    // vbroadcastsd zmm9, qword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0x72, 0xfd, 0x48, 0x19, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vbroadcastsd_zmm_label(9, "L"); });
}

#[test]
fn vmovqd_zmm_label_0() {
    // vmovupd zmm1, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0xf1, 0xfd, 0x48, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_label(1, "L"); });
}

#[test]
fn vmovqd_zmm_label_1() {
    // vmovupd zmm9, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0x71, 0xfd, 0x48, 0x10, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vmovqd_zmm_label(9, "L"); });
}

#[test]
fn vandpd_label_0() {
    // vandpd ymm1, ymm2, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xed, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandpd_label(1, 2, "L"); });
}

#[test]
fn vanddd_label_0() {
    // vandpd xmm1, xmm2, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xe9, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vanddd_label(1, 2, "L"); });
}

#[test]
fn vandqd_label_0() {
    // vandpd zmm1, zmm2, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandqd_label(1, 2, "L"); });
}

#[test]
fn vandpd_label_1() {
    // vandpd ymm9, ymm10, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x2d, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandpd_label(9, 10, "L"); });
}

#[test]
fn vanddd_label_1() {
    // vandpd xmm9, xmm10, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x29, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vanddd_label(9, 10, "L"); });
}

#[test]
fn vandqd_label_1() {
    // vandpd zmm9, zmm10, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0x71, 0xad, 0x48, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandqd_label(9, 10, "L"); });
}

#[test]
fn andpd_label_0() {
    // andpd xmm1, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x48, 0x0f, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.andpd_label(1, "L"); });
}

#[test]
fn andpd_label_1() {
    // andpd xmm9, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x4c, 0x0f, 0x54, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.andpd_label(9, "L"); });
}

#[test]
fn vandnpd_label_0() {
    // vandnpd ymm1, ymm2, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xed, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandnpd_label(1, 2, "L"); });
}

#[test]
fn vandndd_label_0() {
    // vandnpd xmm1, xmm2, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xe9, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandndd_label(1, 2, "L"); });
}

#[test]
fn vandnqd_label_0() {
    // vandnpd zmm1, zmm2, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandnqd_label(1, 2, "L"); });
}

#[test]
fn vandnpd_label_1() {
    // vandnpd ymm9, ymm10, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x2d, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandnpd_label(9, 10, "L"); });
}

#[test]
fn vandndd_label_1() {
    // vandnpd xmm9, xmm10, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x29, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandndd_label(9, 10, "L"); });
}

#[test]
fn vandnqd_label_1() {
    // vandnpd zmm9, zmm10, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0x71, 0xad, 0x48, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vandnqd_label(9, 10, "L"); });
}

#[test]
fn andnpd_label_0() {
    // andnpd xmm1, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x48, 0x0f, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.andnpd_label(1, "L"); });
}

#[test]
fn andnpd_label_1() {
    // andnpd xmm9, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x4c, 0x0f, 0x55, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.andnpd_label(9, "L"); });
}

#[test]
fn vorpd_label_0() {
    // vorpd ymm1, ymm2, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xed, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vorpd_label(1, 2, "L"); });
}

#[test]
fn vordd_label_0() {
    // vorpd xmm1, xmm2, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xe9, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vordd_label(1, 2, "L"); });
}

#[test]
fn vorqd_label_0() {
    // vorpd zmm1, zmm2, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vorqd_label(1, 2, "L"); });
}

#[test]
fn vorpd_label_1() {
    // vorpd ymm9, ymm10, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x2d, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vorpd_label(9, 10, "L"); });
}

#[test]
fn vordd_label_1() {
    // vorpd xmm9, xmm10, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x29, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vordd_label(9, 10, "L"); });
}

#[test]
fn vorqd_label_1() {
    // vorpd zmm9, zmm10, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0x71, 0xad, 0x48, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vorqd_label(9, 10, "L"); });
}

#[test]
fn orpd_label_0() {
    // orpd xmm1, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x48, 0x0f, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.orpd_label(1, "L"); });
}

#[test]
fn orpd_label_1() {
    // orpd xmm9, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x4c, 0x0f, 0x56, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.orpd_label(9, "L"); });
}

#[test]
fn vxorpd_label_0() {
    // vxorpd ymm1, ymm2, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xed, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vxorpd_label(1, 2, "L"); });
}

#[test]
fn vxordd_label_0() {
    // vxorpd xmm1, xmm2, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0xe9, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vxordd_label(1, 2, "L"); });
}

#[test]
fn vxorqd_label_0() {
    // vxorpd zmm1, zmm2, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0xf1, 0xed, 0x48, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vxorqd_label(1, 2, "L"); });
}

#[test]
fn vxorpd_label_1() {
    // vxorpd ymm9, ymm10, ymmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x2d, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vxorpd_label(9, 10, "L"); });
}

#[test]
fn vxordd_label_1() {
    // vxorpd xmm9, xmm10, xmmword ptr [rip+L]
    check_label(DataType::F64, &[0xc5, 0x29, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vxordd_label(9, 10, "L"); });
}

#[test]
fn vxorqd_label_1() {
    // vxorpd zmm9, zmm10, zmmword ptr [rip+L]
    check_label(DataType::F64, &[0x62, 0x71, 0xad, 0x48, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.vxorqd_label(9, 10, "L"); });
}

#[test]
fn xorpd_label_0() {
    // xorpd xmm1, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x48, 0x0f, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.xorpd_label(1, "L"); });
}

#[test]
fn xorpd_label_1() {
    // xorpd xmm9, xmmword ptr [rip+L]
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check_label(DataType::F64, &[0x66, 0x4c, 0x0f, 0x57, 0x0d, 0xc8, 0x00, 0x00, 0x00], |a| { a.xorpd_label(9, "L"); });
}

#[test]
fn call_indirect_0() {
    // call qword ptr [rip+L]
    check_label(DataType::F64, &[0xff, 0x15, 0xc8, 0x00, 0x00, 0x00], |a| { a.call_indirect("L"); });
}

#[test]
fn call_relative_0() {
    // call L
    check_label(DataType::F64, &[0xe8, 0xc8, 0x00, 0x00, 0x00], |a| { a.call_relative("L"); });
}

#[test]
fn jmp_0() {
    // jmp L
    check_label(DataType::F64, &[0xe9, 0xc8, 0x00, 0x00, 0x00], |a| { a.jmp("L"); });
}

#[test]
fn jz_0() {
    // jz L
    check_label(DataType::F64, &[0x0f, 0x84, 0xc8, 0x00, 0x00, 0x00], |a| { a.jz("L"); });
}

#[test]
fn jnz_0() {
    // jnz L
    check_label(DataType::F64, &[0x0f, 0x85, 0xc8, 0x00, 0x00, 0x00], |a| { a.jnz("L"); });
}

#[test]
fn jpe_0() {
    // jpe L
    check_label(DataType::F64, &[0x0f, 0x8a, 0xc8, 0x00, 0x00, 0x00], |a| { a.jpe("L"); });
}

#[test]
fn jpo_0() {
    // jpo L
    check_label(DataType::F64, &[0x0f, 0x8b, 0xc8, 0x00, 0x00, 0x00], |a| { a.jpo("L"); });
}

#[test]
fn js_0() {
    // js L
    check_label(DataType::F64, &[0x0f, 0x88, 0xc8, 0x00, 0x00, 0x00], |a| { a.js("L"); });
}

#[test]
fn jb_0() {
    // jb L
    check_label(DataType::F64, &[0x0f, 0x82, 0xc8, 0x00, 0x00, 0x00], |a| { a.jb("L"); });
}

#[test]
fn jnb_0() {
    // jnb L
    check_label(DataType::F64, &[0x0f, 0x83, 0xc8, 0x00, 0x00, 0x00], |a| { a.jnb("L"); });
}

#[test]
fn f32_vaddsd() {
    // vaddss xmm1, xmm2, xmm3
    check(DataType::F32, &[0xc5, 0xea, 0x58, 0xcb], |a| { a.vaddsd(1, 2, 3); });
}

#[test]
fn f32_vaddsd_hi() {
    // vaddss xmm9, xmm10, xmm11
    check(DataType::F32, &[0xc4, 0x41, 0x2a, 0x58, 0xcb], |a| { a.vaddsd(9, 10, 11); });
}

#[test]
fn f32_vaddpd() {
    // vaddps ymm1, ymm2, ymm3
    check(DataType::F32, &[0xc5, 0xec, 0x58, 0xcb], |a| { a.vaddpd(1, 2, 3); });
}

#[test]
fn f32_vadddd() {
    // vaddps xmm1, xmm2, xmm3
    check(DataType::F32, &[0xc5, 0xe8, 0x58, 0xcb], |a| { a.vadddd(1, 2, 3); });
}

#[test]
fn f32_vmovmskpd() {
    // vmovmskps ecx, ymm3
    check(DataType::F32, &[0xc5, 0xfc, 0x50, 0xcb], |a| { a.vmovmskpd(1, 3); });
}

#[test]
fn f32_vandpd() {
    // vandps ymm1, ymm2, ymm3
    check(DataType::F32, &[0xc5, 0xec, 0x54, 0xcb], |a| { a.vandpd(1, 2, 3); });
}

#[test]
fn f32_movapd() {
    // movaps xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F32, &[0x48, 0x0f, 0x28, 0xca], |a| { a.movapd(1, 2); });
}

#[test]
fn f32_addsd() {
    // addss xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F32, &[0xf3, 0x48, 0x0f, 0x58, 0xca], |a| { a.addsd(1, 2); });
}

#[test]
fn f32_andpd() {
    // andps xmm1, xmm2
    // NOTE: the assembler emits a redundant REX.W (no effect on this SSE instruction); the canonical form has no REX
    check(DataType::F32, &[0x48, 0x0f, 0x54, 0xca], |a| { a.andpd(1, 2); });
}

#[test]
fn f32_vmovsd_load() {
    // vmovss xmm1, dword ptr [rbx+8]
    check(DataType::F32, &[0xc5, 0xfa, 0x10, 0x4b, 0x08], |a| { a.vmovsd_xmm_mem(1, 3, 8); });
}

#[test]
fn f32_vmovpd_load() {
    // vmovups ymm1, ymmword ptr [rbx+8]
    check(DataType::F32, &[0xc5, 0xfc, 0x10, 0x4b, 0x08], |a| { a.vmovpd_ymm_mem(1, 3, 8); });
}
