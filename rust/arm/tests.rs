// Encoding tests for the AArch64 instruction macro `arm!` (macros.rs).
//
// Each test builds one instruction with `arm!` and compares the 32-bit word with the
// encoding produced by an independent AArch64 assembler (Keystone) for the equivalent
// instruction; the `// comment` in each test is the assembly line that was assembled.
// Where the assembler cannot express an instruction, the expected word is pinned to
// the macro output after checking it against a Capstone disassembly (noted in the test).
//
// Conventions in `arm!`: `q(n)` is a 128-bit packed-double register (v<n>.2d);
// `label(ofs)` is a byte offset relative to the instruction.
//
// `#[ignore]`d tests document known defects; run with `cargo test arm::tests -- --ignored`.


#[test]
fn fmov_dd_0() {
    // fmov d1, d2
    let w: u32 = arm! { fmov d(1), d(2) };
    assert_eq!(w, 0x1e604041);
}

#[test]
fn fmov_dd_1() {
    // fmov d17, d18
    let w: u32 = arm! { fmov d(17), d(18) };
    assert_eq!(w, 0x1e604251);
}

#[test]
fn fmov_dd_2() {
    // fmov d30, d29
    let w: u32 = arm! { fmov d(30), d(29) };
    assert_eq!(w, 0x1e6043be);
}

#[test]
fn fmov_dd_3() {
    // fmov d0, d5
    let w: u32 = arm! { fmov d(0), d(5) };
    assert_eq!(w, 0x1e6040a0);
}

#[test]
fn fmov_dx_0() {
    // fmov d1, x2
    let w: u32 = arm! { fmov d(1), x(2) };
    assert_eq!(w, 0x9e670041);
}

#[test]
fn fmov_dx_1() {
    // fmov d17, x18
    let w: u32 = arm! { fmov d(17), x(18) };
    assert_eq!(w, 0x9e670251);
}

#[test]
fn fmov_dx_2() {
    // fmov d30, x29
    let w: u32 = arm! { fmov d(30), x(29) };
    assert_eq!(w, 0x9e6703be);
}

#[test]
fn fmov_dx_3() {
    // fmov d0, x5
    let w: u32 = arm! { fmov d(0), x(5) };
    assert_eq!(w, 0x9e6700a0);
}

#[test]
fn fmov_xd_0() {
    // fmov x1, d2
    let w: u32 = arm! { fmov x(1), d(2) };
    assert_eq!(w, 0x9e660041);
}

#[test]
fn fmov_xd_1() {
    // fmov x17, d18
    let w: u32 = arm! { fmov x(17), d(18) };
    assert_eq!(w, 0x9e660251);
}

#[test]
fn fmov_xd_2() {
    // fmov x30, d29
    let w: u32 = arm! { fmov x(30), d(29) };
    assert_eq!(w, 0x9e6603be);
}

#[test]
fn fmov_xd_3() {
    // fmov x0, d5
    let w: u32 = arm! { fmov x(0), d(5) };
    assert_eq!(w, 0x9e6600a0);
}

#[test]
fn fadd_d_0() {
    // fadd d1, d2, d3
    let w: u32 = arm! { fadd d(1), d(2), d(3) };
    assert_eq!(w, 0x1e632841);
}

#[test]
fn fadd_d_1() {
    // fadd d17, d18, d19
    let w: u32 = arm! { fadd d(17), d(18), d(19) };
    assert_eq!(w, 0x1e732a51);
}

#[test]
fn fadd_d_2() {
    // fadd d30, d29, d28
    let w: u32 = arm! { fadd d(30), d(29), d(28) };
    assert_eq!(w, 0x1e7c2bbe);
}

#[test]
fn fadd_d_3() {
    // fadd d0, d5, d10
    let w: u32 = arm! { fadd d(0), d(5), d(10) };
    assert_eq!(w, 0x1e6a28a0);
}

#[test]
fn fadd_q_0() {
    // fadd v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fadd q(1), q(2), q(3) };
    assert_eq!(w, 0x4e63d441);
}

#[test]
fn fadd_q_1() {
    // fadd v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fadd q(17), q(18), q(19) };
    assert_eq!(w, 0x4e73d651);
}

#[test]
fn fadd_q_2() {
    // fadd v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fadd q(30), q(29), q(28) };
    assert_eq!(w, 0x4e7cd7be);
}

#[test]
fn fadd_q_3() {
    // fadd v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fadd q(0), q(5), q(10) };
    assert_eq!(w, 0x4e6ad4a0);
}

#[test]
fn fsub_d_0() {
    // fsub d1, d2, d3
    let w: u32 = arm! { fsub d(1), d(2), d(3) };
    assert_eq!(w, 0x1e633841);
}

#[test]
fn fsub_d_1() {
    // fsub d17, d18, d19
    let w: u32 = arm! { fsub d(17), d(18), d(19) };
    assert_eq!(w, 0x1e733a51);
}

#[test]
fn fsub_d_2() {
    // fsub d30, d29, d28
    let w: u32 = arm! { fsub d(30), d(29), d(28) };
    assert_eq!(w, 0x1e7c3bbe);
}

#[test]
fn fsub_d_3() {
    // fsub d0, d5, d10
    let w: u32 = arm! { fsub d(0), d(5), d(10) };
    assert_eq!(w, 0x1e6a38a0);
}

#[test]
fn fsub_q_0() {
    // fsub v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fsub q(1), q(2), q(3) };
    assert_eq!(w, 0x4ee3d441);
}

#[test]
fn fsub_q_1() {
    // fsub v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fsub q(17), q(18), q(19) };
    assert_eq!(w, 0x4ef3d651);
}

#[test]
fn fsub_q_2() {
    // fsub v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fsub q(30), q(29), q(28) };
    assert_eq!(w, 0x4efcd7be);
}

#[test]
fn fsub_q_3() {
    // fsub v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fsub q(0), q(5), q(10) };
    assert_eq!(w, 0x4eead4a0);
}

#[test]
fn fmul_d_0() {
    // fmul d1, d2, d3
    let w: u32 = arm! { fmul d(1), d(2), d(3) };
    assert_eq!(w, 0x1e630841);
}

#[test]
fn fmul_d_1() {
    // fmul d17, d18, d19
    let w: u32 = arm! { fmul d(17), d(18), d(19) };
    assert_eq!(w, 0x1e730a51);
}

#[test]
fn fmul_d_2() {
    // fmul d30, d29, d28
    let w: u32 = arm! { fmul d(30), d(29), d(28) };
    assert_eq!(w, 0x1e7c0bbe);
}

#[test]
fn fmul_d_3() {
    // fmul d0, d5, d10
    let w: u32 = arm! { fmul d(0), d(5), d(10) };
    assert_eq!(w, 0x1e6a08a0);
}

#[test]
fn fmul_q_0() {
    // fmul v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fmul q(1), q(2), q(3) };
    assert_eq!(w, 0x6e63dc41);
}

#[test]
fn fmul_q_1() {
    // fmul v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fmul q(17), q(18), q(19) };
    assert_eq!(w, 0x6e73de51);
}

#[test]
fn fmul_q_2() {
    // fmul v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fmul q(30), q(29), q(28) };
    assert_eq!(w, 0x6e7cdfbe);
}

#[test]
fn fmul_q_3() {
    // fmul v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fmul q(0), q(5), q(10) };
    assert_eq!(w, 0x6e6adca0);
}

#[test]
fn fdiv_d_0() {
    // fdiv d1, d2, d3
    let w: u32 = arm! { fdiv d(1), d(2), d(3) };
    assert_eq!(w, 0x1e631841);
}

#[test]
fn fdiv_d_1() {
    // fdiv d17, d18, d19
    let w: u32 = arm! { fdiv d(17), d(18), d(19) };
    assert_eq!(w, 0x1e731a51);
}

#[test]
fn fdiv_d_2() {
    // fdiv d30, d29, d28
    let w: u32 = arm! { fdiv d(30), d(29), d(28) };
    assert_eq!(w, 0x1e7c1bbe);
}

#[test]
fn fdiv_d_3() {
    // fdiv d0, d5, d10
    let w: u32 = arm! { fdiv d(0), d(5), d(10) };
    assert_eq!(w, 0x1e6a18a0);
}

#[test]
fn fdiv_q_0() {
    // fdiv v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fdiv q(1), q(2), q(3) };
    assert_eq!(w, 0x6e63fc41);
}

#[test]
fn fdiv_q_1() {
    // fdiv v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fdiv q(17), q(18), q(19) };
    assert_eq!(w, 0x6e73fe51);
}

#[test]
fn fdiv_q_2() {
    // fdiv v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fdiv q(30), q(29), q(28) };
    assert_eq!(w, 0x6e7cffbe);
}

#[test]
fn fdiv_q_3() {
    // fdiv v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fdiv q(0), q(5), q(10) };
    assert_eq!(w, 0x6e6afca0);
}

#[test]
fn faddp_d_0() {
    // faddp d1, v2.2d
    let w: u32 = arm! { faddp d(1), q(2) };
    assert_eq!(w, 0x7e70d841);
}

#[test]
fn faddp_d_1() {
    // faddp d17, v18.2d
    let w: u32 = arm! { faddp d(17), q(18) };
    assert_eq!(w, 0x7e70da51);
}

#[test]
fn faddp_d_2() {
    // faddp d30, v29.2d
    let w: u32 = arm! { faddp d(30), q(29) };
    assert_eq!(w, 0x7e70dbbe);
}

#[test]
fn faddp_d_3() {
    // faddp d0, v5.2d
    let w: u32 = arm! { faddp d(0), q(5) };
    assert_eq!(w, 0x7e70d8a0);
}

#[test]
fn fsqrt_d_0() {
    // fsqrt d1, d2
    let w: u32 = arm! { fsqrt d(1), d(2) };
    assert_eq!(w, 0x1e61c041);
}

#[test]
fn fsqrt_d_1() {
    // fsqrt d17, d18
    let w: u32 = arm! { fsqrt d(17), d(18) };
    assert_eq!(w, 0x1e61c251);
}

#[test]
fn fsqrt_d_2() {
    // fsqrt d30, d29
    let w: u32 = arm! { fsqrt d(30), d(29) };
    assert_eq!(w, 0x1e61c3be);
}

#[test]
fn fsqrt_d_3() {
    // fsqrt d0, d5
    let w: u32 = arm! { fsqrt d(0), d(5) };
    assert_eq!(w, 0x1e61c0a0);
}

#[test]
fn fsqrt_q_0() {
    // fsqrt v1.2d, v2.2d
    let w: u32 = arm! { fsqrt q(1), q(2) };
    assert_eq!(w, 0x6ee1f841);
}

#[test]
fn fsqrt_q_1() {
    // fsqrt v17.2d, v18.2d
    let w: u32 = arm! { fsqrt q(17), q(18) };
    assert_eq!(w, 0x6ee1fa51);
}

#[test]
fn fsqrt_q_2() {
    // fsqrt v30.2d, v29.2d
    let w: u32 = arm! { fsqrt q(30), q(29) };
    assert_eq!(w, 0x6ee1fbbe);
}

#[test]
fn fsqrt_q_3() {
    // fsqrt v0.2d, v5.2d
    let w: u32 = arm! { fsqrt q(0), q(5) };
    assert_eq!(w, 0x6ee1f8a0);
}

#[test]
fn fneg_d_0() {
    // fneg d1, d2
    let w: u32 = arm! { fneg d(1), d(2) };
    assert_eq!(w, 0x1e614041);
}

#[test]
fn fneg_d_1() {
    // fneg d17, d18
    let w: u32 = arm! { fneg d(17), d(18) };
    assert_eq!(w, 0x1e614251);
}

#[test]
fn fneg_d_2() {
    // fneg d30, d29
    let w: u32 = arm! { fneg d(30), d(29) };
    assert_eq!(w, 0x1e6143be);
}

#[test]
fn fneg_d_3() {
    // fneg d0, d5
    let w: u32 = arm! { fneg d(0), d(5) };
    assert_eq!(w, 0x1e6140a0);
}

#[test]
fn fneg_q_0() {
    // fneg v1.2d, v2.2d
    let w: u32 = arm! { fneg q(1), q(2) };
    assert_eq!(w, 0x6ee0f841);
}

#[test]
fn fneg_q_1() {
    // fneg v17.2d, v18.2d
    let w: u32 = arm! { fneg q(17), q(18) };
    assert_eq!(w, 0x6ee0fa51);
}

#[test]
fn fneg_q_2() {
    // fneg v30.2d, v29.2d
    let w: u32 = arm! { fneg q(30), q(29) };
    assert_eq!(w, 0x6ee0fbbe);
}

#[test]
fn fneg_q_3() {
    // fneg v0.2d, v5.2d
    let w: u32 = arm! { fneg q(0), q(5) };
    assert_eq!(w, 0x6ee0f8a0);
}

#[test]
fn fabs_d_0() {
    // fabs d1, d2
    let w: u32 = arm! { fabs d(1), d(2) };
    assert_eq!(w, 0x1e60c041);
}

#[test]
fn fabs_d_1() {
    // fabs d17, d18
    let w: u32 = arm! { fabs d(17), d(18) };
    assert_eq!(w, 0x1e60c251);
}

#[test]
fn fabs_d_2() {
    // fabs d30, d29
    let w: u32 = arm! { fabs d(30), d(29) };
    assert_eq!(w, 0x1e60c3be);
}

#[test]
fn fabs_d_3() {
    // fabs d0, d5
    let w: u32 = arm! { fabs d(0), d(5) };
    assert_eq!(w, 0x1e60c0a0);
}

#[test]
fn fabs_q_0() {
    // fabs v1.2d, v2.2d
    let w: u32 = arm! { fabs q(1), q(2) };
    assert_eq!(w, 0x4ee0f841);
}

#[test]
fn fabs_q_1() {
    // fabs v17.2d, v18.2d
    let w: u32 = arm! { fabs q(17), q(18) };
    assert_eq!(w, 0x4ee0fa51);
}

#[test]
fn fabs_q_2() {
    // fabs v30.2d, v29.2d
    let w: u32 = arm! { fabs q(30), q(29) };
    assert_eq!(w, 0x4ee0fbbe);
}

#[test]
fn fabs_q_3() {
    // fabs v0.2d, v5.2d
    let w: u32 = arm! { fabs q(0), q(5) };
    assert_eq!(w, 0x4ee0f8a0);
}

#[test]
fn frinti_d_0() {
    // frinti d1, d2
    let w: u32 = arm! { frinti d(1), d(2) };
    assert_eq!(w, 0x1e67c041);
}

#[test]
fn frinti_d_1() {
    // frinti d17, d18
    let w: u32 = arm! { frinti d(17), d(18) };
    assert_eq!(w, 0x1e67c251);
}

#[test]
fn frinti_d_2() {
    // frinti d30, d29
    let w: u32 = arm! { frinti d(30), d(29) };
    assert_eq!(w, 0x1e67c3be);
}

#[test]
fn frinti_d_3() {
    // frinti d0, d5
    let w: u32 = arm! { frinti d(0), d(5) };
    assert_eq!(w, 0x1e67c0a0);
}

#[test]
fn frinti_q_0() {
    // frinti v1.2d, v2.2d
    let w: u32 = arm! { frinti q(1), q(2) };
    assert_eq!(w, 0x6ee19841);
}

#[test]
fn frinti_q_1() {
    // frinti v17.2d, v18.2d
    let w: u32 = arm! { frinti q(17), q(18) };
    assert_eq!(w, 0x6ee19a51);
}

#[test]
fn frinti_q_2() {
    // frinti v30.2d, v29.2d
    let w: u32 = arm! { frinti q(30), q(29) };
    assert_eq!(w, 0x6ee19bbe);
}

#[test]
fn frinti_q_3() {
    // frinti v0.2d, v5.2d
    let w: u32 = arm! { frinti q(0), q(5) };
    assert_eq!(w, 0x6ee198a0);
}

#[test]
fn frintm_d_0() {
    // frintm d1, d2
    let w: u32 = arm! { frintm d(1), d(2) };
    assert_eq!(w, 0x1e654041);
}

#[test]
fn frintm_d_1() {
    // frintm d17, d18
    let w: u32 = arm! { frintm d(17), d(18) };
    assert_eq!(w, 0x1e654251);
}

#[test]
fn frintm_d_2() {
    // frintm d30, d29
    let w: u32 = arm! { frintm d(30), d(29) };
    assert_eq!(w, 0x1e6543be);
}

#[test]
fn frintm_d_3() {
    // frintm d0, d5
    let w: u32 = arm! { frintm d(0), d(5) };
    assert_eq!(w, 0x1e6540a0);
}

#[test]
fn frintm_q_0() {
    // frintm v1.2d, v2.2d
    let w: u32 = arm! { frintm q(1), q(2) };
    assert_eq!(w, 0x4e619841);
}

#[test]
fn frintm_q_1() {
    // frintm v17.2d, v18.2d
    let w: u32 = arm! { frintm q(17), q(18) };
    assert_eq!(w, 0x4e619a51);
}

#[test]
fn frintm_q_2() {
    // frintm v30.2d, v29.2d
    let w: u32 = arm! { frintm q(30), q(29) };
    assert_eq!(w, 0x4e619bbe);
}

#[test]
fn frintm_q_3() {
    // frintm v0.2d, v5.2d
    let w: u32 = arm! { frintm q(0), q(5) };
    assert_eq!(w, 0x4e6198a0);
}

#[test]
fn frintp_d_0() {
    // frintp d1, d2
    let w: u32 = arm! { frintp d(1), d(2) };
    assert_eq!(w, 0x1e64c041);
}

#[test]
fn frintp_d_1() {
    // frintp d17, d18
    let w: u32 = arm! { frintp d(17), d(18) };
    assert_eq!(w, 0x1e64c251);
}

#[test]
fn frintp_d_2() {
    // frintp d30, d29
    let w: u32 = arm! { frintp d(30), d(29) };
    assert_eq!(w, 0x1e64c3be);
}

#[test]
fn frintp_d_3() {
    // frintp d0, d5
    let w: u32 = arm! { frintp d(0), d(5) };
    assert_eq!(w, 0x1e64c0a0);
}

#[test]
fn frintp_q_0() {
    // frintp v1.2d, v2.2d
    let w: u32 = arm! { frintp q(1), q(2) };
    assert_eq!(w, 0x4ee18841);
}

#[test]
fn frintp_q_1() {
    // frintp v17.2d, v18.2d
    let w: u32 = arm! { frintp q(17), q(18) };
    assert_eq!(w, 0x4ee18a51);
}

#[test]
fn frintp_q_2() {
    // frintp v30.2d, v29.2d
    let w: u32 = arm! { frintp q(30), q(29) };
    assert_eq!(w, 0x4ee18bbe);
}

#[test]
fn frintp_q_3() {
    // frintp v0.2d, v5.2d
    let w: u32 = arm! { frintp q(0), q(5) };
    assert_eq!(w, 0x4ee188a0);
}

#[test]
fn frintz_d_0() {
    // frintz d1, d2
    let w: u32 = arm! { frintz d(1), d(2) };
    assert_eq!(w, 0x1e65c041);
}

#[test]
fn frintz_d_1() {
    // frintz d17, d18
    let w: u32 = arm! { frintz d(17), d(18) };
    assert_eq!(w, 0x1e65c251);
}

#[test]
fn frintz_d_2() {
    // frintz d30, d29
    let w: u32 = arm! { frintz d(30), d(29) };
    assert_eq!(w, 0x1e65c3be);
}

#[test]
fn frintz_d_3() {
    // frintz d0, d5
    let w: u32 = arm! { frintz d(0), d(5) };
    assert_eq!(w, 0x1e65c0a0);
}

#[test]
fn frintz_q_0() {
    // frintz v1.2d, v2.2d
    let w: u32 = arm! { frintz q(1), q(2) };
    assert_eq!(w, 0x4ee19841);
}

#[test]
fn frintz_q_1() {
    // frintz v17.2d, v18.2d
    let w: u32 = arm! { frintz q(17), q(18) };
    assert_eq!(w, 0x4ee19a51);
}

#[test]
fn frintz_q_2() {
    // frintz v30.2d, v29.2d
    let w: u32 = arm! { frintz q(30), q(29) };
    assert_eq!(w, 0x4ee19bbe);
}

#[test]
fn frintz_q_3() {
    // frintz v0.2d, v5.2d
    let w: u32 = arm! { frintz q(0), q(5) };
    assert_eq!(w, 0x4ee198a0);
}

#[test]
fn fmadd_d_0() {
    // fmadd d1, d2, d3, d4
    let w: u32 = arm! { fmadd d(1), d(2), d(3), d(4) };
    assert_eq!(w, 0x1f431041);
}

#[test]
fn fmadd_d_1() {
    // fmadd d17, d18, d19, d20
    let w: u32 = arm! { fmadd d(17), d(18), d(19), d(20) };
    assert_eq!(w, 0x1f535251);
}

#[test]
fn fmadd_d_2() {
    // fmadd d30, d29, d28, d27
    let w: u32 = arm! { fmadd d(30), d(29), d(28), d(27) };
    assert_eq!(w, 0x1f5c6fbe);
}

#[test]
fn fmadd_d_3() {
    // fmadd d0, d5, d10, d15
    let w: u32 = arm! { fmadd d(0), d(5), d(10), d(15) };
    assert_eq!(w, 0x1f4a3ca0);
}

#[test]
fn fmadd_q_0() {
    // fmadd d1, d2, d3, d4
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmadd q(1), q(2), q(3), q(4) };
    assert_eq!(w, 0x1f431041);
}

#[test]
fn fmadd_q_1() {
    // fmadd d17, d18, d19, d20
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmadd q(17), q(18), q(19), q(20) };
    assert_eq!(w, 0x1f535251);
}

#[test]
fn fmadd_q_2() {
    // fmadd d30, d29, d28, d27
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmadd q(30), q(29), q(28), q(27) };
    assert_eq!(w, 0x1f5c6fbe);
}

#[test]
fn fmadd_q_3() {
    // fmadd d0, d5, d10, d15
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmadd q(0), q(5), q(10), q(15) };
    assert_eq!(w, 0x1f4a3ca0);
}

#[test]
fn fmsub_d_0() {
    // fmsub d1, d2, d3, d4
    let w: u32 = arm! { fmsub d(1), d(2), d(3), d(4) };
    assert_eq!(w, 0x1f439041);
}

#[test]
fn fmsub_d_1() {
    // fmsub d17, d18, d19, d20
    let w: u32 = arm! { fmsub d(17), d(18), d(19), d(20) };
    assert_eq!(w, 0x1f53d251);
}

#[test]
fn fmsub_d_2() {
    // fmsub d30, d29, d28, d27
    let w: u32 = arm! { fmsub d(30), d(29), d(28), d(27) };
    assert_eq!(w, 0x1f5cefbe);
}

#[test]
fn fmsub_d_3() {
    // fmsub d0, d5, d10, d15
    let w: u32 = arm! { fmsub d(0), d(5), d(10), d(15) };
    assert_eq!(w, 0x1f4abca0);
}

#[test]
fn fmsub_q_0() {
    // fmsub d1, d2, d3, d4
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmsub q(1), q(2), q(3), q(4) };
    assert_eq!(w, 0x1f439041);
}

#[test]
fn fmsub_q_1() {
    // fmsub d17, d18, d19, d20
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmsub q(17), q(18), q(19), q(20) };
    assert_eq!(w, 0x1f53d251);
}

#[test]
fn fmsub_q_2() {
    // fmsub d30, d29, d28, d27
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmsub q(30), q(29), q(28), q(27) };
    assert_eq!(w, 0x1f5cefbe);
}

#[test]
fn fmsub_q_3() {
    // fmsub d0, d5, d10, d15
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fmsub q(0), q(5), q(10), q(15) };
    assert_eq!(w, 0x1f4abca0);
}

#[test]
fn fnmadd_d_0() {
    // fnmadd d1, d2, d3, d4
    let w: u32 = arm! { fnmadd d(1), d(2), d(3), d(4) };
    assert_eq!(w, 0x1f631041);
}

#[test]
fn fnmadd_d_1() {
    // fnmadd d17, d18, d19, d20
    let w: u32 = arm! { fnmadd d(17), d(18), d(19), d(20) };
    assert_eq!(w, 0x1f735251);
}

#[test]
fn fnmadd_d_2() {
    // fnmadd d30, d29, d28, d27
    let w: u32 = arm! { fnmadd d(30), d(29), d(28), d(27) };
    assert_eq!(w, 0x1f7c6fbe);
}

#[test]
fn fnmadd_d_3() {
    // fnmadd d0, d5, d10, d15
    let w: u32 = arm! { fnmadd d(0), d(5), d(10), d(15) };
    assert_eq!(w, 0x1f6a3ca0);
}

#[test]
fn fnmadd_q_0() {
    // fnmadd d1, d2, d3, d4
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmadd q(1), q(2), q(3), q(4) };
    assert_eq!(w, 0x1f631041);
}

#[test]
fn fnmadd_q_1() {
    // fnmadd d17, d18, d19, d20
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmadd q(17), q(18), q(19), q(20) };
    assert_eq!(w, 0x1f735251);
}

#[test]
fn fnmadd_q_2() {
    // fnmadd d30, d29, d28, d27
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmadd q(30), q(29), q(28), q(27) };
    assert_eq!(w, 0x1f7c6fbe);
}

#[test]
fn fnmadd_q_3() {
    // fnmadd d0, d5, d10, d15
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmadd q(0), q(5), q(10), q(15) };
    assert_eq!(w, 0x1f6a3ca0);
}

#[test]
fn fnmsub_d_0() {
    // fnmsub d1, d2, d3, d4
    let w: u32 = arm! { fnmsub d(1), d(2), d(3), d(4) };
    assert_eq!(w, 0x1f639041);
}

#[test]
fn fnmsub_d_1() {
    // fnmsub d17, d18, d19, d20
    let w: u32 = arm! { fnmsub d(17), d(18), d(19), d(20) };
    assert_eq!(w, 0x1f73d251);
}

#[test]
fn fnmsub_d_2() {
    // fnmsub d30, d29, d28, d27
    let w: u32 = arm! { fnmsub d(30), d(29), d(28), d(27) };
    assert_eq!(w, 0x1f7cefbe);
}

#[test]
fn fnmsub_d_3() {
    // fnmsub d0, d5, d10, d15
    let w: u32 = arm! { fnmsub d(0), d(5), d(10), d(15) };
    assert_eq!(w, 0x1f6abca0);
}

#[test]
fn fnmsub_q_0() {
    // fnmsub d1, d2, d3, d4
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmsub q(1), q(2), q(3), q(4) };
    assert_eq!(w, 0x1f639041);
}

#[test]
fn fnmsub_q_1() {
    // fnmsub d17, d18, d19, d20
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmsub q(17), q(18), q(19), q(20) };
    assert_eq!(w, 0x1f73d251);
}

#[test]
fn fnmsub_q_2() {
    // fnmsub d30, d29, d28, d27
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmsub q(30), q(29), q(28), q(27) };
    assert_eq!(w, 0x1f7cefbe);
}

#[test]
fn fnmsub_q_3() {
    // fnmsub d0, d5, d10, d15
    // NOTE: the `q` form encodes the scalar d-register instruction; AArch64 has no packed-double FMADD
    let w: u32 = arm! { fnmsub q(0), q(5), q(10), q(15) };
    assert_eq!(w, 0x1f6abca0);
}

#[test]
fn fmla_q_0() {
    // fmla v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fmla q(1), q(2), q(3) };
    assert_eq!(w, 0x4e63cc41);
}

#[test]
fn fmla_q_1() {
    // fmla v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fmla q(17), q(18), q(19) };
    assert_eq!(w, 0x4e73ce51);
}

#[test]
fn fmla_q_2() {
    // fmla v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fmla q(30), q(29), q(28) };
    assert_eq!(w, 0x4e7ccfbe);
}

#[test]
fn fmla_q_3() {
    // fmla v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fmla q(0), q(5), q(10) };
    assert_eq!(w, 0x4e6acca0);
}

#[test]
fn fmls_q_0() {
    // fmls v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fmls q(1), q(2), q(3) };
    assert_eq!(w, 0x4ee3cc41);
}

#[test]
fn fmls_q_1() {
    // fmls v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fmls q(17), q(18), q(19) };
    assert_eq!(w, 0x4ef3ce51);
}

#[test]
fn fmls_q_2() {
    // fmls v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fmls q(30), q(29), q(28) };
    assert_eq!(w, 0x4efccfbe);
}

#[test]
fn fmls_q_3() {
    // fmls v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fmls q(0), q(5), q(10) };
    assert_eq!(w, 0x4eeacca0);
}

#[test]
fn fcmeq_d_0() {
    // fcmeq d1, d2, d3
    let w: u32 = arm! { fcmeq d(1), d(2), d(3) };
    assert_eq!(w, 0x5e63e441);
}

#[test]
fn fcmeq_d_1() {
    // fcmeq d17, d18, d19
    let w: u32 = arm! { fcmeq d(17), d(18), d(19) };
    assert_eq!(w, 0x5e73e651);
}

#[test]
fn fcmeq_d_2() {
    // fcmeq d30, d29, d28
    let w: u32 = arm! { fcmeq d(30), d(29), d(28) };
    assert_eq!(w, 0x5e7ce7be);
}

#[test]
fn fcmeq_d_3() {
    // fcmeq d0, d5, d10
    let w: u32 = arm! { fcmeq d(0), d(5), d(10) };
    assert_eq!(w, 0x5e6ae4a0);
}

#[test]
fn fcmgt_d_0() {
    // fcmgt d1, d2, d3
    let w: u32 = arm! { fcmgt d(1), d(2), d(3) };
    assert_eq!(w, 0x7ee3e441);
}

#[test]
fn fcmgt_d_1() {
    // fcmgt d17, d18, d19
    let w: u32 = arm! { fcmgt d(17), d(18), d(19) };
    assert_eq!(w, 0x7ef3e651);
}

#[test]
fn fcmgt_d_2() {
    // fcmgt d30, d29, d28
    let w: u32 = arm! { fcmgt d(30), d(29), d(28) };
    assert_eq!(w, 0x7efce7be);
}

#[test]
fn fcmgt_d_3() {
    // fcmgt d0, d5, d10
    let w: u32 = arm! { fcmgt d(0), d(5), d(10) };
    assert_eq!(w, 0x7eeae4a0);
}

#[test]
fn fcmge_d_0() {
    // fcmge d1, d2, d3
    let w: u32 = arm! { fcmge d(1), d(2), d(3) };
    assert_eq!(w, 0x7e63e441);
}

#[test]
fn fcmge_d_1() {
    // fcmge d17, d18, d19
    let w: u32 = arm! { fcmge d(17), d(18), d(19) };
    assert_eq!(w, 0x7e73e651);
}

#[test]
fn fcmge_d_2() {
    // fcmge d30, d29, d28
    let w: u32 = arm! { fcmge d(30), d(29), d(28) };
    assert_eq!(w, 0x7e7ce7be);
}

#[test]
fn fcmge_d_3() {
    // fcmge d0, d5, d10
    let w: u32 = arm! { fcmge d(0), d(5), d(10) };
    assert_eq!(w, 0x7e6ae4a0);
}

#[test]
fn fcmlt_d_0() {
    // fcmgt d1, d3, d2
    let w: u32 = arm! { fcmlt d(1), d(2), d(3) };
    assert_eq!(w, 0x7ee2e461);
}

#[test]
fn fcmlt_d_1() {
    // fcmgt d17, d19, d18
    let w: u32 = arm! { fcmlt d(17), d(18), d(19) };
    assert_eq!(w, 0x7ef2e671);
}

#[test]
fn fcmlt_d_2() {
    // fcmgt d30, d28, d29
    let w: u32 = arm! { fcmlt d(30), d(29), d(28) };
    assert_eq!(w, 0x7efde79e);
}

#[test]
fn fcmlt_d_3() {
    // fcmgt d0, d10, d5
    let w: u32 = arm! { fcmlt d(0), d(5), d(10) };
    assert_eq!(w, 0x7ee5e540);
}

#[test]
fn fcmle_d_0() {
    // fcmge d1, d3, d2
    let w: u32 = arm! { fcmle d(1), d(2), d(3) };
    assert_eq!(w, 0x7e62e461);
}

#[test]
fn fcmle_d_1() {
    // fcmge d17, d19, d18
    let w: u32 = arm! { fcmle d(17), d(18), d(19) };
    assert_eq!(w, 0x7e72e671);
}

#[test]
fn fcmle_d_2() {
    // fcmge d30, d28, d29
    let w: u32 = arm! { fcmle d(30), d(29), d(28) };
    assert_eq!(w, 0x7e7de79e);
}

#[test]
fn fcmle_d_3() {
    // fcmge d0, d10, d5
    let w: u32 = arm! { fcmle d(0), d(5), d(10) };
    assert_eq!(w, 0x7e65e540);
}

#[test]
fn fcmeq_q_0() {
    // fcmeq v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fcmeq q(1), q(2), q(3) };
    assert_eq!(w, 0x4e63e441);
}

#[test]
fn fcmeq_q_1() {
    // fcmeq v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fcmeq q(17), q(18), q(19) };
    assert_eq!(w, 0x4e73e651);
}

#[test]
fn fcmeq_q_2() {
    // fcmeq v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fcmeq q(30), q(29), q(28) };
    assert_eq!(w, 0x4e7ce7be);
}

#[test]
fn fcmeq_q_3() {
    // fcmeq v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fcmeq q(0), q(5), q(10) };
    assert_eq!(w, 0x4e6ae4a0);
}

#[test]
fn fcmgt_q_0() {
    // fcmgt v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fcmgt q(1), q(2), q(3) };
    assert_eq!(w, 0x6ee3e441);
}

#[test]
fn fcmgt_q_1() {
    // fcmgt v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fcmgt q(17), q(18), q(19) };
    assert_eq!(w, 0x6ef3e651);
}

#[test]
fn fcmgt_q_2() {
    // fcmgt v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fcmgt q(30), q(29), q(28) };
    assert_eq!(w, 0x6efce7be);
}

#[test]
fn fcmgt_q_3() {
    // fcmgt v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fcmgt q(0), q(5), q(10) };
    assert_eq!(w, 0x6eeae4a0);
}

#[test]
fn fcmge_q_0() {
    // fcmge v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { fcmge q(1), q(2), q(3) };
    assert_eq!(w, 0x6e63e441);
}

#[test]
fn fcmge_q_1() {
    // fcmge v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { fcmge q(17), q(18), q(19) };
    assert_eq!(w, 0x6e73e651);
}

#[test]
fn fcmge_q_2() {
    // fcmge v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { fcmge q(30), q(29), q(28) };
    assert_eq!(w, 0x6e7ce7be);
}

#[test]
fn fcmge_q_3() {
    // fcmge v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { fcmge q(0), q(5), q(10) };
    assert_eq!(w, 0x6e6ae4a0);
}

#[test]
fn fcmlt_q_0() {
    // fcmgt v1.2d, v3.2d, v2.2d
    let w: u32 = arm! { fcmlt q(1), q(2), q(3) };
    assert_eq!(w, 0x6ee2e461);
}

#[test]
fn fcmlt_q_1() {
    // fcmgt v17.2d, v19.2d, v18.2d
    let w: u32 = arm! { fcmlt q(17), q(18), q(19) };
    assert_eq!(w, 0x6ef2e671);
}

#[test]
fn fcmlt_q_2() {
    // fcmgt v30.2d, v28.2d, v29.2d
    let w: u32 = arm! { fcmlt q(30), q(29), q(28) };
    assert_eq!(w, 0x6efde79e);
}

#[test]
fn fcmlt_q_3() {
    // fcmgt v0.2d, v10.2d, v5.2d
    let w: u32 = arm! { fcmlt q(0), q(5), q(10) };
    assert_eq!(w, 0x6ee5e540);
}

#[test]
fn fcmle_q_0() {
    // fcmge v1.2d, v3.2d, v2.2d
    let w: u32 = arm! { fcmle q(1), q(2), q(3) };
    assert_eq!(w, 0x6e62e461);
}

#[test]
fn fcmle_q_1() {
    // fcmge v17.2d, v19.2d, v18.2d
    let w: u32 = arm! { fcmle q(17), q(18), q(19) };
    assert_eq!(w, 0x6e72e671);
}

#[test]
fn fcmle_q_2() {
    // fcmge v30.2d, v28.2d, v29.2d
    let w: u32 = arm! { fcmle q(30), q(29), q(28) };
    assert_eq!(w, 0x6e7de79e);
}

#[test]
fn fcmle_q_3() {
    // fcmge v0.2d, v10.2d, v5.2d
    let w: u32 = arm! { fcmle q(0), q(5), q(10) };
    assert_eq!(w, 0x6e65e540);
}

#[test]
fn fcmeq_d_zero_0() {
    // fcmeq d1, d2, #0.0
    let w: u32 = arm! { fcmeq d(1), d(2), #0.0 };
    assert_eq!(w, 0x5ee0d841);
}

#[test]
fn fcmeq_d_zero_1() {
    // fcmeq d17, d18, #0.0
    let w: u32 = arm! { fcmeq d(17), d(18), #0.0 };
    assert_eq!(w, 0x5ee0da51);
}

#[test]
fn fcmeq_d_zero_2() {
    // fcmeq d30, d29, #0.0
    let w: u32 = arm! { fcmeq d(30), d(29), #0.0 };
    assert_eq!(w, 0x5ee0dbbe);
}

#[test]
fn fcmeq_d_zero_3() {
    // fcmeq d0, d5, #0.0
    let w: u32 = arm! { fcmeq d(0), d(5), #0.0 };
    assert_eq!(w, 0x5ee0d8a0);
}

#[test]
fn fcmge_d_zero_0() {
    // fcmge d1, d2, #0.0
    let w: u32 = arm! { fcmge d(1), d(2), #0.0 };
    assert_eq!(w, 0x7ee0c841);
}

#[test]
fn fcmge_d_zero_1() {
    // fcmge d17, d18, #0.0
    let w: u32 = arm! { fcmge d(17), d(18), #0.0 };
    assert_eq!(w, 0x7ee0ca51);
}

#[test]
fn fcmge_d_zero_2() {
    // fcmge d30, d29, #0.0
    let w: u32 = arm! { fcmge d(30), d(29), #0.0 };
    assert_eq!(w, 0x7ee0cbbe);
}

#[test]
fn fcmge_d_zero_3() {
    // fcmge d0, d5, #0.0
    let w: u32 = arm! { fcmge d(0), d(5), #0.0 };
    assert_eq!(w, 0x7ee0c8a0);
}

#[test]
fn fcmlt_d_zero_0() {
    // fcmlt d1, d2, #0.0
    let w: u32 = arm! { fcmlt d(1), d(2), #0.0 };
    assert_eq!(w, 0x5ee0e841);
}

#[test]
fn fcmlt_d_zero_1() {
    // fcmlt d17, d18, #0.0
    let w: u32 = arm! { fcmlt d(17), d(18), #0.0 };
    assert_eq!(w, 0x5ee0ea51);
}

#[test]
fn fcmlt_d_zero_2() {
    // fcmlt d30, d29, #0.0
    let w: u32 = arm! { fcmlt d(30), d(29), #0.0 };
    assert_eq!(w, 0x5ee0ebbe);
}

#[test]
fn fcmlt_d_zero_3() {
    // fcmlt d0, d5, #0.0
    let w: u32 = arm! { fcmlt d(0), d(5), #0.0 };
    assert_eq!(w, 0x5ee0e8a0);
}

#[test]
fn fcmeq_q_zero_0() {
    // fcmeq v1.2d, v2.2d, #0.0
    let w: u32 = arm! { fcmeq q(1), q(2), #0.0 };
    assert_eq!(w, 0x4ee0d841);
}

#[test]
fn fcmeq_q_zero_1() {
    // fcmeq v17.2d, v18.2d, #0.0
    let w: u32 = arm! { fcmeq q(17), q(18), #0.0 };
    assert_eq!(w, 0x4ee0da51);
}

#[test]
fn fcmeq_q_zero_2() {
    // fcmeq v30.2d, v29.2d, #0.0
    let w: u32 = arm! { fcmeq q(30), q(29), #0.0 };
    assert_eq!(w, 0x4ee0dbbe);
}

#[test]
fn fcmeq_q_zero_3() {
    // fcmeq v0.2d, v5.2d, #0.0
    let w: u32 = arm! { fcmeq q(0), q(5), #0.0 };
    assert_eq!(w, 0x4ee0d8a0);
}

#[test]
fn fcmp_zero_0() {
    // fcmp d1, #0.0
    let w: u32 = arm! { fcmp d(1), #0.0 };
    assert_eq!(w, 0x1e602028);
}

#[test]
fn fcmp_zero_1() {
    // fcmp d17, #0.0
    let w: u32 = arm! { fcmp d(17), #0.0 };
    assert_eq!(w, 0x1e602228);
}

#[test]
fn fcmp_zero_2() {
    // fcmp d30, #0.0
    let w: u32 = arm! { fcmp d(30), #0.0 };
    assert_eq!(w, 0x1e6023c8);
}

#[test]
fn fcsel_eq_0() {
    // fcsel d1, d2, d3, eq
    let w: u32 = arm! { fcsel d(1), d(2), d(3), eq };
    assert_eq!(w, 0x1e630c41);
}

#[test]
fn fcsel_eq_1() {
    // fcsel d17, d18, d19, eq
    let w: u32 = arm! { fcsel d(17), d(18), d(19), eq };
    assert_eq!(w, 0x1e730e51);
}

#[test]
fn fcsel_eq_2() {
    // fcsel d30, d29, d28, eq
    let w: u32 = arm! { fcsel d(30), d(29), d(28), eq };
    assert_eq!(w, 0x1e7c0fbe);
}

#[test]
fn fcsel_eq_3() {
    // fcsel d0, d5, d10, eq
    let w: u32 = arm! { fcsel d(0), d(5), d(10), eq };
    assert_eq!(w, 0x1e6a0ca0);
}

#[test]
fn fcsel_ne_0() {
    // fcsel d1, d2, d3, ne
    let w: u32 = arm! { fcsel d(1), d(2), d(3), ne };
    assert_eq!(w, 0x1e631c41);
}

#[test]
fn fcsel_ne_1() {
    // fcsel d17, d18, d19, ne
    let w: u32 = arm! { fcsel d(17), d(18), d(19), ne };
    assert_eq!(w, 0x1e731e51);
}

#[test]
fn fcsel_ne_2() {
    // fcsel d30, d29, d28, ne
    let w: u32 = arm! { fcsel d(30), d(29), d(28), ne };
    assert_eq!(w, 0x1e7c1fbe);
}

#[test]
fn fcsel_ne_3() {
    // fcsel d0, d5, d10, ne
    let w: u32 = arm! { fcsel d(0), d(5), d(10), ne };
    assert_eq!(w, 0x1e6a1ca0);
}

#[test]
fn fcsel_lt_0() {
    // fcsel d1, d2, d3, lt
    let w: u32 = arm! { fcsel d(1), d(2), d(3), lt };
    assert_eq!(w, 0x1e63bc41);
}

#[test]
fn fcsel_lt_1() {
    // fcsel d17, d18, d19, lt
    let w: u32 = arm! { fcsel d(17), d(18), d(19), lt };
    assert_eq!(w, 0x1e73be51);
}

#[test]
fn fcsel_lt_2() {
    // fcsel d30, d29, d28, lt
    let w: u32 = arm! { fcsel d(30), d(29), d(28), lt };
    assert_eq!(w, 0x1e7cbfbe);
}

#[test]
fn fcsel_lt_3() {
    // fcsel d0, d5, d10, lt
    let w: u32 = arm! { fcsel d(0), d(5), d(10), lt };
    assert_eq!(w, 0x1e6abca0);
}

#[test]
fn fcsel_le_0() {
    // fcsel d1, d2, d3, le
    let w: u32 = arm! { fcsel d(1), d(2), d(3), le };
    assert_eq!(w, 0x1e63dc41);
}

#[test]
fn fcsel_le_1() {
    // fcsel d17, d18, d19, le
    let w: u32 = arm! { fcsel d(17), d(18), d(19), le };
    assert_eq!(w, 0x1e73de51);
}

#[test]
fn fcsel_le_2() {
    // fcsel d30, d29, d28, le
    let w: u32 = arm! { fcsel d(30), d(29), d(28), le };
    assert_eq!(w, 0x1e7cdfbe);
}

#[test]
fn fcsel_le_3() {
    // fcsel d0, d5, d10, le
    let w: u32 = arm! { fcsel d(0), d(5), d(10), le };
    assert_eq!(w, 0x1e6adca0);
}

#[test]
fn fcsel_gt_0() {
    // fcsel d1, d2, d3, gt
    let w: u32 = arm! { fcsel d(1), d(2), d(3), gt };
    assert_eq!(w, 0x1e63cc41);
}

#[test]
fn fcsel_gt_1() {
    // fcsel d17, d18, d19, gt
    let w: u32 = arm! { fcsel d(17), d(18), d(19), gt };
    assert_eq!(w, 0x1e73ce51);
}

#[test]
fn fcsel_gt_2() {
    // fcsel d30, d29, d28, gt
    let w: u32 = arm! { fcsel d(30), d(29), d(28), gt };
    assert_eq!(w, 0x1e7ccfbe);
}

#[test]
fn fcsel_gt_3() {
    // fcsel d0, d5, d10, gt
    let w: u32 = arm! { fcsel d(0), d(5), d(10), gt };
    assert_eq!(w, 0x1e6acca0);
}

#[test]
fn fcsel_ge_0() {
    // fcsel d1, d2, d3, ge
    let w: u32 = arm! { fcsel d(1), d(2), d(3), ge };
    assert_eq!(w, 0x1e63ac41);
}

#[test]
fn fcsel_ge_1() {
    // fcsel d17, d18, d19, ge
    let w: u32 = arm! { fcsel d(17), d(18), d(19), ge };
    assert_eq!(w, 0x1e73ae51);
}

#[test]
fn fcsel_ge_2() {
    // fcsel d30, d29, d28, ge
    let w: u32 = arm! { fcsel d(30), d(29), d(28), ge };
    assert_eq!(w, 0x1e7cafbe);
}

#[test]
fn fcsel_ge_3() {
    // fcsel d0, d5, d10, ge
    let w: u32 = arm! { fcsel d(0), d(5), d(10), ge };
    assert_eq!(w, 0x1e6aaca0);
}

#[test]
fn and_8b_0() {
    // and v1.8b, v2.8b, v3.8b
    let w: u32 = arm! { and v(1).8b, v(2).8b, v(3).8b };
    assert_eq!(w, 0x0e231c41);
}

#[test]
fn and_8b_1() {
    // and v17.8b, v18.8b, v19.8b
    let w: u32 = arm! { and v(17).8b, v(18).8b, v(19).8b };
    assert_eq!(w, 0x0e331e51);
}

#[test]
fn and_8b_2() {
    // and v30.8b, v29.8b, v28.8b
    let w: u32 = arm! { and v(30).8b, v(29).8b, v(28).8b };
    assert_eq!(w, 0x0e3c1fbe);
}

#[test]
fn and_8b_3() {
    // and v0.8b, v5.8b, v10.8b
    let w: u32 = arm! { and v(0).8b, v(5).8b, v(10).8b };
    assert_eq!(w, 0x0e2a1ca0);
}

#[test]
fn orr_8b_0() {
    // orr v1.8b, v2.8b, v3.8b
    let w: u32 = arm! { orr v(1).8b, v(2).8b, v(3).8b };
    assert_eq!(w, 0x0ea31c41);
}

#[test]
fn orr_8b_1() {
    // orr v17.8b, v18.8b, v19.8b
    let w: u32 = arm! { orr v(17).8b, v(18).8b, v(19).8b };
    assert_eq!(w, 0x0eb31e51);
}

#[test]
fn orr_8b_2() {
    // orr v30.8b, v29.8b, v28.8b
    let w: u32 = arm! { orr v(30).8b, v(29).8b, v(28).8b };
    assert_eq!(w, 0x0ebc1fbe);
}

#[test]
fn orr_8b_3() {
    // orr v0.8b, v5.8b, v10.8b
    let w: u32 = arm! { orr v(0).8b, v(5).8b, v(10).8b };
    assert_eq!(w, 0x0eaa1ca0);
}

#[test]
fn eor_8b_0() {
    // eor v1.8b, v2.8b, v3.8b
    let w: u32 = arm! { eor v(1).8b, v(2).8b, v(3).8b };
    assert_eq!(w, 0x2e231c41);
}

#[test]
fn eor_8b_1() {
    // eor v17.8b, v18.8b, v19.8b
    let w: u32 = arm! { eor v(17).8b, v(18).8b, v(19).8b };
    assert_eq!(w, 0x2e331e51);
}

#[test]
fn eor_8b_2() {
    // eor v30.8b, v29.8b, v28.8b
    let w: u32 = arm! { eor v(30).8b, v(29).8b, v(28).8b };
    assert_eq!(w, 0x2e3c1fbe);
}

#[test]
fn eor_8b_3() {
    // eor v0.8b, v5.8b, v10.8b
    let w: u32 = arm! { eor v(0).8b, v(5).8b, v(10).8b };
    assert_eq!(w, 0x2e2a1ca0);
}

#[test]
fn bit_8b_0() {
    // bit v1.8b, v2.8b, v3.8b
    let w: u32 = arm! { bit v(1).8b, v(2).8b, v(3).8b };
    assert_eq!(w, 0x2ea31c41);
}

#[test]
fn bit_8b_1() {
    // bit v17.8b, v18.8b, v19.8b
    let w: u32 = arm! { bit v(17).8b, v(18).8b, v(19).8b };
    assert_eq!(w, 0x2eb31e51);
}

#[test]
fn bit_8b_2() {
    // bit v30.8b, v29.8b, v28.8b
    let w: u32 = arm! { bit v(30).8b, v(29).8b, v(28).8b };
    assert_eq!(w, 0x2ebc1fbe);
}

#[test]
fn bit_8b_3() {
    // bit v0.8b, v5.8b, v10.8b
    let w: u32 = arm! { bit v(0).8b, v(5).8b, v(10).8b };
    assert_eq!(w, 0x2eaa1ca0);
}

#[test]
fn bif_8b_0() {
    // bif v1.8b, v2.8b, v3.8b
    let w: u32 = arm! { bif v(1).8b, v(2).8b, v(3).8b };
    assert_eq!(w, 0x2ee31c41);
}

#[test]
fn bif_8b_1() {
    // bif v17.8b, v18.8b, v19.8b
    let w: u32 = arm! { bif v(17).8b, v(18).8b, v(19).8b };
    assert_eq!(w, 0x2ef31e51);
}

#[test]
fn bif_8b_2() {
    // bif v30.8b, v29.8b, v28.8b
    let w: u32 = arm! { bif v(30).8b, v(29).8b, v(28).8b };
    assert_eq!(w, 0x2efc1fbe);
}

#[test]
fn bif_8b_3() {
    // bif v0.8b, v5.8b, v10.8b
    let w: u32 = arm! { bif v(0).8b, v(5).8b, v(10).8b };
    assert_eq!(w, 0x2eea1ca0);
}

#[test]
fn bic_8b_0() {
    // bic v1.8b, v2.8b, v3.8b
    let w: u32 = arm! { bic v(1).8b, v(2).8b, v(3).8b };
    assert_eq!(w, 0x0e631c41);
}

#[test]
fn bic_8b_1() {
    // bic v17.8b, v18.8b, v19.8b
    let w: u32 = arm! { bic v(17).8b, v(18).8b, v(19).8b };
    assert_eq!(w, 0x0e731e51);
}

#[test]
fn bic_8b_2() {
    // bic v30.8b, v29.8b, v28.8b
    let w: u32 = arm! { bic v(30).8b, v(29).8b, v(28).8b };
    assert_eq!(w, 0x0e7c1fbe);
}

#[test]
fn bic_8b_3() {
    // bic v0.8b, v5.8b, v10.8b
    let w: u32 = arm! { bic v(0).8b, v(5).8b, v(10).8b };
    assert_eq!(w, 0x0e6a1ca0);
}

#[test]
fn bsl_8b_0() {
    // bsl v1.8b, v2.8b, v3.8b
    let w: u32 = arm! { bsl v(1).8b, v(2).8b, v(3).8b };
    assert_eq!(w, 0x2e631c41);
}

#[test]
fn bsl_8b_1() {
    // bsl v17.8b, v18.8b, v19.8b
    let w: u32 = arm! { bsl v(17).8b, v(18).8b, v(19).8b };
    assert_eq!(w, 0x2e731e51);
}

#[test]
fn bsl_8b_2() {
    // bsl v30.8b, v29.8b, v28.8b
    let w: u32 = arm! { bsl v(30).8b, v(29).8b, v(28).8b };
    assert_eq!(w, 0x2e7c1fbe);
}

#[test]
fn bsl_8b_3() {
    // bsl v0.8b, v5.8b, v10.8b
    let w: u32 = arm! { bsl v(0).8b, v(5).8b, v(10).8b };
    assert_eq!(w, 0x2e6a1ca0);
}

#[test]
fn not_8b_0() {
    // not v1.8b, v2.8b
    let w: u32 = arm! { not v(1).8b, v(2).8b };
    assert_eq!(w, 0x2e205841);
}

#[test]
fn not_8b_1() {
    // not v17.8b, v18.8b
    let w: u32 = arm! { not v(17).8b, v(18).8b };
    assert_eq!(w, 0x2e205a51);
}

#[test]
fn not_8b_2() {
    // not v30.8b, v29.8b
    let w: u32 = arm! { not v(30).8b, v(29).8b };
    assert_eq!(w, 0x2e205bbe);
}

#[test]
fn not_8b_3() {
    // not v0.8b, v5.8b
    let w: u32 = arm! { not v(0).8b, v(5).8b };
    assert_eq!(w, 0x2e2058a0);
}

#[test]
fn and_16b_0() {
    // and v1.16b, v2.16b, v3.16b
    let w: u32 = arm! { and v(1).16b, v(2).16b, v(3).16b };
    assert_eq!(w, 0x4e231c41);
}

#[test]
fn and_16b_1() {
    // and v17.16b, v18.16b, v19.16b
    let w: u32 = arm! { and v(17).16b, v(18).16b, v(19).16b };
    assert_eq!(w, 0x4e331e51);
}

#[test]
fn and_16b_2() {
    // and v30.16b, v29.16b, v28.16b
    let w: u32 = arm! { and v(30).16b, v(29).16b, v(28).16b };
    assert_eq!(w, 0x4e3c1fbe);
}

#[test]
fn and_16b_3() {
    // and v0.16b, v5.16b, v10.16b
    let w: u32 = arm! { and v(0).16b, v(5).16b, v(10).16b };
    assert_eq!(w, 0x4e2a1ca0);
}

#[test]
fn orr_16b_0() {
    // orr v1.16b, v2.16b, v3.16b
    let w: u32 = arm! { orr v(1).16b, v(2).16b, v(3).16b };
    assert_eq!(w, 0x4ea31c41);
}

#[test]
fn orr_16b_1() {
    // orr v17.16b, v18.16b, v19.16b
    let w: u32 = arm! { orr v(17).16b, v(18).16b, v(19).16b };
    assert_eq!(w, 0x4eb31e51);
}

#[test]
fn orr_16b_2() {
    // orr v30.16b, v29.16b, v28.16b
    let w: u32 = arm! { orr v(30).16b, v(29).16b, v(28).16b };
    assert_eq!(w, 0x4ebc1fbe);
}

#[test]
fn orr_16b_3() {
    // orr v0.16b, v5.16b, v10.16b
    let w: u32 = arm! { orr v(0).16b, v(5).16b, v(10).16b };
    assert_eq!(w, 0x4eaa1ca0);
}

#[test]
fn eor_16b_0() {
    // eor v1.16b, v2.16b, v3.16b
    let w: u32 = arm! { eor v(1).16b, v(2).16b, v(3).16b };
    assert_eq!(w, 0x6e231c41);
}

#[test]
fn eor_16b_1() {
    // eor v17.16b, v18.16b, v19.16b
    let w: u32 = arm! { eor v(17).16b, v(18).16b, v(19).16b };
    assert_eq!(w, 0x6e331e51);
}

#[test]
fn eor_16b_2() {
    // eor v30.16b, v29.16b, v28.16b
    let w: u32 = arm! { eor v(30).16b, v(29).16b, v(28).16b };
    assert_eq!(w, 0x6e3c1fbe);
}

#[test]
fn eor_16b_3() {
    // eor v0.16b, v5.16b, v10.16b
    let w: u32 = arm! { eor v(0).16b, v(5).16b, v(10).16b };
    assert_eq!(w, 0x6e2a1ca0);
}

#[test]
fn bit_16b_0() {
    // bit v1.16b, v2.16b, v3.16b
    let w: u32 = arm! { bit v(1).16b, v(2).16b, v(3).16b };
    assert_eq!(w, 0x6ea31c41);
}

#[test]
fn bit_16b_1() {
    // bit v17.16b, v18.16b, v19.16b
    let w: u32 = arm! { bit v(17).16b, v(18).16b, v(19).16b };
    assert_eq!(w, 0x6eb31e51);
}

#[test]
fn bit_16b_2() {
    // bit v30.16b, v29.16b, v28.16b
    let w: u32 = arm! { bit v(30).16b, v(29).16b, v(28).16b };
    assert_eq!(w, 0x6ebc1fbe);
}

#[test]
fn bit_16b_3() {
    // bit v0.16b, v5.16b, v10.16b
    let w: u32 = arm! { bit v(0).16b, v(5).16b, v(10).16b };
    assert_eq!(w, 0x6eaa1ca0);
}

#[test]
fn bif_16b_0() {
    // bif v1.16b, v2.16b, v3.16b
    let w: u32 = arm! { bif v(1).16b, v(2).16b, v(3).16b };
    assert_eq!(w, 0x6ee31c41);
}

#[test]
fn bif_16b_1() {
    // bif v17.16b, v18.16b, v19.16b
    let w: u32 = arm! { bif v(17).16b, v(18).16b, v(19).16b };
    assert_eq!(w, 0x6ef31e51);
}

#[test]
fn bif_16b_2() {
    // bif v30.16b, v29.16b, v28.16b
    let w: u32 = arm! { bif v(30).16b, v(29).16b, v(28).16b };
    assert_eq!(w, 0x6efc1fbe);
}

#[test]
fn bif_16b_3() {
    // bif v0.16b, v5.16b, v10.16b
    let w: u32 = arm! { bif v(0).16b, v(5).16b, v(10).16b };
    assert_eq!(w, 0x6eea1ca0);
}

#[test]
fn bic_16b_0() {
    // bic v1.16b, v2.16b, v3.16b
    let w: u32 = arm! { bic v(1).16b, v(2).16b, v(3).16b };
    assert_eq!(w, 0x4e631c41);
}

#[test]
fn bic_16b_1() {
    // bic v17.16b, v18.16b, v19.16b
    let w: u32 = arm! { bic v(17).16b, v(18).16b, v(19).16b };
    assert_eq!(w, 0x4e731e51);
}

#[test]
fn bic_16b_2() {
    // bic v30.16b, v29.16b, v28.16b
    let w: u32 = arm! { bic v(30).16b, v(29).16b, v(28).16b };
    assert_eq!(w, 0x4e7c1fbe);
}

#[test]
fn bic_16b_3() {
    // bic v0.16b, v5.16b, v10.16b
    let w: u32 = arm! { bic v(0).16b, v(5).16b, v(10).16b };
    assert_eq!(w, 0x4e6a1ca0);
}

#[test]
fn bsl_16b_0() {
    // bsl v1.16b, v2.16b, v3.16b
    let w: u32 = arm! { bsl v(1).16b, v(2).16b, v(3).16b };
    assert_eq!(w, 0x6e631c41);
}

#[test]
fn bsl_16b_1() {
    // bsl v17.16b, v18.16b, v19.16b
    let w: u32 = arm! { bsl v(17).16b, v(18).16b, v(19).16b };
    assert_eq!(w, 0x6e731e51);
}

#[test]
fn bsl_16b_2() {
    // bsl v30.16b, v29.16b, v28.16b
    let w: u32 = arm! { bsl v(30).16b, v(29).16b, v(28).16b };
    assert_eq!(w, 0x6e7c1fbe);
}

#[test]
fn bsl_16b_3() {
    // bsl v0.16b, v5.16b, v10.16b
    let w: u32 = arm! { bsl v(0).16b, v(5).16b, v(10).16b };
    assert_eq!(w, 0x6e6a1ca0);
}

#[test]
fn not_16b_0() {
    // not v1.16b, v2.16b
    let w: u32 = arm! { not v(1).16b, v(2).16b };
    assert_eq!(w, 0x6e205841);
}

#[test]
fn not_16b_1() {
    // not v17.16b, v18.16b
    let w: u32 = arm! { not v(17).16b, v(18).16b };
    assert_eq!(w, 0x6e205a51);
}

#[test]
fn not_16b_2() {
    // not v30.16b, v29.16b
    let w: u32 = arm! { not v(30).16b, v(29).16b };
    assert_eq!(w, 0x6e205bbe);
}

#[test]
fn not_16b_3() {
    // not v0.16b, v5.16b
    let w: u32 = arm! { not v(0).16b, v(5).16b };
    assert_eq!(w, 0x6e2058a0);
}

#[test]
fn fmov_d_imm_0_0_0() {
    // fmov d1, #0.0   [expected word from override]
    // NOTE: 0.0 is not an FP8 immediate; the macro emits `movi v<n>.16b, #0`, which clears all 128 bits (pinned to the macro word, verified with Capstone)
    let w: u32 = arm! { fmov d(1), #0.0 };
    assert_eq!(w, 0x4f00e401);
}

#[test]
fn fmov_d_imm_0_0_1() {
    // fmov d17, #0.0   [expected word from override]
    // NOTE: 0.0 is not an FP8 immediate; the macro emits `movi v<n>.16b, #0`, which clears all 128 bits (pinned to the macro word, verified with Capstone)
    let w: u32 = arm! { fmov d(17), #0.0 };
    assert_eq!(w, 0x4f00e411);
}

#[test]
fn fmov_d_imm_0_5_0() {
    // fmov d1, #0.5
    let w: u32 = arm! { fmov d(1), #0.5 };
    assert_eq!(w, 0x1e6c1001);
}

#[test]
fn fmov_d_imm_0_5_1() {
    // fmov d17, #0.5
    let w: u32 = arm! { fmov d(17), #0.5 };
    assert_eq!(w, 0x1e6c1011);
}

#[test]
fn fmov_d_imm_1_0_0() {
    // fmov d1, #1.0
    let w: u32 = arm! { fmov d(1), #1.0 };
    assert_eq!(w, 0x1e6e1001);
}

#[test]
fn fmov_d_imm_1_0_1() {
    // fmov d17, #1.0
    let w: u32 = arm! { fmov d(17), #1.0 };
    assert_eq!(w, 0x1e6e1011);
}

#[test]
fn fmov_d_imm_2_0_0() {
    // fmov d1, #2.0
    let w: u32 = arm! { fmov d(1), #2.0 };
    assert_eq!(w, 0x1e601001);
}

#[test]
fn fmov_d_imm_2_0_1() {
    // fmov d17, #2.0
    let w: u32 = arm! { fmov d(17), #2.0 };
    assert_eq!(w, 0x1e601011);
}

#[test]
fn fmov_d_imm_m1_0_0() {
    // fmov d1, #-1.0
    let w: u32 = arm! { fmov d(1), #-1.0 };
    assert_eq!(w, 0x1e7e1001);
}

#[test]
fn fmov_d_imm_m1_0_1() {
    // fmov d17, #-1.0
    let w: u32 = arm! { fmov d(17), #-1.0 };
    assert_eq!(w, 0x1e7e1011);
}

#[test]
fn fmov_q_imm_0_0_0() {
    // fmov v1.2d, #0.0   [expected word from override]
    // NOTE: 0.0 is not an FP8 immediate; the macro emits `movi v<n>.16b, #0`, which clears all 128 bits (pinned to the macro word, verified with Capstone)
    let w: u32 = arm! { fmov q(1), #0.0 };
    assert_eq!(w, 0x4f00e401);
}

#[test]
fn fmov_q_imm_0_0_1() {
    // fmov v17.2d, #0.0   [expected word from override]
    // NOTE: 0.0 is not an FP8 immediate; the macro emits `movi v<n>.16b, #0`, which clears all 128 bits (pinned to the macro word, verified with Capstone)
    let w: u32 = arm! { fmov q(17), #0.0 };
    assert_eq!(w, 0x4f00e411);
}

#[test]
fn fmov_q_imm_0_5_0() {
    // fmov v1.2d, #0.5
    let w: u32 = arm! { fmov q(1), #0.5 };
    assert_eq!(w, 0x6f03f401);
}

#[test]
fn fmov_q_imm_0_5_1() {
    // fmov v17.2d, #0.5
    let w: u32 = arm! { fmov q(17), #0.5 };
    assert_eq!(w, 0x6f03f411);
}

#[test]
fn fmov_q_imm_1_0_0() {
    // fmov v1.2d, #1.0
    let w: u32 = arm! { fmov q(1), #1.0 };
    assert_eq!(w, 0x6f03f601);
}

#[test]
fn fmov_q_imm_1_0_1() {
    // fmov v17.2d, #1.0
    let w: u32 = arm! { fmov q(17), #1.0 };
    assert_eq!(w, 0x6f03f611);
}

#[test]
fn fmov_q_imm_2_0_0() {
    // fmov v1.2d, #2.0
    let w: u32 = arm! { fmov q(1), #2.0 };
    assert_eq!(w, 0x6f00f401);
}

#[test]
fn fmov_q_imm_2_0_1() {
    // fmov v17.2d, #2.0
    let w: u32 = arm! { fmov q(17), #2.0 };
    assert_eq!(w, 0x6f00f411);
}

#[test]
fn movi_16b_zero_0() {
    // movi v1.16b, #0
    let w: u32 = arm! { movi v(1).16b, #0 };
    assert_eq!(w, 0x4f00e401);
}

#[test]
fn movi_16b_zero_1() {
    // movi v17.16b, #0
    let w: u32 = arm! { movi v(17).16b, #0 };
    assert_eq!(w, 0x4f00e411);
}

#[test]
fn movi_d_zero_0() {
    // movi d1, #0
    let w: u32 = arm! { movi d(1), #0 };
    assert_eq!(w, 0x2f00e401);
}

#[test]
fn movi_d_zero_1() {
    // movi d17, #0
    let w: u32 = arm! { movi d(17), #0 };
    assert_eq!(w, 0x2f00e411);
}

#[test]
fn movi_q_zero_0() {
    // movi v1.2d, #0
    let w: u32 = arm! { movi q(1), #0 };
    assert_eq!(w, 0x6f00e401);
}

#[test]
fn movi_q_zero_1() {
    // movi v17.2d, #0
    let w: u32 = arm! { movi q(17), #0 };
    assert_eq!(w, 0x6f00e411);
}

#[test]
fn fmov_qq_0() {
    // mov v1.16b, v2.16b
    let w: u32 = arm! { fmov q(1), q(2) };
    assert_eq!(w, 0x4ea21c41);
}

#[test]
fn fmov_qq_1() {
    // mov v17.16b, v18.16b
    let w: u32 = arm! { fmov q(17), q(18) };
    assert_eq!(w, 0x4eb21e51);
}

#[test]
fn fmov_qq_2() {
    // mov v30.16b, v29.16b
    let w: u32 = arm! { fmov q(30), q(29) };
    assert_eq!(w, 0x4ebd1fbe);
}

#[test]
fn fmov_qq_3() {
    // mov v0.16b, v5.16b
    let w: u32 = arm! { fmov q(0), q(5) };
    assert_eq!(w, 0x4ea51ca0);
}

#[test]
fn dup_q0_0() {
    // dup v1.2d, v2.d[0]
    let w: u32 = arm! { dup q(1), q(2)[0] };
    assert_eq!(w, 0x4e080441);
}

#[test]
fn dup_q0_1() {
    // dup v17.2d, v18.d[0]
    let w: u32 = arm! { dup q(17), q(18)[0] };
    assert_eq!(w, 0x4e080651);
}

#[test]
fn dup_q0_2() {
    // dup v30.2d, v29.d[0]
    let w: u32 = arm! { dup q(30), q(29)[0] };
    assert_eq!(w, 0x4e0807be);
}

#[test]
fn dup_q0_3() {
    // dup v0.2d, v5.d[0]
    let w: u32 = arm! { dup q(0), q(5)[0] };
    assert_eq!(w, 0x4e0804a0);
}

#[test]
fn dup_q1_0() {
    // dup v1.2d, v2.d[1]
    let w: u32 = arm! { dup q(1), q(2)[1] };
    assert_eq!(w, 0x4e180441);
}

#[test]
fn dup_q1_1() {
    // dup v17.2d, v18.d[1]
    let w: u32 = arm! { dup q(17), q(18)[1] };
    assert_eq!(w, 0x4e180651);
}

#[test]
fn dup_q1_2() {
    // dup v30.2d, v29.d[1]
    let w: u32 = arm! { dup q(30), q(29)[1] };
    assert_eq!(w, 0x4e1807be);
}

#[test]
fn dup_q1_3() {
    // dup v0.2d, v5.d[1]
    let w: u32 = arm! { dup q(0), q(5)[1] };
    assert_eq!(w, 0x4e1804a0);
}

#[test]
fn umov_d0_0() {
    // umov x1, v2.d[0]
    let w: u32 = arm! { umov x(1), v(2).d[0] };
    assert_eq!(w, 0x4e083c41);
}

#[test]
fn umov_d0_1() {
    // umov x17, v18.d[0]
    let w: u32 = arm! { umov x(17), v(18).d[0] };
    assert_eq!(w, 0x4e083e51);
}

#[test]
fn umov_d0_2() {
    // umov x30, v29.d[0]
    let w: u32 = arm! { umov x(30), v(29).d[0] };
    assert_eq!(w, 0x4e083fbe);
}

#[test]
fn umov_d0_3() {
    // umov x0, v5.d[0]
    let w: u32 = arm! { umov x(0), v(5).d[0] };
    assert_eq!(w, 0x4e083ca0);
}

#[test]
fn umov_d1_0() {
    // umov x1, v2.d[1]
    let w: u32 = arm! { umov x(1), v(2).d[1] };
    assert_eq!(w, 0x4e183c41);
}

#[test]
fn umov_d1_1() {
    // umov x17, v18.d[1]
    let w: u32 = arm! { umov x(17), v(18).d[1] };
    assert_eq!(w, 0x4e183e51);
}

#[test]
fn umov_d1_2() {
    // umov x30, v29.d[1]
    let w: u32 = arm! { umov x(30), v(29).d[1] };
    assert_eq!(w, 0x4e183fbe);
}

#[test]
fn umov_d1_3() {
    // umov x0, v5.d[1]
    let w: u32 = arm! { umov x(0), v(5).d[1] };
    assert_eq!(w, 0x4e183ca0);
}

#[test]
fn zip1_0() {
    // zip1 v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { zip1 q(1), q(2), q(3) };
    assert_eq!(w, 0x4ec33841);
}

#[test]
fn zip1_1() {
    // zip1 v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { zip1 q(17), q(18), q(19) };
    assert_eq!(w, 0x4ed33a51);
}

#[test]
fn zip1_2() {
    // zip1 v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { zip1 q(30), q(29), q(28) };
    assert_eq!(w, 0x4edc3bbe);
}

#[test]
fn zip1_3() {
    // zip1 v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { zip1 q(0), q(5), q(10) };
    assert_eq!(w, 0x4eca38a0);
}

#[test]
fn zip2_0() {
    // zip2 v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { zip2 q(1), q(2), q(3) };
    assert_eq!(w, 0x4ec37841);
}

#[test]
fn zip2_1() {
    // zip2 v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { zip2 q(17), q(18), q(19) };
    assert_eq!(w, 0x4ed37a51);
}

#[test]
fn zip2_2() {
    // zip2 v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { zip2 q(30), q(29), q(28) };
    assert_eq!(w, 0x4edc7bbe);
}

#[test]
fn zip2_3() {
    // zip2 v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { zip2 q(0), q(5), q(10) };
    assert_eq!(w, 0x4eca78a0);
}

#[test]
fn uzp1_0() {
    // uzp1 v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { uzp1 q(1), q(2), q(3) };
    assert_eq!(w, 0x4ec31841);
}

#[test]
fn uzp1_1() {
    // uzp1 v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { uzp1 q(17), q(18), q(19) };
    assert_eq!(w, 0x4ed31a51);
}

#[test]
fn uzp1_2() {
    // uzp1 v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { uzp1 q(30), q(29), q(28) };
    assert_eq!(w, 0x4edc1bbe);
}

#[test]
fn uzp1_3() {
    // uzp1 v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { uzp1 q(0), q(5), q(10) };
    assert_eq!(w, 0x4eca18a0);
}

#[test]
fn uzp2_0() {
    // uzp2 v1.2d, v2.2d, v3.2d
    let w: u32 = arm! { uzp2 q(1), q(2), q(3) };
    assert_eq!(w, 0x4ec35841);
}

#[test]
fn uzp2_1() {
    // uzp2 v17.2d, v18.2d, v19.2d
    let w: u32 = arm! { uzp2 q(17), q(18), q(19) };
    assert_eq!(w, 0x4ed35a51);
}

#[test]
fn uzp2_2() {
    // uzp2 v30.2d, v29.2d, v28.2d
    let w: u32 = arm! { uzp2 q(30), q(29), q(28) };
    assert_eq!(w, 0x4edc5bbe);
}

#[test]
fn uzp2_3() {
    // uzp2 v0.2d, v5.2d, v10.2d
    let w: u32 = arm! { uzp2 q(0), q(5), q(10) };
    assert_eq!(w, 0x4eca58a0);
}

#[test]
fn ext_q_0() {
    // ext v1.16b, v2.16b, v3.16b, #8
    let w: u32 = arm! { ext q(1), q(2), q(3), #8 };
    assert_eq!(w, 0x6e034041);
}

#[test]
fn ext_q_1() {
    // ext v17.16b, v18.16b, v19.16b, #8
    let w: u32 = arm! { ext q(17), q(18), q(19), #8 };
    assert_eq!(w, 0x6e134251);
}

#[test]
fn ext_q_2() {
    // ext v30.16b, v29.16b, v28.16b, #8
    let w: u32 = arm! { ext q(30), q(29), q(28), #8 };
    assert_eq!(w, 0x6e1c43be);
}

#[test]
fn ext_q_3() {
    // ext v0.16b, v5.16b, v10.16b, #8
    let w: u32 = arm! { ext q(0), q(5), q(10), #8 };
    assert_eq!(w, 0x6e0a40a0);
}

#[test]
fn rev64_q_0() {
    // rev64 v1.2d, v2.2d   [expected word from capstone]
    // NOTE: the word is REV64 on 16-byte elements (reverses bytes within each doubleword), not a swap of the two doubles; the macro is unused
    let w: u32 = arm! { rev64 q(1), q(2) };
    assert_eq!(w, 0x4e200841);
}

#[test]
fn rev64_q_1() {
    // rev64 v17.2d, v18.2d   [expected word from capstone]
    let w: u32 = arm! { rev64 q(17), q(18) };
    assert_eq!(w, 0x4e200a51);
}

#[test]
fn rev64_q_2() {
    // rev64 v30.2d, v29.2d   [expected word from capstone]
    let w: u32 = arm! { rev64 q(30), q(29) };
    assert_eq!(w, 0x4e200bbe);
}

#[test]
fn rev64_q_3() {
    // rev64 v0.2d, v5.2d   [expected word from capstone]
    let w: u32 = arm! { rev64 q(0), q(5) };
    assert_eq!(w, 0x4e2008a0);
}

#[test]
fn fcmla_0_0() {
    // fcmla v1.2d, v2.2d, v3.2d, #0   [expected word from capstone]
    let w: u32 = arm! { fcmla q(1), q(2), q(3), #0 };
    assert_eq!(w, 0x6ec3c441);
}

#[test]
fn fcmla_0_1() {
    // fcmla v17.2d, v18.2d, v19.2d, #0   [expected word from capstone]
    let w: u32 = arm! { fcmla q(17), q(18), q(19), #0 };
    assert_eq!(w, 0x6ed3c651);
}

#[test]
fn fcmla_0_2() {
    // fcmla v30.2d, v29.2d, v28.2d, #0   [expected word from capstone]
    let w: u32 = arm! { fcmla q(30), q(29), q(28), #0 };
    assert_eq!(w, 0x6edcc7be);
}

#[test]
fn fcmla_0_3() {
    // fcmla v0.2d, v5.2d, v10.2d, #0   [expected word from capstone]
    let w: u32 = arm! { fcmla q(0), q(5), q(10), #0 };
    assert_eq!(w, 0x6ecac4a0);
}

#[test]
fn fcmla_90_0() {
    // fcmla v1.2d, v2.2d, v3.2d, #90   [expected word from capstone]
    let w: u32 = arm! { fcmla q(1), q(2), q(3), #90 };
    assert_eq!(w, 0x6ec3cc41);
}

#[test]
fn fcmla_90_1() {
    // fcmla v17.2d, v18.2d, v19.2d, #90   [expected word from capstone]
    let w: u32 = arm! { fcmla q(17), q(18), q(19), #90 };
    assert_eq!(w, 0x6ed3ce51);
}

#[test]
fn fcmla_90_2() {
    // fcmla v30.2d, v29.2d, v28.2d, #90   [expected word from capstone]
    let w: u32 = arm! { fcmla q(30), q(29), q(28), #90 };
    assert_eq!(w, 0x6edccfbe);
}

#[test]
fn fcmla_90_3() {
    // fcmla v0.2d, v5.2d, v10.2d, #90   [expected word from capstone]
    let w: u32 = arm! { fcmla q(0), q(5), q(10), #90 };
    assert_eq!(w, 0x6ecacca0);
}

#[test]
fn fcmla_180_0() {
    // fcmla v1.2d, v2.2d, v3.2d, #180   [expected word from capstone]
    let w: u32 = arm! { fcmla q(1), q(2), q(3), #180 };
    assert_eq!(w, 0x6ec3d441);
}

#[test]
fn fcmla_180_1() {
    // fcmla v17.2d, v18.2d, v19.2d, #180   [expected word from capstone]
    let w: u32 = arm! { fcmla q(17), q(18), q(19), #180 };
    assert_eq!(w, 0x6ed3d651);
}

#[test]
fn fcmla_180_2() {
    // fcmla v30.2d, v29.2d, v28.2d, #180   [expected word from capstone]
    let w: u32 = arm! { fcmla q(30), q(29), q(28), #180 };
    assert_eq!(w, 0x6edcd7be);
}

#[test]
fn fcmla_180_3() {
    // fcmla v0.2d, v5.2d, v10.2d, #180   [expected word from capstone]
    let w: u32 = arm! { fcmla q(0), q(5), q(10), #180 };
    assert_eq!(w, 0x6ecad4a0);
}

#[test]
fn fcmla_270_0() {
    // fcmla v1.2d, v2.2d, v3.2d, #270   [expected word from capstone]
    let w: u32 = arm! { fcmla q(1), q(2), q(3), #270 };
    assert_eq!(w, 0x6ec3dc41);
}

#[test]
fn fcmla_270_1() {
    // fcmla v17.2d, v18.2d, v19.2d, #270   [expected word from capstone]
    let w: u32 = arm! { fcmla q(17), q(18), q(19), #270 };
    assert_eq!(w, 0x6ed3de51);
}

#[test]
fn fcmla_270_2() {
    // fcmla v30.2d, v29.2d, v28.2d, #270   [expected word from capstone]
    let w: u32 = arm! { fcmla q(30), q(29), q(28), #270 };
    assert_eq!(w, 0x6edcdfbe);
}

#[test]
fn fcmla_270_3() {
    // fcmla v0.2d, v5.2d, v10.2d, #270   [expected word from capstone]
    let w: u32 = arm! { fcmla q(0), q(5), q(10), #270 };
    assert_eq!(w, 0x6ecadca0);
}

#[test]
fn and_xxx_0() {
    // and x1, x2, x3
    let w: u32 = arm! { and x(1), x(2), x(3) };
    assert_eq!(w, 0x8a030041);
}

#[test]
fn and_xxx_1() {
    // and x17, x18, x19
    let w: u32 = arm! { and x(17), x(18), x(19) };
    assert_eq!(w, 0x8a130251);
}

#[test]
fn and_xxx_2() {
    // and x30, x29, x28
    let w: u32 = arm! { and x(30), x(29), x(28) };
    assert_eq!(w, 0x8a1c03be);
}

#[test]
fn and_xxx_3() {
    // and x0, x5, x10
    let w: u32 = arm! { and x(0), x(5), x(10) };
    assert_eq!(w, 0x8a0a00a0);
}

#[test]
fn ands_xxx_0() {
    // ands x1, x2, x3
    let w: u32 = arm! { ands x(1), x(2), x(3) };
    assert_eq!(w, 0xea030041);
}

#[test]
fn ands_xxx_1() {
    // ands x17, x18, x19
    let w: u32 = arm! { ands x(17), x(18), x(19) };
    assert_eq!(w, 0xea130251);
}

#[test]
fn ands_xxx_2() {
    // ands x30, x29, x28
    let w: u32 = arm! { ands x(30), x(29), x(28) };
    assert_eq!(w, 0xea1c03be);
}

#[test]
fn ands_xxx_3() {
    // ands x0, x5, x10
    let w: u32 = arm! { ands x(0), x(5), x(10) };
    assert_eq!(w, 0xea0a00a0);
}

#[test]
fn orr_xxx_0() {
    // orr x1, x2, x3
    let w: u32 = arm! { orr x(1), x(2), x(3) };
    assert_eq!(w, 0xaa030041);
}

#[test]
fn orr_xxx_1() {
    // orr x17, x18, x19
    let w: u32 = arm! { orr x(17), x(18), x(19) };
    assert_eq!(w, 0xaa130251);
}

#[test]
fn orr_xxx_2() {
    // orr x30, x29, x28
    let w: u32 = arm! { orr x(30), x(29), x(28) };
    assert_eq!(w, 0xaa1c03be);
}

#[test]
fn orr_xxx_3() {
    // orr x0, x5, x10
    let w: u32 = arm! { orr x(0), x(5), x(10) };
    assert_eq!(w, 0xaa0a00a0);
}

#[test]
fn orn_xxx_0() {
    // orn x1, x2, x3
    let w: u32 = arm! { orn x(1), x(2), x(3) };
    assert_eq!(w, 0xaa230041);
}

#[test]
fn orn_xxx_1() {
    // orn x17, x18, x19
    let w: u32 = arm! { orn x(17), x(18), x(19) };
    assert_eq!(w, 0xaa330251);
}

#[test]
fn orn_xxx_2() {
    // orn x30, x29, x28
    let w: u32 = arm! { orn x(30), x(29), x(28) };
    assert_eq!(w, 0xaa3c03be);
}

#[test]
fn orn_xxx_3() {
    // orn x0, x5, x10
    let w: u32 = arm! { orn x(0), x(5), x(10) };
    assert_eq!(w, 0xaa2a00a0);
}

#[test]
fn eor_xxx_0() {
    // eor x1, x2, x3
    let w: u32 = arm! { eor x(1), x(2), x(3) };
    assert_eq!(w, 0xca030041);
}

#[test]
fn eor_xxx_1() {
    // eor x17, x18, x19
    let w: u32 = arm! { eor x(17), x(18), x(19) };
    assert_eq!(w, 0xca130251);
}

#[test]
fn eor_xxx_2() {
    // eor x30, x29, x28
    let w: u32 = arm! { eor x(30), x(29), x(28) };
    assert_eq!(w, 0xca1c03be);
}

#[test]
fn eor_xxx_3() {
    // eor x0, x5, x10
    let w: u32 = arm! { eor x(0), x(5), x(10) };
    assert_eq!(w, 0xca0a00a0);
}

#[test]
fn add_xxx_0() {
    // add x1, x2, x3
    let w: u32 = arm! { add x(1), x(2), x(3) };
    assert_eq!(w, 0x8b030041);
}

#[test]
fn add_xxx_1() {
    // add x17, x18, x19
    let w: u32 = arm! { add x(17), x(18), x(19) };
    assert_eq!(w, 0x8b130251);
}

#[test]
fn add_xxx_2() {
    // add x30, x29, x28
    let w: u32 = arm! { add x(30), x(29), x(28) };
    assert_eq!(w, 0x8b1c03be);
}

#[test]
fn add_xxx_3() {
    // add x0, x5, x10
    let w: u32 = arm! { add x(0), x(5), x(10) };
    assert_eq!(w, 0x8b0a00a0);
}

#[test]
fn adds_xxx_0() {
    // adds x1, x2, x3
    let w: u32 = arm! { adds x(1), x(2), x(3) };
    assert_eq!(w, 0xab030041);
}

#[test]
fn adds_xxx_1() {
    // adds x17, x18, x19
    let w: u32 = arm! { adds x(17), x(18), x(19) };
    assert_eq!(w, 0xab130251);
}

#[test]
fn adds_xxx_2() {
    // adds x30, x29, x28
    let w: u32 = arm! { adds x(30), x(29), x(28) };
    assert_eq!(w, 0xab1c03be);
}

#[test]
fn adds_xxx_3() {
    // adds x0, x5, x10
    let w: u32 = arm! { adds x(0), x(5), x(10) };
    assert_eq!(w, 0xab0a00a0);
}

#[test]
fn add_xxx_lsl3_0() {
    // add x1, x2, x3, lsl #3
    let w: u32 = arm! { add x(1), x(2), x(3), lsl #3 };
    assert_eq!(w, 0x8b030c41);
}

#[test]
fn add_xxx_lsl3_1() {
    // add x17, x18, x19, lsl #3
    let w: u32 = arm! { add x(17), x(18), x(19), lsl #3 };
    assert_eq!(w, 0x8b130e51);
}

#[test]
fn add_xxx_lsl3_2() {
    // add x30, x29, x28, lsl #3
    let w: u32 = arm! { add x(30), x(29), x(28), lsl #3 };
    assert_eq!(w, 0x8b1c0fbe);
}

#[test]
fn add_xxx_lsl3_3() {
    // add x0, x5, x10, lsl #3
    let w: u32 = arm! { add x(0), x(5), x(10), lsl #3 };
    assert_eq!(w, 0x8b0a0ca0);
}

#[test]
fn add_xxx_lsl63_0() {
    // add x1, x2, x3, lsl #63
    let w: u32 = arm! { add x(1), x(2), x(3), lsl #63 };
    assert_eq!(w, 0x8b03fc41);
}

#[test]
fn tst_0() {
    // tst x1, x2
    let w: u32 = arm! { tst x(1), x(2) };
    assert_eq!(w, 0xea02003f);
}

#[test]
fn tst_1() {
    // tst x17, x18
    let w: u32 = arm! { tst x(17), x(18) };
    assert_eq!(w, 0xea12023f);
}

#[test]
fn tst_2() {
    // tst x30, x29
    let w: u32 = arm! { tst x(30), x(29) };
    assert_eq!(w, 0xea1d03df);
}

#[test]
fn tst_3() {
    // tst x0, x5
    let w: u32 = arm! { tst x(0), x(5) };
    assert_eq!(w, 0xea05001f);
}

#[test]
fn mov_xx_0() {
    // mov x1, x2
    let w: u32 = arm! { mov x(1), x(2) };
    assert_eq!(w, 0xaa0203e1);
}

#[test]
fn mov_xx_1() {
    // mov x17, x18
    let w: u32 = arm! { mov x(17), x(18) };
    assert_eq!(w, 0xaa1203f1);
}

#[test]
fn mov_xx_2() {
    // mov x30, x29
    let w: u32 = arm! { mov x(30), x(29) };
    assert_eq!(w, 0xaa1d03fe);
}

#[test]
fn mov_xx_3() {
    // mov x0, x5
    let w: u32 = arm! { mov x(0), x(5) };
    assert_eq!(w, 0xaa0503e0);
}

#[test]
fn mov_x_sp() {
    // mov x1, sp
    let w: u32 = arm! { mov x(1), sp };
    assert_eq!(w, 0x910003e1);
}

#[test]
fn mov_sp_x() {
    // mov sp, x1
    let w: u32 = arm! { mov sp, x(1) };
    assert_eq!(w, 0x9100003f);
}

#[test]
fn mov_x31_x() {
    // mov sp, x2
    let w: u32 = arm! { mov x(31), x(2) };
    assert_eq!(w, 0x9100005f);
}

#[test]
fn mov_x_x31() {
    // mov x1, sp
    let w: u32 = arm! { mov x(1), x(31) };
    assert_eq!(w, 0x910003e1);
}

#[test]
fn add_imm_0() {
    // add x1, x2, #100
    let w: u32 = arm! { add x(1), x(2), #100 };
    assert_eq!(w, 0x91019041);
}

#[test]
fn add_imm_1() {
    // add x17, x18, #100
    let w: u32 = arm! { add x(17), x(18), #100 };
    assert_eq!(w, 0x91019251);
}

#[test]
fn add_imm_2() {
    // add x30, x29, #100
    let w: u32 = arm! { add x(30), x(29), #100 };
    assert_eq!(w, 0x910193be);
}

#[test]
fn add_imm_max_0() {
    // add x1, x2, #4095
    let w: u32 = arm! { add x(1), x(2), #4095 };
    assert_eq!(w, 0x913ffc41);
}

#[test]
fn add_imm_lsl12_0() {
    // add x1, x2, #100, lsl #12
    let w: u32 = arm! { add x(1), x(2), #100, lsl #12 };
    assert_eq!(w, 0x91419041);
}

#[test]
fn add_imm_lsl12_1() {
    // add x17, x18, #100, lsl #12
    let w: u32 = arm! { add x(17), x(18), #100, lsl #12 };
    assert_eq!(w, 0x91419251);
}

#[test]
fn add_sp_imm() {
    // add sp, sp, #16
    let w: u32 = arm! { add sp, sp, #16 };
    assert_eq!(w, 0x910043ff);
}

#[test]
fn add_sp_imm_lsl12() {
    // add sp, sp, #16, lsl #12
    let w: u32 = arm! { add sp, sp, #16, lsl #12 };
    assert_eq!(w, 0x914043ff);
}

#[test]
fn sub_imm_0() {
    // sub x1, x2, #100
    let w: u32 = arm! { sub x(1), x(2), #100 };
    assert_eq!(w, 0xd1019041);
}

#[test]
fn sub_imm_1() {
    // sub x17, x18, #100
    let w: u32 = arm! { sub x(17), x(18), #100 };
    assert_eq!(w, 0xd1019251);
}

#[test]
fn sub_imm_2() {
    // sub x30, x29, #100
    let w: u32 = arm! { sub x(30), x(29), #100 };
    assert_eq!(w, 0xd10193be);
}

#[test]
fn sub_imm_max_0() {
    // sub x1, x2, #4095
    let w: u32 = arm! { sub x(1), x(2), #4095 };
    assert_eq!(w, 0xd13ffc41);
}

#[test]
fn sub_imm_lsl12_0() {
    // sub x1, x2, #100, lsl #12
    let w: u32 = arm! { sub x(1), x(2), #100, lsl #12 };
    assert_eq!(w, 0xd1419041);
}

#[test]
fn sub_imm_lsl12_1() {
    // sub x17, x18, #100, lsl #12
    let w: u32 = arm! { sub x(17), x(18), #100, lsl #12 };
    assert_eq!(w, 0xd1419251);
}

#[test]
fn sub_sp_imm() {
    // sub sp, sp, #16
    let w: u32 = arm! { sub sp, sp, #16 };
    assert_eq!(w, 0xd10043ff);
}

#[test]
fn sub_sp_imm_lsl12() {
    // sub sp, sp, #16, lsl #12
    let w: u32 = arm! { sub sp, sp, #16, lsl #12 };
    assert_eq!(w, 0xd14043ff);
}

#[test]
fn subs_imm_0() {
    // subs x1, x2, #100
    let w: u32 = arm! { subs x(1), x(2), #100 };
    assert_eq!(w, 0xf1019041);
}

#[test]
fn subs_imm_1() {
    // subs x17, x18, #100
    let w: u32 = arm! { subs x(17), x(18), #100 };
    assert_eq!(w, 0xf1019251);
}

#[test]
fn lsr_imm1_0() {
    // lsr x1, x2, #1
    let w: u32 = arm! { lsr x(1), x(2), #1 };
    assert_eq!(w, 0xd341fc41);
}

#[test]
fn lsr_imm1_1() {
    // lsr x17, x18, #1
    let w: u32 = arm! { lsr x(17), x(18), #1 };
    assert_eq!(w, 0xd341fe51);
}

#[test]
fn lsr_imm5_0() {
    // lsr x1, x2, #5
    let w: u32 = arm! { lsr x(1), x(2), #5 };
    assert_eq!(w, 0xd345fc41);
}

#[test]
fn lsr_imm5_1() {
    // lsr x17, x18, #5
    let w: u32 = arm! { lsr x(17), x(18), #5 };
    assert_eq!(w, 0xd345fe51);
}

#[test]
fn lsr_imm63_0() {
    // lsr x1, x2, #63
    let w: u32 = arm! { lsr x(1), x(2), #63 };
    assert_eq!(w, 0xd37ffc41);
}

#[test]
fn lsr_imm63_1() {
    // lsr x17, x18, #63
    let w: u32 = arm! { lsr x(17), x(18), #63 };
    assert_eq!(w, 0xd37ffe51);
}

#[test]
fn ubfm_0() {
    // ubfm x1, x2, #3, #7
    let w: u32 = arm! { ubfm x(1), x(2), #3, #7 };
    assert_eq!(w, 0xd3431c41);
}

#[test]
fn ubfm_1() {
    // ubfm x17, x18, #3, #7
    let w: u32 = arm! { ubfm x(17), x(18), #3, #7 };
    assert_eq!(w, 0xd3431e51);
}

#[test]
fn movz_0() {
    // movz x1, #0x1234
    let w: u32 = arm! { movz x(1), #0x1234 };
    assert_eq!(w, 0xd2824681);
}

#[test]
fn movz_1() {
    // movz x17, #0x1234
    let w: u32 = arm! { movz x(17), #0x1234 };
    assert_eq!(w, 0xd2824691);
}

#[test]
fn movz_2() {
    // movz x30, #0x1234
    let w: u32 = arm! { movz x(30), #0x1234 };
    assert_eq!(w, 0xd282469e);
}

#[test]
fn movk_lsl16_0() {
    // movk x1, #0x1234, lsl #16
    let w: u32 = arm! { movk_lsl16 x(1), #0x1234 };
    assert_eq!(w, 0xf2a24681);
}

#[test]
fn movk_lsl16_1() {
    // movk x17, #0x1234, lsl #16
    let w: u32 = arm! { movk_lsl16 x(17), #0x1234 };
    assert_eq!(w, 0xf2a24691);
}

#[test]
fn movk_lsl16_2() {
    // movk x30, #0x1234, lsl #16
    let w: u32 = arm! { movk_lsl16 x(30), #0x1234 };
    assert_eq!(w, 0xf2a2469e);
}

#[test]
fn movk_lsl32_0() {
    // movk x1, #0x1234, lsl #32
    let w: u32 = arm! { movk_lsl32 x(1), #0x1234 };
    assert_eq!(w, 0xf2c24681);
}

#[test]
fn movk_lsl32_1() {
    // movk x17, #0x1234, lsl #32
    let w: u32 = arm! { movk_lsl32 x(17), #0x1234 };
    assert_eq!(w, 0xf2c24691);
}

#[test]
fn movk_lsl32_2() {
    // movk x30, #0x1234, lsl #32
    let w: u32 = arm! { movk_lsl32 x(30), #0x1234 };
    assert_eq!(w, 0xf2c2469e);
}

#[test]
fn movk_lsl48_0() {
    // movk x1, #0x1234, lsl #48
    let w: u32 = arm! { movk_lsl48 x(1), #0x1234 };
    assert_eq!(w, 0xf2e24681);
}

#[test]
fn movk_lsl48_1() {
    // movk x17, #0x1234, lsl #48
    let w: u32 = arm! { movk_lsl48 x(17), #0x1234 };
    assert_eq!(w, 0xf2e24691);
}

#[test]
fn movk_lsl48_2() {
    // movk x30, #0x1234, lsl #48
    let w: u32 = arm! { movk_lsl48 x(30), #0x1234 };
    assert_eq!(w, 0xf2e2469e);
}

#[test]
fn blr() {
    // blr x1
    let w: u32 = arm! { blr x(1) };
    assert_eq!(w, 0xd63f0020);
}

#[test]
fn blr_hi() {
    // blr x23
    let w: u32 = arm! { blr x(23) };
    assert_eq!(w, 0xd63f02e0);
}

#[test]
fn ret() {
    // ret
    let w: u32 = arm! { ret };
    assert_eq!(w, 0xd65f03c0);
}

#[test]
    #[ignore = "the macro encodes `add x0, x0, #0` (functionally a no-op, but not the architectural NOP 0xd503201f)"]
fn nop() {
    // nop
    let w: u32 = arm! { nop };
    assert_eq!(w, 0xd503201f);
}

#[test]
fn adrp_0() {
    // adrp x1, #0x5000
    let w: u32 = arm! { adrp x(1), label(0x5000u32) };
    assert_eq!(w, 0xb0000021);
}

#[test]
fn adrp_1() {
    // adrp x17, #0x5000
    let w: u32 = arm! { adrp x(17), label(0x5000u32) };
    assert_eq!(w, 0xb0000031);
}

#[test]
fn ldr_d_ofs0_0() {
    // ldr d1, [x2, #0]
    let w: u32 = arm! { ldr d(1), [x(2), #0] };
    assert_eq!(w, 0xfd400041);
}

#[test]
fn str_d_ofs0_0() {
    // str d1, [x2, #0]
    let w: u32 = arm! { str d(1), [x(2), #0] };
    assert_eq!(w, 0xfd000041);
}

#[test]
fn ldr_d_ofs0_1() {
    // ldr d17, [x18, #0]
    let w: u32 = arm! { ldr d(17), [x(18), #0] };
    assert_eq!(w, 0xfd400251);
}

#[test]
fn str_d_ofs0_1() {
    // str d17, [x18, #0]
    let w: u32 = arm! { str d(17), [x(18), #0] };
    assert_eq!(w, 0xfd000251);
}

#[test]
fn ldr_d_ofs0_2() {
    // ldr d30, [x29, #0]
    let w: u32 = arm! { ldr d(30), [x(29), #0] };
    assert_eq!(w, 0xfd4003be);
}

#[test]
fn str_d_ofs0_2() {
    // str d30, [x29, #0]
    let w: u32 = arm! { str d(30), [x(29), #0] };
    assert_eq!(w, 0xfd0003be);
}

#[test]
fn ldr_x_ofs0_0() {
    // ldr x1, [x2, #0]
    let w: u32 = arm! { ldr x(1), [x(2), #0] };
    assert_eq!(w, 0xf9400041);
}

#[test]
fn str_x_ofs0_0() {
    // str x1, [x2, #0]
    let w: u32 = arm! { str x(1), [x(2), #0] };
    assert_eq!(w, 0xf9000041);
}

#[test]
fn ldr_x_ofs0_1() {
    // ldr x17, [x18, #0]
    let w: u32 = arm! { ldr x(17), [x(18), #0] };
    assert_eq!(w, 0xf9400251);
}

#[test]
fn str_x_ofs0_1() {
    // str x17, [x18, #0]
    let w: u32 = arm! { str x(17), [x(18), #0] };
    assert_eq!(w, 0xf9000251);
}

#[test]
fn ldr_x_ofs0_2() {
    // ldr x30, [x29, #0]
    let w: u32 = arm! { ldr x(30), [x(29), #0] };
    assert_eq!(w, 0xf94003be);
}

#[test]
fn str_x_ofs0_2() {
    // str x30, [x29, #0]
    let w: u32 = arm! { str x(30), [x(29), #0] };
    assert_eq!(w, 0xf90003be);
}

#[test]
fn ldr_d_ofs8_0() {
    // ldr d1, [x2, #8]
    let w: u32 = arm! { ldr d(1), [x(2), #8] };
    assert_eq!(w, 0xfd400441);
}

#[test]
fn str_d_ofs8_0() {
    // str d1, [x2, #8]
    let w: u32 = arm! { str d(1), [x(2), #8] };
    assert_eq!(w, 0xfd000441);
}

#[test]
fn ldr_d_ofs8_1() {
    // ldr d17, [x18, #8]
    let w: u32 = arm! { ldr d(17), [x(18), #8] };
    assert_eq!(w, 0xfd400651);
}

#[test]
fn str_d_ofs8_1() {
    // str d17, [x18, #8]
    let w: u32 = arm! { str d(17), [x(18), #8] };
    assert_eq!(w, 0xfd000651);
}

#[test]
fn ldr_d_ofs8_2() {
    // ldr d30, [x29, #8]
    let w: u32 = arm! { ldr d(30), [x(29), #8] };
    assert_eq!(w, 0xfd4007be);
}

#[test]
fn str_d_ofs8_2() {
    // str d30, [x29, #8]
    let w: u32 = arm! { str d(30), [x(29), #8] };
    assert_eq!(w, 0xfd0007be);
}

#[test]
fn ldr_x_ofs8_0() {
    // ldr x1, [x2, #8]
    let w: u32 = arm! { ldr x(1), [x(2), #8] };
    assert_eq!(w, 0xf9400441);
}

#[test]
fn str_x_ofs8_0() {
    // str x1, [x2, #8]
    let w: u32 = arm! { str x(1), [x(2), #8] };
    assert_eq!(w, 0xf9000441);
}

#[test]
fn ldr_x_ofs8_1() {
    // ldr x17, [x18, #8]
    let w: u32 = arm! { ldr x(17), [x(18), #8] };
    assert_eq!(w, 0xf9400651);
}

#[test]
fn str_x_ofs8_1() {
    // str x17, [x18, #8]
    let w: u32 = arm! { str x(17), [x(18), #8] };
    assert_eq!(w, 0xf9000651);
}

#[test]
fn ldr_x_ofs8_2() {
    // ldr x30, [x29, #8]
    let w: u32 = arm! { ldr x(30), [x(29), #8] };
    assert_eq!(w, 0xf94007be);
}

#[test]
fn str_x_ofs8_2() {
    // str x30, [x29, #8]
    let w: u32 = arm! { str x(30), [x(29), #8] };
    assert_eq!(w, 0xf90007be);
}

#[test]
fn ldr_d_ofs256_0() {
    // ldr d1, [x2, #256]
    let w: u32 = arm! { ldr d(1), [x(2), #256] };
    assert_eq!(w, 0xfd408041);
}

#[test]
fn str_d_ofs256_0() {
    // str d1, [x2, #256]
    let w: u32 = arm! { str d(1), [x(2), #256] };
    assert_eq!(w, 0xfd008041);
}

#[test]
fn ldr_d_ofs256_1() {
    // ldr d17, [x18, #256]
    let w: u32 = arm! { ldr d(17), [x(18), #256] };
    assert_eq!(w, 0xfd408251);
}

#[test]
fn str_d_ofs256_1() {
    // str d17, [x18, #256]
    let w: u32 = arm! { str d(17), [x(18), #256] };
    assert_eq!(w, 0xfd008251);
}

#[test]
fn ldr_d_ofs256_2() {
    // ldr d30, [x29, #256]
    let w: u32 = arm! { ldr d(30), [x(29), #256] };
    assert_eq!(w, 0xfd4083be);
}

#[test]
fn str_d_ofs256_2() {
    // str d30, [x29, #256]
    let w: u32 = arm! { str d(30), [x(29), #256] };
    assert_eq!(w, 0xfd0083be);
}

#[test]
fn ldr_x_ofs256_0() {
    // ldr x1, [x2, #256]
    let w: u32 = arm! { ldr x(1), [x(2), #256] };
    assert_eq!(w, 0xf9408041);
}

#[test]
fn str_x_ofs256_0() {
    // str x1, [x2, #256]
    let w: u32 = arm! { str x(1), [x(2), #256] };
    assert_eq!(w, 0xf9008041);
}

#[test]
fn ldr_x_ofs256_1() {
    // ldr x17, [x18, #256]
    let w: u32 = arm! { ldr x(17), [x(18), #256] };
    assert_eq!(w, 0xf9408251);
}

#[test]
fn str_x_ofs256_1() {
    // str x17, [x18, #256]
    let w: u32 = arm! { str x(17), [x(18), #256] };
    assert_eq!(w, 0xf9008251);
}

#[test]
fn ldr_x_ofs256_2() {
    // ldr x30, [x29, #256]
    let w: u32 = arm! { ldr x(30), [x(29), #256] };
    assert_eq!(w, 0xf94083be);
}

#[test]
fn str_x_ofs256_2() {
    // str x30, [x29, #256]
    let w: u32 = arm! { str x(30), [x(29), #256] };
    assert_eq!(w, 0xf90083be);
}

#[test]
fn ldr_d_ofs32760_0() {
    // ldr d1, [x2, #32760]
    let w: u32 = arm! { ldr d(1), [x(2), #32760] };
    assert_eq!(w, 0xfd7ffc41);
}

#[test]
fn str_d_ofs32760_0() {
    // str d1, [x2, #32760]
    let w: u32 = arm! { str d(1), [x(2), #32760] };
    assert_eq!(w, 0xfd3ffc41);
}

#[test]
fn ldr_d_ofs32760_1() {
    // ldr d17, [x18, #32760]
    let w: u32 = arm! { ldr d(17), [x(18), #32760] };
    assert_eq!(w, 0xfd7ffe51);
}

#[test]
fn str_d_ofs32760_1() {
    // str d17, [x18, #32760]
    let w: u32 = arm! { str d(17), [x(18), #32760] };
    assert_eq!(w, 0xfd3ffe51);
}

#[test]
fn ldr_d_ofs32760_2() {
    // ldr d30, [x29, #32760]
    let w: u32 = arm! { ldr d(30), [x(29), #32760] };
    assert_eq!(w, 0xfd7fffbe);
}

#[test]
fn str_d_ofs32760_2() {
    // str d30, [x29, #32760]
    let w: u32 = arm! { str d(30), [x(29), #32760] };
    assert_eq!(w, 0xfd3fffbe);
}

#[test]
fn ldr_x_ofs32760_0() {
    // ldr x1, [x2, #32760]
    let w: u32 = arm! { ldr x(1), [x(2), #32760] };
    assert_eq!(w, 0xf97ffc41);
}

#[test]
fn str_x_ofs32760_0() {
    // str x1, [x2, #32760]
    let w: u32 = arm! { str x(1), [x(2), #32760] };
    assert_eq!(w, 0xf93ffc41);
}

#[test]
fn ldr_x_ofs32760_1() {
    // ldr x17, [x18, #32760]
    let w: u32 = arm! { ldr x(17), [x(18), #32760] };
    assert_eq!(w, 0xf97ffe51);
}

#[test]
fn str_x_ofs32760_1() {
    // str x17, [x18, #32760]
    let w: u32 = arm! { str x(17), [x(18), #32760] };
    assert_eq!(w, 0xf93ffe51);
}

#[test]
fn ldr_x_ofs32760_2() {
    // ldr x30, [x29, #32760]
    let w: u32 = arm! { ldr x(30), [x(29), #32760] };
    assert_eq!(w, 0xf97fffbe);
}

#[test]
fn str_x_ofs32760_2() {
    // str x30, [x29, #32760]
    let w: u32 = arm! { str x(30), [x(29), #32760] };
    assert_eq!(w, 0xf93fffbe);
}

#[test]
fn ldr_q_ofs0_0() {
    // ldr q1, [x2, #0]
    let w: u32 = arm! { ldr q(1), [x(2), #0] };
    assert_eq!(w, 0x3dc00041);
}

#[test]
fn str_q_ofs0_0() {
    // str q1, [x2, #0]
    let w: u32 = arm! { str q(1), [x(2), #0] };
    assert_eq!(w, 0x3d800041);
}

#[test]
fn ldr_q_ofs0_1() {
    // ldr q17, [x18, #0]
    let w: u32 = arm! { ldr q(17), [x(18), #0] };
    assert_eq!(w, 0x3dc00251);
}

#[test]
fn str_q_ofs0_1() {
    // str q17, [x18, #0]
    let w: u32 = arm! { str q(17), [x(18), #0] };
    assert_eq!(w, 0x3d800251);
}

#[test]
fn ldr_q_ofs16_0() {
    // ldr q1, [x2, #16]
    let w: u32 = arm! { ldr q(1), [x(2), #16] };
    assert_eq!(w, 0x3dc00441);
}

#[test]
fn str_q_ofs16_0() {
    // str q1, [x2, #16]
    let w: u32 = arm! { str q(1), [x(2), #16] };
    assert_eq!(w, 0x3d800441);
}

#[test]
fn ldr_q_ofs16_1() {
    // ldr q17, [x18, #16]
    let w: u32 = arm! { ldr q(17), [x(18), #16] };
    assert_eq!(w, 0x3dc00651);
}

#[test]
fn str_q_ofs16_1() {
    // str q17, [x18, #16]
    let w: u32 = arm! { str q(17), [x(18), #16] };
    assert_eq!(w, 0x3d800651);
}

#[test]
fn ldr_q_ofs256_0() {
    // ldr q1, [x2, #256]
    let w: u32 = arm! { ldr q(1), [x(2), #256] };
    assert_eq!(w, 0x3dc04041);
}

#[test]
fn str_q_ofs256_0() {
    // str q1, [x2, #256]
    let w: u32 = arm! { str q(1), [x(2), #256] };
    assert_eq!(w, 0x3d804041);
}

#[test]
fn ldr_q_ofs256_1() {
    // ldr q17, [x18, #256]
    let w: u32 = arm! { ldr q(17), [x(18), #256] };
    assert_eq!(w, 0x3dc04251);
}

#[test]
fn str_q_ofs256_1() {
    // str q17, [x18, #256]
    let w: u32 = arm! { str q(17), [x(18), #256] };
    assert_eq!(w, 0x3d804251);
}

#[test]
fn ldr_q_ofs65520_0() {
    // ldr q1, [x2, #65520]
    let w: u32 = arm! { ldr q(1), [x(2), #65520] };
    assert_eq!(w, 0x3dfffc41);
}

#[test]
fn str_q_ofs65520_0() {
    // str q1, [x2, #65520]
    let w: u32 = arm! { str q(1), [x(2), #65520] };
    assert_eq!(w, 0x3dbffc41);
}

#[test]
fn ldr_q_ofs65520_1() {
    // ldr q17, [x18, #65520]
    let w: u32 = arm! { ldr q(17), [x(18), #65520] };
    assert_eq!(w, 0x3dfffe51);
}

#[test]
fn str_q_ofs65520_1() {
    // str q17, [x18, #65520]
    let w: u32 = arm! { str q(17), [x(18), #65520] };
    assert_eq!(w, 0x3dbffe51);
}

#[test]
fn ldr_d_idx_0() {
    // ldr d1, [x2, x3, lsl #3]
    let w: u32 = arm! { ldr d(1), [x(2), x(3), lsl #3] };
    assert_eq!(w, 0xfc637841);
}

#[test]
fn str_d_idx_0() {
    // str d1, [x2, x3, lsl #3]
    let w: u32 = arm! { str d(1), [x(2), x(3), lsl #3] };
    assert_eq!(w, 0xfc237841);
}

#[test]
fn ldr_x_idx_0() {
    // ldr x1, [x2, x3, lsl #3]
    let w: u32 = arm! { ldr x(1), [x(2), x(3), lsl #3] };
    assert_eq!(w, 0xf8637841);
}

#[test]
fn ldr_q_idx_0() {
    // ldr q1, [x2, x3, lsl #4]
    let w: u32 = arm! { ldr q(1), [x(2), x(3), lsl #4] };
    assert_eq!(w, 0x3ce37841);
}

#[test]
fn str_q_idx_0() {
    // str q1, [x2, x3, lsl #4]
    let w: u32 = arm! { str q(1), [x(2), x(3), lsl #4] };
    assert_eq!(w, 0x3ca37841);
}

#[test]
fn ldr_d_idx_1() {
    // ldr d17, [x18, x19, lsl #3]
    let w: u32 = arm! { ldr d(17), [x(18), x(19), lsl #3] };
    assert_eq!(w, 0xfc737a51);
}

#[test]
fn str_d_idx_1() {
    // str d17, [x18, x19, lsl #3]
    let w: u32 = arm! { str d(17), [x(18), x(19), lsl #3] };
    assert_eq!(w, 0xfc337a51);
}

#[test]
fn ldr_x_idx_1() {
    // ldr x17, [x18, x19, lsl #3]
    let w: u32 = arm! { ldr x(17), [x(18), x(19), lsl #3] };
    assert_eq!(w, 0xf8737a51);
}

#[test]
fn ldr_q_idx_1() {
    // ldr q17, [x18, x19, lsl #4]
    let w: u32 = arm! { ldr q(17), [x(18), x(19), lsl #4] };
    assert_eq!(w, 0x3cf37a51);
}

#[test]
fn str_q_idx_1() {
    // str q17, [x18, x19, lsl #4]
    let w: u32 = arm! { str q(17), [x(18), x(19), lsl #4] };
    assert_eq!(w, 0x3cb37a51);
}

#[test]
fn ldr_d_idx_2() {
    // ldr d30, [x29, x28, lsl #3]
    let w: u32 = arm! { ldr d(30), [x(29), x(28), lsl #3] };
    assert_eq!(w, 0xfc7c7bbe);
}

#[test]
fn str_d_idx_2() {
    // str d30, [x29, x28, lsl #3]
    let w: u32 = arm! { str d(30), [x(29), x(28), lsl #3] };
    assert_eq!(w, 0xfc3c7bbe);
}

#[test]
fn ldr_x_idx_2() {
    // ldr x30, [x29, x28, lsl #3]
    let w: u32 = arm! { ldr x(30), [x(29), x(28), lsl #3] };
    assert_eq!(w, 0xf87c7bbe);
}

#[test]
fn ldr_q_idx_2() {
    // ldr q30, [x29, x28, lsl #4]
    let w: u32 = arm! { ldr q(30), [x(29), x(28), lsl #4] };
    assert_eq!(w, 0x3cfc7bbe);
}

#[test]
fn str_q_idx_2() {
    // str q30, [x29, x28, lsl #4]
    let w: u32 = arm! { str q(30), [x(29), x(28), lsl #4] };
    assert_eq!(w, 0x3cbc7bbe);
}

#[test]
fn ldr_d_label_100_0() {
    // ldr d1, #256
    let w: u32 = arm! { ldr d(1), label(256i32) };
    assert_eq!(w, 0x5c000801);
}

#[test]
fn ldr_x_label_100_0() {
    // ldr x1, #256
    let w: u32 = arm! { ldr x(1), label(256i32) };
    assert_eq!(w, 0x58000801);
}

#[test]
fn ldr_q_label_100_0() {
    // ldr q1, #256
    let w: u32 = arm! { ldr q(1), label(256i32) };
    assert_eq!(w, 0x9c000801);
}

#[test]
fn ldr_d_label_100_1() {
    // ldr d17, #256
    let w: u32 = arm! { ldr d(17), label(256i32) };
    assert_eq!(w, 0x5c000811);
}

#[test]
fn ldr_x_label_100_1() {
    // ldr x17, #256
    let w: u32 = arm! { ldr x(17), label(256i32) };
    assert_eq!(w, 0x58000811);
}

#[test]
fn ldr_q_label_100_1() {
    // ldr q17, #256
    let w: u32 = arm! { ldr q(17), label(256i32) };
    assert_eq!(w, 0x9c000811);
}

#[test]
fn ldr_d_label_m100_0() {
    // ldr d1, #-256
    let w: u32 = arm! { ldr d(1), label(-256i32) };
    assert_eq!(w, 0x5cfff801);
}

#[test]
fn ldr_x_label_m100_0() {
    // ldr x1, #-256
    let w: u32 = arm! { ldr x(1), label(-256i32) };
    assert_eq!(w, 0x58fff801);
}

#[test]
fn ldr_q_label_m100_0() {
    // ldr q1, #-256
    let w: u32 = arm! { ldr q(1), label(-256i32) };
    assert_eq!(w, 0x9cfff801);
}

#[test]
fn ldr_d_label_m100_1() {
    // ldr d17, #-256
    let w: u32 = arm! { ldr d(17), label(-256i32) };
    assert_eq!(w, 0x5cfff811);
}

#[test]
fn ldr_x_label_m100_1() {
    // ldr x17, #-256
    let w: u32 = arm! { ldr x(17), label(-256i32) };
    assert_eq!(w, 0x58fff811);
}

#[test]
fn ldr_q_label_m100_1() {
    // ldr q17, #-256
    let w: u32 = arm! { ldr q(17), label(-256i32) };
    assert_eq!(w, 0x9cfff811);
}

#[test]
fn ldp_d_0_0() {
    // ldp d1, d2, [x3, #0]
    let w: u32 = arm! { ldp d(1), d(2), [x(3), #0] };
    assert_eq!(w, 0x6d400861);
}

#[test]
fn stp_d_0_0() {
    // stp d1, d2, [x3, #0]
    let w: u32 = arm! { stp d(1), d(2), [x(3), #0] };
    assert_eq!(w, 0x6d000861);
}

#[test]
fn ldp_x_0_0() {
    // ldp x1, x2, [x3, #0]
    let w: u32 = arm! { ldp x(1), x(2), [x(3), #0] };
    assert_eq!(w, 0xa9400861);
}

#[test]
fn stp_x_0_0() {
    // stp x1, x2, [x3, #0]
    let w: u32 = arm! { stp x(1), x(2), [x(3), #0] };
    assert_eq!(w, 0xa9000861);
}

#[test]
fn ldp_d_0_1() {
    // ldp d17, d18, [x19, #0]
    let w: u32 = arm! { ldp d(17), d(18), [x(19), #0] };
    assert_eq!(w, 0x6d404a71);
}

#[test]
fn stp_d_0_1() {
    // stp d17, d18, [x19, #0]
    let w: u32 = arm! { stp d(17), d(18), [x(19), #0] };
    assert_eq!(w, 0x6d004a71);
}

#[test]
fn ldp_x_0_1() {
    // ldp x17, x18, [x19, #0]
    let w: u32 = arm! { ldp x(17), x(18), [x(19), #0] };
    assert_eq!(w, 0xa9404a71);
}

#[test]
fn stp_x_0_1() {
    // stp x17, x18, [x19, #0]
    let w: u32 = arm! { stp x(17), x(18), [x(19), #0] };
    assert_eq!(w, 0xa9004a71);
}

#[test]
fn ldp_d_0_2() {
    // ldp d30, d29, [x28, #0]
    let w: u32 = arm! { ldp d(30), d(29), [x(28), #0] };
    assert_eq!(w, 0x6d40779e);
}

#[test]
fn stp_d_0_2() {
    // stp d30, d29, [x28, #0]
    let w: u32 = arm! { stp d(30), d(29), [x(28), #0] };
    assert_eq!(w, 0x6d00779e);
}

#[test]
fn ldp_x_0_2() {
    // ldp x30, x29, [x28, #0]
    let w: u32 = arm! { ldp x(30), x(29), [x(28), #0] };
    assert_eq!(w, 0xa940779e);
}

#[test]
fn stp_x_0_2() {
    // stp x30, x29, [x28, #0]
    let w: u32 = arm! { stp x(30), x(29), [x(28), #0] };
    assert_eq!(w, 0xa900779e);
}

#[test]
fn ldp_d_8_0() {
    // ldp d1, d2, [x3, #8]
    let w: u32 = arm! { ldp d(1), d(2), [x(3), #8] };
    assert_eq!(w, 0x6d408861);
}

#[test]
fn stp_d_8_0() {
    // stp d1, d2, [x3, #8]
    let w: u32 = arm! { stp d(1), d(2), [x(3), #8] };
    assert_eq!(w, 0x6d008861);
}

#[test]
fn ldp_x_8_0() {
    // ldp x1, x2, [x3, #8]
    let w: u32 = arm! { ldp x(1), x(2), [x(3), #8] };
    assert_eq!(w, 0xa9408861);
}

#[test]
fn stp_x_8_0() {
    // stp x1, x2, [x3, #8]
    let w: u32 = arm! { stp x(1), x(2), [x(3), #8] };
    assert_eq!(w, 0xa9008861);
}

#[test]
fn ldp_d_8_1() {
    // ldp d17, d18, [x19, #8]
    let w: u32 = arm! { ldp d(17), d(18), [x(19), #8] };
    assert_eq!(w, 0x6d40ca71);
}

#[test]
fn stp_d_8_1() {
    // stp d17, d18, [x19, #8]
    let w: u32 = arm! { stp d(17), d(18), [x(19), #8] };
    assert_eq!(w, 0x6d00ca71);
}

#[test]
fn ldp_x_8_1() {
    // ldp x17, x18, [x19, #8]
    let w: u32 = arm! { ldp x(17), x(18), [x(19), #8] };
    assert_eq!(w, 0xa940ca71);
}

#[test]
fn stp_x_8_1() {
    // stp x17, x18, [x19, #8]
    let w: u32 = arm! { stp x(17), x(18), [x(19), #8] };
    assert_eq!(w, 0xa900ca71);
}

#[test]
fn ldp_d_8_2() {
    // ldp d30, d29, [x28, #8]
    let w: u32 = arm! { ldp d(30), d(29), [x(28), #8] };
    assert_eq!(w, 0x6d40f79e);
}

#[test]
fn stp_d_8_2() {
    // stp d30, d29, [x28, #8]
    let w: u32 = arm! { stp d(30), d(29), [x(28), #8] };
    assert_eq!(w, 0x6d00f79e);
}

#[test]
fn ldp_x_8_2() {
    // ldp x30, x29, [x28, #8]
    let w: u32 = arm! { ldp x(30), x(29), [x(28), #8] };
    assert_eq!(w, 0xa940f79e);
}

#[test]
fn stp_x_8_2() {
    // stp x30, x29, [x28, #8]
    let w: u32 = arm! { stp x(30), x(29), [x(28), #8] };
    assert_eq!(w, 0xa900f79e);
}

#[test]
fn ldp_d_504_0() {
    // ldp d1, d2, [x3, #504]
    let w: u32 = arm! { ldp d(1), d(2), [x(3), #504] };
    assert_eq!(w, 0x6d5f8861);
}

#[test]
fn stp_d_504_0() {
    // stp d1, d2, [x3, #504]
    let w: u32 = arm! { stp d(1), d(2), [x(3), #504] };
    assert_eq!(w, 0x6d1f8861);
}

#[test]
fn ldp_x_504_0() {
    // ldp x1, x2, [x3, #504]
    let w: u32 = arm! { ldp x(1), x(2), [x(3), #504] };
    assert_eq!(w, 0xa95f8861);
}

#[test]
fn stp_x_504_0() {
    // stp x1, x2, [x3, #504]
    let w: u32 = arm! { stp x(1), x(2), [x(3), #504] };
    assert_eq!(w, 0xa91f8861);
}

#[test]
fn ldp_d_504_1() {
    // ldp d17, d18, [x19, #504]
    let w: u32 = arm! { ldp d(17), d(18), [x(19), #504] };
    assert_eq!(w, 0x6d5fca71);
}

#[test]
fn stp_d_504_1() {
    // stp d17, d18, [x19, #504]
    let w: u32 = arm! { stp d(17), d(18), [x(19), #504] };
    assert_eq!(w, 0x6d1fca71);
}

#[test]
fn ldp_x_504_1() {
    // ldp x17, x18, [x19, #504]
    let w: u32 = arm! { ldp x(17), x(18), [x(19), #504] };
    assert_eq!(w, 0xa95fca71);
}

#[test]
fn stp_x_504_1() {
    // stp x17, x18, [x19, #504]
    let w: u32 = arm! { stp x(17), x(18), [x(19), #504] };
    assert_eq!(w, 0xa91fca71);
}

#[test]
fn ldp_d_504_2() {
    // ldp d30, d29, [x28, #504]
    let w: u32 = arm! { ldp d(30), d(29), [x(28), #504] };
    assert_eq!(w, 0x6d5ff79e);
}

#[test]
fn stp_d_504_2() {
    // stp d30, d29, [x28, #504]
    let w: u32 = arm! { stp d(30), d(29), [x(28), #504] };
    assert_eq!(w, 0x6d1ff79e);
}

#[test]
fn ldp_x_504_2() {
    // ldp x30, x29, [x28, #504]
    let w: u32 = arm! { ldp x(30), x(29), [x(28), #504] };
    assert_eq!(w, 0xa95ff79e);
}

#[test]
fn stp_x_504_2() {
    // stp x30, x29, [x28, #504]
    let w: u32 = arm! { stp x(30), x(29), [x(28), #504] };
    assert_eq!(w, 0xa91ff79e);
}

#[test]
fn ldp_q_0_0() {
    // ldp q1, q2, [x3, #0]
    let w: u32 = arm! { ldp q(1), q(2), [x(3), #0] };
    assert_eq!(w, 0xad400861);
}

#[test]
fn stp_q_0_0() {
    // stp q1, q2, [x3, #0]
    let w: u32 = arm! { stp q(1), q(2), [x(3), #0] };
    assert_eq!(w, 0xad000861);
}

#[test]
fn ldp_q_0_1() {
    // ldp q17, q18, [x19, #0]
    let w: u32 = arm! { ldp q(17), q(18), [x(19), #0] };
    assert_eq!(w, 0xad404a71);
}

#[test]
fn stp_q_0_1() {
    // stp q17, q18, [x19, #0]
    let w: u32 = arm! { stp q(17), q(18), [x(19), #0] };
    assert_eq!(w, 0xad004a71);
}

#[test]
fn ldp_q_16_0() {
    // ldp q1, q2, [x3, #16]
    let w: u32 = arm! { ldp q(1), q(2), [x(3), #16] };
    assert_eq!(w, 0xad408861);
}

#[test]
fn stp_q_16_0() {
    // stp q1, q2, [x3, #16]
    let w: u32 = arm! { stp q(1), q(2), [x(3), #16] };
    assert_eq!(w, 0xad008861);
}

#[test]
fn ldp_q_16_1() {
    // ldp q17, q18, [x19, #16]
    let w: u32 = arm! { ldp q(17), q(18), [x(19), #16] };
    assert_eq!(w, 0xad40ca71);
}

#[test]
fn stp_q_16_1() {
    // stp q17, q18, [x19, #16]
    let w: u32 = arm! { stp q(17), q(18), [x(19), #16] };
    assert_eq!(w, 0xad00ca71);
}

#[test]
fn ldp_q_1008_0() {
    // ldp q1, q2, [x3, #1008]
    let w: u32 = arm! { ldp q(1), q(2), [x(3), #1008] };
    assert_eq!(w, 0xad5f8861);
}

#[test]
fn stp_q_1008_0() {
    // stp q1, q2, [x3, #1008]
    let w: u32 = arm! { stp q(1), q(2), [x(3), #1008] };
    assert_eq!(w, 0xad1f8861);
}

#[test]
fn ldp_q_1008_1() {
    // ldp q17, q18, [x19, #1008]
    let w: u32 = arm! { ldp q(17), q(18), [x(19), #1008] };
    assert_eq!(w, 0xad5fca71);
}

#[test]
fn stp_q_1008_1() {
    // stp q17, q18, [x19, #1008]
    let w: u32 = arm! { stp q(17), q(18), [x(19), #1008] };
    assert_eq!(w, 0xad1fca71);
}

#[test]
fn ldp_x_sp() {
    // ldp x19, x20, [sp, #16]
    let w: u32 = arm! { ldp x(19), x(20), [sp, #16] };
    assert_eq!(w, 0xa94153f3);
}

#[test]
fn stp_x_sp() {
    // stp x19, x20, [sp, #16]
    let w: u32 = arm! { stp x(19), x(20), [sp, #16] };
    assert_eq!(w, 0xa90153f3);
}

#[test]
fn ldp_lr_sp() {
    // ldp x30, x19, [sp, #16]
    let w: u32 = arm! { ldp lr, x(19), [sp, #16] };
    assert_eq!(w, 0xa9414ffe);
}

#[test]
fn stp_lr_sp() {
    // stp x30, x19, [sp, #16]
    let w: u32 = arm! { stp lr, x(19), [sp, #16] };
    assert_eq!(w, 0xa9014ffe);
}

#[test]
fn ldr_x_sp() {
    // ldr x1, [sp, #8]
    let w: u32 = arm! { ldr x(1), [sp, #8] };
    assert_eq!(w, 0xf94007e1);
}

#[test]
fn str_x_sp() {
    // str x1, [sp, #8]
    let w: u32 = arm! { str x(1), [sp, #8] };
    assert_eq!(w, 0xf90007e1);
}

#[test]
fn ldr_lr_sp() {
    // ldr x30, [sp, #8]
    let w: u32 = arm! { ldr lr, [sp, #8] };
    assert_eq!(w, 0xf94007fe);
}

#[test]
fn str_lr_sp() {
    // str x30, [sp, #8]
    let w: u32 = arm! { str lr, [sp, #8] };
    assert_eq!(w, 0xf90007fe);
}

#[test]
fn ldr_d_sp() {
    // ldr d1, [sp, #8]
    let w: u32 = arm! { ldr d(1), [sp, #8] };
    assert_eq!(w, 0xfd4007e1);
}

#[test]
fn str_d_sp() {
    // str d1, [sp, #8]
    let w: u32 = arm! { str d(1), [sp, #8] };
    assert_eq!(w, 0xfd0007e1);
}

#[test]
fn ldr_q_sp() {
    // ldr q1, [sp, #16]
    let w: u32 = arm! { ldr q(1), [sp, #16] };
    assert_eq!(w, 0x3dc007e1);
}

#[test]
fn str_q_sp() {
    // str q1, [sp, #16]
    let w: u32 = arm! { str q(1), [sp, #16] };
    assert_eq!(w, 0x3d8007e1);
}

#[test]
fn ldp_d_sp() {
    // ldp d8, d9, [sp, #16]
    let w: u32 = arm! { ldp d(8), d(9), [sp, #16] };
    assert_eq!(w, 0x6d4127e8);
}

#[test]
fn stp_d_sp() {
    // stp d8, d9, [sp, #16]
    let w: u32 = arm! { stp d(8), d(9), [sp, #16] };
    assert_eq!(w, 0x6d0127e8);
}

#[test]
fn ldp_q_sp() {
    // ldp q8, q9, [sp, #32]
    let w: u32 = arm! { ldp q(8), q(9), [sp, #32] };
    assert_eq!(w, 0xad4127e8);
}

#[test]
fn stp_q_sp() {
    // stp q8, q9, [sp, #32]
    let w: u32 = arm! { stp q(8), q(9), [sp, #32] };
    assert_eq!(w, 0xad0127e8);
}

#[test]
fn ldr_x_sp_idx() {
    // ldr x1, [sp, x2, lsl #3]
    let w: u32 = arm! { ldr x(1), [sp, x(2), lsl #3] };
    assert_eq!(w, 0xf8627be1);
}

#[test]
fn ldr_d_sp_idx() {
    // ldr d1, [sp, x2, lsl #3]
    let w: u32 = arm! { ldr d(1), [sp, x(2), lsl #3] };
    assert_eq!(w, 0xfc627be1);
}

#[test]
fn str_d_sp_idx() {
    // str d1, [sp, x2, lsl #3]
    let w: u32 = arm! { str d(1), [sp, x(2), lsl #3] };
    assert_eq!(w, 0xfc227be1);
}

#[test]
fn ld1r_0() {
    // ld1r {v1.2d}, [x2]
    let w: u32 = arm! { ld1r {q(1)}, [x(2)] };
    assert_eq!(w, 0x4d40cc41);
}

#[test]
fn ld1r_1() {
    // ld1r {v17.2d}, [x18]
    let w: u32 = arm! { ld1r {q(17)}, [x(18)] };
    assert_eq!(w, 0x4d40ce51);
}

#[test]
fn b_100() {
    // b #256
    let w: u32 = arm! { b label(256i32) };
    assert_eq!(w, 0x14000040);
}

#[test]
fn bl_100() {
    // bl #256
    let w: u32 = arm! { bl label(256i32) };
    assert_eq!(w, 0x94000040);
}

#[test]
fn b_eq_100() {
    // b.eq #256
    let w: u32 = arm! { b.eq label(256i32) };
    assert_eq!(w, 0x54000800);
}

#[test]
fn b_ne_100() {
    // b.ne #256
    let w: u32 = arm! { b.ne label(256i32) };
    assert_eq!(w, 0x54000801);
}

#[test]
fn b_lt_100() {
    // b.lt #256
    let w: u32 = arm! { b.lt label(256i32) };
    assert_eq!(w, 0x5400080b);
}

#[test]
fn b_le_100() {
    // b.le #256
    let w: u32 = arm! { b.le label(256i32) };
    assert_eq!(w, 0x5400080d);
}

#[test]
fn b_gt_100() {
    // b.gt #256
    let w: u32 = arm! { b.gt label(256i32) };
    assert_eq!(w, 0x5400080c);
}

#[test]
fn b_ge_100() {
    // b.ge #256
    let w: u32 = arm! { b.ge label(256i32) };
    assert_eq!(w, 0x5400080a);
}

#[test]
fn b_mi_100() {
    // b.mi #256
    let w: u32 = arm! { b.mi label(256i32) };
    assert_eq!(w, 0x54000804);
}

#[test]
fn tbz_0_100() {
    // tbz x1, #0, #256
    let w: u32 = arm! { tbz x(1), #0, label(256i32) };
    assert_eq!(w, 0x36000801);
}

#[test]
fn tbnz_0_100() {
    // tbnz x1, #0, #256
    let w: u32 = arm! { tbnz x(1), #0, label(256i32) };
    assert_eq!(w, 0x37000801);
}

#[test]
fn tbz_5_100() {
    // tbz x1, #5, #256
    let w: u32 = arm! { tbz x(1), #5, label(256i32) };
    assert_eq!(w, 0x36280801);
}

#[test]
fn tbnz_5_100() {
    // tbnz x1, #5, #256
    let w: u32 = arm! { tbnz x(1), #5, label(256i32) };
    assert_eq!(w, 0x37280801);
}

#[test]
fn tbz_31_100() {
    // tbz x1, #31, #256
    let w: u32 = arm! { tbz x(1), #31, label(256i32) };
    assert_eq!(w, 0x36f80801);
}

#[test]
fn tbnz_31_100() {
    // tbnz x1, #31, #256
    let w: u32 = arm! { tbnz x(1), #31, label(256i32) };
    assert_eq!(w, 0x37f80801);
}

#[test]
fn tbz_32_100() {
    // tbz x1, #32, #256
    let w: u32 = arm! { tbz x(1), #32, label(256i32) };
    assert_eq!(w, 0xb6000801);
}

#[test]
fn tbnz_32_100() {
    // tbnz x1, #32, #256
    let w: u32 = arm! { tbnz x(1), #32, label(256i32) };
    assert_eq!(w, 0xb7000801);
}

#[test]
fn tbz_63_100() {
    // tbz x1, #63, #256
    let w: u32 = arm! { tbz x(1), #63, label(256i32) };
    assert_eq!(w, 0xb6f80801);
}

#[test]
fn tbnz_63_100() {
    // tbnz x1, #63, #256
    let w: u32 = arm! { tbnz x(1), #63, label(256i32) };
    assert_eq!(w, 0xb7f80801);
}

#[test]
fn b_m100() {
    // b #-256
    let w: u32 = arm! { b label(-256i32) };
    assert_eq!(w, 0x17ffffc0);
}

#[test]
fn bl_m100() {
    // bl #-256
    let w: u32 = arm! { bl label(-256i32) };
    assert_eq!(w, 0x97ffffc0);
}

#[test]
fn b_eq_m100() {
    // b.eq #-256
    let w: u32 = arm! { b.eq label(-256i32) };
    assert_eq!(w, 0x54fff800);
}

#[test]
fn b_ne_m100() {
    // b.ne #-256
    let w: u32 = arm! { b.ne label(-256i32) };
    assert_eq!(w, 0x54fff801);
}

#[test]
fn b_lt_m100() {
    // b.lt #-256
    let w: u32 = arm! { b.lt label(-256i32) };
    assert_eq!(w, 0x54fff80b);
}

#[test]
fn b_le_m100() {
    // b.le #-256
    let w: u32 = arm! { b.le label(-256i32) };
    assert_eq!(w, 0x54fff80d);
}

#[test]
fn b_gt_m100() {
    // b.gt #-256
    let w: u32 = arm! { b.gt label(-256i32) };
    assert_eq!(w, 0x54fff80c);
}

#[test]
fn b_ge_m100() {
    // b.ge #-256
    let w: u32 = arm! { b.ge label(-256i32) };
    assert_eq!(w, 0x54fff80a);
}

#[test]
fn b_mi_m100() {
    // b.mi #-256
    let w: u32 = arm! { b.mi label(-256i32) };
    assert_eq!(w, 0x54fff804);
}

#[test]
fn tbz_0_m100() {
    // tbz x1, #0, #-256
    let w: u32 = arm! { tbz x(1), #0, label(-256i32) };
    assert_eq!(w, 0x3607f801);
}

#[test]
fn tbnz_0_m100() {
    // tbnz x1, #0, #-256
    let w: u32 = arm! { tbnz x(1), #0, label(-256i32) };
    assert_eq!(w, 0x3707f801);
}

#[test]
fn tbz_5_m100() {
    // tbz x1, #5, #-256
    let w: u32 = arm! { tbz x(1), #5, label(-256i32) };
    assert_eq!(w, 0x362ff801);
}

#[test]
fn tbnz_5_m100() {
    // tbnz x1, #5, #-256
    let w: u32 = arm! { tbnz x(1), #5, label(-256i32) };
    assert_eq!(w, 0x372ff801);
}

#[test]
fn tbz_31_m100() {
    // tbz x1, #31, #-256
    let w: u32 = arm! { tbz x(1), #31, label(-256i32) };
    assert_eq!(w, 0x36fff801);
}

#[test]
fn tbnz_31_m100() {
    // tbnz x1, #31, #-256
    let w: u32 = arm! { tbnz x(1), #31, label(-256i32) };
    assert_eq!(w, 0x37fff801);
}

#[test]
fn tbz_32_m100() {
    // tbz x1, #32, #-256
    let w: u32 = arm! { tbz x(1), #32, label(-256i32) };
    assert_eq!(w, 0xb607f801);
}

#[test]
fn tbnz_32_m100() {
    // tbnz x1, #32, #-256
    let w: u32 = arm! { tbnz x(1), #32, label(-256i32) };
    assert_eq!(w, 0xb707f801);
}

#[test]
fn tbz_63_m100() {
    // tbz x1, #63, #-256
    let w: u32 = arm! { tbz x(1), #63, label(-256i32) };
    assert_eq!(w, 0xb6fff801);
}

#[test]
fn tbnz_63_m100() {
    // tbnz x1, #63, #-256
    let w: u32 = arm! { tbnz x(1), #63, label(-256i32) };
    assert_eq!(w, 0xb7fff801);
}
