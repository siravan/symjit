// Encoding tests for the RISC-V instruction macro `rvv!` (macros.rs).
//
// Each test builds one instruction with `rvv!` and compares the 32-bit word with the
// encoding produced by an independent encoder written from the RISC-V unprivileged ISA
// field layouts (not from the macro's opcode constants); every expected word was also
// verified by disassembling it with Capstone. The `// comment` in each test is the
// assembly line the word encodes.
//
// Conventions in `rvv!`: `x(n)` / `f(n)` are integer / floating-point register numbers.
// `sd`/`fsd` take the *source* register first and the base register second
// (`sd x(src), x(base), imm`). `ble`/`bgt`/`fgt.d`/`fge.d` are pseudo-instructions that
// swap their operands.
//
// `#[ignore]`d tests document known defects; run with `cargo test riscv64::tests -- --ignored`.


#[test]
fn add_0() {
    // add ra, sp, gp
    let w: u32 = rvv! { add x(1), x(2), x(3) };
    assert_eq!(w, 0x003100b3);
}

#[test]
fn add_1() {
    // add a0, a1, a2
    let w: u32 = rvv! { add x(10), x(11), x(12) };
    assert_eq!(w, 0x00c58533);
}

#[test]
fn add_2() {
    // add t6, t5, t4
    let w: u32 = rvv! { add x(31), x(30), x(29) };
    assert_eq!(w, 0x01df0fb3);
}

#[test]
fn add_3() {
    // add t0, t1, t2
    let w: u32 = rvv! { add x(5), x(6), x(7) };
    assert_eq!(w, 0x007302b3);
}

#[test]
fn add_4() {
    // add zero, s0, s1
    let w: u32 = rvv! { add x(0), x(8), x(9) };
    assert_eq!(w, 0x00940033);
}

#[test]
fn sub_0() {
    // sub ra, sp, gp
    let w: u32 = rvv! { sub x(1), x(2), x(3) };
    assert_eq!(w, 0x403100b3);
}

#[test]
fn sub_1() {
    // sub a0, a1, a2
    let w: u32 = rvv! { sub x(10), x(11), x(12) };
    assert_eq!(w, 0x40c58533);
}

#[test]
fn sub_2() {
    // sub t6, t5, t4
    let w: u32 = rvv! { sub x(31), x(30), x(29) };
    assert_eq!(w, 0x41df0fb3);
}

#[test]
fn sub_3() {
    // sub t0, t1, t2
    let w: u32 = rvv! { sub x(5), x(6), x(7) };
    assert_eq!(w, 0x407302b3);
}

#[test]
fn sub_4() {
    // sub zero, s0, s1
    let w: u32 = rvv! { sub x(0), x(8), x(9) };
    assert_eq!(w, 0x40940033);
}

#[test]
fn sll_0() {
    // sll ra, sp, gp
    let w: u32 = rvv! { sll x(1), x(2), x(3) };
    assert_eq!(w, 0x003110b3);
}

#[test]
fn sll_1() {
    // sll a0, a1, a2
    let w: u32 = rvv! { sll x(10), x(11), x(12) };
    assert_eq!(w, 0x00c59533);
}

#[test]
fn sll_2() {
    // sll t6, t5, t4
    let w: u32 = rvv! { sll x(31), x(30), x(29) };
    assert_eq!(w, 0x01df1fb3);
}

#[test]
fn sll_3() {
    // sll t0, t1, t2
    let w: u32 = rvv! { sll x(5), x(6), x(7) };
    assert_eq!(w, 0x007312b3);
}

#[test]
fn sll_4() {
    // sll zero, s0, s1
    let w: u32 = rvv! { sll x(0), x(8), x(9) };
    assert_eq!(w, 0x00941033);
}

#[test]
fn xor_0() {
    // xor ra, sp, gp
    let w: u32 = rvv! { xor x(1), x(2), x(3) };
    assert_eq!(w, 0x003140b3);
}

#[test]
fn xor_1() {
    // xor a0, a1, a2
    let w: u32 = rvv! { xor x(10), x(11), x(12) };
    assert_eq!(w, 0x00c5c533);
}

#[test]
fn xor_2() {
    // xor t6, t5, t4
    let w: u32 = rvv! { xor x(31), x(30), x(29) };
    assert_eq!(w, 0x01df4fb3);
}

#[test]
fn xor_3() {
    // xor t0, t1, t2
    let w: u32 = rvv! { xor x(5), x(6), x(7) };
    assert_eq!(w, 0x007342b3);
}

#[test]
fn xor_4() {
    // xor zero, s0, s1
    let w: u32 = rvv! { xor x(0), x(8), x(9) };
    assert_eq!(w, 0x00944033);
}

#[test]
fn srl_0() {
    // srl ra, sp, gp
    let w: u32 = rvv! { srl x(1), x(2), x(3) };
    assert_eq!(w, 0x003150b3);
}

#[test]
fn srl_1() {
    // srl a0, a1, a2
    let w: u32 = rvv! { srl x(10), x(11), x(12) };
    assert_eq!(w, 0x00c5d533);
}

#[test]
fn srl_2() {
    // srl t6, t5, t4
    let w: u32 = rvv! { srl x(31), x(30), x(29) };
    assert_eq!(w, 0x01df5fb3);
}

#[test]
fn srl_3() {
    // srl t0, t1, t2
    let w: u32 = rvv! { srl x(5), x(6), x(7) };
    assert_eq!(w, 0x007352b3);
}

#[test]
fn srl_4() {
    // srl zero, s0, s1
    let w: u32 = rvv! { srl x(0), x(8), x(9) };
    assert_eq!(w, 0x00945033);
}

#[test]
fn sra_0() {
    // sra ra, sp, gp
    let w: u32 = rvv! { sra x(1), x(2), x(3) };
    assert_eq!(w, 0x403150b3);
}

#[test]
fn sra_1() {
    // sra a0, a1, a2
    let w: u32 = rvv! { sra x(10), x(11), x(12) };
    assert_eq!(w, 0x40c5d533);
}

#[test]
fn sra_2() {
    // sra t6, t5, t4
    let w: u32 = rvv! { sra x(31), x(30), x(29) };
    assert_eq!(w, 0x41df5fb3);
}

#[test]
fn sra_3() {
    // sra t0, t1, t2
    let w: u32 = rvv! { sra x(5), x(6), x(7) };
    assert_eq!(w, 0x407352b3);
}

#[test]
fn sra_4() {
    // sra zero, s0, s1
    let w: u32 = rvv! { sra x(0), x(8), x(9) };
    assert_eq!(w, 0x40945033);
}

#[test]
fn or_0() {
    // or ra, sp, gp
    let w: u32 = rvv! { or x(1), x(2), x(3) };
    assert_eq!(w, 0x003160b3);
}

#[test]
fn or_1() {
    // or a0, a1, a2
    let w: u32 = rvv! { or x(10), x(11), x(12) };
    assert_eq!(w, 0x00c5e533);
}

#[test]
fn or_2() {
    // or t6, t5, t4
    let w: u32 = rvv! { or x(31), x(30), x(29) };
    assert_eq!(w, 0x01df6fb3);
}

#[test]
fn or_3() {
    // or t0, t1, t2
    let w: u32 = rvv! { or x(5), x(6), x(7) };
    assert_eq!(w, 0x007362b3);
}

#[test]
fn or_4() {
    // or zero, s0, s1
    let w: u32 = rvv! { or x(0), x(8), x(9) };
    assert_eq!(w, 0x00946033);
}

#[test]
fn and_0() {
    // and ra, sp, gp
    let w: u32 = rvv! { and x(1), x(2), x(3) };
    assert_eq!(w, 0x003170b3);
}

#[test]
fn and_1() {
    // and a0, a1, a2
    let w: u32 = rvv! { and x(10), x(11), x(12) };
    assert_eq!(w, 0x00c5f533);
}

#[test]
fn and_2() {
    // and t6, t5, t4
    let w: u32 = rvv! { and x(31), x(30), x(29) };
    assert_eq!(w, 0x01df7fb3);
}

#[test]
fn and_3() {
    // and t0, t1, t2
    let w: u32 = rvv! { and x(5), x(6), x(7) };
    assert_eq!(w, 0x007372b3);
}

#[test]
fn and_4() {
    // and zero, s0, s1
    let w: u32 = rvv! { and x(0), x(8), x(9) };
    assert_eq!(w, 0x00947033);
}

#[test]
fn mv_0() {
    // mv ra, sp
    // NOTE: canonical `mv` is `addi rd, rs, 0`; the macro emits the equivalent `add rd, rs, zero`
    let w: u32 = rvv! { mv x(1), x(2) };
    assert_eq!(w, 0x000100b3);
}

#[test]
fn neg_0() {
    // neg ra, sp
    let w: u32 = rvv! { neg x(1), x(2) };
    assert_eq!(w, 0x402000b3);
}

#[test]
fn not_0() {
    // not ra, sp
    let w: u32 = rvv! { not x(1), x(2) };
    assert_eq!(w, 0xfff14093);
}

#[test]
fn mv_1() {
    // mv a0, a1
    // NOTE: canonical `mv` is `addi rd, rs, 0`; the macro emits the equivalent `add rd, rs, zero`
    let w: u32 = rvv! { mv x(10), x(11) };
    assert_eq!(w, 0x00058533);
}

#[test]
fn neg_1() {
    // neg a0, a1
    let w: u32 = rvv! { neg x(10), x(11) };
    assert_eq!(w, 0x40b00533);
}

#[test]
fn not_1() {
    // not a0, a1
    let w: u32 = rvv! { not x(10), x(11) };
    assert_eq!(w, 0xfff5c513);
}

#[test]
fn mv_2() {
    // mv t6, t5
    // NOTE: canonical `mv` is `addi rd, rs, 0`; the macro emits the equivalent `add rd, rs, zero`
    let w: u32 = rvv! { mv x(31), x(30) };
    assert_eq!(w, 0x000f0fb3);
}

#[test]
fn neg_2() {
    // neg t6, t5
    let w: u32 = rvv! { neg x(31), x(30) };
    assert_eq!(w, 0x41e00fb3);
}

#[test]
fn not_2() {
    // not t6, t5
    let w: u32 = rvv! { not x(31), x(30) };
    assert_eq!(w, 0xffff4f93);
}

#[test]
fn mv_3() {
    // mv t0, t1
    // NOTE: canonical `mv` is `addi rd, rs, 0`; the macro emits the equivalent `add rd, rs, zero`
    let w: u32 = rvv! { mv x(5), x(6) };
    assert_eq!(w, 0x000302b3);
}

#[test]
fn neg_3() {
    // neg t0, t1
    let w: u32 = rvv! { neg x(5), x(6) };
    assert_eq!(w, 0x406002b3);
}

#[test]
fn not_3() {
    // not t0, t1
    let w: u32 = rvv! { not x(5), x(6) };
    assert_eq!(w, 0xfff34293);
}

#[test]
fn mv_4() {
    // mv zero, s0
    // NOTE: canonical `mv` is `addi rd, rs, 0`; the macro emits the equivalent `add rd, rs, zero`
    let w: u32 = rvv! { mv x(0), x(8) };
    assert_eq!(w, 0x00040033);
}

#[test]
fn neg_4() {
    // neg zero, s0
    let w: u32 = rvv! { neg x(0), x(8) };
    assert_eq!(w, 0x40800033);
}

#[test]
fn not_4() {
    // not zero, s0
    let w: u32 = rvv! { not x(0), x(8) };
    assert_eq!(w, 0xfff44013);
}

#[test]
fn addi_0() {
    // addi ra, sp, 0
    let w: u32 = rvv! { addi x(1), x(2), 0 };
    assert_eq!(w, 0x00010093);
}

#[test]
fn addi_1() {
    // addi a0, a1, 1
    let w: u32 = rvv! { addi x(10), x(11), 1 };
    assert_eq!(w, 0x00158513);
}

#[test]
fn addi_2() {
    // addi t6, t5, 5
    let w: u32 = rvv! { addi x(31), x(30), 5 };
    assert_eq!(w, 0x005f0f93);
}

#[test]
fn addi_3() {
    // addi t0, t1, 2047
    let w: u32 = rvv! { addi x(5), x(6), 2047 };
    assert_eq!(w, 0x7ff30293);
}

#[test]
fn addi_4() {
    // addi zero, s0, -1
    let w: u32 = rvv! { addi x(0), x(8), -1 };
    assert_eq!(w, 0xfff40013);
}

#[test]
fn addi_5() {
    // addi ra, sp, -2048
    let w: u32 = rvv! { addi x(1), x(2), -2048 };
    assert_eq!(w, 0x80010093);
}

#[test]
fn addi_6() {
    // addi a0, a1, -100
    let w: u32 = rvv! { addi x(10), x(11), -100 };
    assert_eq!(w, 0xf9c58513);
}

#[test]
fn xori_0() {
    // xori ra, sp, 0
    let w: u32 = rvv! { xori x(1), x(2), 0 };
    assert_eq!(w, 0x00014093);
}

#[test]
fn xori_1() {
    // xori a0, a1, 1
    let w: u32 = rvv! { xori x(10), x(11), 1 };
    assert_eq!(w, 0x0015c513);
}

#[test]
fn xori_2() {
    // xori t6, t5, 5
    let w: u32 = rvv! { xori x(31), x(30), 5 };
    assert_eq!(w, 0x005f4f93);
}

#[test]
fn xori_3() {
    // xori t0, t1, 2047
    let w: u32 = rvv! { xori x(5), x(6), 2047 };
    assert_eq!(w, 0x7ff34293);
}

#[test]
fn xori_4() {
    // xori zero, s0, -1
    let w: u32 = rvv! { xori x(0), x(8), -1 };
    assert_eq!(w, 0xfff44013);
}

#[test]
fn xori_5() {
    // xori ra, sp, -2048
    let w: u32 = rvv! { xori x(1), x(2), -2048 };
    assert_eq!(w, 0x80014093);
}

#[test]
fn xori_6() {
    // xori a0, a1, -100
    let w: u32 = rvv! { xori x(10), x(11), -100 };
    assert_eq!(w, 0xf9c5c513);
}

#[test]
fn ori_0() {
    // ori ra, sp, 0
    let w: u32 = rvv! { ori x(1), x(2), 0 };
    assert_eq!(w, 0x00016093);
}

#[test]
fn ori_1() {
    // ori a0, a1, 1
    let w: u32 = rvv! { ori x(10), x(11), 1 };
    assert_eq!(w, 0x0015e513);
}

#[test]
fn ori_2() {
    // ori t6, t5, 5
    let w: u32 = rvv! { ori x(31), x(30), 5 };
    assert_eq!(w, 0x005f6f93);
}

#[test]
fn ori_3() {
    // ori t0, t1, 2047
    let w: u32 = rvv! { ori x(5), x(6), 2047 };
    assert_eq!(w, 0x7ff36293);
}

#[test]
fn ori_4() {
    // ori zero, s0, -1
    let w: u32 = rvv! { ori x(0), x(8), -1 };
    assert_eq!(w, 0xfff46013);
}

#[test]
fn ori_5() {
    // ori ra, sp, -2048
    let w: u32 = rvv! { ori x(1), x(2), -2048 };
    assert_eq!(w, 0x80016093);
}

#[test]
fn ori_6() {
    // ori a0, a1, -100
    let w: u32 = rvv! { ori x(10), x(11), -100 };
    assert_eq!(w, 0xf9c5e513);
}

#[test]
fn andi_0() {
    // andi ra, sp, 0
    let w: u32 = rvv! { andi x(1), x(2), 0 };
    assert_eq!(w, 0x00017093);
}

#[test]
fn andi_1() {
    // andi a0, a1, 1
    let w: u32 = rvv! { andi x(10), x(11), 1 };
    assert_eq!(w, 0x0015f513);
}

#[test]
fn andi_2() {
    // andi t6, t5, 5
    let w: u32 = rvv! { andi x(31), x(30), 5 };
    assert_eq!(w, 0x005f7f93);
}

#[test]
fn andi_3() {
    // andi t0, t1, 2047
    let w: u32 = rvv! { andi x(5), x(6), 2047 };
    assert_eq!(w, 0x7ff37293);
}

#[test]
fn andi_4() {
    // andi zero, s0, -1
    let w: u32 = rvv! { andi x(0), x(8), -1 };
    assert_eq!(w, 0xfff47013);
}

#[test]
fn andi_5() {
    // andi ra, sp, -2048
    let w: u32 = rvv! { andi x(1), x(2), -2048 };
    assert_eq!(w, 0x80017093);
}

#[test]
fn andi_6() {
    // andi a0, a1, -100
    let w: u32 = rvv! { andi x(10), x(11), -100 };
    assert_eq!(w, 0xf9c5f513);
}

#[test]
fn slli_0() {
    // slli ra, sp, 0
    let w: u32 = rvv! { slli x(1), x(2), 0 };
    assert_eq!(w, 0x00011093);
}

#[test]
fn slli_1() {
    // slli a0, a1, 1
    let w: u32 = rvv! { slli x(10), x(11), 1 };
    assert_eq!(w, 0x00159513);
}

#[test]
fn slli_13() {
    // slli t6, t5, 13
    let w: u32 = rvv! { slli x(31), x(30), 13 };
    assert_eq!(w, 0x00df1f93);
}

#[test]
fn slli_31() {
    // slli t0, t1, 31
    let w: u32 = rvv! { slli x(5), x(6), 31 };
    assert_eq!(w, 0x01f31293);
}

#[test]
fn slli_32() {
    // slli zero, s0, 32
    let w: u32 = rvv! { slli x(0), x(8), 32 };
    assert_eq!(w, 0x02041013);
}

#[test]
fn slli_63() {
    // slli ra, sp, 63
    let w: u32 = rvv! { slli x(1), x(2), 63 };
    assert_eq!(w, 0x03f11093);
}

#[test]
fn srli_0() {
    // srli ra, sp, 0
    let w: u32 = rvv! { srli x(1), x(2), 0 };
    assert_eq!(w, 0x00015093);
}

#[test]
fn srli_1() {
    // srli a0, a1, 1
    let w: u32 = rvv! { srli x(10), x(11), 1 };
    assert_eq!(w, 0x0015d513);
}

#[test]
fn srli_13() {
    // srli t6, t5, 13
    let w: u32 = rvv! { srli x(31), x(30), 13 };
    assert_eq!(w, 0x00df5f93);
}

#[test]
fn srli_31() {
    // srli t0, t1, 31
    let w: u32 = rvv! { srli x(5), x(6), 31 };
    assert_eq!(w, 0x01f35293);
}

#[test]
fn srli_32() {
    // srli zero, s0, 32
    let w: u32 = rvv! { srli x(0), x(8), 32 };
    assert_eq!(w, 0x02045013);
}

#[test]
fn srli_63() {
    // srli ra, sp, 63
    let w: u32 = rvv! { srli x(1), x(2), 63 };
    assert_eq!(w, 0x03f15093);
}

#[test]
fn srai_0() {
    // srai ra, sp, 0
    let w: u32 = rvv! { srai x(1), x(2), 0 };
    assert_eq!(w, 0x40015093);
}

#[test]
fn srai_1() {
    // srai a0, a1, 1
    let w: u32 = rvv! { srai x(10), x(11), 1 };
    assert_eq!(w, 0x4015d513);
}

#[test]
fn srai_13() {
    // srai t6, t5, 13
    let w: u32 = rvv! { srai x(31), x(30), 13 };
    assert_eq!(w, 0x40df5f93);
}

#[test]
fn srai_31() {
    // srai t0, t1, 31
    let w: u32 = rvv! { srai x(5), x(6), 31 };
    assert_eq!(w, 0x41f35293);
}

#[test]
fn srai_32() {
    // srai zero, s0, 32
    let w: u32 = rvv! { srai x(0), x(8), 32 };
    assert_eq!(w, 0x42045013);
}

#[test]
fn srai_63() {
    // srai ra, sp, 63
    let w: u32 = rvv! { srai x(1), x(2), 63 };
    assert_eq!(w, 0x43f15093);
}

#[test]
fn nop() {
    // nop
    let w: u32 = rvv! { nop };
    assert_eq!(w, 0x00000013);
}

#[test]
fn ld_0() {
    // ld ra, 0(sp)
    let w: u32 = rvv! { ld x(1), x(2), 0 };
    assert_eq!(w, 0x00013083);
}

#[test]
fn sd_0() {
    // sd ra, 0(sp)
    let w: u32 = rvv! { sd x(1), x(2), 0 };
    assert_eq!(w, 0x00113023);
}

#[test]
fn fld_0() {
    // fld ft1, 0(sp)
    let w: u32 = rvv! { fld f(1), x(2), 0 };
    assert_eq!(w, 0x00013087);
}

#[test]
fn fsd_0() {
    // fsd ft1, 0(sp)
    let w: u32 = rvv! { fsd f(1), x(2), 0 };
    assert_eq!(w, 0x00113027);
}

#[test]
fn ld_1() {
    // ld a0, 8(a1)
    let w: u32 = rvv! { ld x(10), x(11), 8 };
    assert_eq!(w, 0x0085b503);
}

#[test]
fn sd_1() {
    // sd a0, 8(a1)
    let w: u32 = rvv! { sd x(10), x(11), 8 };
    assert_eq!(w, 0x00a5b423);
}

#[test]
fn fld_1() {
    // fld fa0, 8(a1)
    let w: u32 = rvv! { fld f(10), x(11), 8 };
    assert_eq!(w, 0x0085b507);
}

#[test]
fn fsd_1() {
    // fsd fa0, 8(a1)
    let w: u32 = rvv! { fsd f(10), x(11), 8 };
    assert_eq!(w, 0x00a5b427);
}

#[test]
fn ld_2() {
    // ld t6, 2040(t5)
    let w: u32 = rvv! { ld x(31), x(30), 2040 };
    assert_eq!(w, 0x7f8f3f83);
}

#[test]
fn sd_2() {
    // sd t6, 2040(t5)
    let w: u32 = rvv! { sd x(31), x(30), 2040 };
    assert_eq!(w, 0x7fff3c23);
}

#[test]
fn fld_2() {
    // fld ft11, 2040(t5)
    let w: u32 = rvv! { fld f(31), x(30), 2040 };
    assert_eq!(w, 0x7f8f3f87);
}

#[test]
fn fsd_2() {
    // fsd ft11, 2040(t5)
    let w: u32 = rvv! { fsd f(31), x(30), 2040 };
    assert_eq!(w, 0x7fff3c27);
}

#[test]
fn ld_3() {
    // ld t0, -8(t1)
    let w: u32 = rvv! { ld x(5), x(6), -8 };
    assert_eq!(w, 0xff833283);
}

#[test]
fn sd_3() {
    // sd t0, -8(t1)
    let w: u32 = rvv! { sd x(5), x(6), -8 };
    assert_eq!(w, 0xfe533c23);
}

#[test]
fn fld_3() {
    // fld ft5, -8(t1)
    let w: u32 = rvv! { fld f(5), x(6), -8 };
    assert_eq!(w, 0xff833287);
}

#[test]
fn fsd_3() {
    // fsd ft5, -8(t1)
    let w: u32 = rvv! { fsd f(5), x(6), -8 };
    assert_eq!(w, 0xfe533c27);
}

#[test]
fn ld_4() {
    // ld zero, -2048(s0)
    let w: u32 = rvv! { ld x(0), x(8), -2048 };
    assert_eq!(w, 0x80043003);
}

#[test]
fn sd_4() {
    // sd zero, -2048(s0)
    let w: u32 = rvv! { sd x(0), x(8), -2048 };
    assert_eq!(w, 0x80043023);
}

#[test]
fn fld_4() {
    // fld ft0, -2048(s0)
    let w: u32 = rvv! { fld f(0), x(8), -2048 };
    assert_eq!(w, 0x80043007);
}

#[test]
fn fsd_4() {
    // fsd ft0, -2048(s0)
    let w: u32 = rvv! { fsd f(0), x(8), -2048 };
    assert_eq!(w, 0x80043027);
}

#[test]
fn ld_5() {
    // ld ra, 2047(sp)
    let w: u32 = rvv! { ld x(1), x(2), 2047 };
    assert_eq!(w, 0x7ff13083);
}

#[test]
fn sd_5() {
    // sd ra, 2047(sp)
    let w: u32 = rvv! { sd x(1), x(2), 2047 };
    assert_eq!(w, 0x7e113fa3);
}

#[test]
fn fld_5() {
    // fld ft1, 2047(sp)
    let w: u32 = rvv! { fld f(1), x(2), 2047 };
    assert_eq!(w, 0x7ff13087);
}

#[test]
fn fsd_5() {
    // fsd ft1, 2047(sp)
    let w: u32 = rvv! { fsd f(1), x(2), 2047 };
    assert_eq!(w, 0x7e113fa7);
}

#[test]
fn beq_0() {
    // beq ra, sp, 16
    let w: u32 = rvv! { beq x(1), x(2), 16 };
    assert_eq!(w, 0x00208863);
}

#[test]
fn beq_1() {
    // beq a0, a1, -16
    let w: u32 = rvv! { beq x(10), x(11), -16 };
    assert_eq!(w, 0xfeb508e3);
}

#[test]
fn beq_2() {
    // beq t6, t5, 4094
    let w: u32 = rvv! { beq x(31), x(30), 4094 };
    assert_eq!(w, 0x7fef8fe3);
}

#[test]
fn beq_3() {
    // beq t0, t1, -4096
    let w: u32 = rvv! { beq x(5), x(6), -4096 };
    assert_eq!(w, 0x80628063);
}

#[test]
fn beq_4() {
    // beq zero, s0, 2048
    let w: u32 = rvv! { beq x(0), x(8), 2048 };
    assert_eq!(w, 0x008000e3);
}

#[test]
fn beq_5() {
    // beq ra, sp, -2050
    let w: u32 = rvv! { beq x(1), x(2), -2050 };
    assert_eq!(w, 0xfe208f63);
}

#[test]
fn bne_0() {
    // bne ra, sp, 16
    let w: u32 = rvv! { bne x(1), x(2), 16 };
    assert_eq!(w, 0x00209863);
}

#[test]
fn bne_1() {
    // bne a0, a1, -16
    let w: u32 = rvv! { bne x(10), x(11), -16 };
    assert_eq!(w, 0xfeb518e3);
}

#[test]
fn bne_2() {
    // bne t6, t5, 4094
    let w: u32 = rvv! { bne x(31), x(30), 4094 };
    assert_eq!(w, 0x7fef9fe3);
}

#[test]
fn bne_3() {
    // bne t0, t1, -4096
    let w: u32 = rvv! { bne x(5), x(6), -4096 };
    assert_eq!(w, 0x80629063);
}

#[test]
fn bne_4() {
    // bne zero, s0, 2048
    let w: u32 = rvv! { bne x(0), x(8), 2048 };
    assert_eq!(w, 0x008010e3);
}

#[test]
fn bne_5() {
    // bne ra, sp, -2050
    let w: u32 = rvv! { bne x(1), x(2), -2050 };
    assert_eq!(w, 0xfe209f63);
}

#[test]
fn blt_0() {
    // blt ra, sp, 16
    let w: u32 = rvv! { blt x(1), x(2), 16 };
    assert_eq!(w, 0x0020c863);
}

#[test]
fn blt_1() {
    // blt a0, a1, -16
    let w: u32 = rvv! { blt x(10), x(11), -16 };
    assert_eq!(w, 0xfeb548e3);
}

#[test]
fn blt_2() {
    // blt t6, t5, 4094
    let w: u32 = rvv! { blt x(31), x(30), 4094 };
    assert_eq!(w, 0x7fefcfe3);
}

#[test]
fn blt_3() {
    // blt t0, t1, -4096
    let w: u32 = rvv! { blt x(5), x(6), -4096 };
    assert_eq!(w, 0x8062c063);
}

#[test]
fn blt_4() {
    // blt zero, s0, 2048
    let w: u32 = rvv! { blt x(0), x(8), 2048 };
    assert_eq!(w, 0x008040e3);
}

#[test]
fn blt_5() {
    // blt ra, sp, -2050
    let w: u32 = rvv! { blt x(1), x(2), -2050 };
    assert_eq!(w, 0xfe20cf63);
}

#[test]
fn bge_0() {
    // bge ra, sp, 16
    let w: u32 = rvv! { bge x(1), x(2), 16 };
    assert_eq!(w, 0x0020d863);
}

#[test]
fn bge_1() {
    // bge a0, a1, -16
    let w: u32 = rvv! { bge x(10), x(11), -16 };
    assert_eq!(w, 0xfeb558e3);
}

#[test]
fn bge_2() {
    // bge t6, t5, 4094
    let w: u32 = rvv! { bge x(31), x(30), 4094 };
    assert_eq!(w, 0x7fefdfe3);
}

#[test]
fn bge_3() {
    // bge t0, t1, -4096
    let w: u32 = rvv! { bge x(5), x(6), -4096 };
    assert_eq!(w, 0x8062d063);
}

#[test]
fn bge_4() {
    // bge zero, s0, 2048
    let w: u32 = rvv! { bge x(0), x(8), 2048 };
    assert_eq!(w, 0x008050e3);
}

#[test]
fn bge_5() {
    // bge ra, sp, -2050
    let w: u32 = rvv! { bge x(1), x(2), -2050 };
    assert_eq!(w, 0xfe20df63);
}

#[test]
fn bltu_0() {
    // bltu ra, sp, 16
    let w: u32 = rvv! { bltu x(1), x(2), 16 };
    assert_eq!(w, 0x0020e863);
}

#[test]
fn bltu_1() {
    // bltu a0, a1, -16
    let w: u32 = rvv! { bltu x(10), x(11), -16 };
    assert_eq!(w, 0xfeb568e3);
}

#[test]
fn bltu_2() {
    // bltu t6, t5, 4094
    let w: u32 = rvv! { bltu x(31), x(30), 4094 };
    assert_eq!(w, 0x7fefefe3);
}

#[test]
fn bltu_3() {
    // bltu t0, t1, -4096
    let w: u32 = rvv! { bltu x(5), x(6), -4096 };
    assert_eq!(w, 0x8062e063);
}

#[test]
fn bltu_4() {
    // bltu zero, s0, 2048
    let w: u32 = rvv! { bltu x(0), x(8), 2048 };
    assert_eq!(w, 0x008060e3);
}

#[test]
fn bltu_5() {
    // bltu ra, sp, -2050
    let w: u32 = rvv! { bltu x(1), x(2), -2050 };
    assert_eq!(w, 0xfe20ef63);
}

#[test]
fn bgeu_0() {
    // bgeu ra, sp, 16
    let w: u32 = rvv! { bgeu x(1), x(2), 16 };
    assert_eq!(w, 0x0020f863);
}

#[test]
fn bgeu_1() {
    // bgeu a0, a1, -16
    let w: u32 = rvv! { bgeu x(10), x(11), -16 };
    assert_eq!(w, 0xfeb578e3);
}

#[test]
fn bgeu_2() {
    // bgeu t6, t5, 4094
    let w: u32 = rvv! { bgeu x(31), x(30), 4094 };
    assert_eq!(w, 0x7fefffe3);
}

#[test]
fn bgeu_3() {
    // bgeu t0, t1, -4096
    let w: u32 = rvv! { bgeu x(5), x(6), -4096 };
    assert_eq!(w, 0x8062f063);
}

#[test]
fn bgeu_4() {
    // bgeu zero, s0, 2048
    let w: u32 = rvv! { bgeu x(0), x(8), 2048 };
    assert_eq!(w, 0x008070e3);
}

#[test]
fn bgeu_5() {
    // bgeu ra, sp, -2050
    let w: u32 = rvv! { bgeu x(1), x(2), -2050 };
    assert_eq!(w, 0xfe20ff63);
}

#[test]
fn ble_0() {
    // bge sp, ra, 16
    let w: u32 = rvv! { ble x(1), x(2), 16 };
    assert_eq!(w, 0x00115863);
}

#[test]
fn ble_1() {
    // bge a1, a0, -16
    let w: u32 = rvv! { ble x(10), x(11), -16 };
    assert_eq!(w, 0xfea5d8e3);
}

#[test]
fn ble_2() {
    // bge t5, t6, 4094
    let w: u32 = rvv! { ble x(31), x(30), 4094 };
    assert_eq!(w, 0x7fff5fe3);
}

#[test]
fn ble_3() {
    // bge t1, t0, -4096
    let w: u32 = rvv! { ble x(5), x(6), -4096 };
    assert_eq!(w, 0x80535063);
}

#[test]
fn ble_4() {
    // bge s0, zero, 2048
    let w: u32 = rvv! { ble x(0), x(8), 2048 };
    assert_eq!(w, 0x000450e3);
}

#[test]
fn ble_5() {
    // bge sp, ra, -2050
    let w: u32 = rvv! { ble x(1), x(2), -2050 };
    assert_eq!(w, 0xfe115f63);
}

#[test]
fn bgt_0() {
    // blt sp, ra, 16
    let w: u32 = rvv! { bgt x(1), x(2), 16 };
    assert_eq!(w, 0x00114863);
}

#[test]
fn bgt_1() {
    // blt a1, a0, -16
    let w: u32 = rvv! { bgt x(10), x(11), -16 };
    assert_eq!(w, 0xfea5c8e3);
}

#[test]
fn bgt_2() {
    // blt t5, t6, 4094
    let w: u32 = rvv! { bgt x(31), x(30), 4094 };
    assert_eq!(w, 0x7fff4fe3);
}

#[test]
fn bgt_3() {
    // blt t1, t0, -4096
    let w: u32 = rvv! { bgt x(5), x(6), -4096 };
    assert_eq!(w, 0x80534063);
}

#[test]
fn bgt_4() {
    // blt s0, zero, 2048
    let w: u32 = rvv! { bgt x(0), x(8), 2048 };
    assert_eq!(w, 0x000440e3);
}

#[test]
fn bgt_5() {
    // blt sp, ra, -2050
    let w: u32 = rvv! { bgt x(1), x(2), -2050 };
    assert_eq!(w, 0xfe114f63);
}

#[test]
fn jal_0() {
    // jal ra, 256
    let w: u32 = rvv! { jal x(1), 256 };
    assert_eq!(w, 0x100000ef);
}

#[test]
fn j_0() {
    // j 256
    let w: u32 = rvv! { j 256 };
    assert_eq!(w, 0x1000006f);
}

#[test]
fn jal_1() {
    // jal a0, -256
    let w: u32 = rvv! { jal x(10), -256 };
    assert_eq!(w, 0xf01ff56f);
}

#[test]
fn j_1() {
    // j -256
    let w: u32 = rvv! { j -256 };
    assert_eq!(w, 0xf01ff06f);
}

#[test]
fn jal_2() {
    // jal t6, 2046
    let w: u32 = rvv! { jal x(31), 2046 };
    assert_eq!(w, 0x7fe00fef);
}

#[test]
fn j_2() {
    // j 2046
    let w: u32 = rvv! { j 2046 };
    assert_eq!(w, 0x7fe0006f);
}

#[test]
fn jal_3() {
    // jal t0, 1048574
    let w: u32 = rvv! { jal x(5), 1048574 };
    assert_eq!(w, 0x7ffff2ef);
}

#[test]
fn j_3() {
    // j 1048574
    let w: u32 = rvv! { j 1048574 };
    assert_eq!(w, 0x7ffff06f);
}

#[test]
fn jal_4() {
    // jal zero, -1048576
    let w: u32 = rvv! { jal x(0), -1048576 };
    assert_eq!(w, 0x8000006f);
}

#[test]
fn j_4() {
    // j -1048576
    let w: u32 = rvv! { j -1048576 };
    assert_eq!(w, 0x8000006f);
}

#[test]
fn jal_5() {
    // jal ra, 2048
    let w: u32 = rvv! { jal x(1), 2048 };
    assert_eq!(w, 0x001000ef);
}

#[test]
fn j_5() {
    // j 2048
    let w: u32 = rvv! { j 2048 };
    assert_eq!(w, 0x0010006f);
}

#[test]
fn jal_6() {
    // jal a0, -2048
    let w: u32 = rvv! { jal x(10), -2048 };
    assert_eq!(w, 0x801ff56f);
}

#[test]
fn j_6() {
    // j -2048
    let w: u32 = rvv! { j -2048 };
    assert_eq!(w, 0x801ff06f);
}

#[test]
fn jalr_0() {
    // jalr ra, 0(sp)
    let w: u32 = rvv! { jalr x(1), x(2), 0 };
    assert_eq!(w, 0x000100e7);
}

#[test]
fn jalr_1() {
    // jalr a0, 8(a1)
    let w: u32 = rvv! { jalr x(10), x(11), 8 };
    assert_eq!(w, 0x00858567);
}

#[test]
fn jalr_2() {
    // jalr t6, -8(t5)
    let w: u32 = rvv! { jalr x(31), x(30), -8 };
    assert_eq!(w, 0xff8f0fe7);
}

#[test]
fn jalr_3() {
    // jalr t0, 2047(t1)
    let w: u32 = rvv! { jalr x(5), x(6), 2047 };
    assert_eq!(w, 0x7ff302e7);
}

#[test]
fn jalr_4() {
    // jalr zero, -2048(s0)
    let w: u32 = rvv! { jalr x(0), x(8), -2048 };
    assert_eq!(w, 0x80040067);
}

#[test]
fn jr_0() {
    // jr sp
    let w: u32 = rvv! { jr x(2) };
    assert_eq!(w, 0x00010067);
}

#[test]
fn jr_1() {
    // jr a1
    let w: u32 = rvv! { jr x(11) };
    assert_eq!(w, 0x00058067);
}

#[test]
fn jr_2() {
    // jr t5
    let w: u32 = rvv! { jr x(30) };
    assert_eq!(w, 0x000f0067);
}

#[test]
fn jr_3() {
    // jr t1
    let w: u32 = rvv! { jr x(6) };
    assert_eq!(w, 0x00030067);
}

#[test]
fn jr_4() {
    // jr s0
    let w: u32 = rvv! { jr x(8) };
    assert_eq!(w, 0x00040067);
}

#[test]
fn ret() {
    // ret
    let w: u32 = rvv! { ret };
    assert_eq!(w, 0x00008067);
}

#[test]
fn lui_0() {
    // lui ra, 0
    let w: u32 = rvv! { lui x(1), 0 };
    assert_eq!(w, 0x000000b7);
}

#[test]
fn auipc_0() {
    // auipc ra, 0
    let w: u32 = rvv! { auipc x(1), 0 };
    assert_eq!(w, 0x00000097);
}

#[test]
fn lui_1() {
    // lui a0, 1
    let w: u32 = rvv! { lui x(10), 1 };
    assert_eq!(w, 0x00001537);
}

#[test]
fn auipc_1() {
    // auipc a0, 1
    let w: u32 = rvv! { auipc x(10), 1 };
    assert_eq!(w, 0x00001517);
}

#[test]
fn lui_2() {
    // lui t6, 74565
    let w: u32 = rvv! { lui x(31), 74565 };
    assert_eq!(w, 0x12345fb7);
}

#[test]
fn auipc_2() {
    // auipc t6, 74565
    let w: u32 = rvv! { auipc x(31), 74565 };
    assert_eq!(w, 0x12345f97);
}

#[test]
fn lui_3() {
    // lui t0, 524287
    let w: u32 = rvv! { lui x(5), 524287 };
    assert_eq!(w, 0x7ffff2b7);
}

#[test]
fn auipc_3() {
    // auipc t0, 524287
    let w: u32 = rvv! { auipc x(5), 524287 };
    assert_eq!(w, 0x7ffff297);
}

#[test]
fn lui_4() {
    // lui zero, 1048575
    let w: u32 = rvv! { lui x(0), 1048575 };
    assert_eq!(w, 0xfffff037);
}

#[test]
fn auipc_4() {
    // auipc zero, 1048575
    let w: u32 = rvv! { auipc x(0), 1048575 };
    assert_eq!(w, 0xfffff017);
}

#[test]
fn fadd_d_0() {
    // fadd.d ft1, ft2, ft3
    let w: u32 = rvv! { fadd.d f(1), f(2), f(3) };
    assert_eq!(w, 0x023170d3);
}

#[test]
fn fadd_d_1() {
    // fadd.d fa0, fa1, fa2
    let w: u32 = rvv! { fadd.d f(10), f(11), f(12) };
    assert_eq!(w, 0x02c5f553);
}

#[test]
fn fadd_d_2() {
    // fadd.d ft11, ft10, ft9
    let w: u32 = rvv! { fadd.d f(31), f(30), f(29) };
    assert_eq!(w, 0x03df7fd3);
}

#[test]
fn fadd_d_3() {
    // fadd.d ft5, ft6, ft7
    let w: u32 = rvv! { fadd.d f(5), f(6), f(7) };
    assert_eq!(w, 0x027372d3);
}

#[test]
fn fadd_d_4() {
    // fadd.d ft0, fs0, fs1
    let w: u32 = rvv! { fadd.d f(0), f(8), f(9) };
    assert_eq!(w, 0x02947053);
}

#[test]
fn fsub_d_0() {
    // fsub.d ft1, ft2, ft3
    let w: u32 = rvv! { fsub.d f(1), f(2), f(3) };
    assert_eq!(w, 0x0a3170d3);
}

#[test]
fn fsub_d_1() {
    // fsub.d fa0, fa1, fa2
    let w: u32 = rvv! { fsub.d f(10), f(11), f(12) };
    assert_eq!(w, 0x0ac5f553);
}

#[test]
fn fsub_d_2() {
    // fsub.d ft11, ft10, ft9
    let w: u32 = rvv! { fsub.d f(31), f(30), f(29) };
    assert_eq!(w, 0x0bdf7fd3);
}

#[test]
fn fsub_d_3() {
    // fsub.d ft5, ft6, ft7
    let w: u32 = rvv! { fsub.d f(5), f(6), f(7) };
    assert_eq!(w, 0x0a7372d3);
}

#[test]
fn fsub_d_4() {
    // fsub.d ft0, fs0, fs1
    let w: u32 = rvv! { fsub.d f(0), f(8), f(9) };
    assert_eq!(w, 0x0a947053);
}

#[test]
fn fmul_d_0() {
    // fmul.d ft1, ft2, ft3
    let w: u32 = rvv! { fmul.d f(1), f(2), f(3) };
    assert_eq!(w, 0x123170d3);
}

#[test]
fn fmul_d_1() {
    // fmul.d fa0, fa1, fa2
    let w: u32 = rvv! { fmul.d f(10), f(11), f(12) };
    assert_eq!(w, 0x12c5f553);
}

#[test]
fn fmul_d_2() {
    // fmul.d ft11, ft10, ft9
    let w: u32 = rvv! { fmul.d f(31), f(30), f(29) };
    assert_eq!(w, 0x13df7fd3);
}

#[test]
fn fmul_d_3() {
    // fmul.d ft5, ft6, ft7
    let w: u32 = rvv! { fmul.d f(5), f(6), f(7) };
    assert_eq!(w, 0x127372d3);
}

#[test]
fn fmul_d_4() {
    // fmul.d ft0, fs0, fs1
    let w: u32 = rvv! { fmul.d f(0), f(8), f(9) };
    assert_eq!(w, 0x12947053);
}

#[test]
fn fdiv_d_0() {
    // fdiv.d ft1, ft2, ft3
    let w: u32 = rvv! { fdiv.d f(1), f(2), f(3) };
    assert_eq!(w, 0x1a3170d3);
}

#[test]
fn fdiv_d_1() {
    // fdiv.d fa0, fa1, fa2
    let w: u32 = rvv! { fdiv.d f(10), f(11), f(12) };
    assert_eq!(w, 0x1ac5f553);
}

#[test]
fn fdiv_d_2() {
    // fdiv.d ft11, ft10, ft9
    let w: u32 = rvv! { fdiv.d f(31), f(30), f(29) };
    assert_eq!(w, 0x1bdf7fd3);
}

#[test]
fn fdiv_d_3() {
    // fdiv.d ft5, ft6, ft7
    let w: u32 = rvv! { fdiv.d f(5), f(6), f(7) };
    assert_eq!(w, 0x1a7372d3);
}

#[test]
fn fdiv_d_4() {
    // fdiv.d ft0, fs0, fs1
    let w: u32 = rvv! { fdiv.d f(0), f(8), f(9) };
    assert_eq!(w, 0x1a947053);
}

#[test]
fn fadd_d_rm_rne() {
    // fadd.d ft1, ft2, ft3, rne
    let w: u32 = rvv! { fadd.d f(1), f(2), f(3), 0 };
    assert_eq!(w, 0x023100d3);
}

#[test]
fn fadd_d_rm_rtz() {
    // fadd.d ft1, ft2, ft3, rtz
    let w: u32 = rvv! { fadd.d f(1), f(2), f(3), 1 };
    assert_eq!(w, 0x023110d3);
}

#[test]
fn fadd_d_rm_rdn() {
    // fadd.d ft1, ft2, ft3, rdn
    let w: u32 = rvv! { fadd.d f(1), f(2), f(3), 2 };
    assert_eq!(w, 0x023120d3);
}

#[test]
fn fadd_d_rm_rup() {
    // fadd.d ft1, ft2, ft3, rup
    let w: u32 = rvv! { fadd.d f(1), f(2), f(3), 3 };
    assert_eq!(w, 0x023130d3);
}

#[test]
fn fadd_d_rm_rmm() {
    // fadd.d ft1, ft2, ft3, rmm
    let w: u32 = rvv! { fadd.d f(1), f(2), f(3), 4 };
    assert_eq!(w, 0x023140d3);
}

#[test]
fn fsqrt_d_0() {
    // fsqrt.d ft1, ft2
    let w: u32 = rvv! { fsqrt.d f(1), f(2) };
    assert_eq!(w, 0x5a0170d3);
}

#[test]
fn fsqrt_d_1() {
    // fsqrt.d fa0, fa1
    let w: u32 = rvv! { fsqrt.d f(10), f(11) };
    assert_eq!(w, 0x5a05f553);
}

#[test]
fn fsqrt_d_2() {
    // fsqrt.d ft11, ft10
    let w: u32 = rvv! { fsqrt.d f(31), f(30) };
    assert_eq!(w, 0x5a0f7fd3);
}

#[test]
fn fsqrt_d_3() {
    // fsqrt.d ft5, ft6
    let w: u32 = rvv! { fsqrt.d f(5), f(6) };
    assert_eq!(w, 0x5a0372d3);
}

#[test]
fn fsqrt_d_4() {
    // fsqrt.d ft0, fs0
    let w: u32 = rvv! { fsqrt.d f(0), f(8) };
    assert_eq!(w, 0x5a047053);
}

#[test]
fn fsgnj_d_0() {
    // fsgnj.d ft1, ft2, ft3
    let w: u32 = rvv! { fsgnj.d f(1), f(2), f(3) };
    assert_eq!(w, 0x223100d3);
}

#[test]
fn fsgnj_d_1() {
    // fsgnj.d fa0, fa1, fa2
    let w: u32 = rvv! { fsgnj.d f(10), f(11), f(12) };
    assert_eq!(w, 0x22c58553);
}

#[test]
fn fsgnj_d_2() {
    // fsgnj.d ft11, ft10, ft9
    let w: u32 = rvv! { fsgnj.d f(31), f(30), f(29) };
    assert_eq!(w, 0x23df0fd3);
}

#[test]
fn fsgnj_d_3() {
    // fsgnj.d ft5, ft6, ft7
    let w: u32 = rvv! { fsgnj.d f(5), f(6), f(7) };
    assert_eq!(w, 0x227302d3);
}

#[test]
fn fsgnj_d_4() {
    // fsgnj.d ft0, fs0, fs1
    let w: u32 = rvv! { fsgnj.d f(0), f(8), f(9) };
    assert_eq!(w, 0x22940053);
}

#[test]
fn fsgnjn_d_0() {
    // fsgnjn.d ft1, ft2, ft3
    let w: u32 = rvv! { fsgnjn.d f(1), f(2), f(3) };
    assert_eq!(w, 0x223110d3);
}

#[test]
fn fsgnjn_d_1() {
    // fsgnjn.d fa0, fa1, fa2
    let w: u32 = rvv! { fsgnjn.d f(10), f(11), f(12) };
    assert_eq!(w, 0x22c59553);
}

#[test]
fn fsgnjn_d_2() {
    // fsgnjn.d ft11, ft10, ft9
    let w: u32 = rvv! { fsgnjn.d f(31), f(30), f(29) };
    assert_eq!(w, 0x23df1fd3);
}

#[test]
fn fsgnjn_d_3() {
    // fsgnjn.d ft5, ft6, ft7
    let w: u32 = rvv! { fsgnjn.d f(5), f(6), f(7) };
    assert_eq!(w, 0x227312d3);
}

#[test]
fn fsgnjn_d_4() {
    // fsgnjn.d ft0, fs0, fs1
    let w: u32 = rvv! { fsgnjn.d f(0), f(8), f(9) };
    assert_eq!(w, 0x22941053);
}

#[test]
fn fsgnjx_d_0() {
    // fsgnjx.d ft1, ft2, ft3
    let w: u32 = rvv! { fsgnjx.d f(1), f(2), f(3) };
    assert_eq!(w, 0x223120d3);
}

#[test]
fn fsgnjx_d_1() {
    // fsgnjx.d fa0, fa1, fa2
    let w: u32 = rvv! { fsgnjx.d f(10), f(11), f(12) };
    assert_eq!(w, 0x22c5a553);
}

#[test]
fn fsgnjx_d_2() {
    // fsgnjx.d ft11, ft10, ft9
    let w: u32 = rvv! { fsgnjx.d f(31), f(30), f(29) };
    assert_eq!(w, 0x23df2fd3);
}

#[test]
fn fsgnjx_d_3() {
    // fsgnjx.d ft5, ft6, ft7
    let w: u32 = rvv! { fsgnjx.d f(5), f(6), f(7) };
    assert_eq!(w, 0x227322d3);
}

#[test]
fn fsgnjx_d_4() {
    // fsgnjx.d ft0, fs0, fs1
    let w: u32 = rvv! { fsgnjx.d f(0), f(8), f(9) };
    assert_eq!(w, 0x22942053);
}

#[test]
fn fmin_d_0() {
    // fmin.d ft1, ft2, ft3
    let w: u32 = rvv! { fmin.d f(1), f(2), f(3) };
    assert_eq!(w, 0x2a3100d3);
}

#[test]
fn fmin_d_1() {
    // fmin.d fa0, fa1, fa2
    let w: u32 = rvv! { fmin.d f(10), f(11), f(12) };
    assert_eq!(w, 0x2ac58553);
}

#[test]
fn fmin_d_2() {
    // fmin.d ft11, ft10, ft9
    let w: u32 = rvv! { fmin.d f(31), f(30), f(29) };
    assert_eq!(w, 0x2bdf0fd3);
}

#[test]
fn fmin_d_3() {
    // fmin.d ft5, ft6, ft7
    let w: u32 = rvv! { fmin.d f(5), f(6), f(7) };
    assert_eq!(w, 0x2a7302d3);
}

#[test]
fn fmin_d_4() {
    // fmin.d ft0, fs0, fs1
    let w: u32 = rvv! { fmin.d f(0), f(8), f(9) };
    assert_eq!(w, 0x2a940053);
}

#[test]
fn fmax_d_0() {
    // fmax.d ft1, ft2, ft3
    let w: u32 = rvv! { fmax.d f(1), f(2), f(3) };
    assert_eq!(w, 0x2a3110d3);
}

#[test]
fn fmax_d_1() {
    // fmax.d fa0, fa1, fa2
    let w: u32 = rvv! { fmax.d f(10), f(11), f(12) };
    assert_eq!(w, 0x2ac59553);
}

#[test]
fn fmax_d_2() {
    // fmax.d ft11, ft10, ft9
    let w: u32 = rvv! { fmax.d f(31), f(30), f(29) };
    assert_eq!(w, 0x2bdf1fd3);
}

#[test]
fn fmax_d_3() {
    // fmax.d ft5, ft6, ft7
    let w: u32 = rvv! { fmax.d f(5), f(6), f(7) };
    assert_eq!(w, 0x2a7312d3);
}

#[test]
fn fmax_d_4() {
    // fmax.d ft0, fs0, fs1
    let w: u32 = rvv! { fmax.d f(0), f(8), f(9) };
    assert_eq!(w, 0x2a941053);
}

#[test]
fn feq_d_0() {
    // feq.d ra, ft2, ft3
    let w: u32 = rvv! { feq.d x(1), f(2), f(3) };
    assert_eq!(w, 0xa23120d3);
}

#[test]
fn feq_d_1() {
    // feq.d a0, fa1, fa2
    let w: u32 = rvv! { feq.d x(10), f(11), f(12) };
    assert_eq!(w, 0xa2c5a553);
}

#[test]
fn feq_d_2() {
    // feq.d t6, ft10, ft9
    let w: u32 = rvv! { feq.d x(31), f(30), f(29) };
    assert_eq!(w, 0xa3df2fd3);
}

#[test]
fn feq_d_3() {
    // feq.d t0, ft6, ft7
    let w: u32 = rvv! { feq.d x(5), f(6), f(7) };
    assert_eq!(w, 0xa27322d3);
}

#[test]
fn feq_d_4() {
    // feq.d zero, fs0, fs1
    let w: u32 = rvv! { feq.d x(0), f(8), f(9) };
    assert_eq!(w, 0xa2942053);
}

#[test]
fn flt_d_0() {
    // flt.d ra, ft2, ft3
    let w: u32 = rvv! { flt.d x(1), f(2), f(3) };
    assert_eq!(w, 0xa23110d3);
}

#[test]
fn flt_d_1() {
    // flt.d a0, fa1, fa2
    let w: u32 = rvv! { flt.d x(10), f(11), f(12) };
    assert_eq!(w, 0xa2c59553);
}

#[test]
fn flt_d_2() {
    // flt.d t6, ft10, ft9
    let w: u32 = rvv! { flt.d x(31), f(30), f(29) };
    assert_eq!(w, 0xa3df1fd3);
}

#[test]
fn flt_d_3() {
    // flt.d t0, ft6, ft7
    let w: u32 = rvv! { flt.d x(5), f(6), f(7) };
    assert_eq!(w, 0xa27312d3);
}

#[test]
fn flt_d_4() {
    // flt.d zero, fs0, fs1
    let w: u32 = rvv! { flt.d x(0), f(8), f(9) };
    assert_eq!(w, 0xa2941053);
}

#[test]
fn fle_d_0() {
    // fle.d ra, ft2, ft3
    let w: u32 = rvv! { fle.d x(1), f(2), f(3) };
    assert_eq!(w, 0xa23100d3);
}

#[test]
fn fle_d_1() {
    // fle.d a0, fa1, fa2
    let w: u32 = rvv! { fle.d x(10), f(11), f(12) };
    assert_eq!(w, 0xa2c58553);
}

#[test]
fn fle_d_2() {
    // fle.d t6, ft10, ft9
    let w: u32 = rvv! { fle.d x(31), f(30), f(29) };
    assert_eq!(w, 0xa3df0fd3);
}

#[test]
fn fle_d_3() {
    // fle.d t0, ft6, ft7
    let w: u32 = rvv! { fle.d x(5), f(6), f(7) };
    assert_eq!(w, 0xa27302d3);
}

#[test]
fn fle_d_4() {
    // fle.d zero, fs0, fs1
    let w: u32 = rvv! { fle.d x(0), f(8), f(9) };
    assert_eq!(w, 0xa2940053);
}

#[test]
fn fgt_d_0() {
    // flt.d ra, ft3, ft2
    let w: u32 = rvv! { fgt.d x(1), f(2), f(3) };
    assert_eq!(w, 0xa22190d3);
}

#[test]
fn fgt_d_1() {
    // flt.d a0, fa2, fa1
    let w: u32 = rvv! { fgt.d x(10), f(11), f(12) };
    assert_eq!(w, 0xa2b61553);
}

#[test]
fn fgt_d_2() {
    // flt.d t6, ft9, ft10
    let w: u32 = rvv! { fgt.d x(31), f(30), f(29) };
    assert_eq!(w, 0xa3ee9fd3);
}

#[test]
fn fgt_d_3() {
    // flt.d t0, ft7, ft6
    let w: u32 = rvv! { fgt.d x(5), f(6), f(7) };
    assert_eq!(w, 0xa26392d3);
}

#[test]
fn fgt_d_4() {
    // flt.d zero, fs1, fs0
    let w: u32 = rvv! { fgt.d x(0), f(8), f(9) };
    assert_eq!(w, 0xa2849053);
}

#[test]
fn fge_d_0() {
    // fle.d ra, ft3, ft2
    let w: u32 = rvv! { fge.d x(1), f(2), f(3) };
    assert_eq!(w, 0xa22180d3);
}

#[test]
fn fge_d_1() {
    // fle.d a0, fa2, fa1
    let w: u32 = rvv! { fge.d x(10), f(11), f(12) };
    assert_eq!(w, 0xa2b60553);
}

#[test]
fn fge_d_2() {
    // fle.d t6, ft9, ft10
    let w: u32 = rvv! { fge.d x(31), f(30), f(29) };
    assert_eq!(w, 0xa3ee8fd3);
}

#[test]
fn fge_d_3() {
    // fle.d t0, ft7, ft6
    let w: u32 = rvv! { fge.d x(5), f(6), f(7) };
    assert_eq!(w, 0xa26382d3);
}

#[test]
fn fge_d_4() {
    // fle.d zero, fs1, fs0
    let w: u32 = rvv! { fge.d x(0), f(8), f(9) };
    assert_eq!(w, 0xa2848053);
}

#[test]
fn fmv_d_0() {
    // fmv.d ft1, ft2
    let w: u32 = rvv! { fmv.d f(1), f(2) };
    assert_eq!(w, 0x222100d3);
}

#[test]
fn fmv_d_1() {
    // fmv.d fa0, fa1
    let w: u32 = rvv! { fmv.d f(10), f(11) };
    assert_eq!(w, 0x22b58553);
}

#[test]
fn fmv_d_2() {
    // fmv.d ft11, ft10
    let w: u32 = rvv! { fmv.d f(31), f(30) };
    assert_eq!(w, 0x23ef0fd3);
}

#[test]
fn fmv_d_3() {
    // fmv.d ft5, ft6
    let w: u32 = rvv! { fmv.d f(5), f(6) };
    assert_eq!(w, 0x226302d3);
}

#[test]
fn fmv_d_4() {
    // fmv.d ft0, fs0
    let w: u32 = rvv! { fmv.d f(0), f(8) };
    assert_eq!(w, 0x22840053);
}

#[test]
fn fneg_d_0() {
    // fneg.d ft1, ft2
    let w: u32 = rvv! { fneg.d f(1), f(2) };
    assert_eq!(w, 0x222110d3);
}

#[test]
fn fneg_d_1() {
    // fneg.d fa0, fa1
    let w: u32 = rvv! { fneg.d f(10), f(11) };
    assert_eq!(w, 0x22b59553);
}

#[test]
fn fneg_d_2() {
    // fneg.d ft11, ft10
    let w: u32 = rvv! { fneg.d f(31), f(30) };
    assert_eq!(w, 0x23ef1fd3);
}

#[test]
fn fneg_d_3() {
    // fneg.d ft5, ft6
    let w: u32 = rvv! { fneg.d f(5), f(6) };
    assert_eq!(w, 0x226312d3);
}

#[test]
fn fneg_d_4() {
    // fneg.d ft0, fs0
    let w: u32 = rvv! { fneg.d f(0), f(8) };
    assert_eq!(w, 0x22841053);
}

#[test]
fn fabs_d_0() {
    // fabs.d ft1, ft2
    let w: u32 = rvv! { fabs.d f(1), f(2) };
    assert_eq!(w, 0x222120d3);
}

#[test]
fn fabs_d_1() {
    // fabs.d fa0, fa1
    let w: u32 = rvv! { fabs.d f(10), f(11) };
    assert_eq!(w, 0x22b5a553);
}

#[test]
fn fabs_d_2() {
    // fabs.d ft11, ft10
    let w: u32 = rvv! { fabs.d f(31), f(30) };
    assert_eq!(w, 0x23ef2fd3);
}

#[test]
fn fabs_d_3() {
    // fabs.d ft5, ft6
    let w: u32 = rvv! { fabs.d f(5), f(6) };
    assert_eq!(w, 0x226322d3);
}

#[test]
fn fabs_d_4() {
    // fabs.d ft0, fs0
    let w: u32 = rvv! { fabs.d f(0), f(8) };
    assert_eq!(w, 0x22842053);
}

#[test]
fn fcvt_w_d_0() {
    // fcvt.w.d ra, ft2
    let w: u32 = rvv! { fcvt.w.d x(1), f(2) };
    assert_eq!(w, 0xc20170d3);
}

#[test]
fn fcvt_d_w_0() {
    // fcvt.d.w ft1, sp
    let w: u32 = rvv! { fcvt.d.w f(1), x(2) };
    assert_eq!(w, 0xd20100d3);
}

#[test]
fn fcvt_d_l_0() {
    // fcvt.d.l ft1, sp
    let w: u32 = rvv! { fcvt.d.l f(1), x(2) };
    assert_eq!(w, 0xd22170d3);
}

#[test]
fn fmv_x_d_0() {
    // fmv.x.d ra, ft2
    let w: u32 = rvv! { fmv.x.d x(1), f(2) };
    assert_eq!(w, 0xe20100d3);
}

#[test]
fn fmv_d_x_0() {
    // fmv.d.x ft1, sp
    let w: u32 = rvv! { fmv.d.x f(1), x(2) };
    assert_eq!(w, 0xf20100d3);
}

#[test]
fn fcvt_w_d_1() {
    // fcvt.w.d a0, fa1
    let w: u32 = rvv! { fcvt.w.d x(10), f(11) };
    assert_eq!(w, 0xc205f553);
}

#[test]
fn fcvt_d_w_1() {
    // fcvt.d.w fa0, a1
    let w: u32 = rvv! { fcvt.d.w f(10), x(11) };
    assert_eq!(w, 0xd2058553);
}

#[test]
fn fcvt_d_l_1() {
    // fcvt.d.l fa0, a1
    let w: u32 = rvv! { fcvt.d.l f(10), x(11) };
    assert_eq!(w, 0xd225f553);
}

#[test]
fn fmv_x_d_1() {
    // fmv.x.d a0, fa1
    let w: u32 = rvv! { fmv.x.d x(10), f(11) };
    assert_eq!(w, 0xe2058553);
}

#[test]
fn fmv_d_x_1() {
    // fmv.d.x fa0, a1
    let w: u32 = rvv! { fmv.d.x f(10), x(11) };
    assert_eq!(w, 0xf2058553);
}

#[test]
fn fcvt_w_d_2() {
    // fcvt.w.d t6, ft10
    let w: u32 = rvv! { fcvt.w.d x(31), f(30) };
    assert_eq!(w, 0xc20f7fd3);
}

#[test]
fn fcvt_d_w_2() {
    // fcvt.d.w ft11, t5
    let w: u32 = rvv! { fcvt.d.w f(31), x(30) };
    assert_eq!(w, 0xd20f0fd3);
}

#[test]
fn fcvt_d_l_2() {
    // fcvt.d.l ft11, t5
    let w: u32 = rvv! { fcvt.d.l f(31), x(30) };
    assert_eq!(w, 0xd22f7fd3);
}

#[test]
fn fmv_x_d_2() {
    // fmv.x.d t6, ft10
    let w: u32 = rvv! { fmv.x.d x(31), f(30) };
    assert_eq!(w, 0xe20f0fd3);
}

#[test]
fn fmv_d_x_2() {
    // fmv.d.x ft11, t5
    let w: u32 = rvv! { fmv.d.x f(31), x(30) };
    assert_eq!(w, 0xf20f0fd3);
}

#[test]
fn fcvt_w_d_3() {
    // fcvt.w.d t0, ft6
    let w: u32 = rvv! { fcvt.w.d x(5), f(6) };
    assert_eq!(w, 0xc20372d3);
}

#[test]
fn fcvt_d_w_3() {
    // fcvt.d.w ft5, t1
    let w: u32 = rvv! { fcvt.d.w f(5), x(6) };
    assert_eq!(w, 0xd20302d3);
}

#[test]
fn fcvt_d_l_3() {
    // fcvt.d.l ft5, t1
    let w: u32 = rvv! { fcvt.d.l f(5), x(6) };
    assert_eq!(w, 0xd22372d3);
}

#[test]
fn fmv_x_d_3() {
    // fmv.x.d t0, ft6
    let w: u32 = rvv! { fmv.x.d x(5), f(6) };
    assert_eq!(w, 0xe20302d3);
}

#[test]
fn fmv_d_x_3() {
    // fmv.d.x ft5, t1
    let w: u32 = rvv! { fmv.d.x f(5), x(6) };
    assert_eq!(w, 0xf20302d3);
}

#[test]
fn fcvt_w_d_4() {
    // fcvt.w.d zero, fs0
    let w: u32 = rvv! { fcvt.w.d x(0), f(8) };
    assert_eq!(w, 0xc2047053);
}

#[test]
fn fcvt_d_w_4() {
    // fcvt.d.w ft0, s0
    let w: u32 = rvv! { fcvt.d.w f(0), x(8) };
    assert_eq!(w, 0xd2040053);
}

#[test]
fn fcvt_d_l_4() {
    // fcvt.d.l ft0, s0
    let w: u32 = rvv! { fcvt.d.l f(0), x(8) };
    assert_eq!(w, 0xd2247053);
}

#[test]
fn fmv_x_d_4() {
    // fmv.x.d zero, fs0
    let w: u32 = rvv! { fmv.x.d x(0), f(8) };
    assert_eq!(w, 0xe2040053);
}

#[test]
fn fmv_d_x_4() {
    // fmv.d.x ft0, s0
    let w: u32 = rvv! { fmv.d.x f(0), x(8) };
    assert_eq!(w, 0xf2040053);
}

#[test]
fn fcvt_l_d_rm_rne() {
    // fcvt.l.d a0, fa1, rne
    let w: u32 = rvv! { fcvt.l.d x(10), f(11), 0 };
    assert_eq!(w, 0xc2258553);
}

#[test]
fn fcvt_l_d_rm_rtz() {
    // fcvt.l.d a0, fa1, rtz
    let w: u32 = rvv! { fcvt.l.d x(10), f(11), 1 };
    assert_eq!(w, 0xc2259553);
}

#[test]
fn fcvt_l_d_rm_rdn() {
    // fcvt.l.d a0, fa1, rdn
    let w: u32 = rvv! { fcvt.l.d x(10), f(11), 2 };
    assert_eq!(w, 0xc225a553);
}

#[test]
fn fcvt_l_d_rm_rup() {
    // fcvt.l.d a0, fa1, rup
    let w: u32 = rvv! { fcvt.l.d x(10), f(11), 3 };
    assert_eq!(w, 0xc225b553);
}

#[test]
fn fcvt_l_d_rm_rmm() {
    // fcvt.l.d a0, fa1, rmm
    let w: u32 = rvv! { fcvt.l.d x(10), f(11), 4 };
    assert_eq!(w, 0xc225c553);
}

#[test]
fn fmadd_d_0() {
    // fmadd.d ft1, ft2, ft3, ft4
    let w: u32 = rvv! { fmadd.d f(1), f(2), f(3), f(4) };
    assert_eq!(w, 0x223170c3);
}

#[test]
fn fmadd_d_1() {
    // fmadd.d fa0, fa1, fa2, fa3
    let w: u32 = rvv! { fmadd.d f(10), f(11), f(12), f(13) };
    assert_eq!(w, 0x6ac5f543);
}

#[test]
fn fmadd_d_2() {
    // fmadd.d ft11, ft10, ft9, ft8
    let w: u32 = rvv! { fmadd.d f(31), f(30), f(29), f(28) };
    assert_eq!(w, 0xe3df7fc3);
}

#[test]
fn fmadd_d_3() {
    // fmadd.d ft5, ft6, ft7, fs0
    let w: u32 = rvv! { fmadd.d f(5), f(6), f(7), f(8) };
    assert_eq!(w, 0x427372c3);
}

#[test]
fn fmadd_d_4() {
    // fmadd.d ft0, fs0, fs1, fa0
    let w: u32 = rvv! { fmadd.d f(0), f(8), f(9), f(10) };
    assert_eq!(w, 0x52947043);
}

#[test]
fn fmsub_d_0() {
    // fmsub.d ft1, ft2, ft3, ft4
    let w: u32 = rvv! { fmsub.d f(1), f(2), f(3), f(4) };
    assert_eq!(w, 0x223170c7);
}

#[test]
fn fmsub_d_1() {
    // fmsub.d fa0, fa1, fa2, fa3
    let w: u32 = rvv! { fmsub.d f(10), f(11), f(12), f(13) };
    assert_eq!(w, 0x6ac5f547);
}

#[test]
fn fmsub_d_2() {
    // fmsub.d ft11, ft10, ft9, ft8
    let w: u32 = rvv! { fmsub.d f(31), f(30), f(29), f(28) };
    assert_eq!(w, 0xe3df7fc7);
}

#[test]
fn fmsub_d_3() {
    // fmsub.d ft5, ft6, ft7, fs0
    let w: u32 = rvv! { fmsub.d f(5), f(6), f(7), f(8) };
    assert_eq!(w, 0x427372c7);
}

#[test]
fn fmsub_d_4() {
    // fmsub.d ft0, fs0, fs1, fa0
    let w: u32 = rvv! { fmsub.d f(0), f(8), f(9), f(10) };
    assert_eq!(w, 0x52947047);
}

#[test]
fn fnmadd_d_0() {
    // fnmadd.d ft1, ft2, ft3, ft4
    let w: u32 = rvv! { fnmadd.d f(1), f(2), f(3), f(4) };
    assert_eq!(w, 0x223170cf);
}

#[test]
fn fnmadd_d_1() {
    // fnmadd.d fa0, fa1, fa2, fa3
    let w: u32 = rvv! { fnmadd.d f(10), f(11), f(12), f(13) };
    assert_eq!(w, 0x6ac5f54f);
}

#[test]
fn fnmadd_d_2() {
    // fnmadd.d ft11, ft10, ft9, ft8
    let w: u32 = rvv! { fnmadd.d f(31), f(30), f(29), f(28) };
    assert_eq!(w, 0xe3df7fcf);
}

#[test]
fn fnmadd_d_3() {
    // fnmadd.d ft5, ft6, ft7, fs0
    let w: u32 = rvv! { fnmadd.d f(5), f(6), f(7), f(8) };
    assert_eq!(w, 0x427372cf);
}

#[test]
fn fnmadd_d_4() {
    // fnmadd.d ft0, fs0, fs1, fa0
    let w: u32 = rvv! { fnmadd.d f(0), f(8), f(9), f(10) };
    assert_eq!(w, 0x5294704f);
}

#[test]
fn fnmsub_d_0() {
    // fnmsub.d ft1, ft2, ft3, ft4
    let w: u32 = rvv! { fnmsub.d f(1), f(2), f(3), f(4) };
    assert_eq!(w, 0x223170cb);
}

#[test]
fn fnmsub_d_1() {
    // fnmsub.d fa0, fa1, fa2, fa3
    let w: u32 = rvv! { fnmsub.d f(10), f(11), f(12), f(13) };
    assert_eq!(w, 0x6ac5f54b);
}

#[test]
fn fnmsub_d_2() {
    // fnmsub.d ft11, ft10, ft9, ft8
    let w: u32 = rvv! { fnmsub.d f(31), f(30), f(29), f(28) };
    assert_eq!(w, 0xe3df7fcb);
}

#[test]
fn fnmsub_d_3() {
    // fnmsub.d ft5, ft6, ft7, fs0
    let w: u32 = rvv! { fnmsub.d f(5), f(6), f(7), f(8) };
    assert_eq!(w, 0x427372cb);
}

#[test]
fn fnmsub_d_4() {
    // fnmsub.d ft0, fs0, fs1, fa0
    let w: u32 = rvv! { fnmsub.d f(0), f(8), f(9), f(10) };
    assert_eq!(w, 0x5294704b);
}
