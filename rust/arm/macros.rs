macro_rules! rd {
    ($x:expr) => {{
        let x = $x;
        assert!(x < 32);
        x as u32
    }};
}

macro_rules! rn {
    ($x:expr) => {{
        let x = $x;
        assert!(x < 32);
        (x as u32) << 5
    }};
}

macro_rules! rd2 {
    ($x:expr) => {{
        let x = $x;
        assert!(x < 32);
        (x as u32) << 10
    }};
}

macro_rules! rm {
    ($x:expr) => {{
        let x = $x;
        assert!(x < 32);
        (x as u32) << 16
    }};
}

macro_rules! imm {
    ($x:expr) => {{
        let x = $x;
        assert!(x < 4096);
        (x as u32) << 10
    }};
}

macro_rules! imm16 {
    ($x:expr) => {{
        let x = $x;
        assert!(x < 65536);
        (x as u32) << 5
    }};
}

macro_rules! ofs {
    ($x:expr) => {{
        let x = $x;
        assert!((x & 7 == 0) && (x < 32768));
        (x as u32) << 7
    }};
}

macro_rules! of7 {
    ($x:expr) => {{
        let x = $x;
        assert!((x & 7 == 0) && (x <= 504));
        (x as u32) << 12
    }};
}

#[macro_export]
macro_rules! arm {
    // lr/sp substitution rules
    ($op:ident lr, [sp, #$imm:expr]) => {
        arm! { $op x(30), [x(31), #$imm] }
    };
    ($op:ident $($a:ident($x:expr),)+ [sp, #$imm:expr]) => {
        arm! { $op $($a($x),)* [x(31), #$imm] }
    };
    ($op:ident lr, [$b:ident($y:expr), #$imm:expr]) => {
        arm! { $op x(30), [$b($y), #$imm] }
    };
    ($op:ident sp, sp, #$imm:expr, lsl #12) => {
        arm! { $op x(31), x(31), #$imm, lsl #12 }
    };
    ($op:ident sp, sp, #$imm:expr) => {
        arm! { $op x(31), x(31), #$imm }
    };
    (mov x($rd:expr), sp) => {
        arm! { add x($rd), x(31), #0 }
    };

    // main rules
    (fmov d($rd:expr), d($rn:expr)) => {
        0x1e604000 | rd!($rd) | rn!($rn)
    };
    (mov x($rd:expr), x($rm:expr)) => {
        0xaa0003e0 | rd!($rd) | rm!($rm)
    };
    (movz x($rd:expr), #$imm16:expr) => {
        0xd2800000 | rd!($rd) | imm16!($imm16)
    };

    // single register load/store instructions
    (ldr d($rd:expr), [x($rn:expr), #$ofs:expr]) => {
        0xfd400000 | rd!($rd) | rn!($rn) | ofs!($ofs)
    };
    (ldr d($rd:expr), [x($rn:expr), x($rm:expr), lsl #3]) => {
        0xfc607800 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (ldr x($rd:expr), [x($rn:expr), #$ofs:expr]) => {
        0xf9400000 | rd!($rd) | rn!($rn) | ofs!($ofs)
    };

    (ldr d($rd:expr), label) => {
        0x5c000000 | rd!($rd)
    };

    (ldr x($rd:expr), label) => {
        0x58000000 | rd!($rd)
    };

    (str d($rd:expr), [x($rn:expr), #$ofs:expr]) => {
        0xfd000000 | rd!($rd) | rn!($rn) | ofs!($ofs)
    };
    (str d($rd:expr), [x($rn:expr), x($rm:expr), lsl #3]) => {
        0xfc207800 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (str x($rd:expr), [x($rn:expr), #$ofs:expr]) => {
        0xf9000000 | rd!($rd) | rn!($rn) | ofs!($ofs)
    };

    // paired-registers load/store instructions
    (ldp d($rd:expr), d($rd2:expr), [x($rn:expr), #$of7:expr]) => {
        0x6d400000 | rd!($rd) | rd2!($rd2) | rn!($rn) | of7!($of7)
    };
    (ldp x($rd:expr), x($rd2:expr), [x($rn:expr), #$of7:expr]) => {
        0xa9400000 | rd!($rd) | rd2!($rd2) | rn!($rn) | of7!($of7)
    };
    (stp d($rd:expr), d($rd2:expr), [x($rn:expr), #$of7:expr]) => {
        0x6d000000 | rd!($rd) | rd2!($rd2) | rn!($rn) | of7!($of7)
    };
    (stp x($rd:expr), x($rd2:expr), [x($rn:expr), #$of7:expr]) => {
        0xa9000000 | rd!($rd) | rd2!($rd2) | rn!($rn) | of7!($of7)
    };

    // x-registers immediate ops
    (add x($rd:expr), x($rn:expr), #$imm:expr, lsl #12) => {
        0x91400000 | rd!($rd) | rn!($rn) | imm!($imm)
    };
    (add x($rd:expr), x($rn:expr), #$imm:expr) => {
        0x91000000 | rd!($rd) | rn!($rn) | imm!($imm)
    };
    (sub x($rd:expr), x($rn:expr), #$imm:expr, lsl #12) => {
        0xd1400000 | rd!($rd) | rn!($rn) | imm!($imm)
    };
    (sub x($rd:expr), x($rn:expr), #$imm:expr) => {
        0xd1000000 | rd!($rd) | rn!($rn) | imm!($imm)
    };
    // floating point ops
    (fadd d($rd:expr), d($rn:expr), d($rm:expr)) => {
        0x1e602800 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (fsub d($rd:expr), d($rn:expr), d($rm:expr)) => {
        0x1e603800 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (fmul d($rd:expr), d($rn:expr), d($rm:expr)) => {
        0x1e600800 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (fdiv d($rd:expr), d($rn:expr), d($rm:expr)) => {
        0x1e601800 | rd!($rd) | rn!($rn) | rm!($rm)
    };

    (fsqrt d($rd:expr), d($rn:expr)) => {
        0x1e61c000 | rd!($rd) | rn!($rn)
    };
    (fneg d($rd:expr), d($rn:expr)) => {
        0x1e614000 | rd!($rd) | rn!($rn)
    };
    (fabs d($rd:expr), d($rn:expr)) => {
        0x1e60c000 | rd!($rd) | rn!($rn)
    };

    // round double to integral (double-coded integer)
    (frinti d($rd:expr), d($rn:expr)) => {
        0x1e67c000 | rd!($rd) | rn!($rn)
    };

    // floor (round toward minus inf) double to integral (double-coded integer)
    (frintm d($rd:expr), d($rn:expr)) => {
        0x1e654000 | rd!($rd) | rn!($rn)
    };

    // ceiling (round toward positive inf) double to integral (double-coded integer)
    (frintp d($rd:expr), d($rn:expr)) => {
        0x1e64c000 | rd!($rd) | rn!($rn)
    };

    // trunc (round toward zero) double to integral (double-coded integer)
    (frintz d($rd:expr), d($rn:expr)) => {
        0x1e65c000 | rd!($rd) | rn!($rn)
    };


    // logical ops
    (and v($rd:expr).8b, v($rn:expr).8b, v($rm:expr).8b) => {
        0x0e201c00 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (orr v($rd:expr).8b, v($rn:expr).8b, v($rm:expr).8b) => {
        0x0ea01c00 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (eor v($rd:expr).8b, v($rn:expr).8b, v($rm:expr).8b) => {
        0x2e201c00 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (bit v($rd:expr).8b, v($rn:expr).8b, v($rm:expr).8b) => {
        0x2ea01c00 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (bif v($rd:expr).8b, v($rn:expr).8b, v($rm:expr).8b) => {
        0x2ee01c00 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (bsl v($rd:expr).8b, v($rn:expr).8b, v($rm:expr).8b) => {
        0x2e601c00 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (not v($rd:expr).8b, v($rn:expr).8b) => {
        0x2e205800 | rd!($rd) | rn!($rn)
    };

    // comparison
    (fcmeq d($rd:expr), d($rn:expr), d($rm:expr)) => {
        0x5e60e400 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    // note that rm and rn are exchanged for fcmlt and fcmle
    (fcmlt d($rd:expr), d($rm:expr), d($rn:expr)) => {
        0x7ee0e400 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (fcmle d($rd:expr), d($rm:expr), d($rn:expr)) => {
        0x7e60e400 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (fcmgt d($rd:expr), d($rn:expr), d($rm:expr)) => {
        0x7ee0e400 | rd!($rd) | rn!($rn) | rm!($rm)
    };
    (fcmge d($rd:expr), d($rn:expr), d($rm:expr)) => {
        0x7e60e400 | rd!($rd) | rn!($rn) | rm!($rm)
    };

    // compare d(..) with 0.0 and set the flags (NZCV)
    (fcmp d($rn:expr), #0.0) => {
        0x1e602008 | rn!($rn)
    };

    // misc
    (blr x($rn:expr)) => { 0xd63f0000 | rn!($rn) };
    (ret) => { 0xd65f03c0 };
    (fmov d($rd:expr), #0.0) => { 0x9e6703e0 | rd!($rd) };
    (fmov d($rd:expr), #1.0) => { 0x1e6e1000 | rd!($rd) };
    (fmov d($rd:expr), #-1.0) => { 0x1e7e1000 | rd!($rd) };
}

#[test]
fn test_arm() {
    let k = 11;
    let n = 1000;

    assert_eq!(
        arm! {sub sp, sp, #32},
        u32::from_le_bytes([0xFF, 0x83, 0x00, 0xD1])
    );

    assert_eq!(
        arm! {str x(29), [sp, #8]},
        u32::from_le_bytes([0xFD, 0x07, 0x00, 0xF9])
    );
    assert_eq!(
        arm! {str x(30), [sp, #16]},
        u32::from_le_bytes([0xFE, 0x0B, 0x00, 0xF9])
    );
    assert_eq!(
        arm! {str d(8), [sp, #24]},
        u32::from_le_bytes([0xE8, 0x0F, 0x00, 0xFD])
    );
    assert_eq!(
        arm! {mov x(29), x(0)},
        u32::from_le_bytes([0xFD, 0x03, 0x00, 0xAA])
    );

    assert_eq!(
        arm! {stp x(29), x(30), [sp, #16]},
        u32::from_le_bytes([0xFD, 0x7B, 0x01, 0xA9])
    );
    assert_eq!(
        arm! {stp d(8), d(9), [sp, #160]},
        u32::from_le_bytes([0xE8, 0x27, 0x0A, 0x6D])
    );
    assert_eq!(
        arm! {ldp x(19), x(20), [sp, #504]},
        u32::from_le_bytes([0xF3, 0xD3, 0x5F, 0xA9])
    );
    assert_eq!(
        arm! {ldp d(k+1), d(13), [sp, #160]},
        u32::from_le_bytes([0xEC, 0x37, 0x4A, 0x6D])
    );

    assert_eq!(
        arm! {ldr d(0), [x(29), #104]},
        u32::from_le_bytes([0xA0, 0x37, 0x40, 0xFD])
    );
    assert_eq!(
        arm! {fmov d(1), d(0)},
        u32::from_le_bytes([0x01, 0x40, 0x60, 0x1E])
    );
    assert_eq!(
        arm! {fadd d(0), d(0), d(1)},
        u32::from_le_bytes([0x00, 0x28, 0x61, 0x1E])
    );
    assert_eq!(
        arm! {fmul d(0), d(0), d(1)},
        u32::from_le_bytes([0x00, 0x08, 0x61, 0x1E])
    );
    assert_eq!(
        arm! {fsub d(0), d(0), d(1)},
        u32::from_le_bytes([0x00, 0x38, 0x61, 0x1E])
    );

    assert_eq!(
        arm! {fcmeq d(10), d(21), d(9)},
        u32::from_le_bytes([0xAA, 0xE6, 0x69, 0x5E])
    );
    assert_eq!(
        arm! {fcmlt d(k), d(1), d(19)},
        u32::from_le_bytes([0x6B, 0xE6, 0xE1, 0x7E])
    );
    assert_eq!(
        arm! {fcmle d(0), d(k), d(31)},
        u32::from_le_bytes([0xE0, 0xE7, 0x6B, 0x7E])
    );
    assert_eq!(
        arm! {fcmgt d(0), d(k+1), d(19)},
        u32::from_le_bytes([0x80, 0xE5, 0xF3, 0x7E])
    );
    assert_eq!(
        arm! {fcmge d(17), d(30), d(3)},
        u32::from_le_bytes([0xD1, 0xE7, 0x63, 0x7E])
    );

    assert_eq!(
        arm! {fdiv d(0), d(0), d(1)},
        u32::from_le_bytes([0x00, 0x18, 0x61, 0x1E])
    );
    assert_eq!(
        arm! {str d(0), [x(30), #200]},
        u32::from_le_bytes([0xC0, 0x67, 0x00, 0xFD])
    );
    assert_eq!(
        arm! {ldr x(29), [sp, #8]},
        u32::from_le_bytes([0xFD, 0x07, 0x40, 0xF9])
    );
    assert_eq!(
        arm! {ldr x(30), [sp, #16]},
        u32::from_le_bytes([0xFE, 0x0B, 0x40, 0xF9])
    );
    assert_eq!(
        arm! {add sp, sp, #32},
        u32::from_le_bytes([0xFF, 0x83, 0x00, 0x91])
    );

    assert_eq!(
        arm! {and v(2).8b, v(5).8b, v(22).8b},
        u32::from_le_bytes([0xA2, 0x1C, 0x36, 0x0E])
    );
    assert_eq!(
        arm! {orr v(1).8b, v(0).8b, v(k+1).8b},
        u32::from_le_bytes([0x01, 0x1C, 0xAC, 0x0E])
    );
    assert_eq!(
        arm! {eor v(7).8b, v(15).8b, v(31).8b},
        u32::from_le_bytes([0xE7, 0x1D, 0x3F, 0x2E])
    );
    assert_eq!(
        arm! {not v(14).8b, v(24).8b},
        u32::from_le_bytes([0x0E, 0x5B, 0x20, 0x2E])
    );

    assert_eq!(
        arm! {ldr lr, [sp, #n]},
        u32::from_le_bytes([0xFE, 0xF7, 0x41, 0xF9])
    );
    assert_eq!(
        arm! {str lr, [sp, #2*n]},
        u32::from_le_bytes([0xFE, 0xEB, 0x03, 0xF9])
    );
    assert_eq!(
        arm! {blr x(6)},
        u32::from_le_bytes([0xC0, 0x00, 0x3F, 0xD6])
    );
    assert_eq!(arm! {ret}, u32::from_le_bytes([0xC0, 0x03, 0x5F, 0xD6]));

    assert_eq!(
        arm! {fmov d(5), #0.0},
        u32::from_le_bytes([0xE5, 0x03, 0x67, 0x9E])
    );
    assert_eq!(
        arm! {fmov d(15), #1.0},
        u32::from_le_bytes([0x0F, 0x10, 0x6E, 0x1E])
    );
    assert_eq!(
        arm! {fmov d(k), #-1.0},
        u32::from_le_bytes([0x0B, 0x10, 0x7E, 0x1E])
    );
}
