macro_rules! rtype {
    ($rd:expr, $rs1:expr, $rs2:expr, $code:expr) => {{
        let rd = $rd as u32;
        let rs1 = $rs1 as u32;
        let rs2 = $rs2 as u32;

        assert!(rd < 32);
        assert!(rs1 < 32);
        assert!(rs2 < 32);

        $code | (rd << 7) | (rs1 << 15) | (rs2 << 20)
    }};
}

macro_rules! r4type {
    ($rd:expr, $rs1:expr, $rs2:expr, $rs3:expr, $code:expr) => {{
        let rd = $rd as u32;
        let rs1 = $rs1 as u32;
        let rs2 = $rs2 as u32;
        let rs3 = $rs3 as u32;

        assert!(rd < 32);
        assert!(rs1 < 32);
        assert!(rs2 < 32);
        assert!(rs3 < 32);

        $code | (rd << 7) | (rs1 << 15) | (rs2 << 20) | (rs3 << 27)
    }};
}

macro_rules! itype {
    ($rd:expr, $rs1:expr, $imm:expr, $code:expr) => {{
        let rd = $rd as u32;
        let rs1 = $rs1 as u32;
        let imm = $imm as i32;

        assert!(rd < 32);
        assert!(rs1 < 32);

        $code | (rd << 7) | (rs1 << 15) | ((imm & 0x0fff) << 20) as u32
    }};
}

macro_rules! stype {
    ($rs1:expr, $rs2:expr, $imm:expr, $code:expr) => {{
        let rs1 = $rs1 as u32;
        let rs2 = $rs2 as u32;
        let imm = $imm as i32;

        assert!(rs1 < 32);
        assert!(rs2 < 32);

        $code
            | (rs1 << 15)
            | (rs2 << 20)
            | ((imm & 0x01f) << 7) as u32
            | ((imm & 0x0fe0) << 20) as u32
    }};
}

macro_rules! btype {
    ($rs1:expr, $rs2:expr, $imm:expr, $code:expr) => {{
        let rs1 = $rs1 as u32;
        let rs2 = $rs2 as u32;
        let imm = $imm as i32;

        assert!(rs1 < 32);
        assert!(rs2 < 32);

        $code
            | (rs1 << 15)
            | (rs2 << 20)
            | ((imm & 0x001e) << 7) as u32
            | ((imm & 0x07e0) << 20) as u32
            | ((imm & 0x0800) >> 4) as u32
            | ((imm & 0x1000) << 19) as u32
    }};
}

macro_rules! utype {
    ($rd:expr, $imm:expr, $code:expr) => {{
        let rd = $rd as u32;
        assert!(rd < 32);
        $code | (rd << 7) | (($imm as u32) << 12)
    }};
}

macro_rules! jtype {
    ($rd:expr, $imm:expr, $code:expr) => {{
        let rd = $rd as u32;
        let imm = $imm as i32;

        assert!(rd < 32);

        $code
            | (rd << 7)
            | ((imm & 0x000007fe) << 20) as u32
            | ((imm & 0x00000800) << 9) as u32
            | ((imm & 0x000ff000) << 0) as u32
            | ((imm & 0x00100000) << 11) as u32
    }};
}

#[macro_export]
macro_rules! rvv {
    (add x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x00000033)
    };

    (mv x($rd:expr), x($rs1:expr)) => {
        rtype!($rd, $rs1, 0, 0x00000033)
    };

    (sub x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x40000033)
    };

    (neg x($rd:expr), x($rs2:expr)) => {
        rtype!($rd, 0, $rs2, 0x40000033)
    };

    (and x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x00007033)
    };

    (or x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x00006033)
    };

    (xor x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x00004033)
    };

    (sll x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x00001033)
    };

    (srl x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x00005033)
    };

    (sra x($rd:expr), x($rs1:expr), x($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x40005033)
    };

    (addi x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00000013)
    };

    (nop) => {
        itype!(0, 0, 0, 0x00000013)
    };

    (andi x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00007013)
    };

    (ori x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00006013)
    };

    (xori x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00004013)
    };

    (not x($rd:expr), x($rs1:expr)) => {
        itype!($rd, $rs1, -1, 0x00004013)
    };

    (slli x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00001013)
    };

    (srli x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00005013)
    };

    (srai x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x40005013)
    };

    (ld x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00003003)
    };

    (sd x($rs1:expr), x($rs2:expr), $imm:expr) => {
        stype!($rs2, $rs1, $imm, 0x00003023)
    };

    (beq x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs1, $rs2, $imm, 0x00000063)
    };

    (bne x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs1, $rs2, $imm, 0x00001063)
    };

    (blt x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs1, $rs2, $imm, 0x00004063)
    };

    (ble x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs2, $rs1, $imm, 0x00005063) // note: rs1 rs2 exchanged
    };

    (bgt x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs2, $rs1, $imm, 0x00004063) // note: rs1 rs2 exchanged
    };

    (bge x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs1, $rs2, $imm, 0x00005063)
    };

    (bltu x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs1, $rs2, $imm, 0x00006063)
    };

    (bgeu x($rs1:expr), x($rs2:expr), $imm:expr) => {
        btype!($rs1, $rs2, $imm, 0x00007063)
    };

    (jal x($rd:expr), $imm:expr) => {
        jtype!($rd, $imm, 0x0000006f)
    };

    (j $imm:expr) => {
        jtype!(0, $imm, 0x0000006f)
    };

    (jalr x($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00000067)
    };

    (jr x($rs1:expr)) => {
        itype!(0, $rs1, 0, 0x00000067)
    };

    (ret) => {
        itype!(0, 1, 0, 0x00000067)
    };

    (lui x($rd:expr), $imm:expr) => {{
        utype!($rd, $imm, 0x00000037)
    }};

    (auipc x($rd:expr), $imm:expr) => {{
        utype!($rd, $imm, 0x00000017)
    }};

    // float point ops
    (fadd.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x02007053)
    };

    // fadd with rounding-mode, used for round/floor/... functions
    (fadd.d f($rd:expr), f($rs1:expr), f($rs2:expr), $rm:expr) => {
        rtype!($rd, $rs1, $rs2, 0x02000053) | ($rm << 12)
    };

    (fsub.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x0a007053)
    };

    (fmul.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x12007053)
    };

    (fdiv.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x1a007053)
    };

    (fsqrt.d f($rd:expr), f($rs1:expr)) => {
        rtype!($rd, $rs1, 0, 0x5a007053)
    };

    (fsgnj.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x22000053)
    };

    (fsgnjn.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x22001053)
    };

    (fsgnjx.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x22002053)
    };

    (fmin.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x2a000053)
    };

    (fmax.d f($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0x2a001053)
    };

    (feq.d x($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0xa2002053)
    };

    (flt.d x($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0xa2001053)
    };

    (fle.d x($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs1, $rs2, 0xa2000053)
    };

    (fgt.d x($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs2, $rs1, 0xa2001053) // note: rs1 rs2 exchanged
    };

    (fge.d x($rd:expr), f($rs1:expr), f($rs2:expr)) => {
        rtype!($rd, $rs2, $rs1, 0xa2000053) // note: rs1 rs2 exchanged
    };

    (fld f($rd:expr), x($rs1:expr), $imm:expr) => {
        itype!($rd, $rs1, $imm, 0x00003007)
    };

    (fsd f($rd:expr), x($rs1:expr), $imm:expr) => {
        stype!($rs1, $rd, $imm, 0x00003027)
    };

    (fabs.d f($rd:expr), f($rs1:expr)) => {{
        let rs1 = $rs1;
        rtype!($rd, rs1, rs1, 0x22002053)
    }};

    (fmv.d f($rd:expr), f($rs1:expr)) => {{
        let rs1 = $rs1;
        rtype!($rd, rs1, rs1, 0x22000053)
    }};

    (fneg.d f($rd:expr), f($rs1:expr)) => {{
        let rs1 = $rs1;
        rtype!($rd, rs1, rs1, 0x22001053)
    }};

    (fcvt.w.d x($rd:expr), f($rs1:expr)) => {{
        itype!($rd, $rs1, 0, 0xc2007053)
    }};

    (fcvt.d.w f($rd:expr), x($rs1:expr)) => {{
        itype!($rd, $rs1, 0, 0xd2000053)
    }};

    (fcvt.l.d x($rd:expr), f($rs1:expr), $rm:expr) => {{
        itype!($rd, $rs1, 0, 0xc2200053 | ($rm << 12))
    }};

    (fcvt.d.l f($rd:expr), x($rs1:expr)) => {{
        itype!($rd, $rs1, 0, 0xd2207053)
    }};

    (fmv.x.d x($rd:expr), f($rs1:expr)) => {{
        itype!($rd, $rs1, 0, 0xe2000053)
    }};

    (fmv.d.x f($rd:expr), x($rs1:expr)) => {{
        itype!($rd, $rs1, 0, 0xf2000053)
    }};

    // rs1 * rs2 + rs3
    (fmadd.d f($rd:expr), f($rs1:expr), f($rs2:expr), f($rs3:expr)) => {
        r4type!($rd, $rs1, $rs2, $rs3, 0x02007043)
    };

    // rs1 * rs2 - rs3
    (fmsub.d f($rd:expr), f($rs1:expr), f($rs2:expr), f($rs3:expr)) => {
        r4type!($rd, $rs1, $rs2, $rs3, 0x02007047)
    };

    // -rs1 * rs2 - rs3
    (fnmadd.d f($rd:expr), f($rs1:expr), f($rs2:expr), f($rs3:expr)) => {
        r4type!($rd, $rs1, $rs2, $rs3, 0x0200704f)
    };

    // -rs1 * rs2 + rs3
    (fnmsub.d f($rd:expr), f($rs1:expr), f($rs2:expr), f($rs3:expr)) => {
        r4type!($rd, $rs1, $rs2, $rs3, 0x0200704b)
    };
}
