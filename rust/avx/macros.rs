macro_rules! reg {
    (rax) => {
        0
    };
    (rcx) => {
        1
    };
    (rdx) => {
        2
    };
    (rbx) => {
        3
    };
    (rsp) => {
        4
    };
    (rbp) => {
        5
    };
    (rsi) => {
        6
    };
    (rdi) => {
        7
    };
    (r8) => {
        8
    };
    (r9) => {
        9
    };
    (r10) => {
        10
    };
    (r11) => {
        11
    };
    (r12) => {
        12
    };
    (r13) => {
        13
    };
    (r14) => {
        14
    };
    (r15) => {
        15
    };
}

macro_rules! modrm_reg {
    ($reg:expr, $rm:expr) => {{
        0xC0 + (($reg & 7) << 3) + ($rm & 7)
    }};
}

macro_rules! rex {
    ($reg:expr, $rm:expr) => {{
        0x48 + (($rm & 8) >> 3) + (($reg & 8) >> 1)
    }};
}

macro_rules! modrm_mem {
    ($reg:expr, $rm:expr, $offset:expr) => {{
        let reg = $reg;
        let rm = $rm;
        let offset = $offset;

        let mut v = if offset < 128 {
            vec![0x40 + ((reg & 7) << 3) + (rm & 7)]
        } else {
            vec![0x80 + ((reg & 7) << 3) + (rm & 7)]
        };

        if rm == 4 {
            // rsp
            v.push(0x24); // SIB byte
        }

        if offset < 128 {
            v.push(offset as u8)
        } else {
            v.push(offset as u8);
            v.push((offset >> 8) as u8);
            v.push((offset >> 16) as u8);
            v.push((offset >> 24) as u8);
        };

        v
    }};
}

/// This is the two-byte VEX prefix (VEX2) for packed-double (pd)
/// and 256-bit ymm registers
macro_rules! vex2pd {
    ($reg:expr, $vreg:expr) => {{
        let r = ($reg & 8) << 4;
        let vvvv = $vreg << 3;
        vec![0xc5, (r | vvvv | 5) ^ 0xf8]
    }};
}

/// This is the two-byte VEX prefix (VEX2) for packed-double (pd)
/// and 256-bit ymm registers
macro_rules! vex2sd {
    ($reg:expr, $vreg:expr) => {{
        let r = ($reg & 8) << 4;
        let vvvv = $vreg << 3;
        vec![0xc5, (r | vvvv | 3) ^ 0xf8]
    }};
}

/// This is the three-byte VEX prefix (VEX3) for packed-double (pd)
/// and 256-bit ymm registers
macro_rules! vex3pd {
    ($reg:expr, $vreg:expr, $rm:expr) => {{
        let r = ($reg & 8) << 4;
        let b = ($rm & 8) << 2;
        let vvvv = $vreg << 3;
        vec![0xc4, (r | b | 1) ^ 0xe0, (vvvv | 5) ^ 0x78]
    }};
    ($reg:expr, $vreg:expr, $rm:expr, $encoding:expr) => {{
        let r = ($reg & 8) << 4;
        let b = ($rm & 8) << 2;
        let vvvv = $vreg << 3;
        vec![0xc4, (r | b | $encoding) ^ 0xe0, (vvvv | 5) ^ 0x78]
    }};
}

/// This is the three-byte VEX prefix (VEX3) for packed-double (pd)
/// and 256-bit ymm registers
macro_rules! vex3sd {
    ($reg:expr, $vreg:expr, $rm:expr) => {{
        let r = ($reg & 8) << 4;
        let b = ($rm & 8) << 2;
        let vvvv = $vreg << 3;
        vec![0xc4, (r | b | 1) ^ 0xe0, (vvvv | 3) ^ 0x78]
    }};
    ($reg:expr, $vreg:expr, $rm:expr, $encoding:expr) => {{
        let r = ($reg & 8) << 4;
        let b = ($rm & 8) << 2;
        let vvvv = $vreg << 3;
        vec![0xc4, (r | b | $encoding) ^ 0xe0, (vvvv | 3) ^ 0x78]
    }};
}

macro_rules! vex_sd {
    ($reg:expr, $vreg:expr, $rm:expr) => {{
        let rm = $rm;
        if rm < 8 {
            vex2sd!($reg, $vreg)
        } else {
            vex3sd!($reg, $vreg, rm)
        }
    }};
}

macro_rules! vex_pd {
    ($reg:expr, $vreg:expr, $rm:expr) => {{
        let rm = $rm;
        if rm < 8 {
            vex2pd!($reg, $vreg)
        } else {
            vex3pd!($reg, $vreg, rm)
        }
    }};
}

macro_rules! assemble {
    ($($x:expr),+ ;) => {
        {
            vec![$($x),*]
        }
    };
    ($($x:expr),+ ; $y:expr) => {
        {
            let mut v = vec![$($x),*];
            for b in $y { v.push(b); }
            v
        }
    };
    (; $y:expr; $($z:expr),+ ;) => {
        {
            let mut y = $y;
            $(
                y.push($z);
            )*
            y
        }
    };
    (; $y:expr; $($z:expr),+ ; $w:expr) => {
        {
            let mut y = $y;
            $(
                y.push($z);
            )*
            for b in $w { y.push(b); }
            y
        }
    };
}

macro_rules! amd {
    // avx
    (vmovsd xmm($reg:expr), [$rm:ident + $offset:expr]) => {{
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex_sd!(reg, 0, rm); 0x10; modrm_mem!(reg, rm, $offset)]
    }};
    (vmovsd [$rm:ident + $offset:expr], xmm($reg:expr)) => {{
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex_sd!(reg, 0, rm); 0x11; modrm_mem!(reg, rm, $offset)]
    }};
    (vmovapd ymm($reg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, 0, rm); 0x28, modrm_reg!(reg, rm);]
    }};
    (vmovapd ymm($reg:expr), [$rm:ident + $offset:expr]) => {{
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex_pd!(reg, 0, rm); 0x28; modrm_mem!(reg, rm, $offset)]
    }};
    (vmovapd [$rm:ident + $offset:expr], ymm($reg:expr)) => {{
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex_pd!(reg, 0, rm); 0x29; modrm_mem!(reg, rm, $offset)]
    }};
    (vmovupd ymm($reg:expr), [$rm:ident + $offset:expr]) => {{
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex_pd!(reg, 0, rm); 0x10; modrm_mem!(reg, rm, $offset)]
    }};
    (vmovupd [$rm:ident + $offset:expr], ymm($reg:expr)) => {{
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex_pd!(reg, 0, rm); 0x11; modrm_mem!(reg, rm, $offset)]
    }};
    (vbroadcastsd ymm($reg:expr), qword ptr [$rm:ident + $offset:expr]) => {{
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex3pd!(reg, 0, rm, 2); 0x19; modrm_mem!(reg, rm, $offset)]
    }};
    (vaddpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x58, modrm_reg!(reg, rm);]
    }};
    (vsubpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x5c, modrm_reg!(reg, rm);]
    }};
    (vmulpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x59, modrm_reg!(reg, rm);]
    }};
    (vdivpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x5e, modrm_reg!(reg, rm);]
    }};
    (vsqrtpd ymm($reg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, 0, rm); 0x51, modrm_reg!(reg, rm);]
    }};
    (vandpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x54, modrm_reg!(reg, rm);]
    }};
    (vandnpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x55, modrm_reg!(reg, rm);]
    }};
    (vorpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x56, modrm_reg!(reg, rm);]
    }};
    (vxorpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0x57, modrm_reg!(reg, rm);]
    }};
    (vcmpeqpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 0;]
    }};
    (vcmpltpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 1;]
    }};
    (vcmplepd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 2;]
    }};
    (vcmpunordpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 3;]
    }};
    (vcmpneqpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 4;]
    }};
    (vcmpnltpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 5;]
    }};
    (vcmpnlepd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 6;]
    }};
    (vcmpordpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![; vex_pd!(reg, $vreg, rm); 0xc2, modrm_reg!(reg, rm), 7;]
    }};
    (vzeroupper) => {{
        assemble![0xc5, 0xf8, 0x77;]
    }};

    // general registers
    (mov $reg:ident, $rm:ident) => {
        {
            let reg = reg!($reg);
            let rm = reg!($rm);
            assemble![rex!(reg, rm), 0x8b, modrm_reg!(reg, rm);]
        }
    };
    (mov $reg:ident, qword ptr [$rm:ident + $offset:expr]) => {
        {
            let reg = reg!($reg);
            let rm = reg!($rm);
            assemble![rex!(reg, rm), 0x8b; modrm_mem!(reg, rm, $offset)]
        }
    };
    (mov qword ptr [$rm:ident + $offset:expr], $reg:ident) => {
        {
            let reg = reg!($reg);
            let rm = reg!($rm);
            assemble![rex!(reg, rm), 0x89; modrm_mem!(reg, rm, $offset)]
        }
    };

    (call $reg:ident) => {
        {
            let reg = reg!($reg);
            if reg < 8 {
                assemble![0xff, 0xd0 | reg;]
            } else {
                assemble![0x41, 0xff, 0xd0 | (reg & 7);]
            }
        }
    };
    (push $reg:ident) => {
        {
            let reg = reg!($reg);
            if reg < 8 {
                assemble![0x50 | reg;]
            } else {
                assemble![0x41, 0x50 | (reg & 7);]
            }
        }
    };
    (pop $reg:ident) => {
        {
            let reg = reg!($reg);
            if reg < 8 {
                assemble![0x58 | reg;]
            } else {
                assemble![0x41, 0x58 | (reg & 7);]
            }
        }
    };
    (ret) => { assemble![0xc3;] };
    (add rsp, $imm:expr) => {
        {
            let imm = $imm as u32;
            assemble![0x48, 0x81, 0xc4; imm.to_le_bytes()]
        }
    };
    (sub rsp, $imm:expr) => {
        {
            let imm = $imm as u32;
            assemble![0x48, 0x81, 0xec; imm.to_le_bytes()]
        }
    };
}

#[test]
fn test_avx() {
    assert_eq!(vec![0x55], amd! {push rbp});
    assert_eq!(vec![0x53], amd! {push rbx});
    assert_eq!(vec![0x48, 0x8b, 0xef], amd! {mov rbp,rdi});
    assert_eq!(vec![0xc5, 0xfd, 0x28, 0xcd], amd! {vmovapd ymm(1), ymm(5)});
    assert_eq!(
        vec![0xc5, 0xfb, 0x10, 0x95, 0x34, 0x12, 0x00, 0x00],
        amd! {vmovsd xmm(2), [rbp+0x1234]}
    );
    assert_eq!(
        vec![0xc5, 0xfd, 0x28, 0x8d, 0x34, 0x12, 0x00, 0x00],
        amd! {vmovapd ymm(1), [rbp+0x1234]}
    );
    assert_eq!(
        vec![0xc5, 0xfd, 0x11, 0x85, 0x34, 0x12, 0x00, 0x00],
        amd! {vmovupd [rbp+0x1234], ymm(0)}
    );

    assert_eq!(
        vec![0xc5, 0xfd, 0x59, 0xd1],
        amd! {vmulpd ymm(2), ymm(0),ymm(1)}
    );
    assert_eq!(
        vec![0xc5, 0x7d, 0x5e, 0xe1],
        amd! {vdivpd ymm(12), ymm(0), ymm(1)}
    );

    assert_eq!(
        vec![0x48, 0x8b, 0x43, 0x10],
        amd! {mov rax, qword ptr [rbx+0x10]}
    );
    assert_eq!(
        vec![0x48, 0x8b, 0x9b, 0x34, 0x12, 0x00, 0x00],
        amd! {mov rbx, qword ptr [rbx+0x1234]}
    );
    assert_eq!(vec![0xff, 0xd0], amd! {call rax});

    assert_eq!(
        vec![0xc5, 0xfd, 0x57, 0xe1],
        amd! {vxorpd ymm(4),ymm(0),ymm(1)}
    );
    assert_eq!(
        vec![0xc5, 0x7d, 0xc2, 0xf9, 0x05],
        amd! {vcmpnltpd ymm(15),ymm(0),ymm(1)}
    );
    assert_eq!(
        vec![0xc5, 0x95, 0x55, 0xc9],
        amd! {vandnpd ymm(1),ymm(13),ymm(1)}
    );
    assert_eq!(
        vec![0xc5, 0x5d, 0x54, 0xca],
        amd! {vandpd ymm(9),ymm(4),ymm(2)}
    );

    assert_eq!(
        vec![0xc5, 0xdd, 0x56, 0xfd],
        amd! {vorpd  ymm(7),ymm(4),ymm(5)}
    );
    assert_eq!(
        vec![0xc5, 0xfd, 0x57, 0xf9],
        amd! {vxorpd ymm(7),ymm(0),ymm(1)}
    );
    assert_eq!(vec![0xc5, 0xfd, 0x28, 0xf9], amd! {vmovapd ymm(7),ymm(1)});
    assert_eq!(
        vec![0xc5, 0xfd, 0x58, 0xfd],
        amd! {vaddpd ymm(7),ymm(0),ymm(5)}
    );
    assert_eq!(vec![0xc5, 0x7d, 0x51, 0xf9], amd! {vsqrtpd ymm(15),ymm(1)});
    assert_eq!(vec![0xc5, 0xf8, 0x77], amd! {vzeroupper});

    assert_eq!(vec![0x5d], amd! {pop rbp});
    assert_eq!(vec![0xc3], amd! {ret});
    assert_eq!(
        vec![0x48, 0x81, 0xc4, 0x34, 0x12, 0x00, 0x00],
        amd! {add rsp,0x1234}
    );

    assert_eq!(
        vec![0x48, 0x81, 0xec, 0x21, 0x43, 0x00, 0x00],
        amd! {sub rsp,0x4321}
    );
}
