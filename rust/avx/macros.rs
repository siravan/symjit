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
    ($reg, $vreg) => {{
        let R = ($reg & 8) << 4;
        let vvvv = $vreg << 3;
        vec![0xc5, (R | vvvv | 6) ^ 0xf8]
    }};
}

/// This is the three-byte VEX prefix (VEX3) for packed-double (pd) 
/// and 256-bit ymm registers
macro_rules! vex3pd {
    ($reg, $vreg, $rm, $encoding) => {{
        let R = ($reg & 8) << 4;
        let B = ($rm & 8) << 2;
        let vvvv = $vreg << 3;
        vec![0xc4, (R | B | $encoding) ^ 0xe0, (vvvv | 6) ^ 0x78]
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
            $(
                y.push($z);
            )*
            y
        }
    };
    (; $y:expr; $($z:expr),+ ; $w:expr) => {
        {
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
    (vmovapd ymm($reg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, 0); 0x28, modrm_reg!(reg, $rm);]
    };
    (vmovapd ymm($reg:expr), [$rm:ident + $offset:expr]) => {
        let reg = $reg;
        assemble![; vex2pd(reg, 0); 0x28; modrm_mem!(reg, reg!($rm), $offset)]
    };
    (vmovapd [$rm:ident + $offset:expr], ymm($reg:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, 0); 0x29; modrm_mem!(reg, reg!($rm), $offset)]
    };
    (vbroadcastsd ymm($reg:expr), qword ptr [$rm:ident + $offset:expr]) => {
        let reg = $reg;
        let rm = reg!($rm);
        assemble![; vex3pd(reg, 0, rm, 2); 0x19; modrm_mem!(reg, rm, $offset)]
    };
    (vaddpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x58, modrm_reg!($reg, $rm);]
    };
    (vsubpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x5c, modrm_reg!($reg, $rm);]
    };
    (vmulpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x59, modrm_reg!($reg, $rm);]
    };
    (vdivpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x5e, modrm_reg!($reg, $rm);]
    };
    (vsqrtpd ymm($reg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, 0); 0x51, modrm_reg!($reg, $rm);]
    };
    (vandpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x54, modrm_reg!($reg, $rm);]
    };
    (vandnpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x55, modrm_reg!($reg, $rm);]
    };
    (vorpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x56, modrm_reg!($reg, $rm);]
    };
    (vxorpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0x57, modrm_reg!($reg, $rm);]
    };
    (vcmpeqpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 0;]
    };
    (vcmpltpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 1;]
    };
    (vcmplepd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 2;]
    };
    (vcmpunordpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 3;]
    };
    (vcmpneqpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 4;]
    };
    (vcmpnltpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 5;]
    };
    (vcmpnlepd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 6;]
    };
    (vcmpordpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr)) => {
        let reg = $reg;
        assemble![; vex2pd(reg, $vreg); 0xc2, modrm_reg!($reg, $rm), 7;]
    };
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
            assemble![0xff, 0xd0 | reg;]
        }
    };
    (push $reg:ident) => {
        {
            let reg = reg!($reg);
            if reg < 8 {
                assemble![0x50 | reg;]
            } else {
                assemble![0x41, 0x48 | reg;]
            }
        }
    };
    (pop $reg:ident) => {
        {
            let reg = reg!($reg);
            if reg < 8 {
                assemble![0x58 | reg;]
            } else {
                assemble![0x41, 0x50 | reg;]
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
    assert_eq!(
        vec![0xf2, 0x0f, 0x10, 0x45, 0x58],
        amd! {movsd ymm(0),qword ptr [rbp+0x58]}
    );
    assert_eq!(
        vec![0xf2, 0x0f, 0x11, 0x85, 0xf8, 0x00, 0x00, 0x00],
        amd! {movsd qword ptr [rbp+0xf8],ymm(0)}
    );
    assert_eq!(vec![0xf2, 0x0f, 0x59, 0xc1], amd! {mulsd ymm(0),ymm(1)});
    assert_eq!(vec![0xf2, 0x0f, 0x5e, 0xc1], amd! {divsd ymm(0),ymm(1)});
    assert_eq!(
        vec![0x48, 0x8b, 0x43, 0x10],
        amd! {mov rax,qword ptr [rbx+0x10]}
    );
    assert_eq!(
        vec![0x48, 0x8b, 0x9b, 0x34, 0x12, 0x00, 0x00],
        amd! {mov rbx,qword ptr [rbx+0x1234]}
    );
    assert_eq!(vec![0xff, 0xd0], amd! {call rax});
    assert_eq!(vec![0x66, 0x0f, 0x57, 0xc1], amd! {xorpd ymm(0),ymm(1)});
    assert_eq!(
        vec![0xf2, 0x0f, 0xc2, 0xc1, 0x05],
        amd! {cmpnltsd ymm(0),ymm(1)}
    );
    assert_eq!(vec![0x66, 0x0f, 0x55, 0xd9], amd! {andnpd ymm(3),ymm(1)});
    assert_eq!(vec![0x66, 0x0f, 0x54, 0xe2], amd! {andpd ymm(4),ymm(2)});
    assert_eq!(
        vec![0xf2, 0x0f, 0x10, 0x4d, 0x18],
        amd! {movsd  ymm(1),qword ptr [rbp+0x18]}
    );
    assert_eq!(vec![0x66, 0x0f, 0x56, 0xe5], amd! {orpd  ymm(4),ymm(5)});
    assert_eq!(vec![0x66, 0x0f, 0x57, 0xc1], amd! {xorpd ymm(0),ymm(1)});
    assert_eq!(vec![0xf2, 0x0f, 0x10, 0xcc], amd! {movsd ymm(1),ymm(4)});
    assert_eq!(vec![0xf2, 0x0f, 0x58, 0xc1], amd! {addsd ymm(0),ymm(1)});
    assert_eq!(vec![0xf2, 0x0f, 0x10, 0xcd], amd! {movsd ymm(1),ymm(5)});
    assert_eq!(vec![0x66, 0x48, 0x0f, 0x7e, 0xde], amd! {movq rsi,ymm(3)});
    assert_eq!(vec![0x66, 0x48, 0x0f, 0x6e, 0xe9], amd! {movq ymm(5),rcx});
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
