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

// remove the REX byte if it is not needed
// this happens if none of XMM8-XMM15 are used
macro_rules! filter_rex {
    ($v:expr) => {{
        let mut v = $v;
        if (v[0] == 0x66 || v[0] == 0xf2) && v[1] == 0x48 {
            v.remove(1);
        };
        v
    }};
}

macro_rules! assemble {
    ($($x:expr),+ ;) => {
        {
            filter_rex!(vec![$($x),*])
        }
    };
    ($($x:expr),+ ; $y:expr) => {
        {
            let mut v = vec![$($x),*];
            for b in $y { v.push(b); }
            filter_rex!(v)
        }
    };
    (; $y:expr; $($z:expr),+ ;) => {
        {
            $(
                y.push($z);
            )*
            filter_rex!(y)
        }
    };
    (; $y:expr; $($z:expr),+ ; $w:expr) => {
        {
            $(
                y.push($z);
            )*
            for b in $w { y.push(b); }
            filter_rex!(y)
        }
    };
}

macro_rules! amd {
    (movapd xmm($reg:expr), xmm($rm:expr)) => {{
        let reg = $reg;
        let rm = $rm;
        assemble![0x66, rex!(reg, rm), 0x0f, 0x28, modrm_reg!(reg, rm);]
    }};
    (movsd xmm($reg:expr), qword ptr [$rm:ident + $offset:expr]) => {
        {
            let reg = $reg;
            let rm = reg!($rm);
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x10; modrm_mem!(reg, rm, $offset)]
        }
    };
    (movsd qword ptr [$rm:ident + $offset:expr], xmm($reg:expr)) => {
        {
            let reg = $reg;
            let rm = reg!($rm);
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x11; modrm_mem!(reg, rm, $offset)]
        }
    };
    (movq xmm($reg:expr), $rm:ident) => {
        {
            let reg = $reg;
            let rm = reg!($rm);
            assemble![0x66, rex!(reg, rm), 0x0f, 0x6e, modrm_reg!(reg, rm);]
        }
    };
    (movq $rm:ident, xmm($reg:expr)) => {
        {
            let rm = reg!($rm);
            let reg = $reg;
            assemble![0x66, rex!(reg, rm), 0x0f, 0x7e, modrm_reg!(reg, rm);]
        }
    };
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
    (addsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x58, modrm_reg!(reg, rm);]
        }
    };
    (subsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x5c, modrm_reg!(reg, rm);]
        }
    };
    (mulsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x59, modrm_reg!(reg, rm);]
        }
    };
    (divsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x5e, modrm_reg!(reg, rm);]
        }
    };
    (sqrtsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x51, modrm_reg!(reg, rm);]
        }
    };
    (rsqrtsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0x52, modrm_reg!(reg, rm);]
        }
    };
    (andpd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0x66, rex!(reg, rm), 0x0f, 0x54, modrm_reg!(reg, rm);]
        }
    };
    (andnpd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0x66, rex!(reg, rm), 0x0f, 0x55, modrm_reg!(reg, rm);]
        }
    };
    (orpd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0x66, rex!(reg, rm), 0x0f, 0x56, modrm_reg!(reg, rm);]
        }
    };
    (xorpd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0x66, rex!(reg, rm), 0x0f, 0x57, modrm_reg!(reg, rm);]
        }
    };
    (cmpeqsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 0;]
        }
    };
    (cmpltsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg =$reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 1;]
        }
    };
    (cmplesd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 2;]
        }
    };
    (cmpunordsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 3;]
        }
    };
    (cmpneqsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 4;]
        }
    };
    (cmpnltsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 5;]
        }
    };
    (cmpnlesd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 6;]
        }
    };
    (cmpordsd xmm($reg:expr), xmm($rm:expr)) => {
        {
            let reg = $reg;
            let rm = $rm;
            assemble![0xf2, rex!(reg, rm), 0x0f, 0xc2, modrm_reg!(reg, rm), 7;]
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
fn test_amd() {
    assert_eq!(vec![0x55], amd! {push rbp});
    assert_eq!(vec![0x53], amd! {push rbx});
    assert_eq!(vec![0x48, 0x8b, 0xef], amd! {mov rbp,rdi});
    assert_eq!(
        vec![0xf2, 0x0f, 0x10, 0x45, 0x58],
        amd! {movsd xmm(0),qword ptr [rbp+0x58]}
    );
    assert_eq!(
        vec![0xf2, 0x0f, 0x11, 0x85, 0xf8, 0x00, 0x00, 0x00],
        amd! {movsd qword ptr [rbp+0xf8],xmm(0)}
    );
    assert_eq!(vec![0xf2, 0x0f, 0x59, 0xc1], amd! {mulsd xmm(0),xmm(1)});
    assert_eq!(vec![0xf2, 0x0f, 0x5e, 0xc1], amd! {divsd xmm(0),xmm(1)});
    assert_eq!(
        vec![0x48, 0x8b, 0x43, 0x10],
        amd! {mov rax,qword ptr [rbx+0x10]}
    );
    assert_eq!(
        vec![0x48, 0x8b, 0x9b, 0x34, 0x12, 0x00, 0x00],
        amd! {mov rbx,qword ptr [rbx+0x1234]}
    );
    assert_eq!(vec![0xff, 0xd0], amd! {call rax});
    assert_eq!(vec![0x66, 0x0f, 0x57, 0xc1], amd! {xorpd xmm(0),xmm(1)});
    assert_eq!(
        vec![0xf2, 0x0f, 0xc2, 0xc1, 0x05],
        amd! {cmpnltsd xmm(0),xmm(1)}
    );
    assert_eq!(vec![0x66, 0x0f, 0x55, 0xd9], amd! {andnpd xmm(3),xmm(1)});
    assert_eq!(vec![0x66, 0x0f, 0x54, 0xe2], amd! {andpd xmm(4),xmm(2)});
    assert_eq!(
        vec![0xf2, 0x0f, 0x10, 0x4d, 0x18],
        amd! {movsd  xmm(1),qword ptr [rbp+0x18]}
    );
    assert_eq!(vec![0x66, 0x0f, 0x56, 0xe5], amd! {orpd  xmm(4),xmm(5)});
    assert_eq!(vec![0x66, 0x0f, 0x57, 0xc1], amd! {xorpd xmm(0),xmm(1)});
    assert_eq!(vec![0xf2, 0x0f, 0x10, 0xcc], amd! {movsd xmm(1),xmm(4)});
    assert_eq!(vec![0xf2, 0x0f, 0x58, 0xc1], amd! {addsd xmm(0),xmm(1)});
    assert_eq!(vec![0xf2, 0x0f, 0x10, 0xcd], amd! {movsd xmm(1),xmm(5)});
    assert_eq!(vec![0x66, 0x48, 0x0f, 0x7e, 0xde], amd! {movq rsi,xmm(3)});
    assert_eq!(vec![0x66, 0x48, 0x0f, 0x6e, 0xe9], amd! {movq xmm(5),rcx});
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
