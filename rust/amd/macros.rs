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
    ($rm:expr, $reg:expr) => {{
        let rm = $rm;
        let reg = $reg;
        0xC0 + ((reg & 7) << 3) + (rm & 7)
    }};
}

macro_rules! rex {
    ($rm:expr, $reg:expr) => {{
        let rm = $rm;
        let reg = $reg;
        0x48 + ((rm & 8) >> 3) + ((reg & 8) >> 1)
    }};
}

macro_rules! modrm_mem {
    ($reg:expr, $base:ident, $offset:expr) => {{
        let reg = $reg;
        let base = $base;
        let offset = $offset;

        let mut v = if offset < 128 {
            vec![0x40 + (reg << 3) + base]
        } else {
            vec![0x80 + (reg << 3) + base]
        };

        if base == 4 {
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

macro_rules! make_modrm {
    ($($v:expr),+ ; $x:expr, $y:expr) => {
        {
            vec![$($v),*, modrm_reg!($x, $y)]
        }
    };
    ($($v:expr),+ ; $x:expr, $y:expr; $z:expr) => {
        {
            vec![$($v),*, modrm_reg!($x, $y), $z]
        }
    };
    ($($v:expr),+ ; $dst:expr, $base:ident, $offset:expr) => {
        {
            let mut v = vec![$($v),*];
            let base = reg!($base);
            v.extend_from_slice(&modrm_mem!($dst, base, $offset)[..]);
            v
        }
    };
}

macro_rules! amd {
    (movsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x10; $src, $dst]
    };
    (movapd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0x66, 0x0f, 0x28; $src, $dst]
    };
    (movsd xmm($dst:expr), qword ptr [$base:ident + $offset:expr]) => {
        make_modrm![0xf2, 0x0f, 0x10; $dst, $base, $offset]
    };
    (movsd qword ptr [$base:ident + $offset:expr], xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x11; $src, $base, $offset]
    };
    (movq xmm($dst:expr), $src:ident) => {
        {
            let dst = $dst;
            let src = reg!($src);
            make_modrm![0x66, rex!(src, dst), 0x0f, 0x6e; src, dst]
        }
    };
    (movq $dst:ident, xmm($src:expr)) => {
        {
            let dst = reg!($dst);
            let src = $src;
            make_modrm![0x66, rex!(dst, src), 0x0f, 0x7e; dst, src]
        }
    };
    (mov $dst:ident, $src:ident) => {
        {
            let dst = reg!($dst);
            let src = reg!($src);
            make_modrm![rex!(dst, src), 0x89; dst, src]
        }
    };
    (mov $dst:ident, qword ptr [$base:ident + $offset:expr]) => {
        {
            let dst = reg!($dst);
            make_modrm![rex!(0, dst), 0x8b; dst, $base, $offset]
        }
    };
    (mov qword ptr [$base:ident + $offset:expr], $src:ident) => {
        {
            let src = reg!($src);
            make_modrm![rex!(0, src), 0x89; src, $base, $offset]
        }
    };
    (addsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x58; $src, $dst]
    };
    (subsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x5c; $src, $dst]
    };
    (mulsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x59; $src, $dst]
    };
    (divsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x5e; $src, $dst]
    };
    (sqrtsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x51; $src, $dst]
    };
    (rsqrtsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0x52; $src, $dst]
    };
    (andpd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0x66, 0x0f, 0x54; $src, $dst]
    };
    (andnpd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0x66, 0x0f, 0x55; $src, $dst]
    };
    (orpd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0x66, 0x0f, 0x56; $src, $dst]
    };
    (xorpd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0x66, 0x0f, 0x57; $src, $dst]
    };
    (cmpeqsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 0]
    };
    (cmpltsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 1]
    };
    (cmplesd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 2]
    };
    (cmpunordsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 3]
    };
    (cmpneqsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 4]
    };
    (cmpnltsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 5]
    };
    (cmpnlesd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 6]
    };
    (cmpordsd xmm($dst:expr), xmm($src:expr)) => {
        make_modrm![0xf2, 0x0f, 0xc2; $src, $dst; 7]
    };
    (call $src:ident) => {
        {
            let src = reg!($src);
            vec![0xff, 0xd0 | src]
        }
    };
    (push $src:ident) => {
        {
            let src = reg!($src);
            if src < 8 {
                vec![0x50 | src]
            } else {
                vec![0x41, 0x48 | src]
            }
        }
    };
    (pop $dst:ident) => {
        {
            let dst = reg!($dst);
            if dst < 8 {
                vec![0x58 | dst]
            } else {
                vec![0x41, 0x50 | dst]
            }
        }
    };
    (ret) => { vec![0xc3] };
    (add rsp, $imm:expr) => {
        {
            let imm = $imm as u32;
            let mut v = vec![0x48, 0x81, 0xc4];
            v.push(imm as u8);
            v.push((imm >> 8) as u8);
            v.push((imm >> 16) as u8);
            v.push((imm >> 24) as u8);
            v
        }
    };
    (sub rsp, $imm:expr) => {
        {
            let imm = $imm as u32;
            let mut v = vec![0x48, 0x81, 0xec];
            v.push(imm as u8);
            v.push((imm >> 8) as u8);
            v.push((imm >> 16) as u8);
            v.push((imm >> 24) as u8);
            v
        }
    };
}

#[test]
fn test_amd() {
    assert_eq!(vec![0x55], amd! {push rbp});
    assert_eq!(vec![0x53], amd! {push rbx});
    assert_eq!(vec![0x48, 0x89, 0xfd], amd! {mov rbp,rdi});
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
