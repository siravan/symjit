// Tests of the object writers on hand-assembled x86-64 functions. The ELF and Mach-O
// files are parsed back field by field; on x86-64 Linux with a C compiler, the ELF
// object is also linked into a C program (with -lm) and run.

use super::*;

// double twice_sin(double x) { return 2 * sin(x); }
//   sub rsp, 8 ; call sin ; addsd xmm0, xmm0 ; add rsp, 8 ; ret
const TWICE_SIN: [u8; 18] = [
    0x48, 0x83, 0xec, 0x08, 0xe8, 0, 0, 0, 0, 0xf2, 0x0f, 0x58, 0xc0, 0x48, 0x83, 0xc4, 0x08, 0xc3,
];
const TWICE_SIN_CALL: usize = 5;

// double power(double x, double y) { return pow(x, y); }   (tail call: jmp pow)
const POWER: [u8; 5] = [0xe9, 0, 0, 0, 0];
const POWER_CALL: usize = 1;

// double plus_pi(double x) { return x + PI; }, PI read RIP-relative after the code:
//   addsd xmm0, [rip + 8] ; ret ; int3 x 7 ; PI
fn plus_pi() -> Vec<u8> {
    let mut code = vec![0xf2, 0x0f, 0x58, 0x05, 0x08, 0, 0, 0, 0xc3];
    code.extend([0xcc; 7]);
    code.extend(std::f64::consts::PI.to_le_bytes());
    code
}

fn sample(format: Format) -> ObjectBuilder {
    let mut obj = ObjectBuilder::new(Target::new(format, Arch::X86_64));
    obj.add_function("twice_sin", &TWICE_SIN, &[Relocation::call(TWICE_SIN_CALL, "sin")])
        .unwrap();
    obj.add_function("power", &POWER, &[Relocation::call(POWER_CALL, "pow")])
        .unwrap();
    obj.add_function("plus_pi", &plus_pi(), &[]).unwrap();
    obj
}

fn u16_at(b: &[u8], k: usize) -> u16 {
    u16::from_le_bytes(b[k..k + 2].try_into().unwrap())
}

fn u32_at(b: &[u8], k: usize) -> u32 {
    u32::from_le_bytes(b[k..k + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], k: usize) -> u64 {
    u64::from_le_bytes(b[k..k + 8].try_into().unwrap())
}

fn cstr(b: &[u8], k: usize) -> String {
    let end = b[k..].iter().position(|&c| c == 0).unwrap();
    String::from_utf8(b[k..k + end].to_vec()).unwrap()
}

#[test]
fn functions_are_aligned_and_offsets_rebased() {
    let obj = sample(Format::Elf);
    let offsets: Vec<usize> = obj.functions.iter().map(|f| f.offset).collect();
    assert_eq!(offsets, vec![0, 32, 48]);
    assert_eq!(obj.code[18..32], [0xcc; 14]); // int3 padding
    assert_eq!(obj.relocations[1], Relocation::call(32 + POWER_CALL, "pow"));
    assert_eq!(obj.undefined_symbols(), vec!["pow", "sin"]);
}

#[test]
fn invalid_input_is_rejected() {
    let mut obj = ObjectBuilder::new(Target::new(Format::Elf, Arch::X86_64));
    assert!(obj.add_function("2bad", &POWER, &[]).is_err());
    assert!(obj.add_function("f", &POWER, &[Relocation::call(3, "pow")]).is_err()); // out of range
    assert!(obj.add_function("f", &TWICE_SIN, &[Relocation::call(4, "sin")]).is_err()); // not after e8/e9
    assert!(obj.add_function("f", &[0xe8, 1, 0, 0, 0], &[Relocation::call(1, "sin")]).is_err()); // nonzero field
    assert!(obj.add_function("f", &POWER, &[Relocation::call(1, "bad-name")]).is_err());
    obj.add_function("f", &POWER, &[Relocation::call(1, "pow")]).unwrap();
    assert!(obj.add_function("f", &POWER, &[]).is_err()); // duplicate
    assert!(ObjectBuilder::new(Target::new(Format::Elf, Arch::X86_64)).build().is_err()); // empty
}

#[test]
fn host_target() {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_eq!(Target::host().unwrap(), Target::new(Format::Elf, Arch::X86_64));
    }
    if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        assert_eq!(Target::host().unwrap(), Target::new(Format::MachO, Arch::X86_64));
    }
}

/***************************** ELF *****************************/

struct ElfSection {
    name: String,
    ty: u32,
    flags: u64,
    offset: usize,
    size: usize,
    link: u32,
    info: u32,
    align: u64,
    entsize: u64,
}

fn elf_sections(b: &[u8]) -> Vec<ElfSection> {
    let shoff = u64_at(b, 0x28) as usize;
    let shnum = u16_at(b, 0x3c) as usize;
    let shstrndx = u16_at(b, 0x3e) as usize;
    let raw: Vec<usize> = (0..shnum).map(|k| shoff + 64 * k).collect();
    let shstr_off = u64_at(b, raw[shstrndx] + 0x18) as usize;
    raw.iter()
        .map(|&h| ElfSection {
            name: cstr(b, shstr_off + u32_at(b, h) as usize),
            ty: u32_at(b, h + 4),
            flags: u64_at(b, h + 8),
            offset: u64_at(b, h + 0x18) as usize,
            size: u64_at(b, h + 0x20) as usize,
            link: u32_at(b, h + 0x28),
            info: u32_at(b, h + 0x2c),
            align: u64_at(b, h + 0x30),
            entsize: u64_at(b, h + 0x38),
        })
        .collect()
}

#[test]
fn elf_header_and_sections() {
    let obj = sample(Format::Elf);
    let b = obj.build().unwrap();

    assert_eq!(&b[..4], b"\x7fELF");
    assert_eq!(&b[4..7], &[2, 1, 1]); // 64-bit, little-endian, version 1
    assert_eq!(u16_at(&b, 0x10), 1); // ET_REL
    assert_eq!(u16_at(&b, 0x12), 62); // EM_X86_64
    assert_eq!(u16_at(&b, 0x34), 64); // e_ehsize
    assert_eq!(u16_at(&b, 0x3a), 64); // e_shentsize

    let s = elf_sections(&b);
    let names: Vec<&str> = s.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["", ".text", ".rela.text", ".symtab", ".strtab", ".shstrtab", ".note.GNU-stack"]);

    let text = &s[1];
    assert_eq!((text.ty, text.flags, text.align), (1, 0x6, 16)); // PROGBITS, ALLOC|EXECINSTR
    assert_eq!(text.offset % 16, 0);
    assert_eq!(&b[text.offset..text.offset + text.size], &obj.code[..]);

    let rela = &s[2];
    assert_eq!((rela.ty, rela.link, rela.info, rela.entsize, rela.flags), (4, 3, 1, 24, 0x40));
    let symtab = &s[3];
    assert_eq!((symtab.ty, symtab.link, symtab.info, symtab.entsize), (2, 4, 2, 24));
    let note = &s[6];
    assert_eq!((note.ty, note.size, note.flags), (1, 0, 0));
}

#[test]
fn elf_symbols_and_relocations() {
    let obj = sample(Format::Elf);
    let b = obj.build().unwrap();
    let s = elf_sections(&b);
    let (symtab, strtab, rela) = (&s[3], &s[4], &s[2]);

    // (name, info, shndx, value, size)
    let syms: Vec<(String, u8, u16, u64, u64)> = (0..symtab.size / 24)
        .map(|k| {
            let e = symtab.offset + 24 * k;
            (
                cstr(&b, strtab.offset + u32_at(&b, e) as usize),
                b[e + 4],
                u16_at(&b, e + 6),
                u64_at(&b, e + 8),
                u64_at(&b, e + 16),
            )
        })
        .collect();
    assert_eq!(
        syms,
        vec![
            ("".to_string(), 0, 0, 0, 0),
            ("".to_string(), 0x03, 1, 0, 0),          // LOCAL SECTION .text
            ("twice_sin".to_string(), 0x12, 1, 0, 18), // GLOBAL FUNC
            ("power".to_string(), 0x12, 1, 32, 5),
            ("plus_pi".to_string(), 0x12, 1, 48, 24),
            ("pow".to_string(), 0x10, 0, 0, 0), // GLOBAL NOTYPE UNDEF
            ("sin".to_string(), 0x10, 0, 0, 0),
        ]
    );

    // (offset, symbol, type, addend)
    let relocs: Vec<(u64, u64, u64, i64)> = (0..rela.size / 24)
        .map(|k| {
            let e = rela.offset + 24 * k;
            let info = u64_at(&b, e + 8);
            (u64_at(&b, e), info >> 32, info & 0xffff_ffff, u64_at(&b, e + 16) as i64)
        })
        .collect();
    assert_eq!(relocs, vec![(5, 6, 4, -4), (33, 5, 4, -4)]); // R_X86_64_PLT32 sin, pow
}

#[test]
fn elf_links_with_a_c_program() {
    // needs x86-64 Linux and a C compiler (`cc`); skipped otherwise
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return;
    }
    let dir = std::env::temp_dir().join(format!("symjit-linker-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join("funcs.o");
    sample(Format::Elf).write(obj_path.to_str().unwrap()).unwrap();

    let c = r#"
#include <math.h>
#include <stdio.h>
double twice_sin(double x);
double power(double x, double y);
double plus_pi(double x);
int main(void) {
    double a = twice_sin(0.5), b = power(2.0, 10.0), c = plus_pi(1.0);
    printf("%.17g %.17g %.17g\n", a, b, c);
    return !(a == 2 * sin(0.5) && b == 1024.0 && c == 1.0 + 3.141592653589793);
}
"#;
    let c_path = dir.join("main.c");
    std::fs::write(&c_path, c).unwrap();
    let exe = dir.join("main");

    let cc = std::process::Command::new("cc")
        .arg(&c_path)
        .arg(&obj_path)
        .arg("-lm")
        .arg("-o")
        .arg(&exe)
        .output();
    let Ok(cc) = cc else {
        eprintln!("elf_links_with_a_c_program: no C compiler (`cc`), skipped");
        return;
    };
    assert!(cc.status.success(), "cc failed:\n{}", String::from_utf8_lossy(&cc.stderr));
    // the linker must not warn (e.g. about an executable stack or text relocations)
    assert!(cc.stderr.is_empty(), "cc warned:\n{}", String::from_utf8_lossy(&cc.stderr));

    let run = std::process::Command::new(&exe).output().unwrap();
    let out = String::from_utf8_lossy(&run.stdout).to_string();
    assert!(run.status.success(), "wrong results: {}", out);
    let _ = std::fs::remove_dir_all(&dir);
}

/***************************** Mach-O *****************************/

#[test]
fn macho_header_and_load_commands() {
    let obj = sample(Format::MachO);
    let b = obj.build().unwrap();

    assert_eq!(u32_at(&b, 0), 0xfeedfacf);
    assert_eq!(u32_at(&b, 4), 0x0100_0007); // CPU_TYPE_X86_64
    assert_eq!(u32_at(&b, 8), 3); // CPU_SUBTYPE_X86_64_ALL
    assert_eq!(u32_at(&b, 12), 1); // MH_OBJECT
    assert_eq!(u32_at(&b, 16), 4); // ncmds
    let sizeofcmds = u32_at(&b, 20) as usize;

    // walk the load commands
    let mut k = 32;
    let mut cmds = Vec::new();
    while k < 32 + sizeofcmds {
        cmds.push((u32_at(&b, k), k));
        k += u32_at(&b, k + 4) as usize;
    }
    assert_eq!(k, 32 + sizeofcmds);
    let kinds: Vec<u32> = cmds.iter().map(|c| c.0).collect();
    assert_eq!(kinds, [0x19, 0x32, 0x2, 0xb]); // SEGMENT_64, BUILD_VERSION, SYMTAB, DYSYMTAB

    // the section __TEXT,__text
    let sect = cmds[0].1 + 72;
    assert_eq!(cstr(&b, sect), "__text");
    assert_eq!(cstr(&b, sect + 16), "__TEXT");
    let size = u64_at(&b, sect + 40) as usize;
    let offset = u32_at(&b, sect + 48) as usize;
    assert_eq!(u32_at(&b, sect + 52), 4); // 2^4 alignment
    assert_eq!(u32_at(&b, sect + 64), 0x8000_0400); // PURE_INSTRUCTIONS | SOME_INSTRUCTIONS
    assert_eq!(offset % 16, 0);
    assert_eq!(&b[offset..offset + size], &obj.code[..]);
    assert_eq!(u64_at(&b, cmds[0].1 + 32), size as u64); // segment vmsize
    assert_eq!(u64_at(&b, cmds[0].1 + 40), offset as u64); // segment fileoff

    // LC_BUILD_VERSION: macOS 10.13
    let bv = cmds[1].1;
    assert_eq!((u32_at(&b, bv + 8), u32_at(&b, bv + 12)), (1, 0x000a_0d00));

    // LC_DYSYMTAB: 3 defined external symbols, then 2 undefined
    let dy = cmds[3].1;
    let ranges: Vec<u32> = (0..6).map(|i| u32_at(&b, dy + 8 + 4 * i)).collect();
    assert_eq!(ranges, [0, 0, 0, 3, 3, 2]);
}

#[test]
fn macho_symbols_and_relocations() {
    let obj = sample(Format::MachO);
    let b = obj.build().unwrap();

    let sect = 32 + 72;
    let (reloff, nreloc) = (u32_at(&b, sect + 56) as usize, u32_at(&b, sect + 60) as usize);
    let symtab = 32 + 72 + 80 + 24;
    assert_eq!(u32_at(&b, symtab), 0x2);
    let (symoff, nsyms, stroff) = (
        u32_at(&b, symtab + 8) as usize,
        u32_at(&b, symtab + 12) as usize,
        u32_at(&b, symtab + 16) as usize,
    );

    // (name, type, sect, value)
    let syms: Vec<(String, u8, u8, u64)> = (0..nsyms)
        .map(|k| {
            let e = symoff + 16 * k;
            (cstr(&b, stroff + u32_at(&b, e) as usize), b[e + 4], b[e + 5], u64_at(&b, e + 8))
        })
        .collect();
    assert_eq!(
        syms,
        vec![
            ("_plus_pi".to_string(), 0x0f, 1, 48), // N_SECT | N_EXT, sorted by name
            ("_power".to_string(), 0x0f, 1, 32),
            ("_twice_sin".to_string(), 0x0f, 1, 0),
            ("_pow".to_string(), 0x01, 0, 0), // N_UNDF | N_EXT
            ("_sin".to_string(), 0x01, 0, 0),
        ]
    );
    assert_eq!(b[stroff], 0); // the string table starts with an empty name

    // (address, symbolnum, pcrel, length, extern, type), by decreasing address
    let relocs: Vec<(u32, u32, u32, u32, u32, u32)> = (0..nreloc)
        .map(|k| {
            let w = u32_at(&b, reloff + 8 * k + 4);
            (u32_at(&b, reloff + 8 * k), w & 0xff_ffff, (w >> 24) & 1, (w >> 25) & 3, (w >> 27) & 1, w >> 28)
        })
        .collect();
    assert_eq!(relocs, vec![(33, 3, 1, 2, 1, 2), (5, 4, 1, 2, 1, 2)]); // X86_64_RELOC_BRANCH pow, sin
}

/// Writes the sample objects (funcs_elf.o, funcs_macho.o for x86-64, funcs_arm64_elf.o,
/// funcs_arm64_macho.o for AArch64) and a C driver (main.c) to the directory named by
/// SYMJIT_LINKER_SAMPLES, for checking them with external tools, e.g. on an Apple
/// silicon Mac:  cc main.c funcs_arm64_macho.o -o main && ./main
#[test]
fn write_sample_objects() {
    let Ok(dir) = std::env::var("SYMJIT_LINKER_SAMPLES") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    sample(Format::Elf).write(&format!("{}/funcs_elf.o", dir)).unwrap();
    sample(Format::MachO).write(&format!("{}/funcs_macho.o", dir)).unwrap();
    arm_sample(Format::Elf).write(&format!("{}/funcs_arm64_elf.o", dir)).unwrap();
    arm_sample(Format::MachO).write(&format!("{}/funcs_arm64_macho.o", dir)).unwrap();
    let c = r#"#include <math.h>
#include <stdio.h>
double twice_sin(double x);
double power(double x, double y);
double plus_pi(double x);
int main(void) {
    double a = twice_sin(0.5), b = power(2.0, 10.0), c = plus_pi(1.0);
    int ok = a == 2 * sin(0.5) && b == 1024.0 && c == 1.0 + 3.141592653589793;
    printf("%.17g %.17g %.17g %s\n", a, b, c, ok ? "ok" : "WRONG");
    return !ok;
}
"#;
    std::fs::write(format!("{}/main.c", dir), c).unwrap();
}

/***************************** AArch64 *****************************/

fn words(ws: &[u32]) -> Vec<u8> {
    ws.iter().flat_map(|w| w.to_le_bytes()).collect()
}

use crate::arm::object_samples;

fn arm_twice_sin() -> Vec<u8> {
    words(&object_samples::twice_sin())
}
const ARM_TWICE_SIN_CALL: usize = 8;

fn arm_power() -> Vec<u8> {
    words(&object_samples::power())
}
const ARM_POWER_CALL: usize = 8;

fn arm_plus_pi() -> Vec<u8> {
    let mut code = words(&object_samples::plus_pi());
    code.extend(std::f64::consts::PI.to_le_bytes());
    code
}

fn arm_sample(format: Format) -> ObjectBuilder {
    let mut obj = ObjectBuilder::new(Target::new(format, Arch::Aarch64));
    obj.add_function("twice_sin", &arm_twice_sin(), &[Relocation::call(ARM_TWICE_SIN_CALL, "sin")])
        .unwrap();
    obj.add_function("power", &arm_power(), &[Relocation::call(ARM_POWER_CALL, "pow")])
        .unwrap();
    obj.add_function("plus_pi", &arm_plus_pi(), &[]).unwrap();
    obj
}

#[test]
fn arm_relocations_must_be_at_bl() {
    let mut obj = ObjectBuilder::new(Target::new(Format::Elf, Arch::Aarch64));
    let code = arm_twice_sin();
    assert!(obj.add_function("f", &code, &[Relocation::call(4, "sin")]).is_err()); // stp, not bl
    assert!(obj.add_function("f", &code, &[Relocation::call(9, "sin")]).is_err()); // unaligned
    assert!(obj.add_function("f", &code, &[Relocation::call(28, "sin")]).is_err()); // outside
    let mut taken = code.clone();
    taken[8..12].copy_from_slice(&object_samples::bl(64).to_le_bytes());
    assert!(obj.add_function("f", &taken, &[Relocation::call(8, "sin")]).is_err()); // nonzero offset
    obj.add_function("f", &code, &[Relocation::call(8, "sin")]).unwrap();
    assert_eq!(obj.relocations, vec![Relocation::call(8, "sin")]);
}

#[test]
fn arm_elf() {
    let obj = arm_sample(Format::Elf);
    let b = obj.build().unwrap();
    assert_eq!(u16_at(&b, 0x12), 183); // EM_AARCH64

    let s = elf_sections(&b);
    let text = &s[1];
    assert_eq!(&b[text.offset..text.offset + text.size], &obj.code[..]);
    // functions at 0, 32 (28 bytes, padded with udf #0), 64
    assert_eq!(obj.functions.iter().map(|f| f.offset).collect::<Vec<_>>(), [0, 32, 64]);
    assert_eq!(obj.code[28..32], [0; 4]);

    let rela = &s[2];
    let relocs: Vec<(u64, u64, u64, i64)> = (0..rela.size / 24)
        .map(|k| {
            let e = rela.offset + 24 * k;
            let info = u64_at(&b, e + 8);
            (u64_at(&b, e), info >> 32, info & 0xffff_ffff, u64_at(&b, e + 16) as i64)
        })
        .collect();
    // R_AARCH64_CALL26 at the `bl`s, addend 0; symbols 5 = pow, 6 = sin
    assert_eq!(relocs, vec![(8, 6, 283, 0), (40, 5, 283, 0)]);
}

#[test]
fn arm_macho() {
    let obj = arm_sample(Format::MachO);
    let b = obj.build().unwrap();
    assert_eq!(u32_at(&b, 4), 0x0100_000c); // CPU_TYPE_ARM64
    assert_eq!(u32_at(&b, 8), 0); // CPU_SUBTYPE_ARM64_ALL

    let sect = 32 + 72;
    let offset = u32_at(&b, sect + 48) as usize;
    assert_eq!(&b[offset..offset + obj.code.len()], &obj.code[..]);
    let bv = 32 + 72 + 80;
    assert_eq!((u32_at(&b, bv), u32_at(&b, bv + 12)), (0x32, 0x000b_0000)); // macOS 11.0

    let (reloff, nreloc) = (u32_at(&b, sect + 56) as usize, u32_at(&b, sect + 60) as usize);
    let relocs: Vec<(u32, u32, u32, u32, u32, u32)> = (0..nreloc)
        .map(|k| {
            let w = u32_at(&b, reloff + 8 * k + 4);
            (u32_at(&b, reloff + 8 * k), w & 0xff_ffff, (w >> 24) & 1, (w >> 25) & 3, (w >> 27) & 1, w >> 28)
        })
        .collect();
    // ARM64_RELOC_BRANCH26 (2), pc-relative, 4 bytes, external: 3 = _pow, 4 = _sin
    assert_eq!(relocs, vec![(40, 3, 1, 2, 1, 2), (8, 4, 1, 2, 1, 2)]);
}
