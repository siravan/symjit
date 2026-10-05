// ELF64 relocatable object (System V gABI; the x86-64 and AArch64 psABIs for the
// relocations).
//
// Layout: ELF header | .text | .rela.text | .symtab | .strtab | .shstrtab | section headers.
// Sections: 0 null, 1 .text, 2 .rela.text, 3 .symtab, 4 .strtab, 5 .shstrtab,
// 6 .note.GNU-stack (empty: the code does not need an executable stack).
// Symbols: null, the .text section symbol (local), then the functions (global FUNC)
// and the undefined external functions (global NOTYPE).

use std::collections::HashMap;

use super::{Arch, ObjectBuilder, Put, RelocKind};

const EM_X86_64: u16 = 62;
const EM_AARCH64: u16 = 183;
const ET_REL: u16 = 1;

const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;

const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_INFO_LINK: u64 = 0x40;

const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STT_NOTYPE: u8 = 0;
const STT_FUNC: u8 = 2;
const STT_SECTION: u8 = 3;

const R_X86_64_PLT32: u32 = 4;
const R_AARCH64_CALL26: u32 = 283;

const TEXT: u16 = 1; // section index of .text
const SYMTAB: u32 = 3;
const STRTAB: u32 = 4;
const SHSTRTAB: u16 = 5;

const SYM_SIZE: u64 = 24;
const RELA_SIZE: u64 = 24;

struct Symbol {
    name: u32, // offset in .strtab
    info: u8,
    shndx: u16,
    value: u64,
    size: u64,
}

struct Section {
    name: &'static str,
    ty: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    align: u64,
    entsize: u64,
}

pub fn build(obj: &ObjectBuilder) -> Vec<u8> {
    // string table and symbols
    let mut strtab: Vec<u8> = vec![0];
    let mut add_str = |s: &str| -> u32 {
        let k = strtab.len() as u32;
        strtab.extend_from_slice(s.as_bytes());
        strtab.push(0);
        k
    };

    let mut symbols = vec![
        Symbol { name: 0, info: 0, shndx: 0, value: 0, size: 0 },
        Symbol {
            name: 0,
            info: (STB_LOCAL << 4) | STT_SECTION,
            shndx: TEXT,
            value: 0,
            size: 0,
        },
    ];
    let first_global = symbols.len() as u32;
    let mut index: HashMap<String, u32> = HashMap::new();

    for f in obj.functions.iter() {
        index.insert(f.name.clone(), symbols.len() as u32);
        symbols.push(Symbol {
            name: add_str(&obj.target.symbol_name(&f.name)),
            info: (STB_GLOBAL << 4) | STT_FUNC,
            shndx: TEXT,
            value: f.offset as u64,
            size: f.size as u64,
        });
    }
    for s in obj.undefined_symbols() {
        index.insert(s.clone(), symbols.len() as u32);
        symbols.push(Symbol {
            name: add_str(&obj.target.symbol_name(&s)),
            info: (STB_GLOBAL << 4) | STT_NOTYPE,
            shndx: 0,
            value: 0,
            size: 0,
        });
    }

    // the file
    let mut buf: Vec<u8> = vec![0; 64]; // the header is written last

    let text_offset = buf.len() as u64;
    buf.extend_from_slice(&obj.code);

    buf.pad_to(8);
    let rela_offset = buf.len() as u64;
    for r in obj.relocations.iter() {
        let (ty, addend) = match (obj.target.arch, r.kind) {
            // S + A - P with P at the rel32 field, which ends 4 bytes later
            (Arch::X86_64, RelocKind::Call) => (R_X86_64_PLT32, -4i64),
            // S + A - P with P at the `bl`
            (Arch::Aarch64, RelocKind::Call) => (R_AARCH64_CALL26, 0),
        };
        let sym = index[&r.symbol] as u64;
        buf.u64(r.offset as u64);
        buf.u64((sym << 32) | ty as u64);
        buf.u64(addend as u64);
    }
    let rela_size = buf.len() as u64 - rela_offset;

    buf.pad_to(8);
    let symtab_offset = buf.len() as u64;
    for s in symbols.iter() {
        buf.u32(s.name);
        buf.u8(s.info);
        buf.u8(0); // st_other: default visibility
        buf.u16(s.shndx);
        buf.u64(s.value);
        buf.u64(s.size);
    }
    let symtab_size = buf.len() as u64 - symtab_offset;

    let strtab_offset = buf.len() as u64;
    buf.extend_from_slice(&strtab);

    let mut sections = vec![
        Section { name: "", ty: 0, flags: 0, offset: 0, size: 0, link: 0, info: 0, align: 0, entsize: 0 },
        Section {
            name: ".text",
            ty: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            offset: text_offset,
            size: obj.code.len() as u64,
            link: 0,
            info: 0,
            align: super::TEXT_ALIGN as u64,
            entsize: 0,
        },
        Section {
            name: ".rela.text",
            ty: SHT_RELA,
            flags: SHF_INFO_LINK,
            offset: rela_offset,
            size: rela_size,
            link: SYMTAB,
            info: TEXT as u32,
            align: 8,
            entsize: RELA_SIZE,
        },
        Section {
            name: ".symtab",
            ty: SHT_SYMTAB,
            flags: 0,
            offset: symtab_offset,
            size: symtab_size,
            link: STRTAB,
            info: first_global, // one more than the last local symbol
            align: 8,
            entsize: SYM_SIZE,
        },
        Section {
            name: ".strtab",
            ty: SHT_STRTAB,
            flags: 0,
            offset: strtab_offset,
            size: strtab.len() as u64,
            link: 0,
            info: 0,
            align: 1,
            entsize: 0,
        },
        Section { name: ".shstrtab", ty: SHT_STRTAB, flags: 0, offset: 0, size: 0, link: 0, info: 0, align: 1, entsize: 0 },
        Section { name: ".note.GNU-stack", ty: SHT_PROGBITS, flags: 0, offset: 0, size: 0, link: 0, info: 0, align: 1, entsize: 0 },
    ];

    let mut shstrtab: Vec<u8> = vec![0];
    let names: Vec<u32> = sections
        .iter()
        .map(|s| {
            if s.name.is_empty() {
                0
            } else {
                let k = shstrtab.len() as u32;
                shstrtab.extend_from_slice(s.name.as_bytes());
                shstrtab.push(0);
                k
            }
        })
        .collect();

    let shstrtab_offset = buf.len() as u64;
    buf.extend_from_slice(&shstrtab);
    sections[SHSTRTAB as usize].offset = shstrtab_offset;
    sections[SHSTRTAB as usize].size = shstrtab.len() as u64;
    sections[6].offset = buf.len() as u64;

    buf.pad_to(8);
    let shoff = buf.len() as u64;
    for (s, name) in sections.iter().zip(names) {
        buf.u32(name);
        buf.u32(s.ty);
        buf.u64(s.flags);
        buf.u64(0); // sh_addr
        buf.u64(s.offset);
        buf.u64(s.size);
        buf.u32(s.link);
        buf.u32(s.info);
        buf.u64(s.align);
        buf.u64(s.entsize);
    }

    // the ELF header
    let mut h: Vec<u8> = Vec::with_capacity(64);
    h.extend_from_slice(&[0x7f, b'E', b'L', b'F']);
    h.u8(2); // ELFCLASS64
    h.u8(1); // ELFDATA2LSB
    h.u8(1); // EV_CURRENT
    h.u8(0); // ELFOSABI_SYSV
    h.extend_from_slice(&[0; 8]); // ABI version and padding
    h.u16(ET_REL);
    h.u16(match obj.target.arch {
        Arch::X86_64 => EM_X86_64,
        Arch::Aarch64 => EM_AARCH64,
    });
    h.u32(1); // e_version
    h.u64(0); // e_entry
    h.u64(0); // e_phoff
    h.u64(shoff);
    h.u32(0); // e_flags
    h.u16(64); // e_ehsize
    h.u16(0); // e_phentsize
    h.u16(0); // e_phnum
    h.u16(64); // e_shentsize
    h.u16(sections.len() as u16);
    h.u16(SHSTRTAB);
    buf[..64].copy_from_slice(&h);

    buf
}
