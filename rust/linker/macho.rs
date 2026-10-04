// Mach-O relocatable object (MH_OBJECT) for macOS (<mach-o/loader.h>, <mach-o/nlist.h>,
// <mach-o/reloc.h>, <mach-o/x86_64/reloc.h>).
//
// Layout: mach_header_64 | LC_SEGMENT_64 (one section, __TEXT,__text) | LC_BUILD_VERSION |
// LC_SYMTAB | LC_DYSYMTAB | text | relocations | symbol table | string table.
// Symbols (all external, names with a leading `_`): the functions, sorted by name,
// then the undefined external functions, sorted by name.

use std::collections::HashMap;

use super::{Arch, ObjectBuilder, Put, RelocKind};

const MH_MAGIC_64: u32 = 0xfeedfacf;
const MH_OBJECT: u32 = 0x1;
const CPU_TYPE_X86_64: u32 = 0x0100_0007;
const CPU_SUBTYPE_X86_64_ALL: u32 = 3;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const LC_BUILD_VERSION: u32 = 0x32;

const SEGMENT_SIZE: u32 = 72;
const SECTION_SIZE: u32 = 80;
const BUILD_VERSION_SIZE: u32 = 24;
const SYMTAB_SIZE: u32 = 24;
const DYSYMTAB_SIZE: u32 = 80;
const HEADER_SIZE: u32 = 32;

const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x8000_0000;
const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x0000_0400;

const PLATFORM_MACOS: u32 = 1;
// minimum macOS version, encoded xxxx.yy.zz
const MIN_MACOS_X86_64: u32 = (10 << 16) | (13 << 8);

const N_EXT: u8 = 0x01;
const N_SECT: u8 = 0x0e;
const N_UNDF: u8 = 0x00;

const X86_64_RELOC_BRANCH: u32 = 2;

const NLIST_SIZE: usize = 16;

fn name16(s: &str) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..s.len()].copy_from_slice(s.as_bytes());
    b
}

pub fn build(obj: &ObjectBuilder) -> Vec<u8> {
    // symbols: defined (sorted), then undefined (sorted)
    let mut defined: Vec<(String, u64)> = obj
        .functions
        .iter()
        .map(|f| (obj.target.symbol_name(&f.name), f.offset as u64))
        .collect();
    defined.sort();
    let undefined: Vec<String> = obj
        .undefined_symbols()
        .iter()
        .map(|s| obj.target.symbol_name(s))
        .collect();

    let mut strtab: Vec<u8> = vec![0];
    let mut nlist: Vec<(u32, u8, u8, u64)> = Vec::new(); // (strx, type, sect, value)
    let mut index: HashMap<String, u32> = HashMap::new();

    for (name, value) in defined.iter() {
        index.insert(name.clone(), nlist.len() as u32);
        nlist.push((strtab.len() as u32, N_SECT | N_EXT, 1, *value));
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
    }
    for name in undefined.iter() {
        index.insert(name.clone(), nlist.len() as u32);
        nlist.push((strtab.len() as u32, N_UNDF | N_EXT, 0, 0));
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
    }
    strtab.pad_to(8);

    // offsets
    let sizeofcmds = SEGMENT_SIZE + SECTION_SIZE + BUILD_VERSION_SIZE + SYMTAB_SIZE + DYSYMTAB_SIZE;
    let text_offset = super::align_up((HEADER_SIZE + sizeofcmds) as usize, super::TEXT_ALIGN);
    let text_size = obj.code.len();
    let reloc_offset = super::align_up(text_offset + text_size, 8);
    let nreloc = obj.relocations.len();
    let symoff = reloc_offset + 8 * nreloc;
    let stroff = symoff + NLIST_SIZE * nlist.len();

    let mut buf: Vec<u8> = Vec::new();

    // mach_header_64
    buf.u32(MH_MAGIC_64);
    let (cputype, cpusubtype, minos) = match obj.target.arch {
        Arch::X86_64 => (CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL, MIN_MACOS_X86_64),
    };
    buf.u32(cputype);
    buf.u32(cpusubtype);
    buf.u32(MH_OBJECT);
    buf.u32(4); // ncmds
    buf.u32(sizeofcmds);
    buf.u32(0); // flags
    buf.u32(0); // reserved

    // LC_SEGMENT_64, with the unnamed segment of an object file
    buf.u32(LC_SEGMENT_64);
    buf.u32(SEGMENT_SIZE + SECTION_SIZE);
    buf.extend_from_slice(&name16(""));
    buf.u64(0); // vmaddr
    buf.u64(text_size as u64); // vmsize
    buf.u64(text_offset as u64); // fileoff
    buf.u64(text_size as u64); // filesize
    buf.u32(7); // maxprot: rwx
    buf.u32(7); // initprot
    buf.u32(1); // nsects
    buf.u32(0); // flags

    // section_64: __TEXT,__text
    buf.extend_from_slice(&name16("__text"));
    buf.extend_from_slice(&name16("__TEXT"));
    buf.u64(0); // addr
    buf.u64(text_size as u64);
    buf.u32(text_offset as u32);
    buf.u32(super::TEXT_ALIGN.trailing_zeros()); // align, as a power of 2
    buf.u32(if nreloc > 0 { reloc_offset as u32 } else { 0 });
    buf.u32(nreloc as u32);
    buf.u32(S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS);
    buf.u32(0); // reserved1
    buf.u32(0); // reserved2
    buf.u32(0); // reserved3

    // LC_BUILD_VERSION
    buf.u32(LC_BUILD_VERSION);
    buf.u32(BUILD_VERSION_SIZE);
    buf.u32(PLATFORM_MACOS);
    buf.u32(minos);
    buf.u32(minos); // sdk
    buf.u32(0); // ntools

    // LC_SYMTAB
    buf.u32(LC_SYMTAB);
    buf.u32(SYMTAB_SIZE);
    buf.u32(symoff as u32);
    buf.u32(nlist.len() as u32);
    buf.u32(stroff as u32);
    buf.u32(strtab.len() as u32);

    // LC_DYSYMTAB: no local symbols, the defined ones, then the undefined ones
    buf.u32(LC_DYSYMTAB);
    buf.u32(DYSYMTAB_SIZE);
    buf.u32(0); // ilocalsym
    buf.u32(0); // nlocalsym
    buf.u32(0); // iextdefsym
    buf.u32(defined.len() as u32); // nextdefsym
    buf.u32(defined.len() as u32); // iundefsym
    buf.u32(undefined.len() as u32); // nundefsym
    for _ in 0..12 {
        buf.u32(0); // tocoff ... nlocrel: unused in object files
    }

    assert_eq!(buf.len(), (HEADER_SIZE + sizeofcmds) as usize);

    // __text
    buf.resize(text_offset, 0);
    buf.extend_from_slice(&obj.code);

    // relocations, by decreasing address (as compilers emit them)
    buf.resize(reloc_offset, 0);
    let mut relocs: Vec<_> = obj.relocations.iter().collect();
    relocs.sort_by(|a, b| b.offset.cmp(&a.offset));
    for r in relocs {
        let (ty, pcrel, length) = match (obj.target.arch, r.kind) {
            // rel32 field, PC-relative, 4 bytes (length 2); the addend is the field (0)
            (Arch::X86_64, RelocKind::Call) => (X86_64_RELOC_BRANCH, 1, 2),
        };
        let symbolnum = index[&obj.target.symbol_name(&r.symbol)];
        let r_extern = 1;
        buf.u32(r.offset as u32);
        buf.u32(symbolnum | (pcrel << 24) | (length << 25) | (r_extern << 27) | (ty << 28));
    }

    // symbol table (nlist_64)
    for (strx, ty, sect, value) in nlist.iter() {
        buf.u32(*strx);
        buf.u8(*ty);
        buf.u8(*sect);
        buf.u16(0); // n_desc
        buf.u64(*value);
    }

    buf.extend_from_slice(&strtab);
    buf
}
