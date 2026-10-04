// A minimal WebAssembly binary encoder (MVP + the few instructions symjit needs).
//
// Reference: WebAssembly Core Specification 2.0, chapter 5 (Binary Format).
// Opcode encodings are pinned against wabt's `wat2wasm` in `rust/wasm/tests.rs`.

// The instruction set covers what the arithmetic and control-flow lowering needs,
// not only what the generator emits so far.
#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValType {
    I32,
    I64,
    F64,
}

impl ValType {
    pub fn code(self) -> u8 {
        match self {
            ValType::I32 => 0x7f,
            ValType::I64 => 0x7e,
            ValType::F64 => 0x7c,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Empty,
    Value(ValType),
}

/// memarg of a load/store: alignment (log2 bytes) and constant offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemArg {
    pub align: u32,
    pub offset: u32,
}

impl MemArg {
    pub fn f64(offset: u32) -> MemArg {
        MemArg { align: 3, offset }
    }

    pub fn i32(offset: u32) -> MemArg {
        MemArg { align: 2, offset }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Unreachable,
    Nop,
    Block(BlockType),
    Loop(BlockType),
    If(BlockType),
    Else,
    End,
    Br(u32),
    BrIf(u32),
    BrTable(Vec<u32>, u32),
    Return,
    Call(u32),
    Drop,
    Select,

    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    GlobalGet(u32),
    GlobalSet(u32),

    I32Load(MemArg),
    I64Load(MemArg),
    F64Load(MemArg),
    I32Store(MemArg),
    I64Store(MemArg),
    F64Store(MemArg),

    I32Const(i32),
    I64Const(i64),
    F64Const(f64),

    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I64Eqz,

    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,

    I32Add,
    I32Sub,
    I32Mul,
    I32And,
    I32Shl,

    I64Sub,
    I64And,
    I64Or,
    I64Xor,

    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F64Sqrt,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,

    I64ExtendI32U,
    I64ReinterpretF64,
    F64ReinterpretI64,
}

/***************************** LEB128 *****************************/

pub fn leb_u32(buf: &mut Vec<u8>, mut v: u32) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

pub fn leb_i64(buf: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7; // arithmetic shift
        let done = (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0);
        if done {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

pub fn leb_i32(buf: &mut Vec<u8>, v: i32) {
    leb_i64(buf, v as i64);
}

fn name(buf: &mut Vec<u8>, s: &str) {
    leb_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

fn block_type(buf: &mut Vec<u8>, bt: BlockType) {
    match bt {
        BlockType::Empty => buf.push(0x40),
        BlockType::Value(t) => buf.push(t.code()),
    }
}

fn mem_op(buf: &mut Vec<u8>, opcode: u8, m: MemArg) {
    buf.push(opcode);
    leb_u32(buf, m.align);
    leb_u32(buf, m.offset);
}

impl Op {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            Op::Unreachable => buf.push(0x00),
            Op::Nop => buf.push(0x01),
            Op::Block(bt) => {
                buf.push(0x02);
                block_type(buf, *bt);
            }
            Op::Loop(bt) => {
                buf.push(0x03);
                block_type(buf, *bt);
            }
            Op::If(bt) => {
                buf.push(0x04);
                block_type(buf, *bt);
            }
            Op::Else => buf.push(0x05),
            Op::End => buf.push(0x0b),
            Op::Br(l) => {
                buf.push(0x0c);
                leb_u32(buf, *l);
            }
            Op::BrIf(l) => {
                buf.push(0x0d);
                leb_u32(buf, *l);
            }
            Op::BrTable(labels, default) => {
                buf.push(0x0e);
                leb_u32(buf, labels.len() as u32);
                for l in labels {
                    leb_u32(buf, *l);
                }
                leb_u32(buf, *default);
            }
            Op::Return => buf.push(0x0f),
            Op::Call(f) => {
                buf.push(0x10);
                leb_u32(buf, *f);
            }
            Op::Drop => buf.push(0x1a),
            Op::Select => buf.push(0x1b),

            Op::LocalGet(i) => {
                buf.push(0x20);
                leb_u32(buf, *i);
            }
            Op::LocalSet(i) => {
                buf.push(0x21);
                leb_u32(buf, *i);
            }
            Op::LocalTee(i) => {
                buf.push(0x22);
                leb_u32(buf, *i);
            }
            Op::GlobalGet(i) => {
                buf.push(0x23);
                leb_u32(buf, *i);
            }
            Op::GlobalSet(i) => {
                buf.push(0x24);
                leb_u32(buf, *i);
            }

            Op::I32Load(m) => mem_op(buf, 0x28, *m),
            Op::I64Load(m) => mem_op(buf, 0x29, *m),
            Op::F64Load(m) => mem_op(buf, 0x2b, *m),
            Op::I32Store(m) => mem_op(buf, 0x36, *m),
            Op::I64Store(m) => mem_op(buf, 0x37, *m),
            Op::F64Store(m) => mem_op(buf, 0x39, *m),

            Op::I32Const(v) => {
                buf.push(0x41);
                leb_i32(buf, *v);
            }
            Op::I64Const(v) => {
                buf.push(0x42);
                leb_i64(buf, *v);
            }
            Op::F64Const(v) => {
                buf.push(0x44);
                buf.extend_from_slice(&v.to_le_bytes());
            }

            Op::I32Eqz => buf.push(0x45),
            Op::I32Eq => buf.push(0x46),
            Op::I32Ne => buf.push(0x47),
            Op::I32LtS => buf.push(0x48),
            Op::I32LtU => buf.push(0x49),
            Op::I32GtS => buf.push(0x4a),
            Op::I32GtU => buf.push(0x4b),
            Op::I32LeS => buf.push(0x4c),
            Op::I32LeU => buf.push(0x4d),
            Op::I32GeS => buf.push(0x4e),
            Op::I32GeU => buf.push(0x4f),
            Op::I64Eqz => buf.push(0x50),

            Op::F64Eq => buf.push(0x61),
            Op::F64Ne => buf.push(0x62),
            Op::F64Lt => buf.push(0x63),
            Op::F64Gt => buf.push(0x64),
            Op::F64Le => buf.push(0x65),
            Op::F64Ge => buf.push(0x66),

            Op::I32Add => buf.push(0x6a),
            Op::I32Sub => buf.push(0x6b),
            Op::I32Mul => buf.push(0x6c),
            Op::I32And => buf.push(0x71),
            Op::I32Shl => buf.push(0x74),

            Op::I64Sub => buf.push(0x7d),
            Op::I64And => buf.push(0x83),
            Op::I64Or => buf.push(0x84),
            Op::I64Xor => buf.push(0x85),

            Op::F64Abs => buf.push(0x99),
            Op::F64Neg => buf.push(0x9a),
            Op::F64Ceil => buf.push(0x9b),
            Op::F64Floor => buf.push(0x9c),
            Op::F64Trunc => buf.push(0x9d),
            Op::F64Nearest => buf.push(0x9e),
            Op::F64Sqrt => buf.push(0x9f),
            Op::F64Add => buf.push(0xa0),
            Op::F64Sub => buf.push(0xa1),
            Op::F64Mul => buf.push(0xa2),
            Op::F64Div => buf.push(0xa3),
            Op::F64Min => buf.push(0xa4),
            Op::F64Max => buf.push(0xa5),
            Op::F64Copysign => buf.push(0xa6),

            Op::I64ExtendI32U => buf.push(0xad),
            Op::I64ReinterpretF64 => buf.push(0xbd),
            Op::F64ReinterpretI64 => buf.push(0xbf),
        }
    }
}

/***************************** Module *****************************/

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

pub struct Import {
    pub module: String,
    pub field: String,
    pub type_idx: u32,
}

pub struct Global {
    pub ty: ValType,
    pub mutable: bool,
    pub init: i32,
}

#[derive(Clone, Copy)]
pub enum ExportKind {
    Func = 0,
    Memory = 2,
    Global = 3,
}

pub struct Export {
    pub name: String,
    pub kind: ExportKind,
    pub idx: u32,
}

pub struct Function {
    pub type_idx: u32,
    pub name: String,
    /// local declarations after the parameters, run-length encoded
    pub locals: Vec<(u32, ValType)>,
    /// names of all locals (parameters first), for the `name` section
    pub local_names: Vec<String>,
    pub body: Vec<Op>,
}

impl Function {
    /// The size in bytes of the encoded body (local declarations and code).
    pub fn body_size(&self) -> usize {
        let mut b = Vec::new();
        leb_u32(&mut b, self.locals.len() as u32);
        for (n, t) in self.locals.iter() {
            leb_u32(&mut b, *n);
            b.push(t.code());
        }
        for op in self.body.iter() {
            op.encode(&mut b);
        }
        b.len()
    }
}

/// A module with imported functions only (no imported memory/tables/globals),
/// one memory, and i32 globals. Function indices are imports first, then `funcs`.
#[derive(Default)]
pub struct Module {
    pub types: Vec<FuncType>,
    pub imports: Vec<Import>,
    pub funcs: Vec<Function>,
    pub memory_pages: Option<u32>,
    pub globals: Vec<Global>,
    pub exports: Vec<Export>,
}

impl Module {
    pub fn new() -> Module {
        Module::default()
    }

    /// Returns the index of `ty`, adding it if needed.
    pub fn add_type(&mut self, ty: FuncType) -> u32 {
        if let Some(k) = self.types.iter().position(|t| *t == ty) {
            return k as u32;
        }
        self.types.push(ty);
        (self.types.len() - 1) as u32
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

        // 1: type section
        let mut s = Vec::new();
        leb_u32(&mut s, self.types.len() as u32);
        for t in self.types.iter() {
            s.push(0x60);
            leb_u32(&mut s, t.params.len() as u32);
            s.extend(t.params.iter().map(|p| p.code()));
            leb_u32(&mut s, t.results.len() as u32);
            s.extend(t.results.iter().map(|p| p.code()));
        }
        section(&mut buf, 1, &s);

        // 2: import section
        if !self.imports.is_empty() {
            let mut s = Vec::new();
            leb_u32(&mut s, self.imports.len() as u32);
            for im in self.imports.iter() {
                name(&mut s, &im.module);
                name(&mut s, &im.field);
                s.push(0x00); // function import
                leb_u32(&mut s, im.type_idx);
            }
            section(&mut buf, 2, &s);
        }

        // 3: function section
        let mut s = Vec::new();
        leb_u32(&mut s, self.funcs.len() as u32);
        for f in self.funcs.iter() {
            leb_u32(&mut s, f.type_idx);
        }
        section(&mut buf, 3, &s);

        // 5: memory section
        if let Some(pages) = self.memory_pages {
            let mut s = Vec::new();
            leb_u32(&mut s, 1);
            s.push(0x00); // limits: min only
            leb_u32(&mut s, pages);
            section(&mut buf, 5, &s);
        }

        // 6: global section
        if !self.globals.is_empty() {
            let mut s = Vec::new();
            leb_u32(&mut s, self.globals.len() as u32);
            for g in self.globals.iter() {
                s.push(g.ty.code());
                s.push(if g.mutable { 0x01 } else { 0x00 });
                Op::I32Const(g.init).encode(&mut s);
                Op::End.encode(&mut s);
            }
            section(&mut buf, 6, &s);
        }

        // 7: export section
        let mut s = Vec::new();
        leb_u32(&mut s, self.exports.len() as u32);
        for e in self.exports.iter() {
            name(&mut s, &e.name);
            s.push(e.kind as u8);
            leb_u32(&mut s, e.idx);
        }
        section(&mut buf, 7, &s);

        // 10: code section
        let mut s = Vec::new();
        leb_u32(&mut s, self.funcs.len() as u32);
        for f in self.funcs.iter() {
            let mut b = Vec::new();
            leb_u32(&mut b, f.locals.len() as u32);
            for (n, t) in f.locals.iter() {
                leb_u32(&mut b, *n);
                b.push(t.code());
            }
            for op in f.body.iter() {
                op.encode(&mut b);
            }
            leb_u32(&mut s, b.len() as u32);
            s.extend(b);
        }
        section(&mut buf, 10, &s);

        // 0: custom `name` section (function and local names, for wasm2wat/debuggers)
        let mut s = Vec::new();
        name(&mut s, "name");

        let n_imports = self.imports.len() as u32;
        let mut sub = Vec::new();
        leb_u32(&mut sub, n_imports + self.funcs.len() as u32);
        for (k, im) in self.imports.iter().enumerate() {
            leb_u32(&mut sub, k as u32);
            name(&mut sub, &format!("{}.{}", im.module, im.field));
        }
        for (k, f) in self.funcs.iter().enumerate() {
            leb_u32(&mut sub, n_imports + k as u32);
            name(&mut sub, &f.name);
        }
        s.push(1); // function names
        leb_u32(&mut s, sub.len() as u32);
        s.extend(sub);

        let mut sub = Vec::new();
        leb_u32(&mut sub, self.funcs.len() as u32);
        for (k, f) in self.funcs.iter().enumerate() {
            leb_u32(&mut sub, n_imports + k as u32);
            leb_u32(&mut sub, f.local_names.len() as u32);
            for (i, n) in f.local_names.iter().enumerate() {
                leb_u32(&mut sub, i as u32);
                name(&mut sub, n);
            }
        }
        s.push(2); // local names
        leb_u32(&mut s, sub.len() as u32);
        s.extend(sub);

        section(&mut buf, 0, &s);

        buf
    }
}

fn section(buf: &mut Vec<u8>, id: u8, contents: &[u8]) {
    buf.push(id);
    leb_u32(buf, contents.len() as u32);
    buf.extend_from_slice(contents);
}
