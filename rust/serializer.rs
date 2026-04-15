use anyhow::{anyhow, Result};
use std::io::{Read, Write};

use crate::config::Config;
use crate::mir::{BinOp, Instruction, Mir, UniOp};
use crate::symbol::Loc;
use crate::utils::*;

const REG_GENERAL: u8 = 0x40;
const REG_STATIC: u8 = 0x80;
const REG_SPECIAL: u8 = 0xc0;
const REG_RET: u8 = REG_SPECIAL;
const REG_TEMP: u8 = REG_SPECIAL | 0x01;
const REG_LEFT: u8 = REG_SPECIAL | 0x02;
const REG_RIGHT: u8 = REG_SPECIAL | 0x03;

const LOC_MEM: u8 = 0x40;
const LOC_PARAM: u8 = 0x80;
const LOC_STACK: u8 = 0xc0;

/*
 * First byte decoding:
 *  1. Look at the two top bits.
 *      -> if 01, then it is UniOp and the other six bits encode the op.
 *      -> if 10, then it is BinOp and the other six bits encode the op.
 *      -> if 11, then it is OtherOp, bits 3-5 encode the type and 0-2 the op.
 *      -> if 00, general instructions encoded in six lower bits.
 */
const UNI_OP: u8 = 0x40;
const BIN_OP: u8 = 0x80;

const NOP: u8 = 0;
const END: u8 = 1;
const MOV: u8 = 2;
const LOAD: u8 = 3;
const SAVE: u8 = 4;
const LOAD_CONST: u8 = 5;
const LOAD_COMPLEX: u8 = 6;
const SAVE_COMPLEX: u8 = 7;
const BRANCH: u8 = 8;
const BRANCH_IF: u8 = 9;
const BRANCH_ELSE: u8 = 10;
const CALL: u8 = 11;
const LABEL: u8 = 12;
const IFELSE: u8 = 13;

const FUSED_MUL_ADD: u8 = 32;
const FUSED_MUL_SUB: u8 = 33;
const FUSED_NEG_MUL_ADD: u8 = 34;
const FUSED_NEG_MUL_SUB: u8 = 35;

const LOAD_MATH_PLUS: u8 = 36;
const LOAD_MATH_MINUS: u8 = 37;
const LOAD_MATH_TIMES: u8 = 38;
const LOAD_MATH_DIVIDE: u8 = 39;

const LOAD_CONST_MATH_PLUS: u8 = 40;
const LOAD_CONST_MATH_MINUS: u8 = 41;
const LOAD_CONST_MATH_TIMES: u8 = 42;
const LOAD_CONST_MATH_DIVIDE: u8 = 43;

const COMPLEX_BI_PLUS: u8 = 44;
const COMPLEX_BI_MINUS: u8 = 45;
const COMPLEX_BI_TIMES: u8 = 46;
const COMPLEX_BI_DIVIDE: u8 = 47;

const UNIOP_NEG: u8 = UniOp::Neg as u8;
const UNIOP_NOT: u8 = UniOp::Not as u8;
const UNIOP_ABS: u8 = UniOp::Abs as u8;
const UNIOP_ROOT: u8 = UniOp::Root as u8;
const UNIOP_REALROOT: u8 = UniOp::RealRoot as u8;
const UNIOP_RECIP: u8 = UniOp::Recip as u8;
const UNIOP_ROUND: u8 = UniOp::Round as u8;
const UNIOP_FLOOR: u8 = UniOp::Floor as u8;
const UNIOP_CEILING: u8 = UniOp::Ceiling as u8;
const UNIOP_TRUNC: u8 = UniOp::Trunc as u8;
const UNIOP_REAL: u8 = UniOp::Real as u8;
const UNIOP_IMAGINARY: u8 = UniOp::Imaginary as u8;
const UNIOP_CONJUGATE: u8 = UniOp::Conjugate as u8;

const BINOP_PLUS: u8 = BinOp::Plus as u8;
const BINOP_MINUS: u8 = BinOp::Minus as u8;
const BINOP_TIMES: u8 = BinOp::Times as u8;
const BINOP_DIVIDE: u8 = BinOp::Divide as u8;
const BINOP_GREATER_THAN: u8 = BinOp::GreaterThan as u8;
const BINOP_GREATER_THAN_EQUAL: u8 = BinOp::GreaterThanEqual as u8;
const BINOP_LITTLE_THAN: u8 = BinOp::LittleThan as u8;
const BINOP_LITTLE_THAN_EQUAL: u8 = BinOp::LittleThanEqual as u8;
const BINOP_EQUAL: u8 = BinOp::Equal as u8;
const BINOP_NOT_EQUAL: u8 = BinOp::NotEqual as u8;
const BINOP_AND: u8 = BinOp::And as u8;
const BINOP_AND_NOT: u8 = BinOp::AndNot as u8;
const BINOP_OR: u8 = BinOp::Or as u8;
const BINOP_XOR: u8 = BinOp::Xor as u8;
const BINOP_COMPLEX: u8 = BinOp::Complex as u8;

pub struct MirWriter {
    buf: Vec<u8>,
}

impl Default for MirWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl MirWriter {
    pub fn new() -> MirWriter {
        MirWriter { buf: Vec::new() }
    }

    fn push(&mut self, b: u8) {
        self.buf.push(b);
    }

    pub fn serialize(&mut self, mir: &Mir) {
        for ins in mir.code.iter() {
            match ins {
                Instruction::Nop => self.push(NOP),
                Instruction::End => self.push(END),
                Instruction::Uni { op, dst, s1 } => {
                    let op = *op as u8;
                    assert!(op < 64);
                    self.push(UNI_OP | op);
                    self.reg(*dst);
                    self.reg(*s1);
                }
                Instruction::Bi { op, dst, s1, s2 } => {
                    let op = *op as u8;
                    assert!(op < 64);
                    self.push(BIN_OP | op);
                    self.reg(*dst);
                    self.reg(*s1);
                    self.reg(*s2);
                }
                Instruction::Mov { dst, s1 } => {
                    self.push(MOV);
                    self.reg(*dst);
                    self.reg(*s1);
                }
                Instruction::Load { dst, loc } => {
                    self.push(LOAD);
                    self.reg(*dst);
                    self.loc(*loc);
                }
                Instruction::LoadMath { op, dst, s1, loc } => {
                    let op = *op as u8;
                    assert!(op < 4);
                    self.push(LOAD_MATH_PLUS + op);
                    self.reg(*dst);
                    self.reg(*s1);
                    self.loc(*loc);
                }
                Instruction::Save { src, loc } => {
                    self.push(SAVE);
                    self.reg(*src);
                    self.loc(*loc);
                }
                Instruction::LoadConst { dst, idx } => {
                    self.push(LOAD_CONST);
                    self.reg(*dst);
                    self.num(0, *idx);
                }
                Instruction::LoadComplex { xd, yd, loc } => {
                    self.push(LOAD_COMPLEX);
                    self.reg(*xd);
                    self.reg(*yd);
                    self.loc(*loc);
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    self.push(SAVE_COMPLEX);
                    self.reg(*xs);
                    self.reg(*ys);
                    self.loc(*loc);
                }
                Instruction::LoadConstMath { op, dst, s1, idx } => {
                    let op = *op as u8;
                    assert!(op < 4);
                    self.push(LOAD_CONST_MATH_PLUS + op);
                    self.reg(*dst);
                    self.reg(*s1);
                    self.num(0, *idx);
                }
                Instruction::Branch { label } => {
                    self.push(BRANCH);
                    self.string(label);
                }
                Instruction::BranchIf {
                    cond,
                    label,
                    is_else,
                } => {
                    if *is_else {
                        self.push(BRANCH_ELSE);
                    } else {
                        self.push(BRANCH_IF);
                    }
                    self.reg(*cond);
                    self.string(label);
                }
                Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                } => {
                    self.push(IFELSE);
                    self.reg(*dst);
                    self.reg(*true_val);
                    self.reg(*false_val);
                    self.loc(*cond);
                }
                Instruction::Call {
                    label, num_args, ..
                } => {
                    self.push(CALL);
                    assert!(*num_args < 256);
                    self.push(*num_args as u8);
                    self.string(label);
                }
                Instruction::Label { label } => {
                    self.push(LABEL);
                    self.string(label);
                }
                Instruction::Fused { op, dst, a, b, c } => {
                    let op = *op as u8;
                    assert!(op < 4);
                    self.push(FUSED_MUL_ADD + op);
                    self.reg(*dst);
                    self.reg(*a);
                    self.reg(*b);
                    self.reg(*c);
                }
                Instruction::ComplexBi {
                    op,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                } => {
                    let op = *op as u8;
                    assert!(op < 8);
                    self.push(COMPLEX_BI_PLUS + op);
                    self.reg(*xd);
                    self.reg(*yd);
                    self.reg(*x1);
                    self.reg(*y1);
                    self.reg(*x2);
                    self.reg(*y2);
                }
            }
        }
    }

    fn num(&mut self, prefix: u8, mut n: u32) {
        let num_bytes: u8 = 1 + ((32 - n.leading_zeros()) >> 3) as u8;
        self.push(prefix | num_bytes);
        for _ in 0..num_bytes {
            self.push((n & 0xff) as u8);
            n >>= 8;
        }
    }

    fn reg(&mut self, r: Reg) {
        match r {
            Reg::Ret => self.push(REG_RET),
            Reg::Temp => self.push(REG_TEMP),
            Reg::Left => self.push(REG_LEFT),
            Reg::Right => self.push(REG_RIGHT),
            Reg::Gen(r) => self.push(REG_GENERAL | r),
            Reg::Static(s) => self.num(REG_STATIC, s),
        }
    }

    fn loc(&mut self, loc: Loc) {
        match loc {
            Loc::Mem(idx) => self.num(LOC_MEM, idx),
            Loc::Param(idx) => self.num(LOC_PARAM, idx),
            Loc::Stack(idx) => self.num(LOC_STACK, idx),
        };
    }

    fn string(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let len = bytes.len();
        assert!(len < 256);

        self.push(len as u8);
        for b in bytes {
            self.push(*b);
        }
    }
}

pub struct MirReader {
    buf: Vec<u8>,
    pos: usize,
}

impl MirReader {
    pub fn new(buf: Vec<u8>) -> MirReader {
        MirReader { buf, pos: 0 }
    }

    fn pop(&mut self) -> Result<u8> {
        if self.pos < self.buf.len() {
            let c = self.buf[self.pos];
            self.pos += 1;
            Ok(c)
        } else {
            Err(anyhow!("unexpected EOF"))
        }
    }

    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub fn deserialize(&mut self, mir: &mut Mir) -> Result<()> {
        while !self.eof() {
            let header = self.pop()?;

            match header & 0xc0 {
                UNI_OP => self.uniop(mir, header)?,
                BIN_OP => self.binop(mir, header)?,
                _ => self.other(mir, header)?,
            }
        }

        Ok(())
    }

    fn uniop(&mut self, mir: &mut Mir, header: u8) -> Result<()> {
        let dst = self.reg()?;
        let s1 = self.reg()?;

        match header & 0x3f {
            UNIOP_NEG => mir.neg(dst, s1),
            UNIOP_NOT => mir.not(dst, s1),
            UNIOP_ABS => mir.abs(dst, s1),
            UNIOP_ROOT => mir.root(dst, s1),
            UNIOP_REALROOT => mir.real_root(dst, s1),
            UNIOP_RECIP => mir.recip(dst, s1),
            UNIOP_ROUND => mir.round(dst, s1),
            UNIOP_FLOOR => mir.floor(dst, s1),
            UNIOP_CEILING => mir.ceiling(dst, s1),
            UNIOP_TRUNC => mir.trunc(dst, s1),
            UNIOP_REAL => mir.real(dst, s1),
            UNIOP_IMAGINARY => mir.imaginary(dst, s1),
            UNIOP_CONJUGATE => mir.conjugate(dst, s1),
            _ => return Err(anyhow!("undefined UniOp {:x}", header)),
        }

        Ok(())
    }

    fn binop(&mut self, mir: &mut Mir, header: u8) -> Result<()> {
        let dst = self.reg()?;
        let s1 = self.reg()?;
        let s2 = self.reg()?;

        match header & 0x3f {
            BINOP_PLUS => mir.plus(dst, s1, s2),
            BINOP_MINUS => mir.minus(dst, s1, s2),
            BINOP_TIMES => mir.times(dst, s1, s2),
            BINOP_DIVIDE => mir.divide(dst, s1, s2),
            BINOP_GREATER_THAN => mir.gt(dst, s1, s2),
            BINOP_GREATER_THAN_EQUAL => mir.geq(dst, s1, s2),
            BINOP_LITTLE_THAN => mir.lt(dst, s1, s2),
            BINOP_LITTLE_THAN_EQUAL => mir.leq(dst, s1, s2),
            BINOP_EQUAL => mir.eq(dst, s1, s2),
            BINOP_NOT_EQUAL => mir.neq(dst, s1, s2),
            BINOP_AND => mir.and(dst, s1, s2),
            BINOP_AND_NOT => mir.andnot(dst, s1, s2),
            BINOP_OR => mir.or(dst, s1, s2),
            BINOP_XOR => mir.xor(dst, s1, s2),
            BINOP_COMPLEX => mir.complex(dst, s1, s2),
            _ => return Err(anyhow!("undefined BinOp {:x}", header)),
        }

        Ok(())
    }

    pub fn other(&mut self, mir: &mut Mir, header: u8) -> Result<()> {
        match header & 0x3f {
            NOP => mir.nop(),
            END => mir.nop(),
            MOV => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                mir.fmov(dst, s1);
            }
            LOAD => {
                let dst = self.reg()?;
                let loc = self.loc()?;
                match loc {
                    Loc::Mem(idx) => mir.load_mem(dst, idx),
                    Loc::Param(idx) => mir.load_param(dst, idx),
                    Loc::Stack(idx) => mir.load_stack(dst, idx),
                }
            }
            SAVE => {
                let src = self.reg()?;
                let loc = self.loc()?;
                match loc {
                    Loc::Mem(idx) => mir.save_mem(src, idx),
                    Loc::Stack(idx) => mir.save_stack(src, idx),
                    _ => {}
                }
            }
            LOAD_CONST => {
                let dst = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                mir.load_const(dst, idx);
            }
            LOAD_COMPLEX => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let loc = self.loc()?;
                match loc {
                    Loc::Mem(idx) => mir.load_mem_complex(xd, yd, idx),
                    Loc::Param(idx) => mir.load_param_complex(xd, yd, idx),
                    Loc::Stack(idx) => mir.load_stack_complex(xd, yd, idx),
                }
            }
            SAVE_COMPLEX => {
                let xs = self.reg()?;
                let ys = self.reg()?;
                let loc = self.loc()?;
                match loc {
                    Loc::Mem(idx) => mir.save_mem_complex(xs, ys, idx),
                    Loc::Stack(idx) => mir.save_stack_complex(xs, ys, idx),
                    _ => {}
                }
            }
            BRANCH => {
                let label = self.string()?;
                mir.branch(&label);
            }
            BRANCH_IF => {
                let cond = self.reg()?;
                let label = self.string()?;
                mir.branch_if(cond, &label, false);
            }
            BRANCH_ELSE => {
                let cond = self.reg()?;
                let label = self.string()?;
                mir.branch_if(cond, &label, true);
            }
            CALL => {
                let num_args = self.pop()? as usize;
                let op = self.string()?;
                mir.call(&op, num_args)
                    .expect(&format!("op code {:?} not found.", &op));
            }
            LABEL => {
                let label = self.string()?;
                mir.set_label(&label);
            }
            IFELSE => {
                let dst = self.reg()?;
                let true_val = self.reg()?;
                let false_val = self.reg()?;
                let cond = self.loc()?;
                mir.ifelse(dst, true_val, false_val, cond);
            }
            FUSED_MUL_ADD => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                mir.fused_mul_add(dst, a, b, c);
            }
            FUSED_MUL_SUB => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                mir.fused_mul_sub(dst, a, b, c);
            }
            FUSED_NEG_MUL_ADD => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                mir.fused_neg_mul_add(dst, a, b, c);
            }
            FUSED_NEG_MUL_SUB => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                mir.fused_neg_mul_sub(dst, a, b, c);
            }
            LOAD_MATH_PLUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                mir.plus_load(dst, s1, loc);
            }
            LOAD_MATH_MINUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                mir.minus_load(dst, s1, loc);
            }
            LOAD_MATH_TIMES => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                mir.times_load(dst, s1, loc);
            }
            LOAD_MATH_DIVIDE => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                mir.divide_load(dst, s1, loc);
            }
            LOAD_CONST_MATH_PLUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                mir.plus_load_const(dst, s1, idx);
            }
            LOAD_CONST_MATH_MINUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                mir.minus_load_const(dst, s1, idx);
            }
            LOAD_CONST_MATH_TIMES => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                mir.times_load_const(dst, s1, idx);
            }
            LOAD_CONST_MATH_DIVIDE => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                mir.divide_load_const(dst, s1, idx);
            }
            COMPLEX_BI_PLUS => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                mir.plus_complex(xd, yd, x1, y1, x2, y2);
            }
            COMPLEX_BI_MINUS => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                mir.minus_complex(xd, yd, x1, y1, x2, y2);
            }
            COMPLEX_BI_TIMES => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                mir.times_complex(xd, yd, x1, y1, x2, y2);
            }
            COMPLEX_BI_DIVIDE => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                mir.divide_complex(xd, yd, x1, y1, x2, y2);
            }
            _ => return Err(anyhow!("undefined header {:x}", header)),
        }

        Ok(())
    }

    fn num(&mut self, b: u8) -> Result<u32> {
        let num_bytes: u8 = b & 15;
        let mut val: u32 = 0;
        for i in 0..num_bytes {
            val += (self.pop()? as u32) << (8 * i);
        }
        Ok(val)
    }

    fn reg(&mut self) -> Result<Reg> {
        let b = self.pop()?;

        let r = match b & 0xc0 {
            REG_GENERAL => Reg::Gen(b & 0x3f),
            REG_STATIC => Reg::Static(self.num(b)?),
            REG_SPECIAL => match b {
                REG_RET => Reg::Ret,
                REG_TEMP => Reg::Temp,
                REG_LEFT => Reg::Left,
                REG_RIGHT => Reg::Right,
                _ => return Err(anyhow!("undefined Reg type")),
            },
            _ => return Err(anyhow!("undefined Reg type")),
        };

        Ok(r)
    }

    fn loc(&mut self) -> Result<Loc> {
        let b = self.pop()?;

        let loc = match b & 0xc0 {
            LOC_MEM => Loc::Mem(self.num(b)?),
            LOC_PARAM => Loc::Param(self.num(b)?),
            LOC_STACK => Loc::Stack(self.num(b)?),
            _ => return Err(anyhow!("undefined Loc type")),
        };

        Ok(loc)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.pop()? as usize;
        let mut b: Vec<u8> = Vec::with_capacity(len);
        for _ in 0..len {
            b.push(self.pop()?)
        }

        Ok(String::from_utf8(b)?)
    }
}

impl Storage for Mir {
    fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&Self::MAGIC.to_le_bytes())?;

        let mut writer = MirWriter::new();
        writer.serialize(self);

        stream.write_all(&writer.buf.len().to_le_bytes())?;
        stream.write_all(&writer.buf)?;

        stream.write_all(&self.consts.len().to_le_bytes())?;

        for x in self.consts.iter() {
            stream.write_all(&x.to_le_bytes())?;
        }

        Ok(())
    }

    fn load(stream: &mut impl Read, config: &Config) -> Result<Self> {
        let mut bytes: [u8; 8] = [0; 8];

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != Self::MAGIC {
            return Err(anyhow!("invalid magic number (Mir)"));
        }

        stream.read_exact(&mut bytes)?;
        let num_bytes = usize::from_le_bytes(bytes);

        let mut buf: Vec<u8> = vec![0; num_bytes];
        stream.read_exact(&mut buf)?;

        let mut mir = Mir::new(config.clone());
        let mut reader: MirReader = MirReader::new(buf);
        reader.deserialize(&mut mir)?;

        stream.read_exact(&mut bytes)?;
        let num_consts = usize::from_le_bytes(bytes);

        let mut buf = [0; 8];
        let mut consts: Vec<f64> = Vec::new();

        for _ in 0..num_consts {
            stream.read_exact(&mut buf)?;
            consts.push(f64::from_le_bytes(buf));
        }

        mir.add_consts(&consts);
        mir.populate_labels();

        Ok(mir)
    }
}
