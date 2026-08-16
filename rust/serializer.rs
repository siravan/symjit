use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::io::{Read, Write};

use super::config::Config;
use super::mir::{ArithOp, BinOp, FusedOp, Instruction, Mir, UniOp};
use super::symbol::Loc;
use super::utils::*;

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
const LOAD_ARG: u8 = 14;
const SAVE_ARG: u8 = 15;

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
const UNIOP_HALF: u8 = UniOp::Half as u8;
const UNIOP_ISZERO: u8 = UniOp::IsZero as u8;
const UNIOP_ISNOTZERO: u8 = UniOp::IsNotZero as u8;

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

#[derive(Clone)]
pub struct MirWriter {
    buf: Vec<u8>,
    pub ip: usize,
}

impl Default for MirWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl MirWriter {
    pub fn new() -> MirWriter {
        MirWriter {
            buf: Vec::new(),
            ip: 0,
        }
    }

    pub fn push(&mut self, ins: &Instruction) {
        self.ip += 1;
        match ins {
            Instruction::Nop => self.append_byte(NOP),
            Instruction::End => self.append_byte(END),
            Instruction::Uni { op, dst, s1 } => {
                let op = *op as u8;
                assert!(op < 64);
                self.append_byte(UNI_OP | op);
                self.reg(*dst);
                self.reg(*s1);
            }
            Instruction::Bi { op, dst, s1, s2 } => {
                let op = *op as u8;
                assert!(op < 64);
                self.append_byte(BIN_OP | op);
                self.reg(*dst);
                self.reg(*s1);
                self.reg(*s2);
            }
            Instruction::Mov { dst, s1 } => {
                self.append_byte(MOV);
                self.reg(*dst);
                self.reg(*s1);
            }
            Instruction::Load { dst, loc } => {
                self.append_byte(LOAD);
                self.reg(*dst);
                self.loc(*loc);
            }
            Instruction::LoadMath { op, dst, s1, loc } => {
                let op = *op as u8;
                assert!(op < 4);
                self.append_byte(LOAD_MATH_PLUS + op);
                self.reg(*dst);
                self.reg(*s1);
                self.loc(*loc);
            }
            Instruction::Save { src, loc } => {
                self.append_byte(SAVE);
                self.reg(*src);
                self.loc(*loc);
            }
            Instruction::LoadConst { dst, idx } => {
                self.append_byte(LOAD_CONST);
                self.reg(*dst);
                self.num(0, *idx);
            }
            Instruction::LoadComplex { xd, yd, loc } => {
                self.append_byte(LOAD_COMPLEX);
                self.reg(*xd);
                self.reg(*yd);
                self.loc(*loc);
            }
            Instruction::SaveComplex { xs, ys, loc } => {
                self.append_byte(SAVE_COMPLEX);
                self.reg(*xs);
                self.reg(*ys);
                self.loc(*loc);
            }
            Instruction::LoadConstMath { op, dst, s1, idx } => {
                let op = *op as u8;
                assert!(op < 4);
                self.append_byte(LOAD_CONST_MATH_PLUS + op);
                self.reg(*dst);
                self.reg(*s1);
                self.num(0, *idx);
            }
            Instruction::LoadArg {
                arg,
                loc,
                complex: false,
            } => {
                self.append_byte(LOAD_ARG);
                self.append_byte(*arg);
                self.loc(*loc);
            }
            Instruction::LoadArg {
                arg,
                loc,
                complex: true,
            } => {
                self.append_byte(LOAD_ARG);
                self.append_byte(*arg | 0x80);
                self.loc(*loc);
            }
            Instruction::SaveArg {
                arg,
                loc,
                complex: false,
            } => {
                self.append_byte(SAVE_ARG);
                self.append_byte(*arg);
                self.loc(*loc);
            }
            Instruction::SaveArg {
                arg,
                loc,
                complex: true,
            } => {
                self.append_byte(SAVE_ARG);
                self.append_byte(*arg | 0x80);
                self.loc(*loc);
            }
            Instruction::Branch { label } => {
                self.append_byte(BRANCH);
                self.string(label);
            }
            Instruction::BranchIf {
                cond,
                label,
                is_else,
            } => {
                if *is_else {
                    self.append_byte(BRANCH_ELSE);
                } else {
                    self.append_byte(BRANCH_IF);
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
                self.append_byte(IFELSE);
                self.reg(*dst);
                self.reg(*true_val);
                self.reg(*false_val);
                self.loc(*cond);
            }
            Instruction::Call {
                label, num_args, ..
            } => {
                self.append_byte(CALL);
                assert!(*num_args < 256);
                self.append_byte(*num_args as u8);
                self.string(label);
            }
            Instruction::Label { label } => {
                self.append_byte(LABEL);
                self.string(label);
            }
            Instruction::Fused { op, dst, a, b, c } => {
                let op = *op as u8;
                assert!(op < 4);
                self.append_byte(FUSED_MUL_ADD + op);
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
                self.append_byte(COMPLEX_BI_PLUS + op);
                self.reg(*xd);
                self.reg(*yd);
                self.reg(*x1);
                self.reg(*y1);
                self.reg(*x2);
                self.reg(*y2);
            }
        }
    }

    pub fn iter_mut(&mut self) -> MirIterator {
        let buf = std::mem::take(&mut self.buf);
        MirIterator { buf, pos: 0 }
    }

    pub fn iter(&self) -> MirIterator {
        // println!("{} instructions, {} bytes", self.ip, self.buf.len());
        let buf = self.buf.clone();
        MirIterator { buf, pos: 0 }
    }

    fn append_byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    pub fn serialize(&mut self, mir: &Mir) {
        for ins in mir.code.iter() {
            self.push(&ins);
        }
    }

    fn num(&mut self, prefix: u8, mut n: u32) {
        let num_bytes: u8 = 1 + ((32 - n.leading_zeros()) >> 3) as u8;
        self.append_byte(prefix | num_bytes);
        for _ in 0..num_bytes {
            self.append_byte((n & 0xff) as u8);
            n >>= 8;
        }
    }

    fn reg(&mut self, r: Reg) {
        match r {
            Reg::Ret => self.append_byte(REG_RET),
            Reg::Temp => self.append_byte(REG_TEMP),
            Reg::Left => self.append_byte(REG_LEFT),
            Reg::Right => self.append_byte(REG_RIGHT),
            Reg::Gen(r) => self.append_byte(REG_GENERAL | r),
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

        self.append_byte(len as u8);
        for b in bytes {
            self.append_byte(*b);
        }
    }
}

impl Storage for Mir {
    fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&Self::MAGIC.to_le_bytes())?;

        stream.write_all(&self.code.buf.len().to_le_bytes())?;
        stream.write_all(&self.code.buf)?;

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

        stream.read_exact(&mut bytes)?;
        let num_consts = usize::from_le_bytes(bytes);

        let mut consts: Vec<f64> = Vec::new();

        for _ in 0..num_consts {
            stream.read_exact(&mut bytes)?;
            consts.push(f64::from_le_bytes(bytes));
        }

        let mut iter = MirIterator::from_buf(&buf);
        let mut ip = 0;
        while iter.next().is_some() {
            ip += 1;
        }

        let mut mir = Mir {
            code: MirWriter { buf, ip },
            consts,
            labels: HashMap::new(),
            config: config.clone(),
        };

        // mir.add_consts(&consts);
        mir.populate_labels();

        Ok(mir)
    }
}

/* ********************************************** */

pub struct MirIterator {
    buf: Vec<u8>,
    pos: usize,
}

impl Iterator for MirIterator {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.eof() {
            let header = self.pop().unwrap();

            match header & 0xc0 {
                UNI_OP => self.uniop(header).ok(),
                BIN_OP => self.binop(header).ok(),
                _ => self.other(header).ok(),
            }
        } else {
            None
        }
    }
}

impl MirIterator {
    pub fn new(writer: &mut MirWriter) -> MirIterator {
        let buf = std::mem::take(&mut writer.buf);
        MirIterator { buf, pos: 0 }
    }

    pub fn from_buf(buf: &[u8]) -> MirIterator {
        MirIterator {
            buf: buf.to_vec(),
            pos: 0,
        }
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

    fn uniop(&mut self, header: u8) -> Result<Instruction> {
        let dst = self.reg()?;
        let s1 = self.reg()?;

        match header & 0x3f {
            UNIOP_NEG => Ok(Instruction::Uni {
                op: UniOp::Neg,
                dst,
                s1,
            }),
            UNIOP_NOT => Ok(Instruction::Uni {
                op: UniOp::Not,
                dst,
                s1,
            }),
            UNIOP_ABS => Ok(Instruction::Uni {
                op: UniOp::Abs,
                dst,
                s1,
            }),
            UNIOP_ROOT => Ok(Instruction::Uni {
                op: UniOp::Root,
                dst,
                s1,
            }),
            UNIOP_REALROOT => Ok(Instruction::Uni {
                op: UniOp::RealRoot,
                dst,
                s1,
            }),
            UNIOP_RECIP => Ok(Instruction::Uni {
                op: UniOp::Recip,
                dst,
                s1,
            }),
            UNIOP_ROUND => Ok(Instruction::Uni {
                op: UniOp::Round,
                dst,
                s1,
            }),
            UNIOP_FLOOR => Ok(Instruction::Uni {
                op: UniOp::Floor,
                dst,
                s1,
            }),
            UNIOP_CEILING => Ok(Instruction::Uni {
                op: UniOp::Ceiling,
                dst,
                s1,
            }),
            UNIOP_TRUNC => Ok(Instruction::Uni {
                op: UniOp::Trunc,
                dst,
                s1,
            }),
            UNIOP_REAL => Ok(Instruction::Uni {
                op: UniOp::Real,
                dst,
                s1,
            }),
            UNIOP_IMAGINARY => Ok(Instruction::Uni {
                op: UniOp::Imaginary,
                dst,
                s1,
            }),
            UNIOP_CONJUGATE => Ok(Instruction::Uni {
                op: UniOp::Conjugate,
                dst,
                s1,
            }),
            UNIOP_HALF => Ok(Instruction::Uni {
                op: UniOp::Half,
                dst,
                s1,
            }),
            UNIOP_ISZERO => Ok(Instruction::Uni {
                op: UniOp::IsZero,
                dst,
                s1,
            }),
            UNIOP_ISNOTZERO => Ok(Instruction::Uni {
                op: UniOp::IsNotZero,
                dst,
                s1,
            }),
            _ => Err(anyhow!("undefined UnaryOp")),
        }
    }

    fn binop(&mut self, header: u8) -> Result<Instruction> {
        let dst = self.reg()?;
        let s1 = self.reg()?;
        let s2 = self.reg()?;

        match header & 0x3f {
            BINOP_PLUS => Ok(Instruction::Bi {
                op: BinOp::Plus,
                dst,
                s1,
                s2,
            }),
            BINOP_MINUS => Ok(Instruction::Bi {
                op: BinOp::Minus,
                dst,
                s1,
                s2,
            }),
            BINOP_TIMES => Ok(Instruction::Bi {
                op: BinOp::Times,
                dst,
                s1,
                s2,
            }),
            BINOP_DIVIDE => Ok(Instruction::Bi {
                op: BinOp::Divide,
                dst,
                s1,
                s2,
            }),
            BINOP_GREATER_THAN => Ok(Instruction::Bi {
                op: BinOp::GreaterThan,
                dst,
                s1,
                s2,
            }),
            BINOP_GREATER_THAN_EQUAL => Ok(Instruction::Bi {
                op: BinOp::GreaterThanEqual,
                dst,
                s1,
                s2,
            }),
            BINOP_LITTLE_THAN => Ok(Instruction::Bi {
                op: BinOp::LittleThan,
                dst,
                s1,
                s2,
            }),
            BINOP_LITTLE_THAN_EQUAL => Ok(Instruction::Bi {
                op: BinOp::LittleThanEqual,
                dst,
                s1,
                s2,
            }),
            BINOP_EQUAL => Ok(Instruction::Bi {
                op: BinOp::Equal,
                dst,
                s1,
                s2,
            }),
            BINOP_NOT_EQUAL => Ok(Instruction::Bi {
                op: BinOp::NotEqual,
                dst,
                s1,
                s2,
            }),
            BINOP_AND => Ok(Instruction::Bi {
                op: BinOp::And,
                dst,
                s1,
                s2,
            }),
            BINOP_AND_NOT => Ok(Instruction::Bi {
                op: BinOp::AndNot,
                dst,
                s1,
                s2,
            }),
            BINOP_OR => Ok(Instruction::Bi {
                op: BinOp::Or,
                dst,
                s1,
                s2,
            }),
            BINOP_XOR => Ok(Instruction::Bi {
                op: BinOp::Xor,
                dst,
                s1,
                s2,
            }),
            BINOP_COMPLEX => Ok(Instruction::Bi {
                op: BinOp::Complex,
                dst,
                s1,
                s2,
            }),
            _ => Err(anyhow!("undefined BinaryOp")),
        }
    }

    pub fn other(&mut self, header: u8) -> Result<Instruction> {
        match header & 0x3f {
            NOP => Ok(Instruction::Nop),
            END => Ok(Instruction::End),
            MOV => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                Ok(Instruction::Mov { dst, s1 })
            }
            LOAD => {
                let dst = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::Load { dst, loc })
            }
            SAVE => {
                let src = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::Save { src, loc })
            }
            LOAD_CONST => {
                let dst = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                Ok(Instruction::LoadConst { dst, idx })
            }
            LOAD_COMPLEX => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::LoadComplex { xd, yd, loc })
            }
            SAVE_COMPLEX => {
                let xs = self.reg()?;
                let ys = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::SaveComplex { xs, ys, loc })
            }
            LOAD_ARG => {
                let arg = self.pop()?;
                let loc = self.loc()?;
                Ok(Instruction::LoadArg {
                    arg: arg & 0x7f,
                    loc,
                    complex: arg & 0x80 != 0,
                })
            }
            SAVE_ARG => {
                let arg = self.pop()?;
                let loc = self.loc()?;
                Ok(Instruction::SaveArg {
                    arg: arg & 0x7f,
                    loc,
                    complex: arg & 0x80 != 0,
                })
            }
            BRANCH => {
                let label = self.string()?;
                Ok(Instruction::Branch { label })
            }
            BRANCH_IF => {
                let cond = self.reg()?;
                let label = self.string()?;
                Ok(Instruction::BranchIf {
                    cond,
                    label,
                    is_else: false,
                })
            }
            BRANCH_ELSE => {
                let cond = self.reg()?;
                let label = self.string()?;
                Ok(Instruction::BranchIf {
                    cond,
                    label,
                    is_else: true,
                })
            }
            CALL => {
                let num_args = self.pop()? as usize;
                let op = self.string()?;
                Ok(Instruction::Call {
                    label: op.to_string(),
                    num_args,
                })
            }
            LABEL => {
                let label = self.string()?;
                Ok(Instruction::Label { label })
            }
            IFELSE => {
                let dst = self.reg()?;
                let true_val = self.reg()?;
                let false_val = self.reg()?;
                let cond = self.loc()?;
                Ok(Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                })
            }
            FUSED_MUL_ADD => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                Ok(Instruction::Fused {
                    op: FusedOp::MulAdd,
                    dst,
                    a,
                    b,
                    c,
                })
            }
            FUSED_MUL_SUB => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                Ok(Instruction::Fused {
                    op: FusedOp::MulSub,
                    dst,
                    a,
                    b,
                    c,
                })
            }
            FUSED_NEG_MUL_ADD => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                Ok(Instruction::Fused {
                    op: FusedOp::NegMulAdd,
                    dst,
                    a,
                    b,
                    c,
                })
            }
            FUSED_NEG_MUL_SUB => {
                let dst = self.reg()?;
                let a = self.reg()?;
                let b = self.reg()?;
                let c = self.reg()?;
                Ok(Instruction::Fused {
                    op: FusedOp::NegMulSub,
                    dst,
                    a,
                    b,
                    c,
                })
            }
            LOAD_MATH_PLUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::LoadMath {
                    op: ArithOp::Plus,
                    dst,
                    s1,
                    loc,
                })
            }
            LOAD_MATH_MINUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::LoadMath {
                    op: ArithOp::Minus,
                    dst,
                    s1,
                    loc,
                })
            }
            LOAD_MATH_TIMES => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::LoadMath {
                    op: ArithOp::Times,
                    dst,
                    s1,
                    loc,
                })
            }
            LOAD_MATH_DIVIDE => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let loc = self.loc()?;
                Ok(Instruction::LoadMath {
                    op: ArithOp::Divide,
                    dst,
                    s1,
                    loc,
                })
            }
            LOAD_CONST_MATH_PLUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                Ok(Instruction::LoadConstMath {
                    op: ArithOp::Plus,
                    dst,
                    s1,
                    idx,
                })
            }
            LOAD_CONST_MATH_MINUS => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                Ok(Instruction::LoadConstMath {
                    op: ArithOp::Minus,
                    dst,
                    s1,
                    idx,
                })
            }
            LOAD_CONST_MATH_TIMES => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                Ok(Instruction::LoadConstMath {
                    op: ArithOp::Times,
                    dst,
                    s1,
                    idx,
                })
            }
            LOAD_CONST_MATH_DIVIDE => {
                let dst = self.reg()?;
                let s1 = self.reg()?;
                let num_bytes = self.pop()?;
                let idx = self.num(num_bytes)?;
                Ok(Instruction::LoadConstMath {
                    op: ArithOp::Divide,
                    dst,
                    s1,
                    idx,
                })
            }
            COMPLEX_BI_PLUS => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                Ok(Instruction::ComplexBi {
                    op: ArithOp::Plus,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                })
            }
            COMPLEX_BI_MINUS => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                Ok(Instruction::ComplexBi {
                    op: ArithOp::Minus,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                })
            }
            COMPLEX_BI_TIMES => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                Ok(Instruction::ComplexBi {
                    op: ArithOp::Times,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                })
            }
            COMPLEX_BI_DIVIDE => {
                let xd = self.reg()?;
                let yd = self.reg()?;
                let x1 = self.reg()?;
                let y1 = self.reg()?;
                let x2 = self.reg()?;
                let y2 = self.reg()?;
                Ok(Instruction::ComplexBi {
                    op: ArithOp::Divide,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                })
            }
            _ => Err(anyhow!("undefined op code")),
        }
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
