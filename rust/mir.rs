use std::fmt;
use std::fs;
use std::io::Write;

use crate::code::{Func, VirtualTable};
use crate::generator::Generator;
use crate::symbol::Loc;
use crate::utils::{bool_to_f64, Compiled, CompiledFunc, Reg};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniOp {
    Abs,
    Not,
    Neg,
    Recip,
    Root,
    Round,
    Floor,
    Ceiling,
    Trunc,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    Plus,
    Minus,
    Times,
    Divide,
    GreaterThan,
    GreaterThanEqual,
    LittleThan,
    LittleThanEqual,
    Equal,
    NotEqual,
    And,
    AndNot,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FusedOp {
    MulAdd,    // + a * b + c
    NegMulAdd, // - a * b + c
    MulSub,    // a * b - c
    NegMulSub, // -a * b - c
}

#[derive(Clone)]
pub enum Instruction {
    Nop,
    Uni {
        op: UniOp,
        dst: Reg,
        s1: Reg,
    },
    Bi {
        op: BinOp,
        dst: Reg,
        s1: Reg,
        s2: Reg,
    },
    Mov {
        dst: Reg,
        s1: Reg,
    },
    Xchg {
        s1: Reg,
        s2: Reg,
    },
    Load {
        dst: Reg,
        loc: Loc,
    },
    Save {
        dst: Reg,
        loc: Loc,
    },
    LoadConst {
        dst: Reg,
        idx: u32,
    },
    Call {
        label: String,
        f: Func,
    },
    Fused {
        op: FusedOp,
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
    },
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Nop => write!(f, "nop"),
            Instruction::Uni { op, dst, s1 } => write!(f, "{:?} := {:?}({:?})", &dst, &op, &s1),
            Instruction::Bi { op, dst, s1, s2 } => {
                write!(f, "{:?} := {:?} {:?} {:?}", &dst, &s1, &op, &s2)
            }
            Instruction::Call { label, .. } => write!(f, "call {}", &label),
            Instruction::Mov { dst, s1 } => write!(f, "{:?} := {:?}", &dst, &s1),
            Instruction::Xchg { s1, s2 } => write!(f, "xchg {:?} and {:?}", &s1, &s2),
            Instruction::Load { dst, loc } => write!(f, "{:?} := {:?}", &dst, &loc),
            Instruction::Save { dst, loc } => write!(f, "{:?} := {:?}", &loc, &dst),
            Instruction::LoadConst { dst, idx } => write!(f, "{:?} := consts[{:?}]", &dst, idx),
            Instruction::Fused { op, dst, a, b, c } => match op {
                FusedOp::MulAdd => write!(f, "{:?} := {:?} * {:?} + {:?}", &dst, &a, &b, &c),
                FusedOp::NegMulAdd => write!(f, "{:?} := - {:?} * {:?} + {:?}", &dst, &a, &b, &c),
                FusedOp::MulSub => write!(f, "{:?} := {:?} * {:?} - {:?}", &dst, &a, &b, &c),
                FusedOp::NegMulSub => write!(f, "{:?} := - {:?} * {:?} - {:?}", &dst, &a, &b, &c),
            },
        }
    }
}

#[derive(Clone)]
pub struct Mir {
    code: Vec<Instruction>,
    consts: Vec<f64>,
    fastmath: bool,
}

impl fmt::Debug for Mir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ins) in self.code.iter().enumerate() {
            writeln!(f, "{:05}\t{:?}", i, ins)?;
        }
        Ok(())
    }
}

impl Mir {
    pub fn new(fastmath: bool) -> Mir {
        Mir {
            code: Vec::new(),
            consts: Vec::new(),
            fastmath,
        }
    }

    fn push(&mut self, ins: Instruction) {
        self.code.push(ins)
    }
}

impl Mir {
    pub fn three_address(&self) -> bool {
        true
    }

    pub fn add_consts(&mut self, consts: &[f64]) {
        self.consts = consts.to_owned();
    }

    pub fn nop(&mut self) {
        self.push(Instruction::Nop);
    }

    pub fn fmov(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Mov { dst, s1 });
    }

    pub fn fxchg(&mut self, s1: Reg, s2: Reg) {
        self.push(Instruction::Xchg { s1, s2 });
    }

    pub fn load_const(&mut self, dst: Reg, idx: u32) {
        self.push(Instruction::LoadConst { dst, idx })
    }

    pub fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.push(Instruction::Load {
            dst,
            loc: Loc::Mem(idx),
        });
    }

    pub fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.push(Instruction::Save {
            dst,
            loc: Loc::Mem(idx),
        });
    }

    pub fn load_param(&mut self, dst: Reg, idx: u32) {
        self.push(Instruction::Load {
            dst,
            loc: Loc::Param(idx),
        });
    }

    pub fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.push(Instruction::Load {
            dst,
            loc: Loc::Stack(idx),
        });
    }

    pub fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.push(Instruction::Save {
            dst,
            loc: Loc::Stack(idx),
        });
    }

    pub fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    pub fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    pub fn neg(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Neg,
            dst,
            s1,
        });
    }

    pub fn abs(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Abs,
            dst,
            s1,
        });
    }

    pub fn root(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Root,
            dst,
            s1,
        });
    }
    pub fn square(&mut self, dst: Reg, s1: Reg) {
        self.times(dst, s1, s1);
    }

    pub fn cube(&mut self, dst: Reg, s1: Reg) {
        self.times(Reg::Temp, s1, s1);
        self.times(dst, s1, Reg::Temp);
    }

    pub fn recip(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Recip,
            dst,
            s1,
        });
    }

    pub fn not(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Not,
            dst,
            s1,
        });
    }

    pub fn round(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Round,
            dst,
            s1,
        });
    }

    pub fn floor(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Floor,
            dst,
            s1,
        });
    }

    pub fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Ceiling,
            dst,
            s1,
        });
    }

    pub fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Trunc,
            dst,
            s1,
        });
    }

    pub fn frac(&mut self, dst: Reg, s1: Reg) {
        self.floor(Reg::Temp, s1);
        self.minus(dst, s1, Reg::Temp);
    }

    pub fn fmod(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        assert!(dst != Reg::Ret && s1 != Reg::Ret && s2 != Reg::Ret);
        self.divide(Reg::Ret, s1, s2);
        self.floor(Reg::Ret, Reg::Ret);
        self.times(Reg::Ret, Reg::Ret, s2);
        self.minus(dst, s1, Reg::Ret);
    }

    pub fn powi(&mut self, dst: Reg, s1: Reg, power: i32) {
        if power == 0 {
            self.divide(dst, dst, dst); // this is a generic way to make 1, but should be
                                        // overrided by the calling Generator for efficiency
        } else if power > 0 {
            let t = power.trailing_zeros();
            let mut n = power >> (t + 1);
            let mut s = s1;

            // nop is required to prevent a bug caused by load/mov peephole optimization
            self.nop();

            self.fmov(dst, s1);

            while n > 0 {
                self.times(Reg::Temp, s, s);
                s = Reg::Temp;

                if n & 1 != 0 {
                    self.times(dst, dst, Reg::Temp);
                };
                n >>= 1;
            }

            for _ in 0..t {
                self.times(dst, dst, dst);
            }
        } else {
            self.powi(dst, s1, -power);
            self.recip(dst, dst);
        }
    }

    pub fn powi_mod(&mut self, dst: Reg, s1: Reg, power: i32, modulus: Reg) {
        assert!(dst != Reg::Ret && s1 != Reg::Ret);

        if power == 0 {
            self.divide(dst, dst, dst); // this is a generic way to make 1, but should be
                                        // overrided by the calling Generator for efficiency
        } else if power > 0 {
            let t = power.trailing_zeros();
            let mut n = power >> (t + 1);
            let mut s = s1;

            // nop is required to prevent a bug caused by load/mov peephole optimization
            self.nop();

            self.fmov(dst, s);

            while n > 0 {
                self.times(Reg::Temp, s, s);
                self.fmod(Reg::Temp, Reg::Temp, modulus);
                s = Reg::Temp;

                if n & 1 != 0 {
                    self.times(dst, dst, Reg::Temp);
                    self.fmod(dst, dst, modulus);
                };
                n >>= 1;
            }

            for _ in 0..t {
                self.times(dst, dst, dst);
                self.fmod(dst, dst, modulus);
            }
        } else {
            self.powi(dst, s1, -power);
            self.recip(dst, dst);
        }
    }

    pub fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Plus,
            dst,
            s1,
            s2,
        });
    }

    pub fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Minus,
            dst,
            s1,
            s2,
        });
    }

    pub fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Times,
            dst,
            s1,
            s2,
        });
    }

    pub fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Divide,
            dst,
            s1,
            s2,
        });
    }

    pub fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::GreaterThan,
            dst,
            s1,
            s2,
        });
    }

    pub fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::GreaterThanEqual,
            dst,
            s1,
            s2,
        });
    }

    pub fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::LittleThan,
            dst,
            s1,
            s2,
        });
    }

    pub fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::LittleThanEqual,
            dst,
            s1,
            s2,
        });
    }

    pub fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Equal,
            dst,
            s1,
            s2,
        });
    }

    pub fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::NotEqual,
            dst,
            s1,
            s2,
        });
    }

    pub fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::And,
            dst,
            s1,
            s2,
        });
    }

    pub fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::AndNot,
            dst,
            s1,
            s2,
        });
    }

    pub fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Or,
            dst,
            s1,
            s2,
        });
    }

    pub fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Xor,
            dst,
            s1,
            s2,
        });
    }

    pub fn setup_call_unary(&mut self, s1: Reg) {
        if s1 != Reg::Left {
            self.fmov(Reg::Left, s1);
        };
    }

    pub fn setup_call_binary(&mut self, s1: Reg, s2: Reg) {
        if s1 == Reg::Right && s2 == Reg::Left {
            self.fxchg(Reg::Right, Reg::Left);
        } else if s2 == Reg::Left {
            self.fmov(Reg::Right, Reg::Left);
            if s1 != Reg::Left {
                self.fmov(Reg::Left, s1);
            }
        } else {
            if s2 != Reg::Right {
                self.fmov(Reg::Right, s2);
            }
            if s1 != Reg::Left {
                self.fmov(Reg::Left, s1);
            }
        };
    }

    pub fn call(&mut self, op: &str, num_args: usize) {
        let f = VirtualTable::from_str(op).expect("func not found");

        match f {
            Func::Unary(_) => assert!(num_args == 1),
            Func::Binary(_) => assert!(num_args == 2),
        }

        self.push(Instruction::Call {
            f,
            label: op.to_string(),
        });
    }

    pub fn select_if(&mut self, dst: Reg, cond: Reg, s1: Reg) {
        self.and(dst, cond, s1);
    }

    pub fn select_else(&mut self, dst: Reg, cond: Reg, s1: Reg) {
        self.andnot(dst, cond, s1);
    }
}

impl Mir {
    fn get(regs: &[f64], r: Reg) -> f64 {
        match r {
            Reg::Ret | Reg::Left => regs[0],
            Reg::Temp | Reg::Right => regs[1],
            Reg::Gen(r) => regs[r as usize + 2],
        }
    }

    fn set(regs: &mut [f64], r: Reg, val: f64) {
        match r {
            Reg::Ret | Reg::Left => {
                regs[0] = val;
            }
            Reg::Temp | Reg::Right => {
                regs[1] = val;
            }
            Reg::Gen(r) => {
                regs[r as usize + 2] = val;
            }
        }
    }

    fn exec_uniop(regs: &mut [f64], op: UniOp, dst: Reg, s1: Reg) {
        let s1 = Self::get(regs, s1);

        let val = match op {
            UniOp::Neg => -s1,
            UniOp::Not => f64::from_bits(!s1.to_bits()),
            UniOp::Abs => s1.abs(),
            UniOp::Root => s1.sqrt(),
            UniOp::Recip => 1.0 / s1,
            UniOp::Round => s1.round(),
            UniOp::Floor => s1.floor(),
            UniOp::Ceiling => s1.ceil(),
            UniOp::Trunc => s1.trunc(),
        };

        Self::set(regs, dst, val);
    }

    fn exec_binop(regs: &mut [f64], op: BinOp, dst: Reg, s1: Reg, s2: Reg) {
        let s1 = Self::get(regs, s1);
        let s2 = Self::get(regs, s2);

        let val = match op {
            BinOp::Plus => s1 + s2,
            BinOp::Minus => s1 - s2,
            BinOp::Times => s1 * s2,
            BinOp::Divide => s1 / s2,
            BinOp::GreaterThan => bool_to_f64(s1 > s2),
            BinOp::GreaterThanEqual => bool_to_f64(s1 >= s2),
            BinOp::LittleThan => bool_to_f64(s1 < s2),
            BinOp::LittleThanEqual => bool_to_f64(s1 <= s2),
            BinOp::Equal => bool_to_f64(s1 == s2),
            BinOp::NotEqual => bool_to_f64(s1 != s2),
            BinOp::And => f64::from_bits(s1.to_bits() & s2.to_bits()),
            BinOp::AndNot => f64::from_bits(!s1.to_bits() & s2.to_bits()),
            BinOp::Or => f64::from_bits(s1.to_bits() | s2.to_bits()),
            BinOp::Xor => f64::from_bits(s1.to_bits() ^ s2.to_bits()),
        };

        Self::set(regs, dst, val);
    }

    fn exec_fused(regs: &mut [f64], op: FusedOp, dst: Reg, a: Reg, b: Reg, c: Reg) {
        let a = Self::get(regs, a);
        let b = Self::get(regs, b);
        let c = Self::get(regs, c);

        let val = match op {
            FusedOp::MulAdd => a * b + c,
            FusedOp::MulSub => a * b - c,
            FusedOp::NegMulAdd => -a * b + c,
            FusedOp::NegMulSub => -a * b - c,
        };

        Self::set(regs, dst, val);
    }

    fn exec_instruction(
        &mut self,
        mem: &mut [f64],
        stack: &mut [f64],
        regs: &mut [f64],
        params: &[f64],
    ) {
        for ins in self.code.iter() {
            match ins {
                Instruction::Nop => {}
                Instruction::Uni { op, dst, s1 } => {
                    Self::exec_uniop(regs, *op, *dst, *s1);
                }
                Instruction::Bi { op, dst, s1, s2 } => {
                    Self::exec_binop(regs, *op, *dst, *s1, *s2);
                }
                Instruction::Mov { dst, s1 } => {
                    let x = Self::get(regs, *s1);
                    Self::set(regs, *dst, x);
                }
                Instruction::Xchg { s1, s2 } => {
                    let x1 = Self::get(regs, *s1);
                    let x2 = Self::get(regs, *s2);
                    Self::set(regs, *s1, x2);
                    Self::set(regs, *s2, x1);
                }
                Instruction::Load { dst, loc } => {
                    let val = match loc {
                        Loc::Mem(idx) => mem[*idx as usize],
                        Loc::Stack(idx) => stack[*idx as usize],
                        Loc::Param(idx) => params[*idx as usize],
                    };
                    Self::set(regs, *dst, val);
                }
                Instruction::Save { dst, loc } => {
                    let val = Self::get(regs, *dst);
                    match loc {
                        Loc::Mem(idx) => {
                            mem[*idx as usize] = val;
                        }
                        Loc::Stack(idx) => {
                            stack[*idx as usize] = val;
                        }
                        Loc::Param(_) => {
                            unreachable!()
                        }
                    };
                }
                Instruction::LoadConst { dst, idx } => {
                    Self::set(regs, *dst, self.consts[*idx as usize]);
                }
                Instruction::Call { f, .. } => match f {
                    Func::Unary(p) => Self::set(regs, Reg::Ret, p(Self::get(regs, Reg::Left))),
                    Func::Binary(p) => Self::set(
                        regs,
                        Reg::Ret,
                        p(Self::get(regs, Reg::Left), Self::get(regs, Reg::Right)),
                    ),
                },
                Instruction::Fused { op, dst, a, b, c } => {
                    Self::exec_fused(regs, *op, *dst, *a, *b, *c);
                }
            }
        }
    }
}

impl Mir {
    fn rerun_uniop(ir: &mut dyn Generator, op: UniOp, dst: Reg, s1: Reg) {
        match op {
            UniOp::Neg => ir.neg(dst, s1),
            UniOp::Not => ir.not(dst, s1),
            UniOp::Abs => ir.abs(dst, s1),
            UniOp::Root => ir.root(dst, s1),
            UniOp::Recip => ir.recip(dst, s1),
            UniOp::Round => ir.round(dst, s1),
            UniOp::Floor => ir.floor(dst, s1),
            UniOp::Ceiling => ir.ceiling(dst, s1),
            UniOp::Trunc => ir.trunc(dst, s1),
        };
    }

    fn rerun_binop(ir: &mut dyn Generator, op: BinOp, dst: Reg, s1: Reg, s2: Reg) {
        match op {
            BinOp::Plus => ir.plus(dst, s1, s2),
            BinOp::Minus => ir.minus(dst, s1, s2),
            BinOp::Times => ir.times(dst, s1, s2),
            BinOp::Divide => ir.divide(dst, s1, s2),
            BinOp::GreaterThan => ir.gt(dst, s1, s2),
            BinOp::GreaterThanEqual => ir.geq(dst, s1, s2),
            BinOp::LittleThan => ir.lt(dst, s1, s2),
            BinOp::LittleThanEqual => ir.leq(dst, s1, s2),
            BinOp::Equal => ir.eq(dst, s1, s2),
            BinOp::NotEqual => ir.neq(dst, s1, s2),
            BinOp::And => ir.and(dst, s1, s2),
            BinOp::AndNot => ir.andnot(dst, s1, s2),
            BinOp::Or => ir.or(dst, s1, s2),
            BinOp::Xor => ir.xor(dst, s1, s2),
        };
    }

    pub fn rerun(&self, ir: &mut dyn Generator) {
        for ins in self.code.iter() {
            match ins {
                Instruction::Nop => {}
                Instruction::Uni { op, dst, s1 } => {
                    Self::rerun_uniop(ir, *op, *dst, *s1);
                }
                Instruction::Bi { op, dst, s1, s2 } => {
                    Self::rerun_binop(ir, *op, *dst, *s1, *s2);
                }
                Instruction::Mov { dst, s1 } => {
                    if *dst != *s1 {
                        ir.fmov(*dst, *s1);
                    }
                }
                Instruction::Xchg { s1, s2 } => {
                    if *s1 != *s2 {
                        ir.fxchg(*s1, *s2);
                    }
                }
                Instruction::Load { dst, loc } => {
                    match loc {
                        Loc::Mem(idx) => ir.load_mem(*dst, *idx),
                        Loc::Stack(idx) => ir.load_stack(*dst, *idx),
                        Loc::Param(idx) => ir.load_param(*dst, *idx),
                    };
                }
                Instruction::Save { dst, loc } => {
                    match loc {
                        Loc::Mem(idx) => ir.save_mem(*dst, *idx),
                        Loc::Stack(idx) => ir.save_stack(*dst, *idx),
                        Loc::Param(_) => unreachable!(),
                    };
                }
                Instruction::LoadConst { dst, idx } => {
                    ir.load_const(*dst, *idx);
                }
                Instruction::Call { label, f } => match f {
                    Func::Unary(_) => ir.call(label, 1),
                    Func::Binary(_) => ir.call(label, 2),
                },
                Instruction::Fused { op, dst, a, b, c } => match op {
                    FusedOp::MulAdd => ir.fused_mul_add(*dst, *a, *b, *c),
                    FusedOp::MulSub => ir.fused_mul_sub(*dst, *a, *b, *c),
                    FusedOp::NegMulAdd => ir.fused_neg_mul_add(*dst, *a, *b, *c),
                    FusedOp::NegMulSub => ir.fused_neg_mul_sub(*dst, *a, *b, *c),
                },
            }
        }
    }
}

impl Mir {
    fn fuse_op_mov(&self, code: &mut Vec<Instruction>, q0: &Instruction, q1: &Instruction) -> bool {
        /*
         * example:
         *      %0 = Root(%2)
         *      %l = %0
         *      call power
         * becomes
         *      %l = Root(%2)
         *      call power
         */
        if let Instruction::Uni { op, dst, s1 } = *q0 {
            if let Instruction::Mov {
                dst: dst_q1,
                s1: s1_q1,
            } = *q1
            {
                if dst == s1_q1 {
                    code.push(Instruction::Uni {
                        op,
                        dst: dst_q1,
                        s1,
                    });
                    return true;
                }
            }
        };

        /*
         * example:
         *      %0 = %2 Plus %3
         *      %l = %0
         *      call power
         * becomes
         *      %l = %2 Plus %3
         *      call power
         */
        if let Instruction::Bi { op, dst, s1, s2 } = *q0 {
            if let Instruction::Mov {
                dst: dst_q1,
                s1: s1_q1,
            } = *q1
            {
                if dst == s1_q1 {
                    code.push(Instruction::Bi {
                        op,
                        dst: dst_q1,
                        s1,
                        s2,
                    });
                    return true;
                }
            }
        };

        false
    }

    fn fuse_load(&self, code: &mut Vec<Instruction>, q0: &Instruction, q1: &Instruction) -> bool {
        /*
         * example
         *      %0 := Stack[2]
         *      %2 := %0
         * becomes
         *      %2 := Stack[2]
         *
         * note that we assume %0 is not needed anymore. This was not true for powi and
         * powi_mod; therefore, we added nop to prevent this rule from firing for those
         * functions.
         */
        if let Instruction::Load { dst, loc } = *q0 {
            if let Instruction::Mov {
                dst: dst_q1,
                s1: s1_q1,
            } = *q1
            {
                if dst == s1_q1 {
                    code.push(Instruction::Load { dst: dst_q1, loc });
                    return true;
                }
            }
        };

        if let Instruction::LoadConst { dst, idx } = *q0 {
            if let Instruction::Mov {
                dst: dst_q1,
                s1: s1_q1,
            } = *q1
            {
                if dst == s1_q1 {
                    code.push(Instruction::LoadConst { dst: dst_q1, idx });
                    return true;
                }
            }
        };

        false
    }

    fn fuse_save(&self, code: &mut Vec<Instruction>, q0: &Instruction, q1: &Instruction) -> bool {
        /*
         * example
         *      %0 := %1
         *      Mem[4] = %0
         * becomes
         *      Mem[4] := %1
         */
        if let Instruction::Mov { dst, s1 } = *q0 {
            if let Instruction::Save {
                dst: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if dst == dst_q1 {
                    code.push(Instruction::Save {
                        dst: s1,
                        loc: loc_q1,
                    });
                    return true;
                }
            }
        }

        /*
         * example
         *      Stack[6] = %2
         *      %0 = Stack[6]
         * becomes
         *      Stack[6] = %2
         *      %0 := %2
         *
         * note that if we know that Stack[6] is not accessed again, we can remove the
         * first instruction, but this is not yet implemented.
         */
        if let Instruction::Save { dst, loc } = *q0 {
            if let Instruction::Load {
                dst: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if loc == loc_q1 {
                    code.push(q0.clone());
                    code.push(Instruction::Mov {
                        dst: dst_q1,
                        s1: dst,
                    });
                    return true;
                }
            }
        };

        false
    }

    fn fuse_save3(
        &self,
        code: &mut Vec<Instruction>,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> bool {
        /*
         * this combination happens in return from remote function calls
         * examples:
         *      call sin
         *      Stack[10] = %$
         *      %0 = Stack[10]
         *      Mem[5] = %0
         * becomes
         *      call sin
         *      Mem[5] = %$
         */
        if let Instruction::Save { dst, loc } = *q0 {
            if let Instruction::Load {
                dst: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if let Instruction::Save {
                    dst: dst_q2,
                    loc: loc_q2,
                } = *q2
                {
                    if dst == Reg::Ret && loc == loc_q1 && dst_q1 == dst_q2 {
                        code.push(Instruction::Save {
                            dst: Reg::Ret,
                            loc: loc_q2,
                        });
                        return true;
                    }
                }
            }
        };

        false
    }

    fn fuse_fma(&self, code: &mut Vec<Instruction>, q0: &Instruction, q1: &Instruction) -> bool {
        if let Instruction::Bi { op, dst, s1, s2 } = *q0 {
            if let Instruction::Bi {
                op: op_q1,
                dst: dst_q1,
                s1: s1_q1,
                s2: s2_q1,
            } = *q1
            {
                if op == BinOp::Times && op_q1 == BinOp::Plus && s1_q1 == dst {
                    code.push(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s2_q1,
                    });
                    return true;
                }

                if op == BinOp::Times && op_q1 == BinOp::Plus && s2_q1 == dst {
                    code.push(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s1_q1,
                    });
                    return true;
                }

                if op == BinOp::Times && op_q1 == BinOp::Minus && s1_q1 == dst {
                    code.push(Instruction::Fused {
                        op: FusedOp::MulSub,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s2_q1,
                    });
                    return true;
                }

                if op == BinOp::Times && op_q1 == BinOp::Minus && s2_q1 == dst {
                    code.push(Instruction::Fused {
                        op: FusedOp::NegMulAdd,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s1_q1,
                    });
                    return true;
                }
            }
        }

        false
    }

    fn fuse_fma3(
        &self,
        code: &mut Vec<Instruction>,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> bool {
        if let Instruction::Bi { op, dst, s1, s2 } = *q0 {
            if let Instruction::LoadConst {
                dst: dst_q1,
                idx: idx_q1,
            } = *q1
            {
                if let Instruction::Bi {
                    op: op_q2,
                    dst: dst_q2,
                    s1: s1_q2,
                    s2: s2_q2,
                } = *q2
                {
                    if op == BinOp::Times
                        && op_q2 == BinOp::Plus
                        && ((s1_q2 == dst && s2_q2 == dst_q1) || (s1_q2 == dst_q1 && s2_q2 == dst))
                    {
                        code.push(Instruction::LoadConst {
                            dst: Reg::Temp,
                            idx: idx_q1,
                        });
                        code.push(Instruction::Fused {
                            op: FusedOp::MulAdd,
                            dst: dst_q2,
                            a: s1,
                            b: s2,
                            c: Reg::Temp,
                        });
                        return true;
                    }
                }
            }
        }

        if let Instruction::Bi { op, dst, s1, s2 } = *q0 {
            if let Instruction::Load {
                dst: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if let Instruction::Bi {
                    op: op_q2,
                    dst: dst_q2,
                    s1: s1_q2,
                    s2: s2_q2,
                } = *q2
                {
                    if op == BinOp::Times
                        && op_q2 == BinOp::Plus
                        && ((s1_q2 == dst && s2_q2 == dst_q1) || (s1_q2 == dst_q1 && s2_q2 == dst))
                    {
                        code.push(Instruction::Load {
                            dst: Reg::Temp,
                            loc: loc_q1,
                        });
                        code.push(Instruction::Fused {
                            op: FusedOp::MulAdd,
                            dst: dst_q2,
                            a: s1,
                            b: s2,
                            c: Reg::Temp,
                        });
                        return true;
                    }
                }
            }
        }

        false
    }

    fn fuse(
        &self,
        code: &mut Vec<Instruction>,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> usize {
        if self.fuse_op_mov(code, q0, q1)
            || self.fuse_load(code, q0, q1)
            || self.fuse_save(code, q0, q1)
            || (self.fastmath && self.fuse_fma(code, q0, q1))
        {
            2
        } else if self.fuse_save3(code, q0, q1, q2)
            || (self.fastmath && self.fuse_fma3(code, q0, q1, q2))
        {
            3
        } else {
            code.push(q0.clone());
            1
        }
    }

    pub fn optimize_peephole(&mut self) {
        self.push(Instruction::Nop);
        self.push(Instruction::Nop);
        let mut code: Vec<Instruction> = Vec::new();

        let mut i = 0;

        while i < self.code.len() - 2 {
            let d = self.fuse(
                &mut code,
                &self.code[i],
                &self.code[i + 1],
                &self.code[i + 2],
            );
            i += d;
        }

        self.code = code;
    }
}

/********************************************************/

pub struct CompiledMir {
    mir: Mir,
    mem: Vec<f64>,
    stack: Vec<f64>,
    regs: Vec<f64>,
}

impl CompiledMir {
    pub fn new(mir: Mir, mem: Vec<f64>, stack: Vec<f64>) -> CompiledMir {
        let regs = vec![0.0; 16];

        CompiledMir {
            mir,
            mem,
            stack,
            regs,
        }
    }
}

impl Compiled<f64> for CompiledMir {
    fn exec(&mut self, params: &[f64]) {
        self.mir
            .exec_instruction(&mut self.mem, &mut self.stack, &mut self.regs, params);
    }

    fn mem(&self) -> &[f64] {
        &self.mem[..]
    }

    fn mem_mut(&mut self) -> &mut [f64] {
        &mut self.mem[..]
    }

    fn dump(&self, name: &str) {
        let mut fs = fs::File::create(name).unwrap();
        let _ = writeln!(fs, "#!");
        let _ = write!(fs, "{:?}", self.mir);
    }

    fn func(&self) -> CompiledFunc<f64> {
        unreachable!()
    }

    fn support_indirect(&self) -> bool {
        false
    }
}
