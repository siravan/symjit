use std::fmt;
use std::fs;
use std::io::Write;

use crate::code::{Func, VirtualTable};
use crate::generator::Generator;
use crate::symbol::Loc;
use crate::utils::{bool_to_f64, Compiled, CompiledFunc, Reg};

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
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
        }
    }
}

#[derive(Clone)]
pub struct Mir {
    code: Vec<Instruction>,
    consts: Vec<f64>,
}

impl fmt::Debug for Mir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ins) in self.code.iter().enumerate() {
            write!(f, "{:05}\t{:?}\n", i, ins)?;
        }
        Ok(())
    }
}

impl Mir {
    pub fn new() -> Mir {
        Mir {
            code: Vec::new(),
            consts: Vec::new(),
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

    pub fn add_consts(&mut self, consts: &Vec<f64>) {
        self.consts = consts.clone();
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
            if s1 != Reg::Left {
                self.fmov(Reg::Left, s1);
            }
            if s2 != Reg::Right {
                self.fmov(Reg::Right, s2);
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
    fn get(regs: &Vec<f64>, r: Reg) -> f64 {
        match r {
            Reg::Ret | Reg::Left => regs[0],
            Reg::Temp | Reg::Right => regs[1],
            Reg::Gen(r) => regs[r as usize + 2],
        }
    }

    fn set(regs: &mut Vec<f64>, r: Reg, val: f64) {
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

    fn exec_uniop(regs: &mut Vec<f64>, op: UniOp, dst: Reg, s1: Reg) {
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

    fn exec_binop(regs: &mut Vec<f64>, op: BinOp, dst: Reg, s1: Reg, s2: Reg) {
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

    fn exec_instruction(
        &mut self,
        mem: &mut Vec<f64>,
        stack: &mut Vec<f64>,
        regs: &mut Vec<f64>,
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
                    ir.fmov(*dst, *s1);
                }
                Instruction::Xchg { s1, s2 } => {
                    ir.fxchg(*s1, *s2);
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
                    Func::Unary(_) => ir.call(&label, 1),
                    Func::Binary(_) => ir.call(&label, 2),
                },
            }
        }
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
        let _ = write!(fs, "#!\n");
        let _ = write!(fs, "{:?}", self.mir);
    }

    fn func(&self) -> CompiledFunc<f64> {
        unreachable!()
    }

    fn support_indirect(&self) -> bool {
        false
    }
}
