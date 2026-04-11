use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::Write;
use std::rc::Rc;

use anyhow::Result;
use num_complex::Complex;
use petgraph::matrix_graph::Zero;

use crate::code::{Func, VirtualTable};
use crate::complexify::Complexifier;
use crate::config::Config;
use crate::config::SPILL_AREA;
use crate::generator::Generator;
use crate::machine::MachineCode;
use crate::symbol::Loc;
use crate::utils::is_external_func;
use crate::utils::{bool_to_f64, Compiled, CompiledFunc, Reg};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniOp {
    Abs,
    Not,
    Neg,
    Recip,
    Root,
    RealRoot,
    Round,
    Floor,
    Ceiling,
    Trunc,
    Real,
    Imaginary,
    Conjugate,
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
    Complex,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArithOp {
    Plus,
    Minus,
    Times,
    Divide,
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
    Load {
        dst: Reg,
        loc: Loc,
    },
    Save {
        src: Reg,
        loc: Loc,
    },
    LoadComplex {
        xd: Reg,
        yd: Reg,
        loc: Loc,
    },
    SaveComplex {
        xs: Reg,
        ys: Reg,
        loc: Loc,
    },
    LoadConst {
        dst: Reg,
        idx: u32,
    },
    Call {
        label: String,
        f: Func,
        num_args: usize,
    },
    Fused {
        op: FusedOp,
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
    },
    IfElse {
        dst: Reg,
        true_val: Reg,
        false_val: Reg,
        cond: Loc,
    },
    Label {
        label: String,
    },
    Branch {
        label: String,
    },
    BranchIf {
        cond: Reg,
        label: String,
        is_else: bool,
    },
    LoadMath {
        op: ArithOp,
        dst: Reg,
        s1: Reg,
        loc: Loc,
    },
    LoadConstMath {
        op: ArithOp,
        dst: Reg,
        s1: Reg,
        idx: u32,
    },
    ComplexBi {
        op: ArithOp,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
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
            Instruction::Load { dst, loc } => write!(f, "{:?} := {:?}", &dst, &loc),
            Instruction::Save { src, loc } => write!(f, "{:?} := {:?}", &loc, &src),
            Instruction::LoadComplex { xd, yd, loc } => {
                write!(f, "({:?} + {:?}*im) := {:?}", &xd, &yd, &loc)
            }
            Instruction::SaveComplex { xs, ys, loc } => {
                write!(f, "{:?} := ({:?} + {:?}*im)", &loc, &xs, &ys)
            }
            Instruction::LoadConst { dst, idx } => write!(f, "{:?} := consts[{:?}]", &dst, idx),
            Instruction::Fused { op, dst, a, b, c } => match op {
                FusedOp::MulAdd => write!(f, "{:?} := {:?} * {:?} + {:?}", &dst, &a, &b, &c),
                FusedOp::NegMulAdd => write!(f, "{:?} := - {:?} * {:?} + {:?}", &dst, &a, &b, &c),
                FusedOp::MulSub => write!(f, "{:?} := {:?} * {:?} - {:?}", &dst, &a, &b, &c),
                FusedOp::NegMulSub => write!(f, "{:?} := - {:?} * {:?} - {:?}", &dst, &a, &b, &c),
            },
            Instruction::IfElse {
                dst,
                true_val,
                false_val,
                cond,
            } => write!(
                f,
                "{:?} := {:?} ? {:?} : {:?}",
                &dst, cond, &true_val, &false_val
            ),
            Self::Label { label } => write!(f, "{:?}:", &label),
            Self::Branch { label } => write!(f, "goto {:?}", label),
            Self::BranchIf {
                cond,
                label,
                is_else,
            } => {
                if *is_else {
                    write!(f, "if not {:?} goto {:?}", &cond, label)
                } else {
                    write!(f, "if {:?} goto {:?}", &cond, label)
                }
            }
            Self::LoadMath { op, dst, s1, loc } => {
                write!(
                    f,
                    "{:?} := {:?} {:?} {:?} # load/math",
                    &dst, &s1, &op, &loc
                )
            }
            Self::LoadConstMath { op, dst, s1, idx } => {
                write!(
                    f,
                    "{:?} := {:?} {:?} consts[{:?}] # load const/math",
                    &dst, &s1, &op, &idx
                )
            }
            Self::ComplexBi {
                op,
                xd,
                yd,
                x1,
                y1,
                x2,
                y2,
            } => {
                write!(
                    f,
                    "({:?} + {:?}*im) := ({:?} + {:?}*im) {:?} ({:?} + {:?}*im)",
                    &xd, &yd, &x1, &y1, &op, &x2, &y2
                )
            }
        }
    }
}

#[derive(Default)]
pub struct Mir {
    pub code: Vec<Instruction>,
    pub consts: Vec<f64>,
    pub labels: HashMap<String, usize>,
    pub config: Config,
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
    pub fn new(config: Config) -> Mir {
        Mir {
            code: Vec::new(),
            consts: Vec::new(),
            labels: HashMap::new(),
            config,
        }
    }

    fn push(&mut self, ins: Instruction) {
        self.code.push(ins)
    }

    pub fn get_dst(ins: &Instruction) -> Option<u8> {
        match *ins {
            Instruction::Uni {
                dst: Reg::Gen(r), ..
            } => Some(r),
            Instruction::Bi {
                dst: Reg::Gen(r), ..
            } => Some(r),
            Instruction::Mov {
                dst: Reg::Gen(r), ..
            } => Some(r),
            Instruction::Load {
                dst: Reg::Gen(r), ..
            } => Some(r),
            Instruction::LoadConst {
                dst: Reg::Gen(r), ..
            } => Some(r),
            Instruction::Fused {
                dst: Reg::Gen(r), ..
            } => Some(r),
            Instruction::Save { .. } => None,
            Instruction::IfElse {
                dst: Reg::Gen(r), ..
            } => Some(r),
            _ => None,
        }
    }

    pub fn used_registers(&self) -> Vec<u8> {
        let mut mask: u32 = 0;

        for ins in self.code.iter() {
            let r = Self::get_dst(ins);

            if let Some(r) = r {
                mask |= 1 << r;
            }
        }

        let mut used: Vec<u8> = Vec::new();

        // 32 is the max possible logical register count
        for i in 0..32 {
            if mask & (1 << i) != 0 {
                used.push(i);
            }
        }

        used
    }

    pub fn populate_labels(&mut self) {
        let mut labels: HashMap<String, usize> = HashMap::new();

        for (ip, ins) in self.code.iter().enumerate() {
            if let Instruction::Label { label } = ins {
                labels.insert(label.clone(), ip);
            }
        }

        self.labels = labels;
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

    pub fn set_label(&mut self, label: &str) {
        self.push(Instruction::Label {
            label: label.to_string(),
        })
    }

    pub fn branch(&mut self, label: &str) {
        self.push(Instruction::Branch {
            label: label.to_string(),
        });
    }

    pub fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        self.push(Instruction::BranchIf {
            cond,
            label: label.to_string(),
            is_else,
        });
    }

    pub fn fmov(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Mov { dst, s1 });
    }

    pub fn fxchg(&mut self, _s1: Reg, _s2: Reg) {
        panic!("xchg not defined for IR");
        // self.push(Instruction::Xchg { s1, s2 });
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

    pub fn save_mem(&mut self, src: Reg, idx: u32) {
        self.push(Instruction::Save {
            src,
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

    pub fn save_stack(&mut self, src: Reg, idx: u32) {
        self.push(Instruction::Save {
            src,
            loc: Loc::Stack(idx),
        });
    }

    pub fn save_mem_result(&mut self, idx: u32) {
        self.save_mem(Reg::Ret, idx);
    }

    pub fn save_stack_result(&mut self, idx: u32) {
        self.save_stack(Reg::Ret, idx);
    }

    pub fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.push(Instruction::LoadComplex {
            xd,
            yd,
            loc: Loc::Mem(idx),
        });
    }

    pub fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.push(Instruction::SaveComplex {
            xs,
            ys,
            loc: Loc::Mem(idx),
        });
    }

    pub fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.push(Instruction::LoadComplex {
            xd,
            yd,
            loc: Loc::Param(idx),
        });
    }

    pub fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.push(Instruction::LoadComplex {
            xd,
            yd,
            loc: Loc::Stack(idx),
        });
    }

    pub fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.push(Instruction::SaveComplex {
            xs,
            ys,
            loc: Loc::Stack(idx),
        });
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

    pub fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::RealRoot,
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

    pub fn real(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Real,
            dst,
            s1,
        });
    }

    pub fn imaginary(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Imaginary,
            dst,
            s1,
        });
    }

    pub fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Conjugate,
            dst,
            s1,
        });
    }

    pub fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.push(Instruction::Bi {
            op: BinOp::Complex,
            dst,
            s1,
            s2,
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

    pub fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, cond: Loc) {
        self.push(Instruction::IfElse {
            dst,
            true_val,
            false_val,
            cond,
        });
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

    pub fn plus_load(&mut self, dst: Reg, s1: Reg, loc: Loc) {
        self.push(Instruction::LoadMath {
            op: ArithOp::Plus,
            dst,
            s1,
            loc,
        });
    }

    pub fn minus_load(&mut self, dst: Reg, s1: Reg, loc: Loc) {
        self.push(Instruction::LoadMath {
            op: ArithOp::Minus,
            dst,
            s1,
            loc,
        });
    }

    pub fn times_load(&mut self, dst: Reg, s1: Reg, loc: Loc) {
        self.push(Instruction::LoadMath {
            op: ArithOp::Times,
            dst,
            s1,
            loc,
        });
    }

    pub fn divide_load(&mut self, dst: Reg, s1: Reg, loc: Loc) {
        self.push(Instruction::LoadMath {
            op: ArithOp::Divide,
            dst,
            s1,
            loc,
        });
    }

    pub fn plus_load_const(&mut self, dst: Reg, s1: Reg, idx: u32) {
        self.push(Instruction::LoadConstMath {
            op: ArithOp::Plus,
            dst,
            s1,
            idx,
        });
    }

    pub fn minus_load_const(&mut self, dst: Reg, s1: Reg, idx: u32) {
        self.push(Instruction::LoadConstMath {
            op: ArithOp::Minus,
            dst,
            s1,
            idx,
        });
    }

    pub fn times_load_const(&mut self, dst: Reg, s1: Reg, idx: u32) {
        self.push(Instruction::LoadConstMath {
            op: ArithOp::Times,
            dst,
            s1,
            idx,
        });
    }

    pub fn divide_load_const(&mut self, dst: Reg, s1: Reg, idx: u32) {
        self.push(Instruction::LoadConstMath {
            op: ArithOp::Divide,
            dst,
            s1,
            idx,
        });
    }

    pub fn plus_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) {
        self.push(Instruction::ComplexBi {
            op: ArithOp::Plus,
            xd,
            yd,
            x1,
            y1,
            x2,
            y2,
        });
    }

    pub fn minus_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) {
        self.push(Instruction::ComplexBi {
            op: ArithOp::Minus,
            xd,
            yd,
            x1,
            y1,
            x2,
            y2,
        });
    }

    pub fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) {
        self.push(Instruction::ComplexBi {
            op: ArithOp::Times,
            xd,
            yd,
            x1,
            y1,
            x2,
            y2,
        });
    }

    pub fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) {
        self.push(Instruction::ComplexBi {
            op: ArithOp::Divide,
            xd,
            yd,
            x1,
            y1,
            x2,
            y2,
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

    pub fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        let f = self.find_op(op)?;
        self.push(Instruction::Call {
            f,
            label: op.to_string(),
            num_args,
        });

        Ok(())
    }

    pub fn find_op(&self, op: &str) -> Result<Func> {
        let op = if self.config.is_complex() && !is_external_func(op) {
            &format!("cplx_{}", &op)
        } else {
            op
        };

        if let Some(df) = &self.config.df {
            if let Some(f) = df.funcs.get(op) {
                Ok(f.clone())
            } else {
                VirtualTable::from_str(op)
            }
        } else {
            VirtualTable::from_str(op)
        }
    }
}

impl Mir {
    fn get(regs: &[f64], r: Reg) -> f64 {
        match r {
            Reg::Ret | Reg::Left => regs[0],
            Reg::Temp | Reg::Right => regs[1],
            Reg::Gen(r) => regs[r as usize + 2],
            Reg::Static(..) => todo!(),
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
            Reg::Static(..) => todo!(),
        }
    }

    fn exec_uniop(regs: &mut [f64], op: UniOp, dst: Reg, s1: Reg) {
        let s1 = Self::get(regs, s1);

        let val = match op {
            UniOp::Neg => -s1,
            UniOp::Not => f64::from_bits(!s1.to_bits()),
            UniOp::Abs => s1.abs(),
            UniOp::Root => s1.sqrt(),
            UniOp::RealRoot => s1.sqrt(),
            UniOp::Recip => 1.0 / s1,
            UniOp::Round => s1.round(),
            UniOp::Floor => s1.floor(),
            UniOp::Ceiling => s1.ceil(),
            UniOp::Trunc => s1.trunc(),
            UniOp::Real => s1,
            UniOp::Imaginary => 0.0,
            UniOp::Conjugate => s1,
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
            BinOp::Complex => s1,
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

    fn exec_load_math(
        mem: &mut [f64],
        stack: &mut [f64],
        regs: &mut [f64],
        params: &[f64],
        op: ArithOp,
        dst: Reg,
        s1: Reg,
        loc: Loc,
    ) {
        let s1 = Self::get(regs, s1);

        let y = match loc {
            Loc::Mem(idx) => mem[idx as usize],
            Loc::Stack(idx) => stack[idx as usize],
            Loc::Param(idx) => params[idx as usize],
        };

        let val = match op {
            ArithOp::Plus => s1 + y,
            ArithOp::Minus => s1 - y,
            ArithOp::Times => s1 * y,
            ArithOp::Divide => s1 / y,
        };

        Self::set(regs, dst, val);
    }

    fn exec_load_const_math(regs: &mut [f64], op: ArithOp, dst: Reg, s1: Reg, y: f64) {
        let s1 = Self::get(regs, s1);

        let val = match op {
            ArithOp::Plus => s1 + y,
            ArithOp::Minus => s1 - y,
            ArithOp::Times => s1 * y,
            ArithOp::Divide => s1 / y,
        };

        Self::set(regs, dst, val);
    }

    fn exec_complex(
        regs: &mut [f64],
        op: ArithOp,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        let z1 = Complex::new(Self::get(regs, x1), Self::get(regs, y1));
        let z2 = Complex::new(Self::get(regs, x2), Self::get(regs, y2));

        let val = match op {
            ArithOp::Plus => z1 + z2,
            ArithOp::Minus => z1 - z2,
            ArithOp::Times => z1 * z2,
            ArithOp::Divide => z1 / z2,
        };

        Self::set(regs, xd, val.re);
        Self::set(regs, yd, val.im);
    }

    pub fn exec_instruction(
        &self,
        mem: &mut [f64],
        stack: &mut [f64],
        regs: &mut [f64],
        params: &[f64],
    ) {
        let mut ip: usize = 0;
        let n = self.code.len();

        while ip < n {
            let ins = &self.code[ip];

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
                Instruction::Load { dst, loc } => {
                    let val = match loc {
                        Loc::Mem(idx) => mem[*idx as usize],
                        Loc::Stack(idx) => stack[*idx as usize],
                        Loc::Param(idx) => params[*idx as usize],
                    };
                    Self::set(regs, *dst, val);
                }
                Instruction::Save { src, loc } => {
                    let val = Self::get(regs, *src);
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
                Instruction::LoadComplex { xd, yd, loc } => {
                    let (x, y) = match loc {
                        Loc::Mem(idx) => (mem[*idx as usize], mem[1 + *idx as usize]),
                        Loc::Stack(idx) => (stack[*idx as usize], stack[1 + *idx as usize]),
                        Loc::Param(idx) => (params[*idx as usize], params[1 + *idx as usize]),
                    };
                    Self::set(regs, *xd, x);
                    Self::set(regs, *yd, y);
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    let x = Self::get(regs, *xs);
                    let y = Self::get(regs, *ys);
                    match loc {
                        Loc::Mem(idx) => {
                            mem[*idx as usize] = x;
                            mem[1 + *idx as usize] = y;
                        }
                        Loc::Stack(idx) => {
                            stack[*idx as usize] = x;
                            stack[1 + *idx as usize] = y;
                        }
                        Loc::Param(_) => {
                            unreachable!()
                        }
                    };
                }
                Instruction::LoadConst { dst, idx } => {
                    Self::set(regs, *dst, self.consts[*idx as usize]);
                }
                Instruction::Call { f, num_args, .. } => match f {
                    Func::Unary(p) => Self::set(regs, Reg::Ret, p(Self::get(regs, Reg::Left))),
                    Func::Binary(p) => Self::set(
                        regs,
                        Reg::Ret,
                        p(Self::get(regs, Reg::Left), Self::get(regs, Reg::Right)),
                    ),
                    Func::UnaryCplx(p) => {
                        let x =
                            Complex::new(Self::get(regs, Reg::Left), Self::get(regs, Reg::Right));
                        let mut z = Complex::ZERO;
                        p(x.re, x.im, &mut z);
                        Self::set(regs, Reg::Ret, z.re);
                        Self::set(regs, Reg::Temp, z.im);
                    }
                    Func::BinaryCplx(p) => {
                        let x =
                            Complex::new(Self::get(regs, Reg::Left), Self::get(regs, Reg::Right));
                        let y = Complex::new(
                            Self::get(regs, Reg::Gen(0)),
                            Self::get(regs, Reg::Gen(1)),
                        );
                        let mut z = y;
                        p(x.re, x.im, &mut z);
                        Self::set(regs, Reg::Ret, z.re);
                        Self::set(regs, Reg::Temp, z.im);
                    }
                    Func::Slice { env, f_scalar, .. } => unsafe {
                        let f: fn(*const std::ffi::c_void, *const f64, usize, *mut f64) -> bool =
                            std::mem::transmute(*f_scalar);

                        let mut val: Complex<f64> = Complex::default();
                        f(
                            *env,
                            stack.as_ptr().add(SPILL_AREA),
                            *num_args,
                            &mut val as *mut _ as *mut f64,
                        );

                        Self::set(regs, Reg::Ret, val.re);
                        Self::set(regs, Reg::Temp, val.im);
                    },
                },
                Instruction::Fused { op, dst, a, b, c } => {
                    Self::exec_fused(regs, *op, *dst, *a, *b, *c);
                }
                Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                } => {
                    let cond = match cond {
                        Loc::Mem(idx) => mem[*idx as usize],
                        Loc::Stack(idx) => stack[*idx as usize],
                        Loc::Param(idx) => params[*idx as usize],
                    };
                    Self::set(
                        regs,
                        *dst,
                        if cond.is_zero() {
                            Self::get(regs, *false_val)
                        } else {
                            Self::get(regs, *true_val)
                        },
                    )
                }
                Instruction::Label { .. } => {}
                Instruction::Branch { label } => ip = *self.labels.get(label).unwrap() - 1,
                Instruction::BranchIf {
                    cond,
                    label,
                    is_else,
                } => {
                    if (Self::get(regs, *cond) != 0.0) ^ is_else {
                        ip = *self.labels.get(label).unwrap() - 1
                    }
                }
                Instruction::LoadMath { op, dst, s1, loc } => {
                    Self::exec_load_math(mem, stack, regs, params, *op, *dst, *s1, *loc);
                }
                Instruction::LoadConstMath { op, dst, s1, idx } => {
                    Self::exec_load_const_math(regs, *op, *dst, *s1, self.consts[*idx as usize]);
                }
                Instruction::ComplexBi {
                    op,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                } => Self::exec_complex(regs, *op, *xd, *yd, *x1, *y1, *x2, *y2),
            }

            ip += 1;
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
            UniOp::RealRoot => ir.real_root(dst, s1),
            UniOp::Recip => ir.recip(dst, s1),
            UniOp::Round => ir.round(dst, s1),
            UniOp::Floor => ir.floor(dst, s1),
            UniOp::Ceiling => ir.ceiling(dst, s1),
            UniOp::Trunc => ir.trunc(dst, s1),
            UniOp::Real => ir.real(dst, s1),
            UniOp::Imaginary => ir.imaginary(dst, s1),
            UniOp::Conjugate => ir.conjugate(dst, s1),
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
            BinOp::Complex => ir.complex(dst, s1, s2),
        };
    }

    pub fn rerun(&self, ir: &mut dyn Generator) -> Result<()> {
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
                Instruction::Load { dst, loc } => {
                    match loc {
                        Loc::Mem(idx) => ir.load_mem(*dst, *idx),
                        Loc::Stack(idx) => ir.load_stack(*dst, *idx),
                        Loc::Param(idx) => ir.load_param(*dst, *idx),
                    };
                }
                Instruction::Save { src, loc } => {
                    match loc {
                        Loc::Mem(idx) => ir.save_mem(*src, *idx),
                        Loc::Stack(idx) => ir.save_stack(*src, *idx),
                        Loc::Param(_) => unreachable!(),
                    };
                }
                Instruction::LoadComplex { xd, yd, loc } => {
                    match loc {
                        Loc::Mem(idx) => {
                            // ir.load_mem(*xd, *idx);
                            // ir.load_mem(*yd, 1 + *idx);
                            ir.load_mem_complex(*xd, *yd, *idx);
                        }
                        Loc::Stack(idx) => {
                            // ir.load_stack(*xd, *idx);
                            // ir.load_stack(*yd, 1 + *idx);
                            ir.load_stack_complex(*xd, *yd, *idx);
                        }
                        Loc::Param(idx) => {
                            // ir.load_param(*xd, *idx);
                            // ir.load_param(*yd, 1 + *idx);
                            ir.load_param_complex(*xd, *yd, *idx);
                        }
                    };
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    match loc {
                        Loc::Mem(idx) => {
                            // ir.save_mem(*xs, *idx);
                            // ir.save_mem(*ys, 1 + *idx);
                            ir.save_mem_complex(*xs, *ys, *idx);
                        }
                        Loc::Stack(idx) => {
                            // ir.save_stack(*xs, *idx);
                            // ir.save_stack(*ys, 1 + *idx);
                            ir.save_stack_complex(*xs, *ys, *idx);
                        }
                        Loc::Param(_) => unreachable!(),
                    };
                }
                Instruction::LoadConst { dst, idx } => {
                    ir.load_const(*dst, *idx);
                }
                Instruction::Call { label, f, num_args } => match f {
                    Func::Unary(_) => ir.call(label, *num_args)?,
                    Func::Binary(_) => ir.call(label, *num_args)?,
                    Func::UnaryCplx(_) => ir.call_complex(label, *num_args)?,
                    Func::BinaryCplx(_) => ir.call_complex(label, *num_args)?,
                    Func::Slice { .. } => ir.call(label, *num_args)?,
                },
                Instruction::Fused { op, dst, a, b, c } => match op {
                    FusedOp::MulAdd => ir.fused_mul_add(*dst, *a, *b, *c),
                    FusedOp::MulSub => ir.fused_mul_sub(*dst, *a, *b, *c),
                    FusedOp::NegMulAdd => ir.fused_neg_mul_add(*dst, *a, *b, *c),
                    FusedOp::NegMulSub => ir.fused_neg_mul_sub(*dst, *a, *b, *c),
                },
                Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                } => {
                    if let Loc::Stack(idx) = *cond {
                        ir.ifelse(*dst, *true_val, *false_val, idx);
                    } else {
                        panic!("IfElse condition should be stored in the stack");
                    }
                }
                Instruction::Label { label } => ir.set_label(label),
                Instruction::Branch { label } => ir.branch(label),
                Instruction::BranchIf {
                    cond,
                    label,
                    is_else,
                } => ir.branch_if(*cond, label, *is_else),
                Instruction::LoadMath { op, dst, s1, loc } => {
                    let t = if self.config.is_complex() {
                        Reg::Temp
                    } else {
                        Reg::Ret
                    };
                    match loc {
                        Loc::Mem(idx) => ir.load_mem(t, *idx),
                        Loc::Stack(idx) => ir.load_stack(t, *idx),
                        Loc::Param(idx) => ir.load_param(t, *idx),
                    }
                    match op {
                        ArithOp::Plus => ir.plus(*dst, *s1, t),
                        ArithOp::Minus => ir.minus(*dst, *s1, t),
                        ArithOp::Times => ir.times(*dst, *s1, t),
                        ArithOp::Divide => ir.divide(*dst, *s1, t),
                    }
                    ir.fuse_load_math();
                }
                Instruction::LoadConstMath { op, dst, s1, idx } => {
                    let t = if self.config.is_complex() {
                        Reg::Temp
                    } else {
                        Reg::Ret
                    };

                    ir.load_const(t, *idx);

                    match op {
                        ArithOp::Plus => ir.plus(*dst, *s1, t),
                        ArithOp::Minus => ir.minus(*dst, *s1, t),
                        ArithOp::Times => ir.times(*dst, *s1, t),
                        ArithOp::Divide => ir.divide(*dst, *s1, t),
                    }
                    ir.fuse_load_math();
                }
                Instruction::ComplexBi {
                    op,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                } => match op {
                    ArithOp::Plus => {
                        Complexifier::generic_complex_plus(ir, *xd, *yd, *x1, *y1, *x2, *y2)
                    }
                    ArithOp::Minus => {
                        Complexifier::generic_complex_minus(ir, *xd, *yd, *x1, *y1, *x2, *y2)
                    }
                    ArithOp::Times => {
                        if !ir.times_complex(*xd, *yd, *x1, *y1, *x2, *y2) {
                            Complexifier::generic_complex_times(ir, *xd, *yd, *x1, *y1, *x2, *y2)
                        }
                    }
                    ArithOp::Divide => {
                        if !ir.divide_complex(*xd, *yd, *x1, *y1, *x2, *y2) {
                            Complexifier::generic_complex_divide(ir, *xd, *yd, *x1, *y1, *x2, *y2)
                        }
                    }
                },
            }
        }

        Ok(())
    }
}

impl Mir {
    fn fuse_op_mov(&self, q0: &Instruction, q1: &Instruction) -> Vec<Instruction> {
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
                    return vec![Instruction::Uni {
                        op,
                        dst: dst_q1,
                        s1,
                    }];
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
                    return vec![Instruction::Bi {
                        op,
                        dst: dst_q1,
                        s1,
                        s2,
                    }];
                }
            }
        };

        Vec::new()
    }

    fn fuse_load(&self, q0: &Instruction, q1: &Instruction) -> Vec<Instruction> {
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
                    return vec![Instruction::Load { dst: dst_q1, loc }];
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
                    return vec![Instruction::LoadConst { dst: dst_q1, idx }];
                }
            }
        };

        if let Instruction::Load { dst, loc } = *q0 {
            if let Instruction::Save {
                src: src_q1,
                loc: loc_q1,
            } = *q1
            {
                if loc == loc_q1 && dst == src_q1 {
                    return vec![Instruction::Nop];
                }
            }
        };

        Vec::new()
    }

    fn fuse_save(&self, q0: &Instruction, q1: &Instruction) -> Vec<Instruction> {
        // Important: this rule is commented out because of a potential bug,
        // where %0 is needed afterward.
        /*
        /*
        * example
        *      %0 := %1
        *      Mem[4] = %0
        * becomes
        *      Mem[4] := %1
        */
        if let Instruction::Mov { dst, s1 } = *q0 {
            if let Instruction::Save {
                src: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if dst == dst_q1 {
                    code.push(Instruction::Save {
                        src: s1,
                        loc: loc_q1,
                    });
                    return true;
                }
            }
        }
        */

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
        if let Instruction::Save { src, loc } = *q0 {
            if let Instruction::Load {
                dst: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if loc == loc_q1 {
                    return vec![
                        q0.clone(),
                        Instruction::Mov {
                            dst: dst_q1,
                            s1: src,
                        },
                    ];
                }
            }
        };

        Vec::new()
    }

    fn fuse_save3(&self, q0: &Instruction, q1: &Instruction, q2: &Instruction) -> Vec<Instruction> {
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
        if let Instruction::Save { src, loc } = *q0 {
            if let Instruction::Load {
                dst: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if let Instruction::Save {
                    src: dst_q2,
                    loc: loc_q2,
                } = *q2
                {
                    if src == Reg::Ret && loc == loc_q1 && dst_q1 == dst_q2 {
                        return vec![Instruction::Save {
                            src: Reg::Ret,
                            loc: loc_q2,
                        }];
                    }
                }
            }
        };

        Vec::new()
    }

    fn fuse_fma(&self, q0: &Instruction, q1: &Instruction) -> Vec<Instruction> {
        if let Instruction::Bi { op, dst, s1, s2 } = *q0 {
            if let Instruction::Bi {
                op: op_q1,
                dst: dst_q1,
                s1: s1_q1,
                s2: s2_q1,
            } = *q1
            {
                if op == BinOp::Times && op_q1 == BinOp::Plus && s1_q1 == dst {
                    return vec![Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s2_q1,
                    }];
                }

                if op == BinOp::Times && op_q1 == BinOp::Plus && s2_q1 == dst {
                    return vec![Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s1_q1,
                    }];
                }

                if op == BinOp::Times && op_q1 == BinOp::Minus && s1_q1 == dst {
                    return vec![Instruction::Fused {
                        op: FusedOp::MulSub,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s2_q1,
                    }];
                }

                if op == BinOp::Times && op_q1 == BinOp::Minus && s2_q1 == dst {
                    return vec![Instruction::Fused {
                        op: FusedOp::NegMulAdd,
                        dst: dst_q1,
                        a: s1,
                        b: s2,
                        c: s1_q1,
                    }];
                }
            }
        }

        Vec::new()
    }

    fn fuse_fma3(&self, q0: &Instruction, q1: &Instruction, q2: &Instruction) -> Vec<Instruction> {
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
                        return vec![
                            Instruction::LoadConst {
                                dst: Reg::Temp,
                                idx: idx_q1,
                            },
                            Instruction::Fused {
                                op: FusedOp::MulAdd,
                                dst: dst_q2,
                                a: s1,
                                b: s2,
                                c: Reg::Temp,
                            },
                        ];
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
                        return vec![
                            Instruction::Load {
                                dst: Reg::Temp,
                                loc: loc_q1,
                            },
                            Instruction::Fused {
                                op: FusedOp::MulAdd,
                                dst: dst_q2,
                                a: s1,
                                b: s2,
                                c: Reg::Temp,
                            },
                        ];
                    }
                }
            }
        }

        Vec::new()
    }

    fn fuse(
        &self,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> (Vec<Instruction>, usize) {
        let fastmath = self.config.fastmath();
        let is_complex = self.config.is_complex(); // TODO: fix the FMA bug for complex noted on `runtests complex`

        let v = self.fuse_save3(q0, q1, q2);
        if !v.is_empty() {
            return (v, 3);
        }

        if fastmath && !is_complex {
            let v = self.fuse_fma3(q0, q1, q2);
            if !v.is_empty() {
                return (v, 3);
            }

            let v = self.fuse_fma(q0, q1);
            if !v.is_empty() {
                return (v, 2);
            }
        }

        let v = self.fuse_op_mov(q0, q1);
        if !v.is_empty() {
            return (v, 2);
        }

        let v = self.fuse_load(q0, q1);
        if !v.is_empty() {
            return (v, 2);
        }

        let v = self.fuse_save(q0, q1);
        if !v.is_empty() {
            return (v, 2);
        }

        (vec![q0.clone()], 1)
    }

    pub fn optimize_peephole(&mut self) -> bool {
        let mut code: Vec<Instruction> = Vec::new();
        let mut success = false;

        let mut iter = self.code.iter();
        let mut q0 = iter.next();
        let mut q1 = iter.next();
        let mut q2 = iter.next();
        let mut p: Instruction;

        while q0.is_some() {
            let (mut v, d) = self.fuse(
                q0.unwrap_or(&Instruction::Nop),
                q1.unwrap_or(&Instruction::Nop),
                q2.unwrap_or(&Instruction::Nop),
            );
            code.append(&mut v);
            success |= d > 1;

            match d {
                1 => {
                    q0 = q1;
                    q1 = q2;
                    q2 = iter.next();
                }
                2 => {
                    p = code.pop().unwrap();
                    q0 = Some(&p);
                    q1 = q2;
                    q2 = iter.next();
                }
                3 => {
                    p = code.pop().unwrap();
                    q0 = Some(&p);
                    q1 = iter.next();
                    q2 = iter.next();
                }
                _ => unreachable!(),
            }
        }

        self.code = code;
        success
    }
}

/********************************************************/

#[derive(Clone)]
pub struct CompiledMir {
    pub mir: Rc<Mir>,
    pub mem: Vec<f64>,
    pub stack: Vec<f64>,
    pub regs: Vec<f64>,
}

impl CompiledMir {
    pub fn new(mir: Mir, mem: Vec<f64>, stack: Vec<f64>) -> CompiledMir {
        let regs = vec![0.0; 16];

        CompiledMir {
            mir: Rc::new(mir),
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

    fn evaluate(&mut self, args: &[f64], outs: &mut [f64]) {
        self.mir
            .exec_instruction(&mut self.mem, &mut self.stack, &mut self.regs, args);
        outs.copy_from_slice(&self.mem[0..outs.len()]);
    }

    fn evaluate_single(&mut self, args: &[f64]) -> f64 {
        self.mir
            .exec_instruction(&mut self.mem, &mut self.stack, &mut self.regs, args);
        self.mem[0]
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

    fn dumps(&self) -> Vec<u8> {
        let s = format!("{:?}", self.mir);
        s.into_bytes()
    }

    fn func(&self) -> CompiledFunc<f64> {
        unreachable!()
    }

    fn support_indirect(&self) -> bool {
        false
    }

    fn count_lanes(&self) -> usize {
        1
    }

    fn as_machine(&self) -> Option<&MachineCode<f64>> {
        None
    }
}
