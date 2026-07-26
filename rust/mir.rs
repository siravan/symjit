use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::rc::Rc;

use anyhow::Result;
use num_complex::Complex;
use num_traits::identities::Zero;

use crate::code::{Func, VirtualTable};
use crate::complexify::Complexifier;
use crate::config::Config;
use crate::config::SPILL_AREA;
use crate::generator::FuncletType;
use crate::generator::Generator;
use crate::machine::MachineCode;
use crate::serializer::MirWriter;
use crate::symbol::Loc;
use crate::utils::is_external_func;
use crate::utils::{bool_to_f64, Compiled, CompiledFunc, Reg};

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[repr(u8)]
pub enum UniOp {
    Neg,
    Not,
    Abs,
    Root,
    RealRoot,
    Recip,
    Round,
    Floor,
    Ceiling,
    Trunc,
    Real,
    Imaginary,
    Conjugate,
    Half,
    IsZero,
    IsNotZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[repr(u8)]
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

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[repr(u8)]
pub enum ArithOp {
    Plus = 0,
    Minus = 1,
    Times = 2,
    Divide = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[repr(u8)]
pub enum FusedOp {
    MulAdd = 0,    // + a * b + c
    MulSub = 1,    // a * b - c
    NegMulAdd = 2, // - a * b + c
    NegMulSub = 3, // -a * b - c
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum FuncletOp {
    Times,
    Divide,
    Root,
    TimesComplex,
    DivideComplex,
}

#[derive(Clone)]
pub enum Instruction {
    Nop,
    End,
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

impl Hash for Instruction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let s = format!("{:?}", &self);
        s.hash(state);
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Nop => write!(f, "nop"),
            Instruction::End => write!(f, "end"),
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

impl Instruction {
    fn desc(&self) -> String {
        match self {
            Instruction::Nop => "nop".into(),
            Instruction::End => "end".into(),
            Instruction::Uni { op, .. } => format!("uniop {:?}", &op),
            Instruction::Bi { op, .. } => format!("binop {:?}", &op),
            Instruction::Call { label, .. } => format!("call {}", label),
            Instruction::Mov { .. } => "mov".into(),
            Instruction::Load { .. } => "load".into(),
            Instruction::Save { .. } => "save".into(),
            Instruction::LoadComplex { .. } => "load_complex".into(),
            Instruction::SaveComplex { .. } => "save_complex".into(),
            Instruction::LoadConst { .. } => "load_const".into(),
            Instruction::Fused { op, .. } => format!("fused op {:?}", &op),
            Instruction::IfElse { .. } => "if_else".into(),
            Self::Label { .. } => "label".into(),
            Self::Branch { .. } => "branch".into(),
            Self::BranchIf { .. } => "branch_if".into(),
            Self::LoadMath { op, .. } => format!("load math {:?}", &op),
            Self::LoadConstMath { op, .. } => format!("load const math {:?}", &op),
            Self::ComplexBi { op, .. } => format!("complex binop {:?}", &op),
        }
    }
}

#[derive(Default, Clone)]
pub struct Mir {
    pub code: MirWriter,
    pub consts: Vec<f64>,
    pub labels: HashMap<String, usize>,
    pub config: Config,
}

impl fmt::Debug for Mir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "config: {:?}", &self.config)?;
        writeln!(f, "ip: {}", &self.code.ip)?;

        for (i, ins) in self.code.iter().enumerate() {
            writeln!(f, "{:05}\t{:?}", i, ins)?;
        }

        for (i, x) in self.consts.iter().enumerate() {
            writeln!(f, "const[{}] = {:?}", i, x)?;
        }

        Ok(())
    }
}

impl Mir {
    pub const MAGIC: usize = 0x876a9b6323b00c9e;

    pub fn new(config: Config) -> Mir {
        Mir {
            code: MirWriter::new(),
            consts: Vec::new(),
            labels: HashMap::new(),
            config,
        }
    }

    fn push(&mut self, ins: Instruction) {
        self.code.push(&ins)
    }

    pub fn get_dst(ins: &Instruction) -> u32 {
        match *ins {
            Instruction::Uni {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::Bi {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::ComplexBi {
                xd: Reg::Gen(x),
                yd: Reg::Gen(y),
                ..
            } => (1 << x) | (1 << y),
            Instruction::Mov {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::Load {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::LoadConst {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::LoadComplex {
                xd: Reg::Gen(x),
                yd: Reg::Gen(y),
                ..
            } => (1 << x) | (1 << y),
            Instruction::LoadMath {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::LoadConstMath {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::Fused {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::IfElse {
                dst: Reg::Gen(r), ..
            } => 1 << r,
            Instruction::BranchIf {
                cond: Reg::Gen(r), ..
            } => 1 << r,
            _ => 0,
        }
    }

    pub fn used_registers(&self) -> Vec<Reg> {
        let mut mask: u32 = 0;

        for ins in self.code.iter() {
            mask |= Self::get_dst(&ins);
        }

        let mut used: Vec<Reg> = Vec::new();

        // 32 is the max possible logical register count
        for i in 0..32 {
            if mask & (1 << i) != 0 {
                used.push(Reg::Gen(i));
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

    pub fn half(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Half,
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

    pub fn iszero(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::IsZero,
            dst,
            s1,
        });
    }

    pub fn isnotzero(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::IsNotZero,
            dst,
            s1,
        });
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

    pub fn fused_mul_add(&mut self, dst: Reg, a: Reg, b: Reg, c: Reg) {
        self.push(Instruction::Fused {
            op: FusedOp::MulAdd,
            dst,
            a,
            b,
            c,
        })
    }

    pub fn fused_mul_sub(&mut self, dst: Reg, a: Reg, b: Reg, c: Reg) {
        self.push(Instruction::Fused {
            op: FusedOp::MulSub,
            dst,
            a,
            b,
            c,
        })
    }

    pub fn fused_neg_mul_add(&mut self, dst: Reg, a: Reg, b: Reg, c: Reg) {
        self.push(Instruction::Fused {
            op: FusedOp::NegMulAdd,
            dst,
            a,
            b,
            c,
        })
    }
    pub fn fused_neg_mul_sub(&mut self, dst: Reg, a: Reg, b: Reg, c: Reg) {
        self.push(Instruction::Fused {
            op: FusedOp::NegMulSub,
            dst,
            a,
            b,
            c,
        })
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
        let _ = self.find_op(op)?;
        self.push(Instruction::Call {
            label: op.to_string(),
            num_args,
        });

        Ok(())
    }

    pub fn find_op(&self, op: &str) -> Result<Func> {
        if let Some(df) = &self.config.df {
            if let Some(f) = df.funcs.get(op) {
                return Ok(f.clone());
            }
        }

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
            UniOp::Half => s1 / 2.0,
            UniOp::IsZero => bool_to_f64(s1 == 0.0),
            UniOp::IsNotZero => bool_to_f64(s1 != 0.0),
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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
        let prog: Vec<Instruction> = self.code.iter().collect();
        let n = prog.len();

        while ip < n {
            let ins = &prog[ip];

            match ins {
                Instruction::Nop | Instruction::End => {}
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
                Instruction::Call { label, num_args } => {
                    let f = self.find_op(label).unwrap();
                    match &f {
                        Func::Unary(p) => Self::set(regs, Reg::Ret, p(Self::get(regs, Reg::Left))),
                        Func::Binary(p) => Self::set(
                            regs,
                            Reg::Ret,
                            p(Self::get(regs, Reg::Left), Self::get(regs, Reg::Right)),
                        ),
                        Func::UnaryCplx(p) => {
                            let x = Complex::new(
                                Self::get(regs, Reg::Left),
                                Self::get(regs, Reg::Right),
                            );
                            let mut z = Complex::ZERO;
                            p(x.re, x.im, &mut z);
                            Self::set(regs, Reg::Ret, z.re);
                            Self::set(regs, Reg::Temp, z.im);
                        }
                        Func::BinaryCplx(p) => {
                            let x = Complex::new(
                                Self::get(regs, Reg::Left),
                                Self::get(regs, Reg::Right),
                            );
                            let y = Complex::new(
                                Self::get(regs, Reg::Gen(0)),
                                Self::get(regs, Reg::Gen(1)),
                            );
                            let mut z = y;
                            p(x.re, x.im, &mut z);
                            Self::set(regs, Reg::Ret, z.re);
                            Self::set(regs, Reg::Temp, z.im);
                        }
                        Func::PairedUnary(p) => {
                            let pair = p(Self::get(regs, Reg::Ret));
                            Self::set(regs, Reg::Ret, pair.s);
                            Self::set(regs, Reg::Temp, pair.c);
                        }
                        Func::Slice { env, f_scalar, .. } => unsafe {
                            let f: fn(
                                *const std::ffi::c_void,
                                *const f64,
                                usize,
                                *mut f64,
                            ) -> bool = std::mem::transmute(*f_scalar);

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
                    }
                }
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
                    if (Self::get(regs, *cond) == 0.0) ^ is_else {
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

// Funclet Section
impl Mir {
    fn try_funclet(
        &self,
        ir: &mut dyn Generator,
        funclets: &mut HashSet<(FuncletOp, Vec<Reg>)>,
        ins: &Instruction,
    ) -> bool {
        if !self.config.compress() || !self.config.is_complex() {
            return false;
        }

        let support = ir.support_funclet();

        match ins {
            Instruction::Bi {
                op: BinOp::Times,
                dst,
                s1,
                s2,
            } => {
                if matches!(support, FuncletType::Real) {
                    funclets.insert((FuncletOp::Times, vec![*dst, *s1, *s2]));
                    let name = format!("times_{:?}_{:?}_{:?}", dst, s1, s2);
                    ir.call_funclet(&name);
                    true
                } else {
                    false
                }
            }
            Instruction::Bi {
                op: BinOp::Divide,
                dst,
                s1,
                s2,
            } => {
                if matches!(support, FuncletType::Real) {
                    funclets.insert((FuncletOp::Divide, vec![*dst, *s1, *s2]));
                    let name = format!("divide_{:?}_{:?}_{:?}", dst, s1, s2);
                    ir.call_funclet(&name);
                    true
                } else {
                    false
                }
            }
            Instruction::LoadMath {
                op: ArithOp::Times,
                dst,
                s1,
                loc,
            } => {
                if matches!(support, FuncletType::Real) {
                    let s2 = Reg::Temp;
                    match loc {
                        Loc::Mem(idx) => ir.load_mem(s2, *idx),
                        Loc::Stack(idx) => ir.load_stack(s2, *idx),
                        Loc::Param(idx) => ir.load_param(s2, *idx),
                    }
                    funclets.insert((FuncletOp::Times, vec![*dst, *s1, s2]));
                    let name = format!("times_{:?}_{:?}_{:?}", dst, s1, s2);
                    ir.call_funclet(&name);
                    true
                } else {
                    false
                }
            }
            Instruction::LoadMath {
                op: ArithOp::Divide,
                dst,
                s1,
                loc,
            } => {
                if matches!(support, FuncletType::Real) {
                    let s2 = Reg::Temp;
                    match loc {
                        Loc::Mem(idx) => ir.load_mem(s2, *idx),
                        Loc::Stack(idx) => ir.load_stack(s2, *idx),
                        Loc::Param(idx) => ir.load_param(s2, *idx),
                    }
                    funclets.insert((FuncletOp::Divide, vec![*dst, *s1, s2]));
                    let name = format!("divide_{:?}_{:?}_{:?}", dst, s1, s2);
                    ir.call_funclet(&name);
                    true
                } else {
                    false
                }
            }
            Instruction::Uni {
                op: UniOp::Root,
                dst,
                s1,
            } => {
                if matches!(support, FuncletType::Real) {
                    funclets.insert((FuncletOp::Root, vec![*dst, *s1]));
                    let name = format!("root_{:?}_{:?}", dst, s1);
                    ir.call_funclet(&name);
                    true
                } else {
                    false
                }
            }
            Instruction::ComplexBi {
                op: ArithOp::Times,
                xd,
                yd,
                x1,
                y1,
                x2,
                y2,
            } => {
                if matches!(support, FuncletType::Complex) {
                    funclets.insert((FuncletOp::TimesComplex, vec![*xd, *yd, *x1, *y1, *x2, *y2]));
                    let name = format!(
                        "times_complex_{:?}_{:?}_{:?}_{:?}_{:?}_{:?}",
                        xd, yd, x1, y1, x2, y2
                    );
                    ir.call_funclet(&name);
                    true
                } else {
                    false
                }
            }
            Instruction::ComplexBi {
                op: ArithOp::Divide,
                xd,
                yd,
                x1,
                y1,
                x2,
                y2,
            } => {
                if matches!(support, FuncletType::Complex) {
                    funclets.insert((FuncletOp::DivideComplex, vec![*xd, *yd, *x1, *y1, *x2, *y2]));
                    let name = format!(
                        "divide_complex_{:?}_{:?}_{:?}_{:?}_{:?}_{:?}",
                        xd, yd, x1, y1, x2, y2
                    );
                    ir.call_funclet(&name);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn create_funclets(
        &self,
        ir: &mut dyn Generator,
        funclets: HashSet<(FuncletOp, Vec<Reg>)>,
    ) -> Result<()> {
        if funclets.is_empty() {
            return Ok(());
        }

        ir.branch("@funclets");

        let mut funclets: Vec<_> = funclets.into_iter().collect();
        funclets.sort_unstable();

        for f in &funclets {
            // println!("{:?}", &f);
            match f {
                (FuncletOp::Times, args) => {
                    let dst = args[0];
                    let s1 = args[1];
                    let s2 = args[2];
                    let name = format!("times_{:?}_{:?}_{:?}", dst, s1, s2);
                    ir.set_label(&name);
                    ir.times(dst, s1, s2);
                    ir.ret();
                }
                (FuncletOp::Divide, args) => {
                    let dst = args[0];
                    let s1 = args[1];
                    let s2 = args[2];
                    let name = format!("divide_{:?}_{:?}_{:?}", dst, s1, s2);
                    ir.set_label(&name);
                    ir.divide(dst, s1, s2);
                    ir.ret();
                }
                (FuncletOp::Root, args) => {
                    let dst = args[0];
                    let s1 = args[1];
                    let name = format!("root_{:?}_{:?}", dst, s1);
                    ir.set_label(&name);
                    ir.root(dst, s1);
                    ir.ret();
                }
                (FuncletOp::TimesComplex, args) => {
                    let xd = args[0];
                    let yd = args[1];
                    let x1 = args[2];
                    let y1 = args[3];
                    let x2 = args[4];
                    let y2 = args[5];
                    let name = format!(
                        "times_complex_{:?}_{:?}_{:?}_{:?}_{:?}_{:?}",
                        xd, yd, x1, y1, x2, y2
                    );
                    ir.set_label(&name);
                    ir.times_complex(xd, yd, x1, y1, x2, y2);
                    ir.ret();
                }
                (FuncletOp::DivideComplex, args) => {
                    let xd = args[0];
                    let yd = args[1];
                    let x1 = args[2];
                    let y1 = args[3];
                    let x2 = args[4];
                    let y2 = args[5];
                    let name = format!(
                        "divide_complex_{:?}_{:?}_{:?}_{:?}_{:?}_{:?}",
                        xd, yd, x1, y1, x2, y2
                    );
                    ir.set_label(&name);
                    ir.divide_complex(xd, yd, x1, y1, x2, y2);
                    ir.ret();
                }
            }
        }

        ir.set_label("@funclets");

        Ok(())
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
            UniOp::Half => ir.half(dst, s1),
            UniOp::IsZero => {
                ir.xor(Reg::Temp, Reg::Temp, Reg::Temp);
                ir.eq(dst, s1, Reg::Temp);
            }
            UniOp::IsNotZero => {
                ir.xor(Reg::Temp, Reg::Temp, Reg::Temp);
                ir.neq(dst, s1, Reg::Temp);
            }
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
        let mut funclets: HashSet<(FuncletOp, Vec<Reg>)> = HashSet::new();

        let mut iter = self.code.iter().peekable();

        while let Some(ins) = iter.next() {
            if self.try_funclet(ir, &mut funclets, &ins) {
                continue;
            }

            match &ins {
                Instruction::Nop | Instruction::End => {}
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
                            if self.config.direct_arena() {
                                crate::direct::emit_direct_complex_destination(
                                    ir,
                                    *xs,
                                    *ys,
                                    *idx,
                                    self.config.direct_arena_operation(),
                                    self.config.direct_arena_identity_output(),
                                )?;
                            } else {
                                // ir.save_mem(*xs, *idx);
                                // ir.save_mem(*ys, 1 + *idx);
                                ir.save_mem_complex(*xs, *ys, *idx);
                            }
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
                Instruction::Call { label, num_args } => {
                    let f = self.find_op(label).unwrap();
                    match f {
                        Func::Unary(_) => ir.call(label, *num_args)?,
                        Func::Binary(_) => ir.call(label, *num_args)?,
                        Func::UnaryCplx(_) => ir.call_complex(label, *num_args)?,
                        Func::BinaryCplx(_) => ir.call_complex(label, *num_args)?,
                        Func::PairedUnary(_) => ir.call(label, *num_args)?,
                        Func::Slice { .. } => ir.call(label, *num_args)?,
                    }
                }
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
                    if matches!(op, ArithOp::Times) && ir.support_times2() {
                        if let Some(Instruction::LoadMath {
                            op: ArithOp::Times, ..
                        }) = iter.peek()
                        {
                            if let Some(Instruction::LoadMath {
                                dst: d2,
                                s1: s2,
                                loc: l2,
                                ..
                            }) = iter.next()
                            {
                                ir.times2_loc(*dst, *s1, *loc, d2, s2, l2);
                                continue;
                            }
                        }
                    }

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

        self.create_funclets(ir, funclets)?;

        Ok(())
    }
}

impl Instruction {
    fn dst(&self) -> Reg {
        match self {
            Instruction::Bi { dst, .. } => *dst,
            Instruction::Uni { dst, .. } => *dst,
            Instruction::Fused { dst, .. } => *dst,
            Instruction::IfElse { dst, .. } => *dst,
            Instruction::Load { dst, .. } => *dst,
            Instruction::LoadConst { dst, .. } => *dst,
            Instruction::LoadConstMath { dst, .. } => *dst,
            Instruction::LoadMath { dst, .. } => *dst,
            Instruction::Mov { dst, .. } => *dst,
            _ => panic!("Instruction {:?} does not have a dst field.", self),
        }
    }

    fn src(&self) -> Reg {
        if let Instruction::Save { src, .. } = self {
            *src
        } else {
            panic!("Instruction {:?} does not have a src field.", self)
        }
    }

    fn s1(&self) -> Reg {
        match self {
            Instruction::Bi { s1, .. } => *s1,
            Instruction::Uni { s1, .. } => *s1,
            Instruction::LoadConstMath { s1, .. } => *s1,
            Instruction::LoadMath { s1, .. } => *s1,
            Instruction::Mov { s1, .. } => *s1,
            _ => panic!("Instruction {:?} does not have an s1 field.", self),
        }
    }

    fn s2(&self) -> Reg {
        if let Instruction::Bi { s2, .. } = self {
            *s2
        } else {
            panic!("Instruction {:?} does not have an s2 field.", self)
        }
    }

    fn loc(&self) -> Loc {
        match self {
            Instruction::Load { loc, .. } => *loc,
            Instruction::LoadMath { loc, .. } => *loc,
            Instruction::Save { loc, .. } => *loc,
            _ => panic!("Instruction {:?} does not have a loc field.", self),
        }
    }

    fn label(&self) -> Option<String> {
        match self {
            Instruction::Label { label } => Some(label.to_string()),
            Instruction::Branch { label } => Some(label.to_string()),
            Instruction::BranchIf { label, .. } => Some(label.to_string()),
            Instruction::Call { label, .. } => Some(label.to_string()),
            _ => None,
        }
    }

    fn regs(&self) -> Vec<Reg> {
        match self {
            Instruction::Bi { dst, s1, s2, .. } => vec![*dst, *s1, *s2],
            Instruction::Uni { dst, s1, .. } => vec![*dst, *s1],
            Instruction::Mov { dst, s1 } => vec![*dst, *s1],
            Instruction::BranchIf { cond, .. } => vec![*cond],
            Instruction::Load { dst, .. } | Instruction::LoadConst { dst, .. } => vec![*dst],
            Instruction::LoadComplex { xd, yd, .. } => vec![*xd, *yd],
            Instruction::LoadMath { dst, s1, .. } => vec![*dst, *s1],
            Instruction::LoadConstMath { dst, s1, .. } => vec![*dst, *s1],
            Instruction::Save { src, .. } => vec![*src],
            Instruction::SaveComplex { xs, ys, .. } => vec![*xs, *ys],
            Instruction::Fused { dst, a, b, c, .. } => vec![*dst, *a, *b, *c],
            Instruction::ComplexBi {
                xd,
                yd,
                x1,
                y1,
                x2,
                y2,
                ..
            } => vec![*xd, *yd, *x1, *y1, *x2, *y2],
            Instruction::IfElse {
                dst,
                true_val,
                false_val,
                ..
            } => vec![*dst, *true_val, *false_val],
            _ => Vec::new(),
        }
    }
}

impl Mir {
    fn fuse_op_mov(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        /*
         * example:
         *      %0 = Root(%2)
         *      %l = %0
         *      call power
         * becomes
         *      %l = Root(%2)
         *      call power
         */
        if let Instruction::Uni { op, .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Uni {
                        op,
                        dst: q1.dst(),
                        s1: q0.s1(),
                    });
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
        if let Instruction::Bi { op, .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Bi {
                        op,
                        dst: q1.dst(),
                        s1: q0.s1(),
                        s2: q0.s2(),
                    });
                }
            }
        };

        None
    }

    fn fuse_goto(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Branch { .. } = *q0 {
            if let Instruction::Label { .. } = *q1 {
                if q0.label().unwrap() == q1.label().unwrap() {
                    return Some(q1.clone());
                }
            }
        };

        None
    }

    fn fuse_load(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
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
        if let Instruction::Load { .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Load {
                        dst: q1.dst(),
                        loc: q0.loc(),
                    });
                }
            }
        };

        if let Instruction::LoadConst { idx, .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::LoadConst { dst: q1.dst(), idx });
                }
            }
        };

        if let Instruction::Load { .. } = *q0 {
            if let Instruction::Save { .. } = *q1 {
                if q0.loc() == q1.loc() && q0.dst() == q1.src() {
                    return Some(Instruction::Nop);
                }
            }
        };

        None
    }

    fn fuse_save(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
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
        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if q0.loc() == q1.loc() {
                    code.push(q0);
                    return Some(Instruction::Mov {
                        dst: q1.dst(),
                        s1: q0.src(),
                    });
                }
            }
        };

        if let Instruction::Save { .. } = *q0 {
            if let Instruction::LoadMath { op, .. } = *q1 {
                if q0.loc() == q1.loc() {
                    code.push(q0);
                    return Some(Instruction::Bi {
                        op: match op {
                            ArithOp::Plus => BinOp::Plus,
                            ArithOp::Minus => BinOp::Minus,
                            ArithOp::Times => BinOp::Times,
                            ArithOp::Divide => BinOp::Divide,
                        },
                        dst: q1.dst(),
                        s1: q1.s1(),
                        s2: q0.src(),
                    });
                }
            }
        }

        None
    }

    fn fuse_save3(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
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
        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Save { .. } = *q2 {
                    if q0.src() == Reg::Ret && q0.loc() == q1.loc() && q1.dst() == q2.src() {
                        return Some(Instruction::Save {
                            src: Reg::Ret,
                            loc: q2.loc(),
                        });
                    }
                }
            }
        };

        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Load { .. } = *q2 {
                    if q0.loc() == q2.loc() && q1.dst() != q2.dst() {
                        code.push(q0);
                        code.push(&Instruction::Mov {
                            dst: q2.dst(),
                            s1: q0.src(),
                        });
                        return Some(q1.clone());
                    }
                }
            }
        };

        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::LoadMath { op, .. } = *q2 {
                    if q0.loc() == q2.loc() && q2.s1() == q1.dst() && q0.src() != Reg::Ret {
                        code.push(q0);
                        code.push(&Instruction::Load {
                            dst: Reg::Ret,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Bi {
                            op: match op {
                                ArithOp::Plus => BinOp::Plus,
                                ArithOp::Minus => BinOp::Minus,
                                ArithOp::Times => BinOp::Times,
                                ArithOp::Divide => BinOp::Divide,
                            },
                            dst: q2.dst(),
                            s1: Reg::Ret,
                            s2: q0.src(),
                        });
                    }
                }
            }
        }

        None
    }

    fn fuse_recip(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Uni {
            op: UniOp::Recip, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Times, ..
            } = *q1
            {
                if q0.dst() == q1.s2() {
                    return Some(Instruction::Bi {
                        op: BinOp::Divide,
                        dst: q1.dst(),
                        s1: q1.s1(),
                        s2: q0.s1(),
                    });
                } else if q0.dst() == q1.s1() {
                    return Some(Instruction::Bi {
                        op: BinOp::Divide,
                        dst: q1.dst(),
                        s1: q1.s2(),
                        s2: q0.s1(),
                    });
                }
            }
        }

        if let Instruction::Uni { op: UniOp::Neg, .. } = *q0 {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q1
            {
                if q0.dst() == q1.s2() {
                    return Some(Instruction::Bi {
                        op: BinOp::Minus,
                        dst: q1.dst(),
                        s1: q1.s1(),
                        s2: q0.s1(),
                    });
                } else if q0.dst() == q1.s1() {
                    return Some(Instruction::Bi {
                        op: BinOp::Minus,
                        dst: q1.dst(),
                        s1: q1.s2(),
                        s2: q0.s1(),
                    });
                }
            }
        }

        None
    }

    fn fuse_recip3(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Uni {
            op: UniOp::Recip, ..
        } = *q0
        {
            if !q1.regs().contains(&q0.dst()) && !q1.regs().contains(&q0.s1()) {
                if let Instruction::Bi {
                    op: BinOp::Times, ..
                } = *q2
                {
                    if q0.dst() == q2.s2() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Divide,
                            dst: q2.dst(),
                            s1: q2.s1(),
                            s2: q0.s1(),
                        });
                    } else if q0.dst() == q2.s1() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Divide,
                            dst: q2.dst(),
                            s1: q2.s2(),
                            s2: q0.s1(),
                        });
                    }
                }
            }
        }

        if let Instruction::Uni { op: UniOp::Neg, .. } = *q0 {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q2
            {
                if !q1.regs().contains(&q0.dst()) && !q1.regs().contains(&q0.s1()) {
                    if q0.dst() == q2.s2() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Minus,
                            dst: q2.dst(),
                            s1: q2.s1(),
                            s2: q0.s1(),
                        });
                    } else if q0.dst() == q2.s1() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Minus,
                            dst: q2.dst(),
                            s1: q2.s2(),
                            s2: q0.s1(),
                        });
                    }
                }
            }
        }

        None
    }

    fn fuse_zero(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Uni {
            op: UniOp::IsZero, ..
        } = *q0
        {
            if let Instruction::Uni { op: UniOp::Neg, .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Uni {
                        op: UniOp::IsNotZero,
                        dst: q1.dst(),
                        s1: q0.s1(),
                    });
                }
            }
        }

        if let Instruction::Uni { op: UniOp::Not, .. } = *q0 {
            if let Instruction::Uni {
                op: UniOp::IsZero, ..
            } = *q1
            {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Uni {
                        op: UniOp::IsNotZero,
                        dst: q1.dst(),
                        s1: q0.s1(),
                    });
                }
            }
        }

        None
    }

    fn fuse_fma(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        // TODO: fix the FMA bug for complex noted on `runtests complex`
        if !self.config.fastmath()
        /*|| self.config.is_complex()*/
        {
            return None;
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s1(),
                    });
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Minus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::MulSub,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::NegMulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s1(),
                    });
                }
            }
        }

        if let Instruction::LoadMath {
            op: ArithOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s1(),
                    });
                }
            }
        }

        if let Instruction::LoadMath {
            op: ArithOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Minus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::MulSub,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::NegMulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s1(),
                    });
                }
            }
        }

        None
    }

    fn fuse_fma3(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
        // TODO: fix the FMA bug for complex noted on `runtests complex`
        if !self.config.fastmath()
        /*|| self.config.is_complex()*/
        {
            return None;
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::LoadConst { idx, .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Plus, ..
                } = *q2
                {
                    if (q2.s1() == q0.dst() && q2.s2() == q1.dst())
                        || (q2.s1() == q1.dst() && q2.s2() == q0.dst())
                    {
                        code.push(&Instruction::LoadConst {
                            dst: Reg::Temp,
                            idx,
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Plus, ..
                } = *q2
                {
                    if (q2.s1() == q0.dst() && q2.s2() == q1.dst())
                        || (q2.s1() == q1.dst() && q2.s2() == q0.dst())
                    {
                        code.push(&Instruction::Load {
                            dst: Reg::Temp,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::LoadConst { idx, .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Minus, ..
                } = *q2
                {
                    if q2.s1() == q0.dst() && q2.s2() == q1.dst() {
                        code.push(&Instruction::LoadConst {
                            dst: Reg::Temp,
                            idx,
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulSub,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    } else if q2.s1() == q1.dst() && q2.s2() == q0.dst() {
                        code.push(&Instruction::LoadConst {
                            dst: Reg::Temp,
                            idx,
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::NegMulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Minus, ..
                } = *q2
                {
                    if q2.s1() == q0.dst() && q2.s2() == q1.dst() {
                        code.push(&Instruction::Load {
                            dst: Reg::Temp,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulSub,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    } else if q2.s1() == q1.dst() && q2.s2() == q0.dst() {
                        code.push(&Instruction::Load {
                            dst: Reg::Temp,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::NegMulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        None
    }

    fn fuse_times2(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
        if !self.config.is_complex() {
            return None;
        }

        if let Instruction::LoadMath {
            op: ArithOp::Times, ..
        } = *q0
        {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::LoadMath {
                    op: ArithOp::Times, ..
                } = *q2
                {
                    if q1.dst() == q2.s1() {
                        code.push(q1);
                        code.push(q0);
                        return Some(q2.clone());
                    }
                }
            }
        };

        None
    }

    fn fuse_times2_5(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
        q3: &Instruction,
        q4: &Instruction,
    ) -> Option<Instruction> {
        if !self.config.is_complex() {
            return None;
        }

        if let Instruction::Load { .. } = *q0 {
            if let Instruction::LoadMath {
                op: ArithOp::Times, ..
            } = *q1
            {
                if let Instruction::Save { .. } = *q2 {
                    if let Instruction::Load { .. } = *q3 {
                        if let Instruction::LoadMath {
                            op: ArithOp::Times, ..
                        } = *q4
                        {
                            if q0.dst() == q1.s1()
                                && q1.dst() == q2.src()
                                && q3.dst() == q4.s1()
                                && q2.loc() != q3.loc()
                                && q2.loc() != q4.loc()
                            {
                                code.push(&Instruction::Load {
                                    dst: Reg::Ret,
                                    loc: q0.loc(),
                                });
                                code.push(q3);
                                code.push(&Instruction::LoadMath {
                                    op: ArithOp::Times,
                                    dst: Reg::Ret,
                                    s1: Reg::Ret,
                                    loc: q1.loc(),
                                });
                                code.push(q4);
                                return Some(Instruction::Save {
                                    src: Reg::Ret,
                                    loc: q2.loc(),
                                });
                            }
                        }
                    }
                }
            }
        }

        None
    }

    fn fuse1(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
        q3: &Instruction,
        q4: &Instruction,
    ) -> (Instruction, usize) {
        if let Some(v) = self.fuse_times2_5(code, q0, q1, q2, q3, q4) {
            (v, 5)
        } else if let Some(v) = self.fuse_save3(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_fma3(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_times2(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_recip3(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_fma(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_op_mov(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_load(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_save(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_recip(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_zero(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_goto(code, q0, q1) {
            (v, 2)
        } else {
            (Instruction::Nop, 0)
        }
    }

    pub fn optimize_peephole(&mut self, _stage: usize) -> bool {
        let mut success = false;

        let mut code = MirWriter::new();
        let mut iter = self.code.iter_mut();
        let mut q0: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q1: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q2: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q3: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q4: Instruction = iter.next().unwrap_or(Instruction::End).clone();

        while !matches!(q0, Instruction::End) {
            let (top, num_consumed) = self.fuse1(&mut code, &q0, &q1, &q2, &q3, &q4);
            success |= num_consumed > 1;

            match num_consumed {
                // no matches. move to the next item.
                0 => {
                    code.push(&q0);
                    q0 = q1;
                    q1 = q2;
                    q2 = q3;
                    q3 = q4;
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0 and q1 match.
                2 => {
                    q0 = top;
                    q1 = q2;
                    q2 = q3;
                    q3 = q4;
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0, q1, and q2 match.
                3 => {
                    q0 = top;
                    q1 = q3;
                    q2 = q4;
                    q3 = iter.next().unwrap_or(Instruction::End).clone();
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0, q1, q2, and q3 match.
                4 => {
                    q0 = top;
                    q1 = q4;
                    q2 = iter.next().unwrap_or(Instruction::End).clone();
                    q3 = iter.next().unwrap_or(Instruction::End).clone();
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0, q1, q2, q3, q4 match.
                5 => {
                    q0 = top;
                    q1 = iter.next().unwrap_or(Instruction::End).clone();
                    q2 = iter.next().unwrap_or(Instruction::End).clone();
                    q3 = iter.next().unwrap_or(Instruction::End).clone();
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                _ => unreachable!(),
            }
        }

        self.code = code;
        success
    }
}

impl Mir {
    pub fn print_stats(&self, name: &str, size: usize) {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut times2: usize = 0;
        let mut stack_count: usize = 0;
        let mut hist: [usize; 32] = [0; 32];

        let mut iter = self.code.iter().peekable();

        while let Some(ins) = iter.next() {
            match ins {
                Instruction::LoadMath {
                    op: ArithOp::Times, ..
                } => {
                    if let Some(Instruction::LoadMath {
                        op: ArithOp::Times, ..
                    }) = iter.peek()
                    {
                        times2 += 1;
                    }
                }
                Instruction::Save {
                    loc: Loc::Stack(idx),
                    ..
                }
                | Instruction::SaveComplex {
                    loc: Loc::Stack(idx),
                    ..
                } => {
                    stack_count = stack_count.max(idx as usize);
                }
                _ => {}
            }

            for r in ins.regs() {
                if let Reg::Gen(r) = r {
                    hist[r as usize] += 1;
                }
            }

            let desc = ins.desc();
            match counts.get_mut(&desc) {
                Some(k) => {
                    *k += 1;
                }
                None => {
                    counts.insert(desc, 1);
                }
            }
        }

        let mut fs = fs::File::create(name).unwrap();
        let _ = writeln!(fs, "---------------------------------");
        let _ = writeln!(fs, "#! STATS");
        let _ = writeln!(fs, "version = {}", env!("CARGO_PKG_VERSION"));
        let _ = writeln!(fs, "{} instructions", self.code.ip);
        let _ = writeln!(fs, "{} stack count", stack_count);
        let _ = writeln!(fs, "--------------");

        for (k, v) in counts.iter() {
            let _ = writeln!(fs, "{} x {}", k, v);
        }
        let _ = writeln!(fs, "times2 x {}", times2);
        let _ = writeln!(fs, "compiled size {} bytes", size);
        let _ = writeln!(fs, "---------------------------------");

        /*
        for i in 0..32 {
            println!("reg({}) usage is {}", i, hist[i]);
        }
        */

        // let name = name.replace("_stats.txt", "_config.toml");
        // self.config.to_toml(&name);
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
        let _ = writeln!(fs, "#! bytecode");
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
