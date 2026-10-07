use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::rc::Rc;

use anyhow::Result;

use super::code::{Func, VirtualTable};
use super::complexify::Complexifier;
use super::config::Config;
use super::generator::Generator;
use super::machine::MachineCode;
use super::serializer::MirWriter;
use super::symbol::Loc;
use super::utils::{Compiled, CompiledFunc, Reg};

mod interpreter;
mod peephole;

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
    Sign,
    Abs2,
    TimesI,
    TimesNegI,
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

/*
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum FuncletOp {
    Times,
    Divide,
    Root,
    TimesComplex,
    DivideComplex,
}
*/

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
    LoadArgs {
        locs: Vec<Loc>,
        complex: bool,
        ultra: bool,
    },
    SaveArgs {
        num_args: u8,
        complex: bool,
        ultra: bool,
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
        let s = format!("{:?}", self);
        s.hash(state);
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Nop => write!(f, "nop"),
            Instruction::End => write!(f, "end"),
            Instruction::Uni { op, dst, s1 } => write!(f, "{:?} := {:?}({:?})", dst, op, s1),
            Instruction::Bi { op, dst, s1, s2 } => {
                write!(f, "{:?} := {:?} {:?} {:?}", dst, s1, op, s2)
            }
            Instruction::Call { label, .. } => write!(f, "call {}", label),
            Instruction::Mov { dst, s1 } => write!(f, "{:?} := {:?}", dst, s1),
            Instruction::Load { dst, loc } => write!(f, "{:?} := {:?}", dst, loc),
            Instruction::Save { src, loc } => write!(f, "{:?} := {:?}", loc, src),
            Instruction::LoadComplex { xd, yd, loc } => {
                write!(f, "({:?} + {:?}*im) := {:?}", xd, yd, loc)
            }
            Instruction::SaveComplex { xs, ys, loc } => {
                write!(f, "{:?} := ({:?} + {:?}*im)", loc, xs, ys)
            }
            Instruction::LoadConst { dst, idx } => write!(f, "{:?} := consts[{:?}]", dst, idx),
            Instruction::LoadArgs {
                locs,
                complex: false,
                ultra,
            } => write!(f, "Load Args {:?}; real, {}", locs, ultra),
            Instruction::LoadArgs {
                locs,
                complex: true,
                ultra,
            } => write!(f, "Load Args {:?}; complex, {}", locs, ultra),
            Instruction::SaveArgs {
                num_args,
                complex: false,
                ultra,
            } => write!(f, "Save Args; n = {}; real, {}", num_args, ultra),
            Instruction::SaveArgs {
                num_args,
                complex: true,
                ultra,
            } => write!(f, "Save Args; n = {}; complex, {}", num_args, ultra),
            Instruction::Fused { op, dst, a, b, c } => match op {
                FusedOp::MulAdd => write!(f, "{:?} := {:?} * {:?} + {:?}", dst, a, b, c),
                FusedOp::NegMulAdd => write!(f, "{:?} := - {:?} * {:?} + {:?}", dst, a, b, c),
                FusedOp::MulSub => write!(f, "{:?} := {:?} * {:?} - {:?}", dst, a, b, c),
                FusedOp::NegMulSub => write!(f, "{:?} := - {:?} * {:?} - {:?}", dst, a, b, c),
            },
            Instruction::IfElse {
                dst,
                true_val,
                false_val,
                cond,
            } => write!(
                f,
                "{:?} := {:?} ? {:?} : {:?}",
                dst, cond, true_val, false_val
            ),
            Self::Label { label } => write!(f, "{:?}:", label),
            Self::Branch { label } => write!(f, "goto {:?}", label),
            Self::BranchIf {
                cond,
                label,
                is_else,
            } => {
                if *is_else {
                    write!(f, "if not {:?} goto {:?}", cond, label)
                } else {
                    write!(f, "if {:?} goto {:?}", cond, label)
                }
            }
            Self::LoadMath { op, dst, s1, loc } => {
                write!(f, "{:?} := {:?} {:?} {:?} # load/math", dst, s1, op, loc)
            }
            Self::LoadConstMath { op, dst, s1, idx } => {
                write!(
                    f,
                    "{:?} := {:?} {:?} consts[{:?}] # load const/math",
                    dst, s1, op, idx
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
                    xd, yd, x1, y1, op, x2, y2
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
            Instruction::Uni { op, .. } => format!("uniop {:?}", op),
            Instruction::Bi { op, .. } => format!("binop {:?}", op),
            Instruction::Call { label, .. } => format!("call {}", label),
            Instruction::Mov { .. } => "mov".into(),
            Instruction::Load { .. } => "load".into(),
            Instruction::Save { .. } => "save".into(),
            Instruction::LoadComplex { .. } => "load_complex".into(),
            Instruction::SaveComplex { .. } => "save_complex".into(),
            Instruction::LoadConst { .. } => "load_const".into(),
            Instruction::LoadArgs { complex: false, .. } => "load_arg".into(),
            Instruction::LoadArgs { complex: true, .. } => "load_arg_complex".into(),
            Instruction::SaveArgs { complex: false, .. } => "save_arg".into(),
            Instruction::SaveArgs { complex: true, .. } => "save_arg_complex".into(),
            Instruction::Fused { op, .. } => format!("fused op {:?}", op),
            Instruction::IfElse { .. } => "if_else".into(),
            Self::Label { .. } => "label".into(),
            Self::Branch { .. } => "branch".into(),
            Self::BranchIf { .. } => "branch_if".into(),
            Self::LoadMath { op, .. } => format!("load math {:?}", op),
            Self::LoadConstMath { op, .. } => format!("load const math {:?}", op),
            Self::ComplexBi { op, .. } => format!("complex binop {:?}", op),
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
        writeln!(f, "config: {:?}", self.config)?;
        writeln!(f, "ip: {}", self.code.ip)?;

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
        let mut mask: u32 = if self.config.compress() { !0 } else { 0 };

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

    pub fn load_args(&mut self, locs: Vec<Loc>, ultra: bool) {
        self.push(Instruction::LoadArgs {
            locs,
            complex: false,
            ultra,
        })
    }

    pub fn save_args(&mut self, num_args: u8, ultra: bool) {
        self.push(Instruction::SaveArgs {
            num_args,
            complex: false,
            ultra,
        })
    }

    pub fn load_args_complex(&mut self, locs: Vec<Loc>, ultra: bool) {
        self.push(Instruction::LoadArgs {
            locs,
            complex: true,
            ultra,
        })
    }

    pub fn save_args_complex(&mut self, num_args: u8, ultra: bool) {
        self.push(Instruction::SaveArgs {
            num_args,
            complex: true,
            ultra,
        })
    }

    pub fn neg(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Neg,
            dst,
            s1,
        });
    }

    pub fn sign(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Sign,
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

    pub fn abs2(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::Abs2,
            dst,
            s1,
        });
    }

    pub fn times_i(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::TimesI,
            dst,
            s1,
        });
    }

    pub fn times_neg_i(&mut self, dst: Reg, s1: Reg) {
        self.push(Instruction::Uni {
            op: UniOp::TimesNegI,
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
        // num_args == 0 means a subroutine call
        if num_args > 0 {
            let _ = self.find_op(op)?;
        }
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

        let op = if self.config.is_complex() && !self.config.is_external_func(op) {
            &format!("cplx_{}", op)
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
    fn rerun_uniop(ir: &mut dyn Generator, op: UniOp, dst: Reg, s1: Reg) {
        match op {
            UniOp::Neg => ir.neg(dst, s1),
            UniOp::Not => ir.not(dst, s1),
            UniOp::Abs => ir.abs(dst, s1),
            UniOp::Abs2 => ir.abs2(dst, s1),
            UniOp::TimesI => ir.times_i(dst, s1),
            UniOp::TimesNegI => ir.times_neg_i(dst, s1),
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
            UniOp::Sign => ir.sign(dst, s1),
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
        // let mut funclets: HashSet<(FuncletOp, Vec<Reg>)> = HashSet::new();

        let mut iter = self.code.iter().peekable();

        while let Some(ins) = iter.next() {
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
                            ir.load_mem_complex(*xd, *yd, *idx);
                        }
                        Loc::Stack(idx) => {
                            ir.load_stack_complex(*xd, *yd, *idx);
                        }
                        Loc::Param(idx) => {
                            ir.load_param_complex(*xd, *yd, *idx);
                        }
                    };
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    match loc {
                        Loc::Mem(idx) => {
                            ir.save_mem_complex(*xs, *ys, *idx);
                        }
                        Loc::Stack(idx) => {
                            ir.save_stack_complex(*xs, *ys, *idx);
                        }
                        Loc::Param(_) => unreachable!(),
                    };
                }
                Instruction::LoadConst { dst, idx } => {
                    ir.load_const(*dst, *idx);
                }
                Instruction::LoadArgs {
                    locs,
                    ultra,
                    complex,
                } => {
                    if *complex {
                        ir.load_args_complex(locs.clone(), *ultra);
                    } else {
                        ir.load_args(locs.clone(), *ultra);
                    }
                }
                Instruction::SaveArgs {
                    num_args,
                    ultra,
                    complex,
                } => {
                    if *complex {
                        ir.save_args_complex(*num_args, *ultra);
                    } else {
                        ir.save_args(*num_args, *ultra);
                    }
                }
                Instruction::Call { label, num_args } => {
                    if *num_args == 0 {
                        ir.call_funclet(label);
                    } else {
                        let f = self.find_op(label).unwrap();
                        match f {
                            Func::Unary(_)
                            | Func::Binary(_)
                            | Func::PairedUnary(_)
                            | Func::Slice { .. }
                            | Func::App(_) => ir.call(label, *num_args)?,
                            Func::UnaryCplx(_) | Func::BinaryCplx(_) => {
                                ir.call_complex(label, *num_args)?
                            }
                            Func::Recursive => ir.call("@self", *num_args)?,
                        }
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
                Instruction::Branch { label } => {
                    if label == ".ret" {
                        ir.ret();
                    } else {
                        ir.branch(label)
                    }
                }
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

        Ok(())
    }
}

impl Instruction {
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

        let name = name.replace("_stats.txt", "_config.toml");
        self.config.to_toml(&name);
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
