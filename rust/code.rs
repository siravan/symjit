use crate::register::Word;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Proc(pub usize);

#[derive(Clone)]
pub enum Instruction {
    Unary {
        op: String,
        x: Word,
        dst: Word,
        p: Proc,
    },
    Binary {
        op: String,
        x: Word,
        y: Word,
        dst: Word,
        p: Proc,
    },
    IfElse {
        x1: Word,
        x2: Word,
        cond: Word,
        dst: Word,
    },
    Num {
        val: f64,
        dst: Word,
    },
    Var {
        name: String,
        reg: Word,
    },
    Eq {
        dst: Word,
    },
    Nop,
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Instruction::Unary { op, x, dst, .. } => {
                write!(f, "r{:<6}← {}(r{})", dst.0, op, x.0)
            }
            Instruction::Binary { op, x, y, dst, .. } => {
                write!(f, "r{:<6}← r{} {} r{}", dst.0, x.0, op, y.0)
            }
            Instruction::IfElse { x1, x2, cond, dst } => {
                write!(f, "r{:<6}← r{} ? r{} : r{}", dst.0, cond.0, x1.0, x2.0)
            }
            Instruction::Num { val, dst } => write!(f, "r{:<6}= {}", dst.0, val),
            Instruction::Var { name, reg } => write!(f, "r{:<6}:: {}", reg.0, name),
            Instruction::Eq { dst } => write!(f, "r{:<6}= ?", dst.0),
            Instruction::Nop => write!(f, "nop"),
        }
    }
}

impl std::fmt::Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

pub type BinaryFunc = fn(f64, f64) -> f64;

pub struct Code {}

impl Code {
    pub fn from_str(op: &str) -> BinaryFunc {
        match op {
            "nop" => Code::nop,
            "mov" => Code::mov,
            "plus" => Code::plus,
            "minus" => Code::minus,
            "neg" => Code::neg,
            "times" => Code::times,
            "divide" => Code::divide,
            "rem" => Code::rem,
            "power" => Code::power,
            "gt" => Code::gt,
            "geq" => Code::geq,
            "lt" => Code::lt,
            "leq" => Code::leq,
            "eq" => Code::eq,
            "neq" => Code::neq,
            "and" => Code::and,
            "or" => Code::or,
            "xor" => Code::xor,
            "if_pos" => Code::if_pos,
            "if_neg" => Code::if_neg,
            "sin" => Code::sin,
            "cos" => Code::cos,
            "tan" => Code::tan,
            "csc" => Code::csc,
            "sec" => Code::sec,
            "cot" => Code::cot,
            "arcsin" => Code::asin,
            "arccos" => Code::acos,
            "arctan" => Code::atan,
            "exp" => Code::exp,
            "ln" => Code::ln,
            "log" => Code::log,
            "root" => Code::root,
            "ifelse" => Code::nop,
            _ => {
                let msg = format!("op_code {} not found", op);
                panic!("{}", msg)
            }
        }
    }

    pub fn nop(_x: f64, _y: f64) -> f64 {
        0.0
    }

    pub fn mov(x: f64, _y: f64) -> f64 {
        x
    }

    pub fn plus(x: f64, y: f64) -> f64 {
        x + y
    }

    pub fn minus(x: f64, y: f64) -> f64 {
        x - y
    }

    pub fn neg(x: f64, _y: f64) -> f64 {
        -x
    }

    pub fn times(x: f64, y: f64) -> f64 {
        x * y
    }

    pub fn divide(x: f64, y: f64) -> f64 {
        x / y
    }

    pub fn rem(x: f64, y: f64) -> f64 {
        x % y
    }

    pub fn power(x: f64, y: f64) -> f64 {
        x.powf(y)
    }

    pub fn gt(x: f64, y: f64) -> f64 {
        if x > y {
            1.0
        } else {
            -1.0
        }
    }

    pub fn geq(x: f64, y: f64) -> f64 {
        if x >= y {
            1.0
        } else {
            -1.0
        }
    }

    pub fn lt(x: f64, y: f64) -> f64 {
        if x < y {
            1.0
        } else {
            -1.0
        }
    }

    pub fn leq(x: f64, y: f64) -> f64 {
        if x <= y {
            1.0
        } else {
            -1.0
        }
    }

    pub fn eq(x: f64, y: f64) -> f64 {
        if x == y {
            1.0
        } else {
            -1.0
        }
    }

    pub fn neq(x: f64, y: f64) -> f64 {
        if x != y {
            1.0
        } else {
            -1.0
        }
    }

    pub fn and(x: f64, y: f64) -> f64 {
        if x > 0.0 && y > 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn or(x: f64, y: f64) -> f64 {
        if x > 0.0 || y > 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn xor(x: f64, y: f64) -> f64 {
        if x * y < 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn if_pos(x: f64, y: f64) -> f64 {
        if x > 0.0 {
            y
        } else {
            0.0
        }
    }

    pub fn if_neg(x: f64, y: f64) -> f64 {
        if x < 0.0 {
            y
        } else {
            0.0
        }
    }

    pub fn sin(x: f64, _y: f64) -> f64 {
        x.sin()
    }

    pub fn cos(x: f64, _y: f64) -> f64 {
        x.cos()
    }

    pub fn tan(x: f64, _y: f64) -> f64 {
        x.tan()
    }

    pub fn csc(x: f64, _y: f64) -> f64 {
        1.0 / x.sin()
    }

    pub fn sec(x: f64, _y: f64) -> f64 {
        1.0 / x.cos()
    }

    pub fn cot(x: f64, _y: f64) -> f64 {
        1.0 / x.tan()
    }

    pub fn asin(x: f64, _y: f64) -> f64 {
        x.asin()
    }

    pub fn acos(x: f64, _y: f64) -> f64 {
        x.acos()
    }

    pub fn atan(x: f64, _y: f64) -> f64 {
        x.atan()
    }

    pub fn exp(x: f64, _y: f64) -> f64 {
        x.exp()
    }

    pub fn ln(x: f64, _y: f64) -> f64 {
        x.ln()
    }

    pub fn log(x: f64, _y: f64) -> f64 {
        x.log(10.0)
    }

    pub fn root(x: f64, _y: f64) -> f64 {
        x.sqrt()
    }
}
