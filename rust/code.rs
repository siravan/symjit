use crate::register::Word;
use anyhow::{anyhow, Result};
use num_traits::Float;

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

pub type BinaryFunc<T> = extern "C" fn(T, T) -> T;

pub struct VirtualTable<T>(T);

impl<T: Float> VirtualTable<T> {
    /// Creates a VirtualTable (a Vec of references to functions)
    /// from a function table (a Vec of function names)
    pub fn from_names(ft: &[String]) -> Vec<BinaryFunc<T>> {
        let mut vt: Vec<BinaryFunc<T>> = Vec::new();

        for f in ft.iter() {
            vt.push(Self::from_str(f).unwrap());
        }

        vt
    }

    /// Confirms that all the names in ft are valid,
    /// i.e., corresponds to actual functions    
    pub fn confirm(ft: &[String]) -> Result<()> {
        for f in ft.iter() {
            let _ = Self::from_str(f)?;
        }
        Ok(())
    }

    // Finds the function reference for op
    fn from_str(op: &str) -> Result<BinaryFunc<T>> {
        let f = match op {
            "nop" => Self::nop,
            "mov" => Self::mov,
            "plus" => Self::plus,
            "minus" => Self::minus,
            "neg" => Self::neg,
            "times" => Self::times,
            "divide" => Self::divide,
            "rem" => Self::rem,
            "power" => Self::power,
            "gt" => Self::gt,
            "geq" => Self::geq,
            "lt" => Self::lt,
            "leq" => Self::leq,
            "eq" => Self::eq,
            "neq" => Self::neq,
            "abs" => Self::abs,
            "and" => Self::and,
            "or" => Self::or,
            "xor" => Self::xor,
            "if_pos" => Self::if_pos,
            "if_neg" => Self::if_neg,
            "sin" => Self::sin,
            "cos" => Self::cos,
            "tan" => Self::tan,
            "csc" => Self::csc,
            "sec" => Self::sec,
            "cot" => Self::cot,
            "sinh" => Self::sinh,
            "cosh" => Self::cosh,
            "tanh" => Self::tanh,
            "csch" => Self::csch,
            "sech" => Self::sech,
            "coth" => Self::coth,
            "arcsin" => Self::asin,
            "arccos" => Self::acos,
            "arctan" => Self::atan,
            "arcsinh" => Self::asinh,
            "arccosh" => Self::acosh,
            "arctanh" => Self::atanh,
            "exp" => Self::exp,
            "ln" => Self::ln,
            "log" => Self::log,
            "root" => Self::root,
            "ifelse" => Self::nop,
            "square" => Self::square,
            "cube" => Self::cube,
            "recip" => Self::recip,
            _ => {
                return Err(anyhow!("op_code {} not found", op));
            }
        };

        Ok(f)
    }

    pub extern "C" fn nop(_x: T, _y: T) -> T {
        T::zero()
    }

    pub extern "C" fn mov(x: T, _y: T) -> T {
        x
    }

    pub extern "C" fn plus(x: T, y: T) -> T {
        x + y
    }

    pub extern "C" fn minus(x: T, y: T) -> T {
        x - y
    }

    pub extern "C" fn neg(x: T, _y: T) -> T {
        -x
    }

    pub extern "C" fn abs(x: T, _y: T) -> T {
        x.abs()
    }

    pub extern "C" fn times(x: T, y: T) -> T {
        x * y
    }

    pub extern "C" fn divide(x: T, y: T) -> T {
        x / y
    }

    pub extern "C" fn rem(x: T, y: T) -> T {
        x % y
    }

    pub extern "C" fn power(x: T, y: T) -> T {
        x.powf(y)
    }

    pub extern "C" fn gt(x: T, y: T) -> T {
        if x > y {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn geq(x: T, y: T) -> T {
        if x >= y {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn lt(x: T, y: T) -> T {
        if x < y {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn leq(x: T, y: T) -> T {
        if x <= y {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn eq(x: T, y: T) -> T {
        if x == y {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn neq(x: T, y: T) -> T {
        if x != y {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn and(x: T, y: T) -> T {
        if x > T::zero() && y > T::zero() {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn or(x: T, y: T) -> T {
        if x > T::zero() || y > T::zero() {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn xor(x: T, y: T) -> T {
        if x * y < T::zero() {
            T::one()
        } else {
            -T::one()
        }
    }

    pub extern "C" fn if_pos(x: T, y: T) -> T {
        if x > T::zero() {
            y
        } else {
            T::zero()
        }
    }

    pub extern "C" fn if_neg(x: T, y: T) -> T {
        if x < T::zero() {
            y
        } else {
            T::zero()
        }
    }

    pub extern "C" fn sin(x: T, _y: T) -> T {
        x.sin()
    }

    pub extern "C" fn cos(x: T, _y: T) -> T {
        x.cos()
    }

    pub extern "C" fn tan(x: T, _y: T) -> T {
        x.tan()
    }

    pub extern "C" fn csc(x: T, _y: T) -> T {
        T::one() / x.sin()
    }

    pub extern "C" fn sec(x: T, _y: T) -> T {
        T::one() / x.cos()
    }

    pub extern "C" fn cot(x: T, _y: T) -> T {
        T::one() / x.tan()
    }

    pub extern "C" fn sinh(x: T, _y: T) -> T {
        x.sinh()
    }

    pub extern "C" fn cosh(x: T, _y: T) -> T {
        x.cosh()
    }

    pub extern "C" fn tanh(x: T, _y: T) -> T {
        x.tanh()
    }

    pub extern "C" fn csch(x: T, _y: T) -> T {
        T::one() / x.sinh()
    }

    pub extern "C" fn sech(x: T, _y: T) -> T {
        T::one() / x.cosh()
    }

    pub extern "C" fn coth(x: T, _y: T) -> T {
        T::one() / x.tanh()
    }

    pub extern "C" fn asin(x: T, _y: T) -> T {
        x.asin()
    }

    pub extern "C" fn acos(x: T, _y: T) -> T {
        x.acos()
    }

    pub extern "C" fn atan(x: T, _y: T) -> T {
        x.atan()
    }

    pub extern "C" fn asinh(x: T, _y: T) -> T {
        x.asinh()
    }

    pub extern "C" fn acosh(x: T, _y: T) -> T {
        x.acosh()
    }

    pub extern "C" fn atanh(x: T, _y: T) -> T {
        x.atanh()
    }

    pub extern "C" fn exp(x: T, _y: T) -> T {
        x.exp()
    }

    pub extern "C" fn ln(x: T, _y: T) -> T {
        x.ln()
    }

    pub extern "C" fn log(x: T, _y: T) -> T {
        x.log10()
    }

    pub extern "C" fn root(x: T, _y: T) -> T {
        x.sqrt()
    }

    pub extern "C" fn square(x: T, _y: T) -> T {
        x * x
    }

    pub extern "C" fn cube(x: T, _y: T) -> T {
        x * x * x
    }

    pub extern "C" fn recip(x: T, _y: T) -> T {
        T::one() / x
    }
}
