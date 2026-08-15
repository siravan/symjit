use std::fmt;

use num_complex::Complex;
use num_rational::Rational64;
use num_traits::{FromPrimitive, One, Zero};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuiltinSymbol(pub u32);

impl<'de> serde::Deserialize<'de> for BuiltinSymbol {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id: u32 = u32::deserialize(deserializer)?;
        Ok(BuiltinSymbol(id))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum Slot {
    /// An entry in the list of parameters.
    Param(usize),
    /// An entry in the list of constants.
    Const(usize),
    /// An entry in the list of temporary storage.
    Temp(usize),
    /// An entry in the list of results.
    Out(usize),
    /// Static-Single-Assignment Form
    Static(usize),
    Arg(usize),
}

#[derive(Clone)]
struct VecSlot(Vec<Slot>);

#[derive(Debug, Clone, Deserialize)]
pub enum Instruction {
    /// `Add(o, [i0,...,i_n])` means `o = i0 + ... + i_n`.
    Add(Slot, Vec<Slot>, usize),
    /// `Mul(o, [i0,...,i_n])` means `o = i0 * ... * i_n`.
    Mul(Slot, Vec<Slot>, usize),
    /// `Pow(o, b, e)` means `o = b^e`.
    Pow(Slot, Slot, i64, bool),
    /// `Powf(o, b, e)` means `o = b^e`.
    Powf(Slot, Slot, Slot, bool),
    /// A function that has a known evaluator or is external, given a symbol name, tags, and arguments.
    /// `Fun(o, (s, t, a), is_real)` means `o = s(t, a)`.
    /// The `is_real` flag indicates whether the function is expected to yield a real number.
    /// Fun(Slot, Box<(Symbol, Vec<String>, Vec<Slot>)>, bool),
    ///
    /// Note that Symjit uses the following simplified version of Fun:
    Fun(Slot, String, Vec<Slot>, bool),
    /// `ExternalFun(o, s, a,...)` means `o = s(a, ...)`, where `s` is an external function.
    ExternalFun(Slot, String, Vec<Slot>),
    /// `Assign(o, v)` means `o = v`.
    Assign(Slot, Slot),
    /// `IfElse(cond, label)` means jump to `label` if `cond` is zero.
    IfElse(Slot, usize),
    /// Unconditional jump to `label`.
    Goto(usize),
    /// A position in the instruction list to jump to.
    Label(usize),
    /// `Join(o, cond, t, f)` means `o = cond ? t : f`.
    Join(Slot, Slot, Slot, Slot),
}

#[derive(Debug, Clone, Deserialize)]
pub enum Value {
    Single(f64),
}

impl Value {
    fn value(&self) -> f64 {
        let Value::Single(x) = self;
        *x
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Rational {
    pub numerator: Value,
    pub denominator: Value,
}

impl Rational {
    fn value(&self) -> f64 {
        self.numerator.value() / self.denominator.value()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ComplexRational {
    pub re: Rational,
    pub im: Rational,
}

impl ComplexRational {
    fn value(&self) -> Complex<f64> {
        Complex::new(self.re.value(), self.im.value())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ConstType {
    Complex(ComplexRational),
    Single(f64),
}

impl ConstType {
    pub fn value(&self) -> Complex<f64> {
        match self {
            ConstType::Single(x) => Complex::new(*x, 0.0),
            ConstType::Complex(x) => x.value(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SymbolicaModel(pub Vec<Instruction>, pub usize, pub Vec<ConstType>);

/***************************************************************/

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Slot::Param(idx) => write!(f, "('param', {})", *idx),
            Slot::Out(idx) => write!(f, "('out', {})", *idx),
            Slot::Temp(idx) => write!(f, "('temp', {})", *idx),
            Slot::Const(idx) => write!(f, "('const', {})", *idx),
            _ => write!(f, "?"),
        }
    }
}

impl fmt::Display for VecSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v: Vec<String> = self.0.iter().map(|s| s.to_string()).collect();
        write!(f, "[{}]", v.join(","))
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Add(lhs, args, num_reals) => {
                write!(
                    f,
                    "('add', {}, {}, {})",
                    lhs,
                    VecSlot(args.clone()),
                    num_reals
                )
            }
            Instruction::Mul(lhs, args, num_reals) => {
                write!(
                    f,
                    "('mul', {}, {}, {})",
                    lhs,
                    VecSlot(args.clone()),
                    num_reals
                )
            }
            Instruction::Pow(lhs, arg, p, is_real) => write!(
                f,
                "('pow', {}, {}, {}, {})",
                lhs,
                arg,
                p,
                if *is_real { "True" } else { "False" }
            ),
            Instruction::Powf(lhs, arg, p, is_real) => write!(
                f,
                "('powf', {}, {}, {}, {})",
                lhs,
                arg,
                p,
                if *is_real { "True" } else { "False" }
            ),
            Instruction::Assign(lhs, rhs) => write!(f, "('assign', {}, {})", lhs, rhs),
            Instruction::Label(id) => write!(f, "('label', {})", id),
            Instruction::Goto(id) => write!(f, "('goto', {})", id),
            Instruction::IfElse(cond, id) => write!(f, "('if_else', {}, {})", cond, id),
            Instruction::Fun(lhs, fun, args, is_real) => {
                let fun = fun.strip_prefix("symbolica_").unwrap_or_else(|| fun);
                write!(
                    f,
                    "('fun', {}, {}, [], {}, {})",
                    lhs,
                    fun,
                    VecSlot(args.clone()),
                    if *is_real { "True" } else { "False" }
                )
            }
            Instruction::Join(lhs, cond, true_val, false_val) => write!(
                f,
                "('join', {}, {}, {}, {})",
                lhs, cond, true_val, false_val
            ),
            _ => Ok(()),
        }
    }
}

pub fn rationalize_complex(z: Complex<f64>) -> String {
    if z.is_zero() {
        "0".into()
    } else if z.im.is_zero() {
        let x = Rational64::from_f64(z.re).unwrap();
        if x.denom().is_one() {
            x.numer().to_string()
        } else {
            format!("{}/{}", x.numer(), x.denom())
        }
    } else if z.re.is_zero() {
        let y = Rational64::from_f64(z.im).unwrap();
        if y.denom().is_one() {
            format!("{}𝑖", y.numer())
        } else {
            format!("{}𝑖/{}", y.numer(), y.denom())
        }
    } else {
        let x = Rational64::from_f64(z.re).unwrap();
        let y = Rational64::from_f64(z.im).unwrap();

        if z.im.is_sign_negative() {
            format!("{}/{}-{}𝑖/{}", x.numer(), x.denom(), y.numer(), y.denom())
        } else {
            format!("{}/{}+{}𝑖/{}", x.numer(), x.denom(), y.numer(), y.denom())
        }
    }
}
