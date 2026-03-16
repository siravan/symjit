use num_complex::Complex;
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
    /// `Fun(o, s, a)` means `o = s(a)`, where `s` is assumed to
    /// be a built-in function such as `sin`.
    Fun(Slot, BuiltinSymbol, Slot, bool),
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
