use serde::Deserialize;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::ops::{BitAnd, BitOr, BitXor};
use std::ops::{Neg, Not};

use anyhow::{anyhow, Result};

use crate::model::{Equation, Variable};
use crate::utils::bool_to_f64;

/// The elements of the top-level expression trees
///
/// Expr is used to create variables and expressions to pass to
/// Symjit to compile.
///
/// The Python/Sympy interface generates a JSON string, encoding the
/// model, and pass it to the Rust code to deserialize. The Rust interface
/// (`Compiler`) directly uses various functions to compose the trees.
///
/// # Examples
///
/// ```rust
/// let x = Expr::var("x");     # create a new variable
/// let c = Expr::from(2.5);    # create a new constant (f64)
/// let expr = &x * &(x.sin() + &c)
/// ...
/// ```
///
/// Note that the overloaded operators expect `&Expr`; therefore, the need
/// for taking reference (adding `&` in from the intermediate expressions).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Expr {
    Tree { op: String, args: Vec<Expr> },
    Const { val: f64 },
    Var { name: String },
}

impl From<f64> for Expr {
    fn from(val: f64) -> Expr {
        Expr::Const { val }
    }
}

impl From<f32> for Expr {
    fn from(val: f32) -> Expr {
        Expr::Const { val: val.into() }
    }
}

impl From<i32> for Expr {
    fn from(val: i32) -> Expr {
        Expr::Const { val: val.into() }
    }
}

impl Expr {
    /// Creates a new variable named `name`.
    pub fn var(name: &str) -> Expr {
        Expr::Var {
            name: name.to_string(),
        }
    }

    /// Create a unary operation: `op(arg)`.
    ///
    /// To create a user-defined unary function, you need to register
    /// the function with `Compiler` using `def_unary`.
    pub fn unary(op: &str, arg: &Expr) -> Expr {
        Expr::Tree {
            op: op.to_string(),
            args: vec![arg.clone()],
        }
    }

    /// Creates a binary operations `op(l, r)`.
    ///
    /// To create a user-defined binary function, you need to register
    /// the function with `Compiler` using `def_binary`.
    pub fn binary(op: &str, l: &Expr, r: &Expr) -> Expr {
        Expr::Tree {
            op: op.to_string(),
            args: vec![l.clone(), r.clone()],
        }
    }

    /// Creates a ternary operation: `op(l, c, r)`.
    pub fn ternary(op: &str, l: &Expr, c: &Expr, r: &Expr) -> Expr {
        Expr::Tree {
            op: op.to_string(),
            args: vec![l.clone(), c.clone(), r.clone()],
        }
    }

    /// Creates an n-ary operation: `op(args...)`.
    pub fn nary(op: &str, args: &[&Expr]) -> Expr {
        Expr::Tree {
            op: op.to_string(),
            args: args.iter().map(|x| (*x).clone()).collect::<Vec<Expr>>(),
        }
    }

    /// Creates an equation lhs ~ rhs.
    pub fn equation(lhs: &Expr, rhs: &Expr) -> Equation {
        Equation {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }
    }

    /// Converts a variable Expr to a Variable type needed by the next stage.
    pub fn to_variable(&self) -> Result<Variable> {
        if let Expr::Var { name } = self {
            Ok(Variable {
                name: name.to_string(),
            })
        } else {
            Err(anyhow!("cannot convert {:?} to a Variable", self))
        }
    }

    // overloaded binary operations

    pub fn add(&self, other: &Expr) -> Expr {
        Expr::binary("plus", self, other)
    }

    pub fn sub(&self, other: &Self) -> Expr {
        Expr::binary("minus", self, other)
    }

    pub fn mul(&self, other: &Self) -> Expr {
        Expr::binary("times", self, other)
    }

    pub fn div(&self, other: &Self) -> Expr {
        Expr::binary("divide", self, other)
    }

    pub fn rem(&self, other: &Self) -> Expr {
        Expr::binary("rem", self, other)
    }

    pub fn bitand(&self, other: &Self) -> Expr {
        Expr::binary("and", self, other)
    }

    pub fn bitor(&self, other: &Self) -> Expr {
        Expr::binary("or", self, other)
    }

    pub fn bitxor(&self, other: &Self) -> Expr {
        Expr::binary("xor", self, other)
    }

    // comparison operations

    /// Comparison `==`
    pub fn eq(&self, other: &Expr) -> Expr {
        Self::binary("eq", self, other)
    }

    /// Comparison `!=`
    pub fn ne(&self, other: &Expr) -> Expr {
        Self::binary("neq", self, other)
    }

    /// Comparison `<`
    pub fn lt(&self, other: &Expr) -> Expr {
        Self::binary("lt", self, other)
    }

    /// Comparison `<=`
    pub fn le(&self, other: &Expr) -> Expr {
        Self::binary("leq", self, other)
    }

    /// Comparison `>`
    pub fn gt(&self, other: &Expr) -> Expr {
        Self::binary("gt", self, other)
    }

    /// Comparison `>=`
    pub fn ge(&self, other: &Expr) -> Expr {
        Self::binary("geq", self, other)
    }

    // non-overloaded binaries
    pub fn min(&self, other: &Expr) -> Expr {
        Self::binary("min", self, other)
    }

    pub fn max(&self, other: &Expr) -> Expr {
        Self::binary("max", self, other)
    }

    pub fn pow(&self, other: &Expr) -> Expr {
        Self::binary("power", self, other)
    }

    /// Ternary select operations: `if self { true_val} else {false_valve}`
    /// Note that this is not a short-circuited operation.
    pub fn ifelse(&self, true_val: &Expr, false_val: &Expr) -> Expr {
        Self::ternary("ifelse", self, true_val, false_val)
    }

    /// Sums `self` for `var` in `start`..=`end`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Expr::var("x");
    /// let i = Expr::var("i");
    /// let p = i.sum(&i, &Expr::from(1), &x);
    /// let mut comp = Compiler::new();
    /// let mut func = comp.compile(&[x], &[p])?;
    /// println!("{}", func.call([5]))  // prints [15.0]
    /// ```
    ///
    /// Note that the range is `start` to `end` inclusive to remain
    /// consistent with SymPy usage.
    ///
    /// # Warning
    ///
    /// `var` should be a unique variable over the whole model.
    pub fn sum(&self, var: &Expr, start: &Expr, end: &Expr) -> Expr {
        Self::nary("Sum", &[self, var, start, end])
    }

    /// Calculates the product of `self` for `var` in `start`..=`end`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Expr::var("x");
    /// let i = Expr::var("i");
    /// let p = i.prod(&i, &Expr::from(1), &x); // this is the factorial function
    /// let mut comp = Compiler::new();
    /// let mut func = comp.compile(&[x], &[p])?;
    /// println!("{}", func.call([5]))  // prints [120.0]
    /// ```
    ///
    /// Note that the range is `start` to `end` inclusive to remain
    /// consistent with SymPy usage.
    ///
    /// `var` should be a unique variable over the whole model.
    pub fn prod(&self, var: &Expr, start: &Expr, end: &Expr) -> Expr {
        Self::nary("Product", &[self, var, start, end])
    }

    // unary operations
    pub fn not(&self) -> Expr {
        Expr::unary("not", self)
    }

    pub fn neg(&self) -> Expr {
        Expr::unary("neg", self)
    }

    /// self^2
    pub fn square(&self) -> Expr {
        Expr::unary("square", self)
    }

    /// self^3
    pub fn cube(&self) -> Expr {
        Expr::unary("cube", self)
    }

    /// 1 / self
    pub fn recip(&self) -> Expr {
        Expr::unary("recip", self)
    }

    pub fn round(&self) -> Expr {
        Expr::unary("round", self)
    }

    pub fn trunc(&self) -> Expr {
        Expr::unary("trunc", self)
    }

    pub fn floor(&self) -> Expr {
        Expr::unary("floor", self)
    }

    pub fn ceil(&self) -> Expr {
        Expr::unary("ceiling", self)
    }

    /// The fractional part of a number: `self - floor(self)`
    pub fn fract(&self) -> Expr {
        Expr::unary("frac", self)
    }

    /// Heaviside functions. It returns 1 if self is >=0;
    /// otherwise returns 0.
    ///
    /// It can be used for conditional operations.
    pub fn heaviside(&self) -> Expr {
        Self::unary("heaviside", self)
    }

    pub fn sqrt(&self) -> Expr {
        Expr::unary("root", self)
    }

    pub fn abs(&self) -> Expr {
        Expr::unary("abs", self)
    }

    pub fn sin(&self) -> Expr {
        Expr::unary("sin", self)
    }

    pub fn cos(&self) -> Expr {
        Expr::unary("cos", self)
    }

    pub fn tan(&self) -> Expr {
        Expr::unary("tan", self)
    }

    pub fn csc(&self) -> Expr {
        Expr::unary("csc", self)
    }

    pub fn sec(&self) -> Expr {
        Expr::unary("sec", self)
    }

    pub fn cot(&self) -> Expr {
        Expr::unary("cot", self)
    }

    pub fn sinh(&self) -> Expr {
        Expr::unary("sinh", self)
    }

    pub fn cosh(&self) -> Expr {
        Expr::unary("cosh", self)
    }

    pub fn tanh(&self) -> Expr {
        Expr::unary("tanh", self)
    }

    pub fn csch(&self) -> Expr {
        Expr::unary("csch", self)
    }

    pub fn sech(&self) -> Expr {
        Expr::unary("sech", self)
    }

    pub fn coth(&self) -> Expr {
        Expr::unary("coth", self)
    }

    pub fn asin(&self) -> Expr {
        Expr::unary("arcsin", self)
    }

    pub fn acos(&self) -> Expr {
        Expr::unary("arccos", self)
    }

    pub fn atan(&self) -> Expr {
        Expr::unary("arctan", self)
    }

    pub fn asinh(&self) -> Expr {
        Expr::unary("arcsinh", self)
    }

    pub fn acosh(&self) -> Expr {
        Expr::unary("arccosh", self)
    }

    pub fn atanh(&self) -> Expr {
        Expr::unary("arctanh", self)
    }

    pub fn sinc(&self) -> Expr {
        Expr::unary("sinc", self)
    }

    pub fn cbrt(&self) -> Expr {
        Expr::unary("cbrt", self)
    }

    pub fn exp(&self) -> Expr {
        Expr::unary("exp", self)
    }

    pub fn ln(&self) -> Expr {
        Expr::unary("ln", self)
    }

    pub fn log10(&self) -> Expr {
        Expr::unary("log", self)
    }

    pub fn exp_m1(&self) -> Expr {
        Expr::unary("expm1", self)
    }

    pub fn ln_1p(&self) -> Expr {
        Expr::unary("log1p", self)
    }

    pub fn log2(&self) -> Expr {
        Expr::unary("log2", self)
    }

    pub fn exp2(&self) -> Expr {
        Expr::unary("exp2", self)
    }

    pub fn erf(&self) -> Expr {
        Expr::unary("erf", self)
    }

    pub fn erfc(&self) -> Expr {
        Expr::unary("erfc", self)
    }

    pub fn gamma(&self) -> Expr {
        Expr::unary("gamma", self)
    }

    /// Log gamma
    pub fn lgam(&self) -> Expr {
        Expr::unary("loggamma", self)
    }

    pub fn si(&self) -> Expr {
        Expr::unary("si", self)
    }

    pub fn ci(&self) -> Expr {
        Expr::unary("ci", self)
    }

    pub fn shi(&self) -> Expr {
        Expr::unary("shi", self)
    }

    pub fn chi(&self) -> Expr {
        Expr::unary("chi", self)
    }
}

impl Add for &Expr {
    type Output = Expr;

    fn add(self, other: Self) -> Expr {
        Expr::add(self, other)
    }
}

impl Sub for &Expr {
    type Output = Expr;

    fn sub(self, other: Self) -> Expr {
        Expr::sub(self, other)
    }
}

impl Mul for &Expr {
    type Output = Expr;

    fn mul(self, other: Self) -> Expr {
        Expr::mul(self, other)
    }
}

impl Div for &Expr {
    type Output = Expr;

    fn div(self, other: Self) -> Expr {
        Expr::div(self, other)
    }
}

impl Rem for &Expr {
    type Output = Expr;

    fn rem(self, other: Self) -> Expr {
        Expr::rem(self, other)
    }
}

impl BitAnd for &Expr {
    type Output = Expr;

    fn bitand(self, other: Self) -> Expr {
        Expr::bitand(self, other)
    }
}

impl BitOr for &Expr {
    type Output = Expr;

    fn bitor(self, other: Self) -> Expr {
        Expr::bitor(self, other)
    }
}

impl BitXor for &Expr {
    type Output = Expr;

    fn bitxor(self, other: Self) -> Expr {
        Expr::bitxor(self, other)
    }
}

impl Neg for &Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        Expr::neg(self)
    }
}

impl Not for &Expr {
    type Output = Expr;

    fn not(self) -> Expr {
        Expr::neg(self)
    }
}

pub fn var(name: &str) -> Expr {
    Expr::var(name)
}

pub fn double(val: f64) -> Expr {
    Expr::from(val)
}

pub fn int(val: i32) -> Expr {
    Expr::from(val)
}

pub fn boolean(val: bool) -> Expr {
    Expr::from(bool_to_f64(val))
}
