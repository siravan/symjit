//! The values of the interpreter and their operations. Numbers known while the kernel is
//! built (`F64`, `Int`, `C64`, `Bool`) are computed in Rust, as the code the `#[kernel]`
//! macro leaves alone; kernel values (`Real`, `Cplx`, `Mask`) go through the operators of
//! `symjit::kernel`, which add instructions to the kernel, so that the kernel is the one the
//! macro builds.

use std::rc::Rc;

use symjit::kernel::{ops, Cplx, IntoCplx, IntoReal, KernelBuilder, Mask, Real, Value};
use symjit::{Complex, Slot};

use super::syntax::Ty;

/// A closure: its parameters, body and the variables it captured (by value).
pub struct Closure<'k> {
    pub params: Vec<syn::Pat>,
    pub output: Option<Ty>,
    pub body: syn::Expr,
    pub env: Vec<(String, Val<'k>)>,
}

#[derive(Clone)]
pub enum Val<'k> {
    Unit,
    Bool(bool),
    Int(i64),
    F64(f64),
    C64(Complex<f64>),
    Real(Real<'k>),
    Cplx(Cplx<'k>),
    Mask(Mask<'k>),
    Tuple(Vec<Val<'k>>),
    Array(Vec<Val<'k>>),
    /// `start..end` (`end` exclusive)
    Range(i64, i64),
    Closure(Rc<Closure<'k>>),
}

pub type VResult<'k> = Result<Val<'k>, String>;

impl<'k> Val<'k> {
    /// The type, for messages.
    pub fn describe(&self) -> String {
        match self {
            Val::Unit => "()".into(),
            Val::Bool(_) => "`bool`".into(),
            Val::Int(_) => "an integer".into(),
            Val::F64(_) => "`f64` (a number known when the kernel is built)".into(),
            Val::C64(_) => "`Complex<f64>` (a number known when the kernel is built)".into(),
            Val::Real(_) => "a real kernel value (`f64`)".into(),
            Val::Cplx(_) => "a complex kernel value (`Complex<f64>`)".into(),
            Val::Mask(_) => "a kernel comparison (`bool`)".into(),
            Val::Tuple(_) => "a tuple".into(),
            Val::Array(_) => "an array".into(),
            Val::Range(..) => "a range".into(),
            Val::Closure(_) => "a closure".into(),
        }
    }

    /// Whether it is or holds a kernel value.
    pub fn is_kernel(&self) -> bool {
        match self {
            Val::Real(_) | Val::Cplx(_) | Val::Mask(_) => true,
            Val::Tuple(vs) | Val::Array(vs) => vs.iter().any(|v| v.is_kernel()),
            _ => false,
        }
    }
}

/// A list of slots as a kernel value of any shape, for `KernelBuilder::if_else` and
/// `while_loop` (whose closures return it) and `compile`. `from_slots` takes all of them.
pub struct Slots<'k>(pub Vec<Slot>, std::marker::PhantomData<&'k ()>);

impl<'k> Slots<'k> {
    pub fn new(slots: Vec<Slot>) -> Self {
        Slots(slots, std::marker::PhantomData)
    }
}

impl<'k> Value<'k> for Slots<'k> {
    fn slots(&self, out: &mut Vec<Slot>) {
        out.extend(self.0.iter().copied());
    }

    fn from_slots(_: &'k KernelBuilder, slots: &mut dyn Iterator<Item = Slot>) -> Self {
        Slots::new(slots.collect())
    }

    fn from_params(_: &'k KernelBuilder) -> Self {
        Slots::new(Vec::new())
    }
}

/// The shape of a kernel value made of slots.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Unit,
    Real,
    Cplx,
    Mask,
    Tuple(Vec<Shape>),
    Array(Vec<Shape>),
}

impl Shape {
    pub fn describe(&self) -> String {
        match self {
            Shape::Unit => "()".into(),
            Shape::Real => "f64".into(),
            Shape::Cplx => "Complex<f64>".into(),
            Shape::Mask => "bool".into(),
            Shape::Tuple(s) => format!(
                "({})",
                s.iter()
                    .map(|s| s.describe())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Shape::Array(s) => match s.first() {
                Some(e) => format!("[{}; {}]", e.describe(), s.len()),
                None => "[_; 0]".into(),
            },
        }
    }
}

/// The slots of `v`; with `lift`, numbers become constants of the kernel (the branches of a
/// kernel `if`), otherwise they are an error (the state of a kernel loop).
pub fn to_slots<'k>(
    k: &'k KernelBuilder,
    v: &Val<'k>,
    lift: bool,
    out: &mut Vec<Slot>,
) -> Result<Shape, String> {
    let number = |v: &Val<'k>| {
        format!(
            "{} is not a kernel value; a variable updated in a kernel loop must be one \
             (e.g. `let mut i = 0.0`, not `0`)",
            v.describe()
        )
    };
    Ok(match v {
        Val::Unit => Shape::Unit,
        Val::Real(r) => {
            r.slots(out);
            Shape::Real
        }
        Val::Cplx(c) => {
            c.slots(out);
            Shape::Cplx
        }
        Val::Mask(m) => {
            m.slots(out);
            Shape::Mask
        }
        Val::F64(x) if lift => {
            k.constant(*x).slots(out);
            Shape::Real
        }
        Val::C64(z) if lift => {
            z.into_cplx(k).slots(out);
            Shape::Cplx
        }
        Val::Bool(b) if lift => {
            k.mask(*b).slots(out);
            Shape::Mask
        }
        Val::Tuple(vs) => Shape::Tuple(
            vs.iter()
                .map(|v| to_slots(k, v, lift, out))
                .collect::<Result<_, _>>()?,
        ),
        Val::Array(vs) => Shape::Array(
            vs.iter()
                .map(|v| to_slots(k, v, lift, out))
                .collect::<Result<_, _>>()?,
        ),
        Val::Int(_) if lift => {
            return Err(
                "an integer is not a kernel value; the branches of a kernel `if` must give \
                 floats (`1.0`, not `1`), and a variable assigned in them must be a float"
                    .into(),
            )
        }
        v if !lift => return Err(number(v)),
        v => return Err(format!("{} cannot be a kernel value", v.describe())),
    })
}

/// The value of `shape` made of the next slots of `it`.
pub fn from_slots<'k>(
    k: &'k KernelBuilder,
    shape: &Shape,
    it: &mut dyn Iterator<Item = Slot>,
) -> Val<'k> {
    match shape {
        Shape::Unit => Val::Unit,
        Shape::Real => Val::Real(Real::from_slots(k, it)),
        Shape::Cplx => Val::Cplx(Cplx::from_slots(k, it)),
        Shape::Mask => Val::Mask(Mask::from_slots(k, it)),
        Shape::Tuple(s) => Val::Tuple(s.iter().map(|s| from_slots(k, s, it)).collect()),
        Shape::Array(s) => Val::Array(s.iter().map(|s| from_slots(k, s, it)).collect()),
    }
}

/// The shape both branches of a kernel `if` are converted to (`ops::Unify`).
pub fn unify(a: &Shape, b: &Shape) -> Option<Shape> {
    match (a, b) {
        (Shape::Unit, Shape::Unit) => Some(Shape::Unit),
        (Shape::Real, Shape::Real) => Some(Shape::Real),
        (Shape::Real | Shape::Cplx, Shape::Real | Shape::Cplx) => Some(Shape::Cplx),
        (Shape::Mask, Shape::Mask) => Some(Shape::Mask),
        (Shape::Tuple(x), Shape::Tuple(y)) if x.len() == y.len() => Some(Shape::Tuple(
            x.iter()
                .zip(y)
                .map(|(a, b)| unify(a, b))
                .collect::<Option<_>>()?,
        )),
        (Shape::Array(x), Shape::Array(y)) if x.len() == y.len() => Some(Shape::Array(
            x.iter()
                .zip(y)
                .map(|(a, b)| unify(a, b))
                .collect::<Option<_>>()?,
        )),
        _ => None,
    }
}

/// `v` converted to `ty` (`ops::lift`): numbers become constants of the kernel. Types that
/// are not handles (`Ty::Other`) take the value as it is.
pub fn lift<'k>(k: &'k KernelBuilder, v: Val<'k>, ty: &Ty) -> VResult<'k> {
    Ok(match (ty, v) {
        (Ty::Real, Val::Real(r)) => Val::Real(r),
        (Ty::Real, Val::F64(x)) => Val::Real(k.constant(x)),
        (Ty::Cplx, Val::Cplx(c)) => Val::Cplx(c),
        (Ty::Cplx, Val::Real(r)) => Val::Cplx(r.into_cplx(k)),
        (Ty::Cplx, Val::F64(x)) => Val::Cplx(x.into_cplx(k)),
        (Ty::Cplx, Val::C64(z)) => Val::Cplx(z.into_cplx(k)),
        (Ty::Mask, Val::Mask(m)) => Val::Mask(m),
        (Ty::Mask, Val::Bool(b)) => Val::Mask(k.mask(b)),
        (Ty::Unit, Val::Unit) => Val::Unit,
        (Ty::Tuple(ts), Val::Tuple(vs)) if ts.len() == vs.len() => Val::Tuple(
            ts.iter()
                .zip(vs)
                .map(|(t, v)| lift(k, v, t))
                .collect::<Result<_, _>>()?,
        ),
        (Ty::Array(t, n), Val::Array(vs)) if *n == vs.len() => Val::Array(
            vs.into_iter()
                .map(|v| lift(k, v, t))
                .collect::<Result<_, _>>()?,
        ),
        (Ty::Other(name), v) => {
            let int = matches!(
                name.as_str(),
                "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
            );
            if int && !matches!(v, Val::Int(_)) {
                return Err(format!("expected `{}`, found {}", name, v.describe()));
            }
            v
        }
        (ty, v) => {
            return Err(format!(
                "expected {}, found {}",
                describe_ty(ty),
                v.describe()
            ))
        }
    })
}

pub fn describe_ty(ty: &Ty) -> String {
    match ty {
        Ty::Real => "`f64`".into(),
        Ty::Cplx => "`Complex<f64>`".into(),
        Ty::Mask => "`bool`".into(),
        Ty::Unit => "`()`".into(),
        Ty::Tuple(ts) => format!(
            "({})",
            ts.iter().map(describe_ty).collect::<Vec<_>>().join(", ")
        ),
        Ty::Array(t, n) => format!("[{}; {}]", describe_ty(t), n),
        Ty::Other(name) => format!("`{}`", name),
    }
}

/// The value of an assignment to a variable holding `old` (`ops::coerce`): numbers assigned
/// to kernel values become constants.
pub fn coerce<'k>(k: &'k KernelBuilder, old: &Val<'k>, new: Val<'k>) -> VResult<'k> {
    let mismatch = |new: &Val<'k>| {
        Err(format!(
            "cannot assign {} to a variable holding {}{}",
            new.describe(),
            old.describe(),
            if matches!(old, Val::F64(_)) && new.is_kernel() {
                "; make it a kernel value: `let mut x = 0.0` (a float literal) or \
                 `let mut x: f64 = ..`"
            } else {
                ""
            }
        ))
    };
    Ok(match (old, new) {
        (Val::Real(_), Val::F64(x)) => Val::Real(k.constant(x)),
        (Val::Real(_), v @ Val::Real(_)) => v,
        (Val::Cplx(_), Val::F64(x)) => Val::Cplx(x.into_cplx(k)),
        (Val::Cplx(_), Val::Real(r)) => Val::Cplx(r.into_cplx(k)),
        (Val::Cplx(_), Val::C64(z)) => Val::Cplx(z.into_cplx(k)),
        (Val::Cplx(_), v @ Val::Cplx(_)) => v,
        (Val::Mask(_), Val::Bool(b)) => Val::Mask(k.mask(b)),
        (Val::Mask(_), v @ Val::Mask(_)) => v,
        (Val::F64(_), v @ Val::F64(_)) => v,
        (Val::C64(_), v @ Val::C64(_)) => v,
        (Val::Int(_), v @ Val::Int(_)) => v,
        (Val::Bool(_), v @ Val::Bool(_)) => v,
        (Val::Tuple(os), Val::Tuple(ns)) if os.len() == ns.len() => Val::Tuple(
            os.iter()
                .zip(ns)
                .map(|(o, n)| coerce(k, o, n))
                .collect::<Result<_, _>>()?,
        ),
        (Val::Array(os), Val::Array(ns)) if os.len() == ns.len() => Val::Array(
            os.iter()
                .zip(ns)
                .map(|(o, n)| coerce(k, o, n))
                .collect::<Result<_, _>>()?,
        ),
        (Val::Range(..), v @ Val::Range(..)) => v,
        (Val::Closure(_), v @ Val::Closure(_)) => v,
        (Val::Unit, Val::Unit) => Val::Unit,
        (_, new) => return mismatch(&new),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl Op {
    fn symbol(self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::Rem => "%",
            Op::BitAnd => "&",
            Op::BitOr => "|",
            Op::BitXor => "^",
            Op::Shl => "<<",
            Op::Shr => ">>",
        }
    }
}

macro_rules! arith {
    ($a:expr, $b:expr, $op:tt, $mismatch:expr) => {
        match ($a, $b) {
            (Val::F64(x), Val::F64(y)) => Val::F64(x $op y),
            (Val::C64(x), Val::C64(y)) => Val::C64(x $op y),
            (Val::C64(x), Val::F64(y)) => Val::C64(x $op y),
            (Val::F64(x), Val::C64(y)) => Val::C64(x $op y),
            (Val::Real(x), Val::Real(y)) => Val::Real(x $op y),
            (Val::Real(x), Val::F64(y)) => Val::Real(x $op y),
            (Val::F64(x), Val::Real(y)) => Val::Real(x $op y),
            (Val::Cplx(x), Val::Cplx(y)) => Val::Cplx(x $op y),
            (Val::Cplx(x), Val::Real(y)) => Val::Cplx(x $op y),
            (Val::Cplx(x), Val::F64(y)) => Val::Cplx(x $op y),
            (Val::Cplx(x), Val::C64(y)) => Val::Cplx(x $op y),
            (Val::Real(x), Val::Cplx(y)) => Val::Cplx(x $op y),
            (Val::F64(x), Val::Cplx(y)) => Val::Cplx(x $op y),
            (Val::C64(x), Val::Cplx(y)) => Val::Cplx(x $op y),
            (a, b) => return $mismatch(a, b),
        }
    };
}

/// `a op b`.
pub fn binary<'k>(op: Op, a: Val<'k>, b: Val<'k>) -> VResult<'k> {
    let mismatch = |a: Val<'k>, b: Val<'k>| -> VResult<'k> {
        Err(format!(
            "cannot apply `{}` to {} and {}",
            op.symbol(),
            a.describe(),
            b.describe()
        ))
    };

    if let (Val::Int(x), Val::Int(y)) = (&a, &b) {
        let (x, y) = (*x, *y);
        let r = match op {
            Op::Add => x.checked_add(y),
            Op::Sub => x.checked_sub(y),
            Op::Mul => x.checked_mul(y),
            Op::Div => x.checked_div(y),
            Op::Rem => x.checked_rem(y),
            Op::BitAnd => Some(x & y),
            Op::BitOr => Some(x | y),
            Op::BitXor => Some(x ^ y),
            Op::Shl => u32::try_from(y).ok().and_then(|y| x.checked_shl(y)),
            Op::Shr => u32::try_from(y).ok().and_then(|y| x.checked_shr(y)),
        };
        return r.map(Val::Int).ok_or_else(|| {
            format!(
                "integer overflow or division by zero in `{} {} {}`",
                x,
                op.symbol(),
                y
            )
        });
    }

    Ok(match op {
        Op::Add => arith!(a, b, +, mismatch),
        Op::Sub => arith!(a, b, -, mismatch),
        Op::Mul => arith!(a, b, *, mismatch),
        Op::Div => arith!(a, b, /, mismatch),
        Op::Rem => match (a, b) {
            (Val::F64(x), Val::F64(y)) => Val::F64(x % y),
            (a @ (Val::Real(_) | Val::Cplx(_)), b) | (a, b @ (Val::Real(_) | Val::Cplx(_))) => {
                return Err(format!(
                    "kernel values have no `%` ({} % {}): write `x - (x / y).floor() * y`, or \
                     use `x.modulo(y)` (floored)",
                    a.describe(),
                    b.describe()
                ))
            }
            (a, b) => return mismatch(a, b),
        },
        Op::BitAnd | Op::BitOr | Op::BitXor => match (a, b) {
            (Val::Bool(x), Val::Bool(y)) => Val::Bool(match op {
                Op::BitAnd => x & y,
                Op::BitOr => x | y,
                _ => x ^ y,
            }),
            (Val::Mask(x), Val::Mask(y)) => Val::Mask(match op {
                Op::BitAnd => x & y,
                Op::BitOr => x | y,
                _ => x ^ y,
            }),
            (a, b) => return mismatch(a, b),
        },
        Op::Shl | Op::Shr => return mismatch(a, b),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// `a < b` and the like (`ops::lt`, ...): a `Mask` for kernel values, a `bool` for numbers.
pub fn compare<'k>(cmp: Cmp, a: Val<'k>, b: Val<'k>) -> VResult<'k> {
    fn rust<T: PartialOrd>(cmp: Cmp, x: T, y: T) -> bool {
        match cmp {
            Cmp::Lt => x < y,
            Cmp::Le => x <= y,
            Cmp::Gt => x > y,
            Cmp::Ge => x >= y,
            Cmp::Eq => x == y,
            Cmp::Ne => x != y,
        }
    }
    macro_rules! mask {
        ($x:expr, $y:expr) => {
            Val::Mask(match cmp {
                Cmp::Lt => ops::lt($x, $y),
                Cmp::Le => ops::le($x, $y),
                Cmp::Gt => ops::gt($x, $y),
                Cmp::Ge => ops::ge($x, $y),
                Cmp::Eq => ops::eq($x, $y),
                Cmp::Ne => ops::ne($x, $y),
            })
        };
    }
    Ok(match (a, b) {
        (Val::Int(x), Val::Int(y)) => Val::Bool(rust(cmp, x, y)),
        (Val::F64(x), Val::F64(y)) => Val::Bool(rust(cmp, x, y)),
        (Val::Bool(x), Val::Bool(y)) => Val::Bool(rust(cmp, x, y)),
        (Val::C64(x), Val::C64(y)) if matches!(cmp, Cmp::Eq | Cmp::Ne) => {
            Val::Bool((x == y) == (cmp == Cmp::Eq))
        }
        (Val::Real(x), Val::Real(y)) => mask!(x, y),
        (Val::Real(x), Val::F64(y)) => mask!(x, y),
        (Val::F64(x), Val::Real(y)) => mask!(x, y),
        (a, b) => {
            return Err(format!(
                "cannot compare {} with {}",
                a.describe(),
                b.describe()
            ))
        }
    })
}

/// `a && b` and `a || b` (`ops::and`, `ops::or`): both sides are evaluated.
pub fn logic<'k>(and: bool, a: Val<'k>, b: Val<'k>) -> VResult<'k> {
    Ok(match (a, b) {
        (Val::Bool(x), Val::Bool(y)) => Val::Bool(if and { ops::and(x, y) } else { ops::or(x, y) }),
        (Val::Mask(x), Val::Mask(y)) => Val::Mask(if and { ops::and(x, y) } else { ops::or(x, y) }),
        (Val::Mask(x), Val::Bool(y)) => Val::Mask(if and { ops::and(x, y) } else { ops::or(x, y) }),
        (Val::Bool(x), Val::Mask(y)) => Val::Mask(if and { ops::and(x, y) } else { ops::or(x, y) }),
        (a, b) => {
            return Err(format!(
                "`{}` needs two `bool`s or kernel comparisons, not {} and {}",
                if and { "&&" } else { "||" },
                a.describe(),
                b.describe()
            ))
        }
    })
}

pub fn neg<'k>(v: Val<'k>) -> VResult<'k> {
    Ok(match v {
        Val::Int(x) => Val::Int(x.checked_neg().ok_or("integer overflow in `-`")?),
        Val::F64(x) => Val::F64(-x),
        Val::C64(z) => Val::C64(-z),
        Val::Real(r) => Val::Real(-r),
        Val::Cplx(c) => Val::Cplx(-c),
        v => return Err(format!("cannot negate {}", v.describe())),
    })
}

pub fn not<'k>(v: Val<'k>) -> VResult<'k> {
    Ok(match v {
        Val::Bool(b) => Val::Bool(!b),
        Val::Mask(m) => Val::Mask(!m),
        Val::Int(x) => Val::Int(!x),
        v => return Err(format!("cannot apply `!` to {}", v.describe())),
    })
}

/// `Complex::new(re, im)` (`ops::complex`).
pub fn complex<'k>(k: &'k KernelBuilder, re: Val<'k>, im: Val<'k>) -> VResult<'k> {
    Ok(match (re, im) {
        (Val::F64(x), Val::F64(y)) => Val::C64(Complex::new(x, y)),
        (Val::Real(x), Val::Real(y)) => Val::Cplx(ops::complex(k, x, y)),
        (Val::Real(x), Val::F64(y)) => Val::Cplx(ops::complex(k, x, y)),
        (Val::F64(x), Val::Real(y)) => Val::Cplx(ops::complex(k, x, y)),
        (a, b) => {
            return Err(format!(
                "`Complex::new` takes two `f64`, not {} and {}",
                a.describe(),
                b.describe()
            ))
        }
    })
}

/// `v as ty`.
pub fn cast<'k>(v: Val<'k>, ty: &str) -> VResult<'k> {
    let int = matches!(
        ty,
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
    );
    Ok(match (v, ty) {
        (Val::Int(x), "f64") => Val::F64(x as f64),
        (Val::F64(x), "f64") => Val::F64(x),
        (Val::Int(x), _) if int => Val::Int(x),
        (Val::F64(x), _) if int => Val::Int(x as i64),
        (Val::Bool(b), _) if int => Val::Int(b as i64),
        (v, ty) => {
            return Err(format!(
                "cannot cast {} as `{}`{}",
                v.describe(),
                ty,
                if v.is_kernel() {
                    " (kernel values cannot be cast)"
                } else {
                    ""
                }
            ))
        }
    })
}

fn arg_count(name: &str, args: &[Val], n: usize) -> Result<(), String> {
    if args.len() == n {
        Ok(())
    } else {
        Err(format!(
            "`{}` takes {} argument{}, not {}",
            name,
            n,
            if n == 1 { "" } else { "s" },
            args.len()
        ))
    }
}

fn int_arg(name: &str, v: &Val) -> Result<i64, String> {
    match v {
        Val::Int(i) => Ok(*i),
        v => Err(format!("`{}` takes an integer, not {}", name, v.describe())),
    }
}

fn f64_arg(name: &str, v: &Val) -> Result<f64, String> {
    match v {
        Val::F64(x) => Ok(*x),
        v => Err(format!(
            "`f64::{}` takes an `f64`, not {}{}",
            name,
            v.describe(),
            if v.is_kernel() {
                " (the receiver is a number known when the kernel is built; call the method \
                 on the kernel value instead)"
            } else {
                ""
            }
        )),
    }
}

/// A `Real` from a real kernel value or a number (`IntoReal`).
fn real_arg<'k>(k: &'k KernelBuilder, name: &str, v: &Val<'k>) -> Result<Real<'k>, String> {
    match v {
        Val::Real(r) => Ok(*r),
        Val::F64(x) => Ok(x.into_real(k)),
        v => Err(format!("`{}` takes an `f64`, not {}", name, v.describe())),
    }
}

/// A `Cplx` from a kernel value or a number (`IntoCplx`).
fn cplx_arg<'k>(k: &'k KernelBuilder, name: &str, v: &Val<'k>) -> Result<Cplx<'k>, String> {
    match v {
        Val::Cplx(c) => Ok(*c),
        Val::Real(r) => Ok(r.into_cplx(k)),
        Val::F64(x) => Ok(x.into_cplx(k)),
        Val::C64(z) => Ok(z.into_cplx(k)),
        v => Err(format!(
            "`{}` takes a `Complex<f64>`, not {}",
            name,
            v.describe()
        )),
    }
}

/// `recv.name(args)`.
pub fn method<'k>(
    k: &'k KernelBuilder,
    recv: Val<'k>,
    name: &str,
    args: Vec<Val<'k>>,
) -> VResult<'k> {
    // methods every value has
    if matches!(name, "clone" | "copied" | "cloned" | "to_owned" | "into") && args.is_empty() {
        return Ok(recv);
    }

    match recv {
        Val::Real(r) => real_method(k, r, name, &args),
        Val::Cplx(c) => cplx_method(k, c, name, &args),
        Val::Mask(m) => match name {
            "select" => {
                arg_count(name, &args, 2)?;
                let a = real_arg(k, name, &args[0])?;
                let b = real_arg(k, name, &args[1])?;
                Ok(Val::Real(m.select(a, b)))
            }
            "select_complex" => {
                arg_count(name, &args, 2)?;
                let a = cplx_arg(k, name, &args[0])?;
                let b = cplx_arg(k, name, &args[1])?;
                Ok(Val::Cplx(m.select_complex(a, b)))
            }
            "to_real" => {
                arg_count(name, &args, 0)?;
                Ok(Val::Real(m.to_real()))
            }
            _ => Err(format!(
                "a kernel comparison (`bool`) has no method `{}` (it has `select`, \
                     `select_complex` and `to_real`)",
                name
            )),
        },
        Val::F64(x) => f64_method(x, name, &args),
        Val::C64(z) => c64_method(z, name, &args),
        Val::Int(i) => int_method(i, name, &args),
        Val::Array(vs) => seq_method(vs, name, &args),
        Val::Range(a, b) => seq_method((a..b.max(a)).map(Val::Int).collect(), name, &args),
        v => Err(format!("{} has no method `{}`", v.describe(), name)),
    }
}

fn real_method<'k>(k: &'k KernelBuilder, r: Real<'k>, name: &str, args: &[Val<'k>]) -> VResult<'k> {
    macro_rules! nullary {
        ($($m:ident),*) => {
            match name {
                $(stringify!($m) => {
                    arg_count(name, args, 0)?;
                    return Ok(Val::Real(r.$m()));
                })*
                _ => {}
            }
        };
    }
    nullary!(
        abs,
        recip,
        floor,
        ceil,
        round_ties_even,
        trunc,
        fract,
        pow2i,
        getexp,
        getmant,
        exp,
        exp2,
        exp_m1,
        sin,
        cos,
        tan,
        atan,
        sinh,
        cosh,
        tanh,
        asinh,
        erf,
        erfc,
        ln,
        log10,
        log2,
        ln_1p,
        asin,
        acos,
        acosh,
        atanh,
        cbrt,
        round,
        sqrt
    );
    Ok(match name {
        "to_complex" => {
            arg_count(name, args, 0)?;
            Val::Cplx(r.to_complex())
        }
        "powi" => {
            arg_count(name, args, 1)?;
            let p = int_arg(name, &args[0])?;
            let p = i32::try_from(p).map_err(|_| format!("`powi` exponent {} is too large", p))?;
            Val::Real(r.powi(p))
        }
        "powf" | "modulo" | "min" | "max" => {
            arg_count(name, args, 1)?;
            let y = real_arg(k, name, &args[0])?;
            Val::Real(match name {
                "powf" => r.powf(y),
                "modulo" => r.modulo(y),
                "min" => r.min(y),
                _ => r.max(y),
            })
        }
        "lt" | "le" | "gt" | "ge" | "eq" | "ne" => {
            arg_count(name, args, 1)?;
            let y = real_arg(k, name, &args[0])?;
            Val::Mask(match name {
                "lt" => r.lt(y),
                "le" => r.le(y),
                "gt" => r.gt(y),
                "ge" => r.ge(y),
                "eq" => r.eq(y),
                _ => r.ne(y),
            })
        }
        _ => {
            return Err(format!(
                "a real kernel value (`f64`) has no method `{}` in kernels",
                name
            ))
        }
    })
}

fn cplx_method<'k>(k: &'k KernelBuilder, c: Cplx<'k>, name: &str, args: &[Val<'k>]) -> VResult<'k> {
    macro_rules! nullary {
        ($ctor:ident: $($m:ident),*) => {
            match name {
                $(stringify!($m) => {
                    arg_count(name, args, 0)?;
                    return Ok(Val::$ctor(c.$m()));
                })*
                _ => {}
            }
        };
    }
    nullary!(Real: re, im, norm_sqr, norm);
    nullary!(
        Cplx: conj, recip, sqrt, exp, ln, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan
    );
    Ok(match name {
        "powi" => {
            arg_count(name, args, 1)?;
            let p = int_arg(name, &args[0])?;
            let p = i32::try_from(p).map_err(|_| format!("`powi` exponent {} is too large", p))?;
            Val::Cplx(c.powi(p))
        }
        "powc" => {
            arg_count(name, args, 1)?;
            Val::Cplx(c.powc(cplx_arg(k, name, &args[0])?))
        }
        _ => {
            return Err(format!(
                "a complex kernel value (`Complex<f64>`) has no method `{}` in kernels",
                name
            ))
        }
    })
}

fn f64_method<'k>(x: f64, name: &str, args: &[Val<'k>]) -> VResult<'k> {
    macro_rules! nullary {
        ($($m:ident),*) => {
            match name {
                $(stringify!($m) => {
                    arg_count(name, args, 0)?;
                    return Ok(Val::F64(x.$m()));
                })*
                _ => {}
            }
        };
    }
    macro_rules! unary {
        ($($m:ident),*) => {
            match name {
                $(stringify!($m) => {
                    arg_count(name, args, 1)?;
                    return Ok(Val::F64(x.$m(f64_arg(name, &args[0])?)));
                })*
                _ => {}
            }
        };
    }
    nullary!(
        abs,
        signum,
        sqrt,
        cbrt,
        exp,
        exp2,
        exp_m1,
        ln,
        log10,
        log2,
        ln_1p,
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        tanh,
        asinh,
        acosh,
        atanh,
        floor,
        ceil,
        round,
        round_ties_even,
        trunc,
        fract,
        recip,
        to_degrees,
        to_radians
    );
    unary!(powf, atan2, hypot, min, max, log, copysign, rem_euclid, div_euclid);
    Ok(match name {
        "powi" => {
            arg_count(name, args, 1)?;
            Val::F64(x.powi(int_arg(name, &args[0])? as i32))
        }
        "mul_add" => {
            arg_count(name, args, 2)?;
            Val::F64(x.mul_add(f64_arg(name, &args[0])?, f64_arg(name, &args[1])?))
        }
        "clamp" => {
            arg_count(name, args, 2)?;
            Val::F64(x.clamp(f64_arg(name, &args[0])?, f64_arg(name, &args[1])?))
        }
        "is_nan" => Val::Bool(x.is_nan()),
        "is_finite" => Val::Bool(x.is_finite()),
        "is_infinite" => Val::Bool(x.is_infinite()),
        "is_sign_negative" => Val::Bool(x.is_sign_negative()),
        "is_sign_positive" => Val::Bool(x.is_sign_positive()),
        _ => return Err(format!("`f64` has no method `{}` in sjc", name)),
    })
}

fn c64_method<'k>(z: Complex<f64>, name: &str, args: &[Val<'k>]) -> VResult<'k> {
    macro_rules! nullary {
        ($ctor:ident: $($m:ident),*) => {
            match name {
                $(stringify!($m) => {
                    arg_count(name, args, 0)?;
                    return Ok(Val::$ctor(z.$m()));
                })*
                _ => {}
            }
        };
    }
    nullary!(F64: norm_sqr, norm, arg, l1_norm);
    nullary!(
        C64: conj, inv, sqrt, exp, ln, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan
    );
    Ok(match name {
        "powi" => {
            arg_count(name, args, 1)?;
            Val::C64(z.powi(int_arg(name, &args[0])? as i32))
        }
        "powf" | "scale" | "unscale" => {
            arg_count(name, args, 1)?;
            let y = f64_arg(name, &args[0])?;
            Val::C64(match name {
                "powf" => z.powf(y),
                "scale" => z.scale(y),
                _ => z.unscale(y),
            })
        }
        "powc" => {
            arg_count(name, args, 1)?;
            match &args[0] {
                Val::C64(w) => Val::C64(z.powc(*w)),
                v => {
                    return Err(format!(
                        "`powc` takes a `Complex<f64>`, not {}",
                        v.describe()
                    ))
                }
            }
        }
        _ => return Err(format!("`Complex<f64>` has no method `{}` in sjc", name)),
    })
}

fn int_method<'k>(i: i64, name: &str, args: &[Val<'k>]) -> VResult<'k> {
    Ok(match name {
        "abs" => Val::Int(i.abs()),
        "signum" => Val::Int(i.signum()),
        "pow" => {
            arg_count(name, args, 1)?;
            let p = u32::try_from(int_arg(name, &args[0])?).map_err(|_| "a negative exponent")?;
            Val::Int(i.checked_pow(p).ok_or("integer overflow in `pow`")?)
        }
        "min" | "max" | "rem_euclid" | "div_euclid" => {
            arg_count(name, args, 1)?;
            let j = int_arg(name, &args[0])?;
            Val::Int(match name {
                "min" => i.min(j),
                "max" => i.max(j),
                _ if j == 0 => return Err("integer division by zero".into()),
                "rem_euclid" => i.rem_euclid(j),
                _ => i.div_euclid(j),
            })
        }
        _ => return Err(format!("integers have no method `{}` in sjc", name)),
    })
}

/// The methods of arrays and ranges (as iterators): `iter`, `enumerate`, `rev`, `zip`, ...
fn seq_method<'k>(vs: Vec<Val<'k>>, name: &str, args: &[Val<'k>]) -> VResult<'k> {
    Ok(match name {
        "iter" | "into_iter" | "to_vec" => {
            arg_count(name, args, 0)?;
            Val::Array(vs)
        }
        "len" => {
            arg_count(name, args, 0)?;
            Val::Int(vs.len() as i64)
        }
        "rev" => {
            arg_count(name, args, 0)?;
            Val::Array(vs.into_iter().rev().collect())
        }
        "enumerate" => {
            arg_count(name, args, 0)?;
            Val::Array(
                vs.into_iter()
                    .enumerate()
                    .map(|(i, v)| Val::Tuple(vec![Val::Int(i as i64), v]))
                    .collect(),
            )
        }
        "step_by" => {
            arg_count(name, args, 1)?;
            let n = int_arg(name, &args[0])?;
            if n <= 0 {
                return Err("`step_by` needs a positive step".into());
            }
            Val::Array(vs.into_iter().step_by(n as usize).collect())
        }
        "skip" | "take" => {
            arg_count(name, args, 1)?;
            let n = int_arg(name, &args[0])?.max(0) as usize;
            Val::Array(if name == "skip" {
                vs.into_iter().skip(n).collect()
            } else {
                vs.into_iter().take(n).collect()
            })
        }
        "zip" => {
            arg_count(name, args, 1)?;
            let other: Vec<Val> = match &args[0] {
                Val::Array(ws) => ws.clone(),
                Val::Range(a, b) => (*a..(*b).max(*a)).map(Val::Int).collect(),
                v => return Err(format!("cannot zip with {}", v.describe())),
            };
            Val::Array(
                vs.into_iter()
                    .zip(other)
                    .map(|(a, b)| Val::Tuple(vec![a, b]))
                    .collect(),
            )
        }
        _ => {
            return Err(format!(
                "arrays and ranges have no method `{}` in sjc",
                name
            ))
        }
    })
}

/// `z.re` and `z.im` (`ops::re`, `ops::im`).
pub fn part<'k>(v: Val<'k>, re: bool) -> VResult<'k> {
    Ok(match v {
        Val::C64(z) => Val::F64(if re { z.re } else { z.im }),
        Val::Cplx(c) => Val::Real(if re { ops::re(c) } else { ops::im(c) }),
        v => {
            return Err(format!(
                "{} has no field `{}`",
                v.describe(),
                if re { "re" } else { "im" }
            ))
        }
    })
}
