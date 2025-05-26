use anyhow::{anyhow, Result};
use num_traits::Float;

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

    fn transmute(f: fn(f64) -> f64) -> BinaryFunc<T> {
        unsafe { std::mem::transmute::<fn(f64) -> f64, BinaryFunc<T>>(f) }
    }

    // Finds the function reference for op
    pub fn from_str(op: &str) -> Result<BinaryFunc<T>> {
        let f = match op {
            "nop" => Self::nop,
            "mov" => Self::mov,
            "plus" => Self::plus,
            "minus" => Self::minus,
            "neg" => Self::neg,
            "times" => Self::times,
            "divide" => Self::divide,
            "rem" => Self::rem,
            "power" => unsafe {
                std::mem::transmute::<fn(f64, f64) -> f64, BinaryFunc<T>>(libm::pow)
            },
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
            "sin" => Self::transmute(libm::sin),
            "cos" => Self::transmute(libm::cos),
            "tan" => Self::transmute(libm::tan),
            "csc" => Self::csc,
            "sec" => Self::sec,
            "cot" => Self::cot,
            "sinh" => Self::transmute(libm::sinh),
            "cosh" => Self::transmute(libm::cosh),
            "tanh" => Self::transmute(libm::tanh),
            "csch" => Self::csch,
            "sech" => Self::sech,
            "coth" => Self::coth,
            "arcsin" => Self::transmute(libm::asin),
            "arccos" => Self::transmute(libm::acos),
            "arctan" => Self::transmute(libm::atan),
            "arcsinh" => Self::transmute(libm::asinh),
            "arccosh" => Self::transmute(libm::acosh),
            "arctanh" => Self::transmute(libm::atanh),
            "exp" => Self::transmute(libm::exp),
            "ln" => Self::transmute(libm::log),
            "log" => Self::transmute(libm::log10),
            "root" => Self::transmute(libm::sqrt),
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
