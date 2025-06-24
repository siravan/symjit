use anyhow::{anyhow, Result};
use num_traits::Float;

pub type BinaryFunc<T> = extern "C" fn(T, T) -> T;

pub struct VirtualTable<T>(T);

impl<T: Float> VirtualTable<T> {
    // Finds the function reference for op
    pub fn from_str(op: &str) -> Result<BinaryFunc<T>> {
        let f = match op {
            "power" => Self::power, //unsafe {
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
            "expm1" => Self::expm1,
            "log1p" => Self::log1p,
            "exp2" => Self::exp2,
            "log2" => Self::log2,
            _ => {
                return Err(anyhow!("op_code {} not found", op));
            }
        };

        Ok(f)
    }

    pub extern "C" fn power(x: T, y: T) -> T {
        x.powf(y)
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

    pub extern "C" fn expm1(x: T, _y: T) -> T {
        x.exp_m1()
    }

    pub extern "C" fn log1p(x: T, _y: T) -> T {
        x.ln_1p()
    }

    pub extern "C" fn exp2(x: T, _y: T) -> T {
        x.exp2()
    }

    pub extern "C" fn log2(x: T, _y: T) -> T {
        x.log2()
    }
}
