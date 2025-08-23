use anyhow::{anyhow, Result};
//use spec_math::{cephes64, Gamma};

pub type BinaryFunc = extern "C" fn(f64, f64) -> f64;

pub struct VirtualTable;

impl VirtualTable {
    // Finds the function reference for op
    pub fn from_str(op: &str) -> Result<BinaryFunc> {
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
            //"erf" => Self::erf,
            //"erfc" => Self::erfc,
            //"gamma" => Self::gamma,
            //"loggamma" => Self::loggamma,
            //"Si" => Self::si,
            //"Ci" => Self::ci,
            //"Shi" => Self::shi,
            //"Chi" => Self::chi,
            _ => {
                return Err(anyhow!("op_code {} not found", op));
            }
        };

        Ok(f)
    }

    pub extern "C" fn power(x: f64, y: f64) -> f64 {
        x.powf(y)
    }

    pub extern "C" fn sin(x: f64, _y: f64) -> f64 {
        x.sin()
    }

    pub extern "C" fn cos(x: f64, _y: f64) -> f64 {
        x.cos()
    }

    pub extern "C" fn tan(x: f64, _y: f64) -> f64 {
        x.tan()
    }

    pub extern "C" fn csc(x: f64, _y: f64) -> f64 {
        1.0 / x.sin()
    }

    pub extern "C" fn sec(x: f64, _y: f64) -> f64 {
        1.0 / x.cos()
    }

    pub extern "C" fn cot(x: f64, _y: f64) -> f64 {
        1.0 / x.tan()
    }

    pub extern "C" fn sinh(x: f64, _y: f64) -> f64 {
        x.sinh()
    }

    pub extern "C" fn cosh(x: f64, _y: f64) -> f64 {
        x.cosh()
    }

    pub extern "C" fn tanh(x: f64, _y: f64) -> f64 {
        x.tanh()
    }

    pub extern "C" fn csch(x: f64, _y: f64) -> f64 {
        1.0 / x.sinh()
    }

    pub extern "C" fn sech(x: f64, _y: f64) -> f64 {
        1.0 / x.cosh()
    }

    pub extern "C" fn coth(x: f64, _y: f64) -> f64 {
        1.0 / x.tanh()
    }

    pub extern "C" fn asin(x: f64, _y: f64) -> f64 {
        x.asin()
    }

    pub extern "C" fn acos(x: f64, _y: f64) -> f64 {
        x.acos()
    }

    pub extern "C" fn atan(x: f64, _y: f64) -> f64 {
        x.atan()
    }

    pub extern "C" fn asinh(x: f64, _y: f64) -> f64 {
        x.asinh()
    }

    pub extern "C" fn acosh(x: f64, _y: f64) -> f64 {
        x.acosh()
    }

    pub extern "C" fn atanh(x: f64, _y: f64) -> f64 {
        x.atanh()
    }

    pub extern "C" fn exp(x: f64, _y: f64) -> f64 {
        x.exp()
    }

    pub extern "C" fn ln(x: f64, _y: f64) -> f64 {
        x.ln()
    }

    pub extern "C" fn log(x: f64, _y: f64) -> f64 {
        x.log10()
    }

    pub extern "C" fn expm1(x: f64, _y: f64) -> f64 {
        x.exp_m1()
    }

    pub extern "C" fn log1p(x: f64, _y: f64) -> f64 {
        x.ln_1p()
    }

    pub extern "C" fn exp2(x: f64, _y: f64) -> f64 {
        x.exp2()
    }

    pub extern "C" fn log2(x: f64, _y: f64) -> f64 {
        x.log2()
    }

    /*
    pub extern "C" fn loggamma(x: f64, _y: f64) -> f64 {
        x.lgamma()
    }

    pub extern "C" fn erf(x: f64, _y: f64) -> f64 {
        cephes64::erf(x)
    }

    pub extern "C" fn erfc(x: f64, _y: f64) -> f64 {
        cephes64::erfc(x)
    }

    pub extern "C" fn si(x: f64, _y: f64) -> f64 {
        let (s, _) = cephes64::sici(x);
        s
    }

    pub extern "C" fn ci(x: f64, _y: f64) -> f64 {
        let (_, c) = cephes64::sici(x);
        c
    }

    pub extern "C" fn shi(x: f64, _y: f64) -> f64 {
        let (s, _) = cephes64::shichi(x);
        s
    }

    pub extern "C" fn chi(x: f64, _y: f64) -> f64 {
        let (_, c) = cephes64::shichi(x);
        c
    }
    */
}
