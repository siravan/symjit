use anyhow::{anyhow, Result};
use num_complex::Complex;
use spec_math::cephes64;
use std::fmt;

pub type UnaryFunc = extern "C" fn(f64) -> f64;
pub type BinaryFunc = extern "C" fn(f64, f64) -> f64;
pub type UnaryFuncCplx = extern "C" fn(Complex<f64>) -> Complex<f64>;
pub type BinaryFuncCplx = extern "C" fn(Complex<f64>, Complex<f64>) -> Complex<f64>;

#[derive(Clone)]
pub enum Func {
    Unary(UnaryFunc),
    Binary(BinaryFunc),
    UnaryCplx(UnaryFuncCplx),
    BinaryCplx(BinaryFuncCplx),
}

impl Func {
    pub fn func_ptr(&self) -> u64 {
        match self {
            Func::Unary(f) => *f as usize as u64,
            Func::Binary(f) => *f as usize as u64,
            Func::UnaryCplx(f) => *f as usize as u64,
            Func::BinaryCplx(f) => *f as usize as u64,
        }
    }
}

impl fmt::Debug for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<function pointer>")
    }
}

pub struct VirtualTable;

impl VirtualTable {
    // Finds the function reference for op
    pub fn from_str(op: &str) -> Result<Func> {
        let f = match op {
            "sin" => Func::Unary(Self::sin),
            "sinc" => Func::Unary(Self::sinc),
            "cos" => Func::Unary(Self::cos),
            "tan" => Func::Unary(Self::tan),
            "csc" => Func::Unary(Self::csc),
            "sec" => Func::Unary(Self::sec),
            "cot" => Func::Unary(Self::cot),
            "sinh" => Func::Unary(Self::sinh),
            "cosh" => Func::Unary(Self::cosh),
            "tanh" => Func::Unary(Self::tanh),
            "csch" => Func::Unary(Self::csch),
            "sech" => Func::Unary(Self::sech),
            "coth" => Func::Unary(Self::coth),
            "arcsin" => Func::Unary(Self::asin),
            "arccos" => Func::Unary(Self::acos),
            "arctan" => Func::Unary(Self::atan),
            "arcsinh" => Func::Unary(Self::asinh),
            "arccosh" => Func::Unary(Self::acosh),
            "arctanh" => Func::Unary(Self::atanh),
            "cbrt" => Func::Unary(Self::cbrt),
            "exp" => Func::Unary(Self::exp),
            "ln" => Func::Unary(Self::ln),
            "log" => Func::Unary(Self::log),
            "expm1" => Func::Unary(Self::expm1),
            "log1p" => Func::Unary(Self::log1p),
            "exp2" => Func::Unary(Self::exp2),
            "log2" => Func::Unary(Self::log2),
            "erf" => Func::Unary(Self::erf),
            "erfc" => Func::Unary(Self::erfc),
            "gamma" => Func::Unary(Self::gamma),
            "loggamma" => Func::Unary(Self::loggamma),
            "Si" => Func::Unary(Self::si),
            "Ci" => Func::Unary(Self::ci),
            "Shi" => Func::Unary(Self::shi),
            "Chi" => Func::Unary(Self::chi),
            // Binary Functions
            "power" => Func::Binary(Self::power),
            "atan2" => Func::Binary(Self::atan2),
            // Unary Complex Functions
            "cplx_sin" => Func::UnaryCplx(Self::cplx_sin),
            "cplx_cos" => Func::UnaryCplx(Self::cplx_cos),
            "cplx_tan" => Func::UnaryCplx(Self::cplx_tan),
            "cplx_csc" => Func::UnaryCplx(Self::cplx_csc),
            "cplx_sec" => Func::UnaryCplx(Self::cplx_sec),
            "cplx_cot" => Func::UnaryCplx(Self::cplx_cot),
            "cplx_sinh" => Func::UnaryCplx(Self::cplx_sinh),
            "cplx_cosh" => Func::UnaryCplx(Self::cplx_cosh),
            "cplx_tanh" => Func::UnaryCplx(Self::cplx_tanh),
            "cplx_csch" => Func::UnaryCplx(Self::cplx_csch),
            "cplx_sech" => Func::UnaryCplx(Self::cplx_sech),
            "cplx_coth" => Func::UnaryCplx(Self::cplx_coth),
            "cplx_arcsin" => Func::UnaryCplx(Self::cplx_asin),
            "cplx_arccos" => Func::UnaryCplx(Self::cplx_acos),
            "cplx_arctan" => Func::UnaryCplx(Self::cplx_atan),
            "cplx_arcsinh" => Func::UnaryCplx(Self::cplx_asinh),
            "cplx_arccosh" => Func::UnaryCplx(Self::cplx_acosh),
            "cplx_arctanh" => Func::UnaryCplx(Self::cplx_atanh),
            "cplx_cbrt" => Func::UnaryCplx(Self::cplx_cbrt),
            "cplx_exp" => Func::UnaryCplx(Self::cplx_exp),
            "cplx_ln" => Func::UnaryCplx(Self::cplx_ln),
            "cplx_log" => Func::UnaryCplx(Self::cplx_log),
            _ => {
                return Err(anyhow!("op_code {} not found", op));
            }
        };

        Ok(f)
    }

    pub extern "C" fn power(x: f64, y: f64) -> f64 {
        x.powf(y)
    }

    pub extern "C" fn atan2(x: f64, y: f64) -> f64 {
        x.atan2(y)
    }

    pub extern "C" fn sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            x.sin() / x
        }
    }

    pub extern "C" fn sin(x: f64) -> f64 {
        x.sin()
    }

    pub extern "C" fn cos(x: f64) -> f64 {
        x.cos()
    }

    pub extern "C" fn tan(x: f64) -> f64 {
        x.tan()
    }

    pub extern "C" fn csc(x: f64) -> f64 {
        1.0 / x.sin()
    }

    pub extern "C" fn sec(x: f64) -> f64 {
        1.0 / x.cos()
    }

    pub extern "C" fn cot(x: f64) -> f64 {
        1.0 / x.tan()
    }

    pub extern "C" fn sinh(x: f64) -> f64 {
        x.sinh()
    }

    pub extern "C" fn cosh(x: f64) -> f64 {
        x.cosh()
    }

    pub extern "C" fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    pub extern "C" fn csch(x: f64) -> f64 {
        1.0 / x.sinh()
    }

    pub extern "C" fn sech(x: f64) -> f64 {
        1.0 / x.cosh()
    }

    pub extern "C" fn coth(x: f64) -> f64 {
        1.0 / x.tanh()
    }

    pub extern "C" fn asin(x: f64) -> f64 {
        x.asin()
    }

    pub extern "C" fn acos(x: f64) -> f64 {
        x.acos()
    }

    pub extern "C" fn atan(x: f64) -> f64 {
        x.atan()
    }

    pub extern "C" fn asinh(x: f64) -> f64 {
        x.asinh()
    }

    pub extern "C" fn acosh(x: f64) -> f64 {
        x.acosh()
    }

    pub extern "C" fn atanh(x: f64) -> f64 {
        x.atanh()
    }

    pub extern "C" fn cbrt(x: f64) -> f64 {
        x.cbrt()
    }

    pub extern "C" fn exp(x: f64) -> f64 {
        x.exp()
    }

    pub extern "C" fn ln(x: f64) -> f64 {
        x.ln()
    }

    pub extern "C" fn log(x: f64) -> f64 {
        x.log10()
    }

    pub extern "C" fn expm1(x: f64) -> f64 {
        x.exp_m1()
    }

    pub extern "C" fn log1p(x: f64) -> f64 {
        x.ln_1p()
    }

    pub extern "C" fn exp2(x: f64) -> f64 {
        x.exp2()
    }

    pub extern "C" fn log2(x: f64) -> f64 {
        x.log2()
    }

    pub extern "C" fn gamma(x: f64) -> f64 {
        cephes64::gamma(x)
    }

    pub extern "C" fn loggamma(x: f64) -> f64 {
        cephes64::lgam(x)
    }

    pub extern "C" fn erf(x: f64) -> f64 {
        cephes64::erf(x)
    }

    pub extern "C" fn erfc(x: f64) -> f64 {
        cephes64::erfc(x)
    }

    pub extern "C" fn si(x: f64) -> f64 {
        let (s, _) = cephes64::sici(x);
        s
    }

    pub extern "C" fn ci(x: f64) -> f64 {
        let (_, c) = cephes64::sici(x);
        c
    }

    pub extern "C" fn shi(x: f64) -> f64 {
        let (s, _) = cephes64::shichi(x);
        s
    }

    pub extern "C" fn chi(x: f64) -> f64 {
        let (_, c) = cephes64::shichi(x);
        c
    }

    pub extern "C" fn cplx_sin(z: Complex<f64>) -> Complex<f64> {
        z.sin()
    }

    pub extern "C" fn cplx_cos(z: Complex<f64>) -> Complex<f64> {
        z.cos()
    }

    pub extern "C" fn cplx_tan(z: Complex<f64>) -> Complex<f64> {
        z.tan()
    }

    pub extern "C" fn cplx_csc(z: Complex<f64>) -> Complex<f64> {
        z.sin().inv()
    }

    pub extern "C" fn cplx_sec(z: Complex<f64>) -> Complex<f64> {
        z.cos().inv()
    }

    pub extern "C" fn cplx_cot(z: Complex<f64>) -> Complex<f64> {
        z.tan().inv()
    }

    pub extern "C" fn cplx_sinh(z: Complex<f64>) -> Complex<f64> {
        z.sinh()
    }

    pub extern "C" fn cplx_cosh(z: Complex<f64>) -> Complex<f64> {
        z.cosh()
    }

    pub extern "C" fn cplx_tanh(z: Complex<f64>) -> Complex<f64> {
        z.tanh()
    }

    pub extern "C" fn cplx_csch(z: Complex<f64>) -> Complex<f64> {
        z.sinh().inv()
    }

    pub extern "C" fn cplx_sech(z: Complex<f64>) -> Complex<f64> {
        z.cosh().inv()
    }

    pub extern "C" fn cplx_coth(z: Complex<f64>) -> Complex<f64> {
        z.tanh().inv()
    }

    pub extern "C" fn cplx_asin(z: Complex<f64>) -> Complex<f64> {
        z.asin()
    }

    pub extern "C" fn cplx_acos(z: Complex<f64>) -> Complex<f64> {
        z.acos()
    }

    pub extern "C" fn cplx_atan(z: Complex<f64>) -> Complex<f64> {
        z.atan()
    }

    pub extern "C" fn cplx_asinh(z: Complex<f64>) -> Complex<f64> {
        z.asinh()
    }

    pub extern "C" fn cplx_acosh(z: Complex<f64>) -> Complex<f64> {
        z.acosh()
    }

    pub extern "C" fn cplx_atanh(z: Complex<f64>) -> Complex<f64> {
        z.atanh()
    }

    pub extern "C" fn cplx_cbrt(z: Complex<f64>) -> Complex<f64> {
        z.cbrt()
    }

    pub extern "C" fn cplx_exp(z: Complex<f64>) -> Complex<f64> {
        z.exp()
    }

    pub extern "C" fn cplx_ln(z: Complex<f64>) -> Complex<f64> {
        z.ln()
    }

    pub extern "C" fn cplx_log(z: Complex<f64>) -> Complex<f64> {
        z.log10()
    }
}
