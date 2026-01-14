use anyhow::{anyhow, Result};
use num_complex::Complex;
use spec_math::cephes64;
use std::fmt;

/*
 * Design Note:
 *
 * Unary complex functions are defined as:
 *      `extern "C" fn(f64, f64, &mut Complex<f64>)`
 *
 * Initially, we tried `extern "C" fn(Complex<f64>) -> Complex<f64>`.
 * However, Complex is a user-defined structure in Rust and not an
 * intrinsic type. Therefore, the C-mapping is variable. This worked
 * fine on Linux systems but not Windows that passes structures in
 * stack and not registers.
 *
 * Next, we tried `fn(Complex<f64>) -> Complex<f64>`. In practice, this
 * works but has a main shortcoming. It depends on the details of
 * Rust ABI, which is not stable. In fact, this only works when the
 * crate is compiled in release mode and not debug mode.
 *
 * Therefore, we settled on the current version, where the function
 * arguments are defined explicitely and the C-mapping is unambiguous.
 * This version has the additional benefit of allowing call elision
 * (not implemented yet).
 *
 */

pub type UnaryFunc = extern "C" fn(f64) -> f64;
pub type BinaryFunc = extern "C" fn(f64, f64) -> f64;
pub type UnaryFuncCplx = extern "C" fn(f64, f64, &mut Complex<f64>);
pub type BinaryFuncCplx = extern "C" fn(f64, f64, &mut Complex<f64>);

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
            "cplx_sinc" => Func::UnaryCplx(Self::cplx_sinc),
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
            "cplx_root" => Func::UnaryCplx(Self::cplx_root),
            "cplx_cbrt" => Func::UnaryCplx(Self::cplx_cbrt),
            "cplx_exp" => Func::UnaryCplx(Self::cplx_exp),
            "cplx_ln" => Func::UnaryCplx(Self::cplx_ln),
            "cplx_log" => Func::UnaryCplx(Self::cplx_log),
            // Complex Binary Functions
            "cplx_power" => Func::BinaryCplx(Self::cplx_power),
            _ => {
                return Err(anyhow!("op_code {} is not found or is not supported", op));
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

    /************** Complex Functions ***************/

    pub extern "C" fn cplx_sinc(xr: f64, xi: f64, z: &mut Complex<f64>) {
        let x = Complex::new(xr, xi);
        if x == Complex::ZERO {
            *z = Complex::ONE;
        } else {
            *z = x.sin() / x;
        }
    }

    pub extern "C" fn cplx_sin(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).sin();
    }

    pub extern "C" fn cplx_cos(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cos();
    }

    pub extern "C" fn cplx_tan(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).tan();
    }

    pub extern "C" fn cplx_csc(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).sin().inv();
    }

    pub extern "C" fn cplx_sec(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cos().inv();
    }

    pub extern "C" fn cplx_cot(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).tan().inv();
    }

    pub extern "C" fn cplx_sinh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).sinh();
    }

    pub extern "C" fn cplx_cosh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cosh();
    }

    pub extern "C" fn cplx_tanh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).tanh();
    }

    pub extern "C" fn cplx_csch(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).sinh().inv();
    }

    pub extern "C" fn cplx_sech(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cosh().inv();
    }

    pub extern "C" fn cplx_coth(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).tanh().inv();
    }

    pub extern "C" fn cplx_asin(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).asin();
    }

    pub extern "C" fn cplx_acos(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).acos();
    }

    pub extern "C" fn cplx_atan(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).atan();
    }

    pub extern "C" fn cplx_asinh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).asinh();
    }

    pub extern "C" fn cplx_acosh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).acosh();
    }

    pub extern "C" fn cplx_atanh(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).atanh();
    }

    pub extern "C" fn cplx_root(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).sqrt();
    }

    pub extern "C" fn cplx_cbrt(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).cbrt();
    }

    pub extern "C" fn cplx_exp(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).exp();
    }

    pub extern "C" fn cplx_ln(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).ln();
    }

    pub extern "C" fn cplx_log(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).ln();
    }

    pub extern "C" fn cplx_power(xr: f64, xi: f64, z: &mut Complex<f64>) {
        *z = Complex::new(xr, xi).powc(*z);
    }
}
