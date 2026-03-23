use num_complex::Complex;
use wide::{f64x2, f64x4};

#[derive(Clone, Debug)]
pub enum ElemType {
    RealF64(f64),
    ComplexF64(Complex<f64>),
    RealF64x2(f64x2),
    ComplexF64x2(Complex<f64x2>),
    RealF64x4(f64x4),
    ComplexF64x4(Complex<f64x4>),
}

pub trait Element: Default {
    fn get_type(x: Self) -> ElemType;
}

impl Element for f64 {
    fn get_type(x: Self) -> ElemType {
        ElemType::RealF64(x)
    }
}

impl Element for Complex<f64> {
    fn get_type(x: Self) -> ElemType {
        ElemType::ComplexF64(x)
    }
}

impl Element for f64x2 {
    fn get_type(x: Self) -> ElemType {
        ElemType::RealF64x2(x)
    }
}

impl Element for Complex<f64x2> {
    fn get_type(x: Self) -> ElemType {
        ElemType::ComplexF64x2(x)
    }
}

impl Element for f64x4 {
    fn get_type(x: Self) -> ElemType {
        ElemType::RealF64x4(x)
    }
}

impl Element for Complex<f64x4> {
    fn get_type(x: Self) -> ElemType {
        ElemType::ComplexF64x4(x)
    }
}
