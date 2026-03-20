use num_complex::Complex;
use wide::{f64x2, f64x4};

#[derive(Clone, Debug)]
pub enum ElemType {
    None,
    RealF64,
    ComplexF64,
    RealF64x2,
    ComplexF64x2,
    RealF64x4,
    ComplexF64x4,
}

pub trait Element {
    fn get_type() -> ElemType;
}

impl Element for f64 {
    fn get_type() -> ElemType {
        ElemType::RealF64
    }
}

impl Element for Complex<f64> {
    fn get_type() -> ElemType {
        ElemType::ComplexF64
    }
}

impl Element for f64x2 {
    fn get_type() -> ElemType {
        ElemType::RealF64x2
    }
}

impl Element for Complex<f64x2> {
    fn get_type() -> ElemType {
        ElemType::ComplexF64x2
    }
}

impl Element for f64x4 {
    fn get_type() -> ElemType {
        ElemType::RealF64x4
    }
}

impl Element for Complex<f64x4> {
    fn get_type() -> ElemType {
        ElemType::ComplexF64x4
    }
}
