// Kernels whose calls are errors when they are compiled (tests/source.rs): by the macro and by
// sjc.

use symjit::kernel::kernel;
use symjit::Complex;

#[kernel]
pub fn complex_callee(z: Complex<f64>) -> Complex<f64> {
    z * z
}

#[kernel]
pub fn real_caller(x: f64) -> f64 {
    complex_callee(Complex::new(x, 0.0)).re
}

#[kernel]
pub fn two_outputs(x: f64) -> (f64, f64) {
    (x, -x)
}

#[kernel]
pub fn calls_two_outputs(x: f64) -> f64 {
    let (a, b) = two_outputs(x);
    a + b
}

#[kernel]
pub fn ping(x: f64) -> f64 {
    if x > 0.0 {
        pong(x - 1.0)
    } else {
        x
    }
}

#[kernel]
pub fn pong(x: f64) -> f64 {
    ping(x) + 1.0
}
