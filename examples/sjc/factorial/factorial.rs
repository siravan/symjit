use symjit::kernel::kernel;
use symjit::Config;

#[kernel]
fn fact(n: f64) -> f64 {
    if n <= 0.0 {
        1.0
    } else {
        n * fact(n - 1.0)
    }
}
