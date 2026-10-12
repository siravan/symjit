use symjit::kernel::kernel;

#[kernel]
fn sign(x: f64) -> f64 {
    let mut s = 1;
    if x < 0.0 {
        s = -1;
    }
    x * s as f64
}

fn main() {}
