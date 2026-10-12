use symjit::kernel::kernel;

#[kernel]
fn shapes(x: f64) -> f64 {
    let (a, b) = if x < 0.0 { (x, -x) } else { x };
    a + b
}

fn main() {}
