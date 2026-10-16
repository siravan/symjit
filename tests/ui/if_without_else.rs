use symjit::kernel::kernel;

#[kernel]
fn no_else(x: f64) -> f64 {
    let y = if x < 0.0 { -x };
    y
}

fn main() {}
