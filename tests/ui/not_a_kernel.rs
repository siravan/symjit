use symjit::kernel::kernel;

fn helper(x: f64) -> f64 {
    x * 2.0
}

#[kernel]
fn calls_helper(x: f64) -> f64 {
    helper(x) + 1.0
}

fn main() {}
