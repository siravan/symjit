use symjit::kernel::kernel;

#[kernel]
fn scaled(x: f64, n: i32) -> f64 {
    x * n as f64
}

fn main() {}
