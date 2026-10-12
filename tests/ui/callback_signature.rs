use symjit::kernel::kernel;

#[kernel(callback)]
fn three(x: f64, y: f64, z: f64) -> f64 {
    x + y + z
}

fn main() {}
