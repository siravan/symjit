use symjit::kernel::kernel;

#[kernel(inline)]
fn down(x: f64) -> f64 {
    if x > 0.0 {
        down(x - 1.0)
    } else {
        x
    }
}

fn main() {}
