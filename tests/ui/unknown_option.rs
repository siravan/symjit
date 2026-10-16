use symjit::kernel::kernel;

#[kernel(fast)]
fn f(x: f64) -> f64 {
    x
}

fn main() {}
