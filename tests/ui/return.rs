use symjit::kernel::kernel;

#[kernel]
fn early(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    x
}

fn main() {}
