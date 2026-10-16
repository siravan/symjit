use symjit::kernel::kernel;

#[kernel]
fn rounded(x: f64) -> i64 {
    x as i64
}

fn main() {}
