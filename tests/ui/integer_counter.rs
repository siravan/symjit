use symjit::kernel::kernel;

#[kernel]
fn count(x: f64) -> f64 {
    let mut y = x;
    let mut n = 0;
    while y > 1.0 {
        y = y * 0.5;
        n += 1;
    }
    y
}

fn main() {}
