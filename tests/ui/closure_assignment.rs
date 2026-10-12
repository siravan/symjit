use symjit::kernel::kernel;

#[kernel]
fn accumulate(x: f64, y: f64) -> f64 {
    let mut sum = 0.0;
    if x < y {
        let mut add = |v: f64| sum += v;
        add(x);
    }
    sum
}

fn main() {}
