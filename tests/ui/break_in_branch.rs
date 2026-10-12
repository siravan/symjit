use symjit::kernel::kernel;

#[kernel]
fn first_negative(x: f64, y: f64) -> f64 {
    let mut found = 0.0;
    for v in [x, y] {
        if v < 0.0 {
            found = v;
            break;
        }
    }
    found
}

fn main() {}
