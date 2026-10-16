#[kernel]
fn fibonacci(n: f64) -> f64 {
    let mut a = 0.0;
    let mut b = 1.0;
    let mut k = n;

    while k > 1.0 {
        (a, b) = (b, a + b);
        k -= 1.0;
    }

    b
}
