use symjit::kernel::kernel;
use symjit::Config;

#[kernel]
fn fact(n: f64) -> f64 {
    if n <= 0.0 {
        1.0
    } else {
        n * fact(n - 1.0)
    }
}

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

fn main() -> anyhow::Result<()> {
    let f = fact::compile(Config::default())?.seal()?;
    assert_eq!(f.evaluate_single(&[8.0]), fact(8.0));

    let g = fibonacci::compile(Config::default())?.seal()?;
    assert_eq!(g.evaluate_single(&[15.0]), fibonacci(15.0));

    Ok(())
}
