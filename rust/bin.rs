use anyhow::Result;
use symjit::{int, var, Compiler, Expr, FastFunc};

fn test_simple() -> Result<()> {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let p = &x + &y;
    let q = &x * &y;

    let mut comp = Compiler::new();
    comp.opt_level(2); // optional
    let mut app = comp.compile(&[x, y], &[p, q])?;
    let v = app.call(&[3.0, 5.0]);
    println!("simple\t{:?}", &v);

    Ok(())
}

fn test_pi_viete(silent: bool) -> Result<()> {
    let x = var("x");
    let mut u = int(1);

    for i in 0..50 {
        let mut t = x.clone();

        for _ in 0..i {
            t = &x + &(&x * &t.sqrt());
        }

        u = &u * &t.sqrt();
    }

    let mut app = Compiler::new().compile(&[x], &[&int(2) / &u])?;
    let res = app.call(&[0.5]);

    if !silent {
        // println!("{:?}", &u);
        println!("pi = \t{:?}", res[0]);
    }

    Ok(())
}

fn test_loops() -> Result<()> {
    let x = var("x");
    let n = var("n");
    let i = var("i");
    let j = var("j");

    // u = x^j / factorial(j) for j in j in 0..=50
    let u = x
        .pow(&j)
        .div(&i.prod(&i, &int(1), &j))
        .sum(&j, &int(0), &int(50));

    // numer = if j % 2 == 0 { 4 } else { -4 }
    let numer = j.rem(&int(2)).eq(&int(0)).ifelse(&int(4), &int(-4));
    // denom = j * 2 + 1
    let denom = j.mul(&int(2)).add(&int(1));
    // v = numer / denom for j in 0..=100000000
    let v = (&numer / &denom).sum(&j, &int(0), &n);

    let mut app = Compiler::new().compile(&[x, n], &[u, v])?;
    let res = app.call(&[2.0, 100000000.0]);

    println!("e^2 = \t{:?}", res[0]);
    println!("pi = \t{:?}", res[1]);

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn test_simd() -> Result<()> {
    use std::arch::x86_64::_mm256_loadu_pd;

    let x = Expr::var("x");
    let p = Expr::var("p"); // parameter

    let expr = &x.square() * &p;
    let mut comp = Compiler::new();
    let mut app = comp.compile_params(&[x], &[expr], &[p])?;

    let a = &[1.0, 2.0, 3.0, 4.0];
    let a = unsafe { _mm256_loadu_pd(a.as_ptr()) };
    let res = app.call_simd_params(&[a], &[5.0])?;
    println!("simd\t{:?}", &res);
    Ok(())
}

fn test_fast() -> Result<()> {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let z = Expr::var("z");
    let p = &x * &(&y - &z).pow(&Expr::from(2));

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x, y, z], &[p])?;
    let f = app.fast_func()?;

    if let FastFunc::F3(f, _) = f {
        let v = f(3.0, 5.0, 9.0);
        println!("fast\t{:?}", &v);
    }

    Ok(())
}

fn test_fact() -> Result<()> {
    let x = Expr::var("x");
    let i = Expr::var("i");
    let p = i.prod(&i, &Expr::from(1), &x);

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x], &[p])?;
    let f = app.fast_func()?;

    if let FastFunc::F1(f, _) = f {
        let v = f(6.0);
        println!("fact\t6! = {:?}", &v);
    }

    Ok(())
}

extern "C" fn f(x: f64) -> f64 {
    x.exp()
}

extern "C" fn g(x: f64, y: f64) -> f64 {
    x.ln() * y
}

fn test_external(p: i32) -> Result<()> {
    let x = Expr::var("x");
    let u = Expr::unary("f_", &x);
    let v = &x * &Expr::binary("g_", &u, &x);

    let mut comp = Compiler::new();
    comp.def_unary("f_", f);
    comp.def_binary("g_", g);
    let mut app = comp.compile(&[x], &[v])?;
    let res = app.call(&[p as f64]);
    println!("f({}) = \t{:?}", p, &res); // it should be 5.0 ^ 3

    Ok(())
}

fn test_memory(n: usize) -> Result<()> {
    for _ in 0..n {
        test_pi_viete(true)?;
    }
    Ok(())
}

pub fn main() -> Result<()> {
    test_simple()?;
    test_pi_viete(false)?;
    test_loops()?;
    test_fast()?;
    test_fact()?;

    for p in 0..50 {
        test_external(p)?;
    }

    #[cfg(target_arch = "x86_64")]
    test_simd()?;

    print!("testing memory leaks...");
    test_memory(1000)?;
    println!("pass!");

    Ok(())
}
