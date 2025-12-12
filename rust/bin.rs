use anyhow::{anyhow, Result};
use symjit::{compiler::FastFunc, Compiler, CompilerType, Expr};

fn test_simple() -> Result<()> {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let p = &x + &y;
    let q = &x * &y;

    let mut comp = Compiler::new();
    comp.opt_level(2); // optional
    let mut app = comp.compile(&[x, y], &[p, q])?;
    let v = app.call(&[3.0, 5.0]);
    println!("{:?}", &v);

    Ok(())
}

fn viete(x: Expr, n: usize) -> Expr {
    let mut p = Expr::from(1);

    for i in 0..n {
        let mut t = x.clone();
        for j in 0..i {
            t = &x + &(&x * &t.sqrt());
        }
        p = &p * &t.sqrt();
    }

    p
}

pub fn test_pi() -> Result<()> {
    let x = Expr::var("x");
    let p = viete(x.clone(), 20);

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x], &[&Expr::from(2) / &p])?;
    let v = app.call(&[0.5]);
    println!("{:?}", &v);

    Ok(())
}

#[cfg(target_arch = "x86_64")]
pub fn test_simd() -> Result<()> {
    use std::arch::x86_64::_mm256_loadu_pd;

    let x = Expr::var("x");
    let p = Expr::var("p"); // parameter

    let expr = &x.square() * &p;
    let mut comp = Compiler::new();
    let mut app = comp.compile_params(&[x], &[expr], &[p])?;

    let v = vec![1.0, 2.0, 3.0, 4.0];
    let p = unsafe { vec![_mm256_loadu_pd(v.as_ptr())] };
    let q = app.call_simd_params(&p, &[5.0])?;
    println!("{:?}", &q);
    Ok(())
}

fn test_fast() -> Result<()> {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let z = Expr::var("z");
    let p = &x * &(&y - &z).pow(&Expr::from(2));

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x, y, z], &[p])?;
    let f = app.fast_func().ok_or(anyhow!("not a fast function"))?;

    if let FastFunc::F3(f, _) = f {
        let v = f(3.0, 5.0, 9.0);
        println!("{:?}", &v);
    }

    Ok(())
}

fn test_fact() -> Result<()> {
    let x = Expr::var("x");
    let i = Expr::var("i");
    let p = i.prod(&i, &Expr::from(1), &x);

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x], &[p])?;
    let f = app.fast_func().ok_or(anyhow!("not a fast function"))?;

    if let FastFunc::F1(f, _) = f {
        let v = f(6.0);
        println!("6! = {:?}", &v);
    }

    Ok(())
}

fn test() -> Option<fn(f64) -> f64> {
    let x = Expr::var("x");
    let i = Expr::var("i");
    let p = i.prod(&i, &Expr::from(1), &x);

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x], &[p]).unwrap();
    let f = app.fast_func().unwrap();

    if let FastFunc::F1(f, _) = f {
        Some(f)
    } else {
        None
    }
}

pub fn main() -> Result<()> {
    for _ in 0..2000 {
        test_simple()?;
        test_pi()?;
        test_fast()?;
        test_fact()?;
    }

    let f = test().unwrap();
    println!("{}", f(8.0));

    if cfg!(target_arch = "x86_64") {
        test_simd()?;
    }

    Ok(())
}
