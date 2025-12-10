use anyhow::Result;
use symjit::{Compiler, CompilerType, Expr};

fn test_simple() -> Result<()> {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let p = &x + &y;
    let q = &x * &y;

    let mut comp = Compiler::new();
    comp.opt_level(2); // optional
    let mut func = comp.compile(&[x, y], &[p, q])?;
    let v = func.call(&[3.0, 5.0]);
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
    let mut func = comp.compile(&[x], &[&Expr::from(2) / &p])?;
    let v = func.call(&[0.5]);
    println!("{:?}", &v);

    Ok(())
}

#[cfg(target_arch = "x86_64")]
pub fn test_simd() -> Result<()> {
    use std::arch::x86_64::_mm256_loadu_pd;

    let x = Expr::var("x");
    let p = x.square();
    let mut comp = Compiler::new();
    let mut func = comp.compile(&[x], &[p])?;

    let v = vec![1.0, 2.0, 3.0, 4.0];
    let p = unsafe { vec![_mm256_loadu_pd(v.as_ptr())] };
    let q = unsafe { func.call_simd(&p).unwrap() };
    println!("{:?}", &q);
    Ok(())
}

pub fn main() -> Result<()> {
    test_simple()?;
    test_pi()?;

    if cfg!(target_arch = "x86_64") {
        test_simd()?;
    }

    Ok(())
}
