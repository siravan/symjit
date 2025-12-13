use anyhow::Result;
use symjit::{Compiler, Expr, FastFunc};

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

pub fn test_pi(silent: bool) -> Result<()> {
    let x = Expr::var("x");
    let p = viete(x.clone(), 20);

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x], &[&Expr::from(2) / &p])?;
    let v = app.call(&[0.5]);

    if !silent {
        println!("pi\t{:?}", &v);
    }

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
    println!("simd\t{:?}", &q);
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

fn test_external() -> Result<()> {
    let x = Expr::var("x");
    let p = Expr::unary("f_", &x);
    let q = &x * &Expr::binary("g_", &p, &x);

    let mut comp = Compiler::new();
    comp.def_unary("f_", f);
    comp.def_binary("g_", g);
    let mut app = comp.compile(&[x], &[q])?;
    let v = app.call(&[5.0]);
    println!("funs\t{:?}", &v); // it should be 5.0 ^ 3

    Ok(())
}

fn test_memory(n: usize) -> Result<()> {
    for _ in 0..n {
        test_pi(true)?;
    }
    Ok(())
}

pub fn main() -> Result<()> {
    test_simple()?;
    test_pi(false)?;
    test_fast()?;
    test_fact()?;
    test_external()?;

    if cfg!(target_arch = "x86_64") {
        test_simd()?;
    }

    print!("testing memory leaks...");
    test_memory(10000)?;
    println!("pass!");

    Ok(())
}
