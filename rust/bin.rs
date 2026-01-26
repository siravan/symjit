use anyhow::Result;
use num_complex::Complex;
use rand::{self, Rng};
use std::fs;
use symjit::{int, var, Compiler, Config, Expr, FastFunc};

use symbolica::{
    atom::{self, AtomCore},
    evaluate::{FunctionMap, OptimizationSettings},
    parse, symbol,
};

fn test_simple() -> Result<()> {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let p = &x + &y;
    let q = &x * &y;

    let mut config = Config::default();
    config.set_opt_level(2); // optional
    let mut comp = Compiler::with_config(config);

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

fn test_symbolica_scalar() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];

    let mut f = FunctionMap::new();
    f.add_conditional(symbol!("if")).unwrap();

    let tests = vec![
        ("x + y^2", &[2.0, 5.0]),
        ("x - 4.0", &[-4.0, 10.0]),
        ("sin(x) + y", &[1.0, 5.0]),
        ("x^10 / y^3", &[2.0, 10.0]),
        ("if(y, x*x + y, x + 3)", &[5.0, 0.0]),
        ("if(y, x*x, x + 3)", &[5.0, 2.0]),
        ("if(y, x*x + y, x + 3)", &[5.0, 2.0]),
        ("x^2 + y^2", &[3.0, 4.0]),
        ("x^3 + y^3", &[5.0, 6.0]),
        ("x^30 + y^30", &[2.0, 3.0]),
    ];

    let mut outs = vec![0.0];

    for (input, args) in tests {
        let eval = parse!(input)
            .evaluator(&f, &params, OptimizationSettings::default())
            .unwrap();
        let json = serde_json::to_string(&eval.export_instructions())?;

        let mut comp = Compiler::new();
        let mut app = comp.translate(&json)?;

        app.evaluate(args, &mut outs);
        let v = eval.map_coeff(&|x| x.re.to_f64()).evaluate_single(args);
        println!("{:?}({:?}) => {} vs {}", &input, args, outs[0], v);
    }

    Ok(())
}

fn test_symbolica_complex() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let eval = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap();
    let json = serde_json::to_string(&eval.export_instructions())?;

    let mut config = Config::default();
    config.set_complex(true);
    let mut comp = Compiler::with_config(config);
    let mut app = comp.translate(&json)?;

    let u = app.evaluate_single(&[Complex::new(2.0, 1.0), Complex::new(-2.0, 4.0)]);
    println!("{:?}", u);

    Ok(())
}

fn test_symbolica_external() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("sinh"), "sinh".to_string())
        .unwrap();

    let eval = parse!("sinh(x+y)")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap();
    let json = serde_json::to_string(&eval.export_instructions())?;

    let mut comp = Compiler::new();
    let mut app = comp.translate(&json)?;

    let u = app.evaluate_single(&[2.0, -3.0]);
    println!("sinh(x+y)[2.0, -3.0] => {:?} vs {:?}", u, f64::sinh(-1.0));

    Ok(())
}

fn test_symbolica_doc_1() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let eval = parse!("x + y^2")
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings::default(),
        )
        .unwrap();

    let json = serde_json::to_string(&eval.export_instructions())?;
    let mut comp = Compiler::new();
    let mut app = comp.translate(&json)?;
    assert!(app.evaluate_single(&[2.0, 3.0]) == 11.0);
    Ok(())
}

fn test_symbolica_doc_2() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let eval = parse!("x + y^2")
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings::default(),
        )
        .unwrap();

    let json = serde_json::to_string(&eval.export_instructions())?;
    let mut config = Config::default();
    config.set_complex(true);
    let mut comp = Compiler::with_config(config);
    let mut app = comp.translate(&json)?;
    let v = vec![Complex::new(2.0, 1.0), Complex::new(-1.0, 3.0)];
    assert!(app.evaluate_single(&v) == Complex::new(-6.0, -5.0));
    Ok(())
}

fn test_symbolica_doc_3() -> Result<()> {
    let params = vec![parse!("x")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("sinh"), "sinh".to_string())
        .unwrap();

    let eval = parse!("sinh(x)")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap();

    let json = serde_json::to_string(&eval.export_instructions())?;
    let mut comp = Compiler::new();
    let mut app = comp.translate(&json)?;
    assert!(app.evaluate_single(&[1.5]) == f64::sinh(1.5));
    Ok(())
}

fn test_symbolica_opt2_bug() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let eval = parse!("x^3 + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap();
    let json = serde_json::to_string(&eval.export_instructions())?;

    let mut config = Config::default();
    config.set_opt_level(1);
    let mut comp = Compiler::with_config(config);
    let mut app = comp.translate(&json)?;

    let u = app.evaluate_single(&[2.0, 3.0]);
    assert!(u == 35.0);

    let mut config = Config::default();
    config.set_opt_level(2);
    let mut comp = Compiler::with_config(config);
    let mut app = comp.translate(&json)?;

    let u = app.evaluate_single(&[2.0, 3.0]);
    assert!(u == 35.0);

    Ok(())
}

fn test_f13() -> Result<()> {
    let f13 = fs::read_to_string("f13.txt")?;

    let params: Vec<atom::Atom> = [
        "alpha", "amuq", "ammu", "xcp1", "e1245", "xcp4", "e3e2", "e1234", "e2345", "e1235",
        "e1345", "amel2", "e2e1", "e5e2", "e4e2", "e3e1", "e4e1", "e5e1", "ammu2", "amuq2", "e5e3",
        "e4e3", "x5", "x6", "x1", "x3", "x4", "xcp3", "xcp2",
    ]
    .iter()
    .map(|v| parse!(v))
    .collect();

    let mut evaluator = parse!(f13)
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings::default(),
        )
        .unwrap()
        .map_coeff(&|x| x.to_real().unwrap().to_f64());

    let mut rng = rand::rng();
    let inputs: Vec<f64> = params.iter().map(|_| rng.random::<f64>()).collect();

    let y1 = evaluator.evaluate_single(&inputs);

    let mut comp = Compiler::new();
    let json = serde_json::to_string(&evaluator.export_instructions())?;
    let mut app = comp.translate(&json)?;

    let y2 = app.evaluate_single(&inputs);

    assert!((y1 - y2).abs() < 1e-10);

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

    // print!("testing memory leaks...");
    // test_memory(1000)?;
    // println!("pass!");

    test_symbolica_doc_1()?;
    println!("doc test #1 passed");

    test_symbolica_doc_2()?;
    println!("doc test #2 passed");

    test_symbolica_doc_3()?;
    println!("doc test #3 passed");

    test_symbolica_scalar()?;
    test_symbolica_complex()?;
    test_symbolica_external()?;

    test_symbolica_opt2_bug()?;
    println!("opt_level 2 bug test passed");

    test_f13()?;
    println!("f13 passed");

    Ok(())
}
