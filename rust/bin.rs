mod testing {
    use anyhow::Result;
    use symjit::{int, var, Compiler, Config, Defuns, Expr, FastFunc};

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

        let mut df = Defuns::new();
        df.add_unary("f_", f);
        df.add_binary("g_", g);
        let mut config = Config::from_defuns(df)?;
        config.set_option("use_simd", "true")?;
        let mut comp = Compiler::with_config(config);

        let mut app = comp.compile(&[x], &[v])?;
        let res = app.call(&[p as f64]);
        let expected = (p * p * p) as f64;
        assert_eq!(res[0], expected);

        Ok(())
    }

    fn test_memory(n: usize) -> Result<()> {
        for _ in 0..n {
            test_pi_viete(true)?;
        }
        Ok(())
    }

    fn pass(what: &str) {
        println!("**** test {:?} passed. ****", what);
    }

    pub fn main() -> Result<()> {
        test_simple()?;
        pass("simple");

        test_pi_viete(false)?;
        pass("pi_viete");

        test_loops()?;
        pass("loops");

        test_fast()?;
        pass("fast");

        test_fact()?;
        pass("fact");

        test_memory(100)?;
        pass("memory");

        for p in 0..50 {
            test_external(p)?;
        }
        pass("external");

        Ok(())
    }

    /*
     * For profiling using Samply,
     * 1. Build as cargo build --profile profiling
     * 2. Profile as samply record ./target/profiling/symjit
     */
    pub fn profile() -> Result<()> {
        let model = std::fs::read_to_string("3loop_instructions_2.txt")?;
        for _ in 0..10 {
            let mut compiler = Compiler::new();
            compiler.translate(model.clone(), 0)?;
        }
        println!("compiled ok!");
        Ok(())
    }
}

use anyhow::Result;

// run `cargo run --release --features testing` for testing
fn main() -> Result<()> {
    testing::main()?;
    // testing::profile()?;
    Ok(())
}
