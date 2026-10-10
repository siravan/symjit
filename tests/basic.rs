use std::f64;

use anyhow::Result;
use num_complex::Complex;
use symjit::{
    int, var, Compiled, Compiler, CompilerType, Composer, Config, Defuns, Expr, FastFunc,
    PlaneDescriptor, Slot, Translator,
};

#[test]
fn test_simple() -> Result<()> {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let p = &x + &y;
    let q = &x * &y;

    let mut config = Config::default();
    config.set_opt_level(2); // optional
    let mut comp = Compiler::with_config(config);

    let mut app = comp.compile(&[x, y], &[p, q])?;
    let res = app.call(&[3.0, 5.0]);

    assert!(res[0] == 8.0 && res[1] == 15.0);

    Ok(())
}

#[test]
fn test_pi_viete() -> Result<()> {
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

    assert!((res[0] - f64::consts::PI).abs() < 1e-14);

    Ok(())
}

#[test]
fn test_loops() -> Result<()> {
    let x = var("x");
    let n = var("n");
    let i = var("i");
    let j = var("j");

    let u = x
        .pow(&j)
        .div(&i.prod(&i, &int(1), &j))
        .sum(&j, &int(0), &int(50));

    let numer = j.rem(&int(2)).eq(&int(0)).ifelse(&int(4), &int(-4));
    let denom = j.mul(&int(2)).add(&int(1));
    let v = (&numer / &denom).sum(&j, &int(0), &n);

    let mut app = Compiler::new().compile(&[x, n], &[u, v])?;
    let res = app.call(&[2.0, 100000000.0]);

    assert!((res[0] - f64::exp(2.0)).abs() < 1e-10);
    assert!((res[1] - f64::consts::PI).abs() < 1e-6);

    Ok(())
}

#[test]
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
        assert_eq!(v, 48.0);
    }

    Ok(())
}

#[test]
fn test_fact() -> Result<()> {
    let x = Expr::var("x");
    let i = Expr::var("i");
    let p = i.prod(&i, &Expr::from(1), &x);

    let mut comp = Compiler::new();
    let mut app = comp.compile(&[x], &[p])?;
    let f = app.fast_func()?;

    if let FastFunc::F1(f, _) = f {
        let v = f(6.0);
        assert_eq!(v, 720.0);
    }

    Ok(())
}

extern "C" fn f(x: f64) -> f64 {
    x.exp()
}

extern "C" fn g(x: f64, y: f64) -> f64 {
    x.ln() * y
}

fn test_external_inner(p: i32) -> Result<()> {
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

#[test]
fn test_external() -> Result<()> {
    for p in 0..50 {
        test_external_inner(p)?;
    }
    Ok(())
}

fn compile_external_evaluators(direct: bool) -> Result<symjit::Application> {
    let f: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> = Box::new(|args| {
        let y = args[0];
        3.0 + y * y + y
    });

    let g: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> = Box::new(|args| {
        let y = args[0];
        3.0 + y * y + y
    });

    let mut defuns = Defuns::new();
    defuns.add_sliced_func("f", f)?;
    defuns.add_sliced_func("g", g)?;

    let mut config = Config::default();
    config.set_complex(false);
    config.set_direct(direct);
    config.set_defuns(defuns);

    // Compile f(y) + f(x). The call order makes the incorrect result match the original report.
    let mut translator = Translator::new(config);
    translator.set_num_params(2);
    translator.append_fun(&Slot::Temp(0), "f", &[Slot::Param(1)], true)?;
    translator.append_fun(&Slot::Temp(1), "f", &[Slot::Param(0)], true)?;
    translator.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Temp(1)], 2)?;

    Ok(translator.compile()?)
}

// external call bug fixed in v2.22.1
#[test]
fn test_external_evaluators() -> Result<()> {
    let args = [5.0, 2.0];
    let expected = 42.0;
    let optimized = compile_external_evaluators(false)?.evaluate_single(&args);
    let direct = compile_external_evaluators(true)?.evaluate_single(&args);

    assert_eq!(optimized, expected, "optimized translation is incorrect");
    assert_eq!(direct, expected);
    Ok(())
}

#[test]
fn test_multiple() -> Result<()> {
    let mut config = Config::default();
    config.set_complex(false);

    // Compile f(y) + f(x). The call order makes the incorrect result match the original report.
    let mut translator = Translator::new(config);
    translator.set_num_params(2);
    translator.append_add(&Slot::Out(0), &[Slot::Param(0), Slot::Param(1)], 0)?;
    translator.append_mul(&Slot::Out(1), &[Slot::Param(0), Slot::Out(0)], 0)?;

    let app = translator.compile()?.seal()?;

    let args = [5.0, 2.0];
    let mut outs = vec![0.0; 2];

    app.evaluate(&args, &mut outs);

    assert_eq!(outs[0], args[0] + args[1]);
    assert_eq!(outs[1], (args[0] + args[1]) * args[0]);

    Ok(())
}

#[test]
fn test_output_reuse() -> Result<()> {
    let mut config = Config::default();
    config.set_complex(false);

    // Compile f(y) + f(x). The call order makes the incorrect result match the original report.
    let mut translator = Translator::new(config);
    translator.set_num_params(1);
    translator.append_assign(&Slot::Out(0), &Slot::Param(0))?;
    translator.append_assign(&Slot::Out(1), &Slot::Out(0))?;

    let app = translator.compile()?;

    // app.dump("output_reuse.bytecode.txt", "bytecode");

    let args = [5.0, 2.0];
    let mut outs = vec![0.0; 2];

    app.evaluate(&args, &mut outs);

    assert_eq!(outs[0], args[0]);
    assert_eq!(outs[1], args[0]);

    Ok(())
}

// AI-generated test unused-trailing-inputs bug
#[test]
fn declared_unused_inputs_do_not_shift_outputs() -> Result<()> {
    for complex in [false, true] {
        for opt in [0, 2] {
            for all_unused in [false, true] {
                let mut config = Config::new(CompilerType::Native, 0)?;
                config.set_symbolica(true);
                config.set_opt_level(opt);
                config.set_complex(complex);
                config.set_simd(true);
                config.set_direct_arena(true);
                config.set_direct_arena_identity_output(true);
                let mut translator = Translator::new(config);
                translator.set_num_params(2);
                let source = if all_unused {
                    Slot::Const(
                        translator
                            .append_constant(Complex::new(7.0, if complex { 9.0 } else { 0.0 }))?,
                    )
                } else {
                    Slot::Param(0)
                };
                if all_unused {
                    translator.append_assign(&Slot::Out(0), &source)?;
                } else {
                    translator.append_add(
                        &Slot::Out(0),
                        &[source, source],
                        if complex { 0 } else { 2 },
                    )?;
                }
                let mut application = translator.compile()?;
                application.prepare_simd();
                let app = application.seal()?;
                let lanes = app.compiled_simd.as_ref().map_or(1, |v| v.count_lanes());
                let width = if complex { 2 } else { 1 };
                // All input/output planes are disjoint. The unused input is a sentinel.
                let mut planes: Vec<Vec<f64>> = (0..3 * width)
                    .map(|i| {
                        vec![
                            if i < width {
                                3.0 + i as f64
                            } else if i < 2 * width {
                                17.0 + i as f64
                            } else {
                                f64::NAN
                            };
                            lanes
                        ]
                    })
                    .collect();
                let table: Vec<_> = planes
                    .iter_mut()
                    .map(|plane| unsafe {
                        PlaneDescriptor::from_raw_parts(plane.as_mut_ptr(), lanes)
                    })
                    .collect();
                for (mode, kernel) in [
                    ("scalar", app.scalar_plane_kernel()),
                    ("SIMD", app.simd_plane_kernel()),
                ] {
                    let kernel = kernel.expect("this reproducer requires scalar and SIMD kernels");
                    for i in 0..width {
                        planes[i].fill(3.0 + i as f64);
                    }
                    for i in width..2 * width {
                        planes[i].fill(17.0 + i as f64);
                    }
                    for plane in &mut planes[2 * width..] {
                        plane.fill(f64::NAN);
                    }
                    let status =
                        unsafe { kernel(std::ptr::null(), table.as_ptr(), 0, app.params.as_ptr()) };
                    assert_eq!(status, 0);
                    let overwritten = (0..2 * width).any(|i| {
                        planes[i].iter().any(|value| {
                            *value
                                != if i < width {
                                    3.0 + i as f64
                                } else {
                                    17.0 + i as f64
                                }
                        })
                    });
                    assert!(
                        !overwritten,
                        "input modified: complex={complex}, O{opt}, {mode}, constant={all_unused}"
                    );
                    for i in 0..width {
                        let expected = if all_unused {
                            7.0 + 2.0 * i as f64
                        } else {
                            6.0 + 2.0 * i as f64
                        };
                        for lane in 0..if mode == "scalar" { 1 } else { lanes } {
                            assert_eq!(planes[2 * width + i][lane], expected,
                                       "output: complex={complex}, O{opt}, {mode}, constant={all_unused}, lane={lane}");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_zero_params() -> Result<()> {
    let simd = std::env::args().nth(1).as_deref() != Some("scalar");
    let mut config = Config::default();
    config.set_complex(true);
    config.set_fastmath(false);
    config.set_direct(false);
    config.set_opt_level(2);
    config.set_simd(simd);
    config.enable_simd512(false);
    config.set_simd_branch(false);
    config.set_threads(false);

    let mut t = Translator::new(config);
    t.set_num_params(0);
    t.append_constant(Complex::new(2.0, 0.0)).unwrap();
    t.append_constant(Complex::new(std::f64::consts::PI, 0.0))
        .unwrap();
    t.append_assign(&Slot::Out(0), &Slot::Const(0)).unwrap();
    t.append_assign(&Slot::Out(1), &Slot::Const(1)).unwrap();
    let app = t.compile().unwrap().seal().unwrap();
    let expected = [
        Complex::new(2.0, 0.0),
        Complex::new(std::f64::consts::PI, 0.0),
    ];
    let mut single = [Complex::new(0.0, 0.0); 2];
    app.evaluate(&[], &mut single);
    assert_eq!(single, expected);
    let mut output = [Complex::new(0.0, 0.0); 10];
    app.evaluate_matrix(&[], &mut output, 5);
    for row in output.chunks_exact(2) {
        assert_eq!(row, expected);
    }

    Ok(())
}

#[test]
fn test_hyperbolic() -> Result<()> {
    let mut config = Config::default();
    config.set_complex(true);

    let mut t = Translator::new(config);
    t.set_num_params(1);
    t.append_fun(&Slot::Temp(0), "tanh", &[Slot::Param(0)], false)?;
    t.append_fun(&Slot::Temp(1), "coth", &[Slot::Param(0)], false)?;
    t.append_fun(&Slot::Temp(2), "sech", &[Slot::Param(0)], false)?;
    t.append_fun(&Slot::Temp(3), "csch", &[Slot::Param(0)], false)?;
    t.append_add(
        &Slot::Out(0),
        &[Slot::Temp(0), Slot::Temp(1), Slot::Temp(2), Slot::Temp(3)],
        0,
    )?;

    let app = t.compile().unwrap().seal().unwrap();
    let args = [Complex::new(1000000000000.0, 0.0); 1];
    let mut outs = [Complex::new(0.0, 0.0); 1];

    app.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(2.0, 0.0));

    Ok(())
}
