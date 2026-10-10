use anyhow::Result;
pub use num_complex::{Complex, ComplexFloat};
use symjit::{Composer, Config, Defuns, Slot, Translator};

#[test]
fn test_simple() -> Result<()> {
    let mut config = Config::default();
    config.set_complex(false);
    config.set_num_args(1000);

    let mut ev = Translator::new(config);
    ev.set_num_params(2);
    ev.append_add(&Slot::Out(0), &[Slot::Param(0), Slot::Param(1)], 0)?;
    let f = ev.compile()?.seal()?;

    let mut outs = [0.0];
    f.evaluate(&[2.0, 5.0], &mut outs);

    assert_eq!(outs[0], 7.0);
    Ok(())
}

fn compile_external_evaluators(direct: bool, complex: bool) -> Result<symjit::Application> {
    let mut config = Config::default();
    config.set_complex(complex);
    config.set_direct(direct);
    config.set_debug_scalar(false);

    let mut inner = Translator::new(config);
    inner.set_num_params(2);
    inner.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Param(1)], 0)?;
    let f = inner.compile()?.seal()?;

    let mut defuns = Defuns::new();
    defuns.add_applet("f", f);

    let mut config = Config::default();
    config.set_complex(complex);
    config.set_direct(direct);
    config.set_debug_scalar(false);
    config.set_defuns(defuns);
    // config.set_debug_bytecode(!complex && !direct);

    let mut outer = Translator::new(config);
    outer.set_num_params(2);
    outer.append_constant(Complex::new(7.0, -2.0))?;

    outer.append_fun(
        &Slot::Temp(0),
        "f",
        &[Slot::Param(0), Slot::Const(0)],
        false,
    )?;

    outer.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Param(1)], 0)?;
    Ok(outer.compile()?)
}

#[test]
fn test_external_evaluators_real() -> Result<()> {
    let args = [5.0, 2.0];
    let expected = 37.0;
    let optimized = compile_external_evaluators(false, false)?.evaluate_single(&args);
    let direct = compile_external_evaluators(true, false)?.evaluate_single(&args);

    assert_eq!(optimized, expected, "optimized translation is incorrect");
    assert_eq!(direct, expected);
    Ok(())
}

#[test]
fn test_external_evaluators_complex() -> Result<()> {
    let args = [Complex::new(5.0, -2.0), Complex::new(2.0, 3.0)];
    let expected = Complex::new(33.0, -21.0);
    let optimized = compile_external_evaluators(false, true)?.evaluate_single(&args);
    let direct = compile_external_evaluators(true, true)?.evaluate_single(&args);

    assert_eq!(optimized, expected, "optimized translation is incorrect");
    assert_eq!(direct, expected);
    Ok(())
}

#[test]
fn test_external_evaluators_real_simd() -> Result<()> {
    let args: Vec<f64> = (0..100).map(|x| f64::from(x)).collect();
    let expected: Vec<f64> = (0..100)
        .step_by(2)
        .map(|x| f64::from(7 * x + x + 1))
        .collect();

    let mut optimized = vec![0.0; 50];
    let mut direct = vec![0.0; 50];

    compile_external_evaluators(false, false)?.evaluate_matrix(&args, &mut optimized, 50);
    compile_external_evaluators(true, false)?.evaluate_matrix(&args, &mut direct, 50);

    assert_eq!(
        optimized[10], expected[10],
        "optimized translation is incorrect"
    );
    assert_eq!(direct[10], expected[10]);
    Ok(())
}

#[test]
fn test_external_evaluators_complex_simd() -> Result<()> {
    let args: Vec<Complex<f64>> = (0..100).map(|x| Complex::new(x as f64, x as f64)).collect();
    let mut expected = vec![Complex::new(0.0, 0.0); 50];
    let mut optimized = vec![Complex::new(0.0, 0.0); 50];
    let mut direct = vec![Complex::new(0.0, 0.0); 50];

    for i in 0..50 {
        expected[i] = Complex::new(7.0, -2.0) * args[2 * i] + args[2 * i + 1];
    }

    compile_external_evaluators(false, true)?.evaluate_matrix(&args, &mut optimized, 50);
    compile_external_evaluators(true, true)?.evaluate_matrix(&args, &mut direct, 50);

    assert_eq!(
        optimized[10], expected[10],
        "optimized translation is incorrect"
    );
    assert_eq!(direct[10], expected[10]);
    Ok(())
}

#[test]
fn test_factorial() -> Result<()> {
    let mut config = Config::default();
    config.set_complex(false);

    let mut df = Defuns::new();
    df.add_self("f");
    config.set_defuns(df);

    let mut ev = Translator::new(config);
    ev.set_num_params(1);
    ev.append_constant(Complex::new(-1.0, 0.0))?;
    ev.append_constant(Complex::new(1.0, 0.0))?;

    ev.append_if_else(&Slot::Param(0), 5)?;
    ev.append_add(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(0)], 0)?;
    ev.append_fun(&Slot::Temp(1), "f", &[Slot::Temp(0)], false)?;
    ev.append_mul(&Slot::Temp(2), &[Slot::Param(0), Slot::Temp(1)], 0)?;
    ev.append_goto(7)?;
    ev.append_label(5)?;
    ev.append_assign(&Slot::Temp(3), &Slot::Const(1))?;
    ev.append_label(7)?;
    ev.append_join(
        &Slot::Out(0),
        &Slot::Param(0),
        &Slot::Temp(2),
        &Slot::Temp(3),
    )?;

    let app = ev.compile()?.seal()?;

    let args = [6.0];
    let mut outs = [0.0];

    app.evaluate(&args, &mut outs);

    assert!(outs[0] == 720.0);
    Ok(())
}

#[test]
fn test_args() -> Result<()> {
    let num_args: usize = 1000;
    let mut config = Config::default();
    config.set_complex(false);
    let mut ev = Translator::new(config);

    ev.set_num_params(num_args);
    ev.append_constant(Complex::new(0.0, 0.0))?;
    ev.append_assign(&Slot::Temp(0), &Slot::Const(0))?;

    for i in 0..num_args {
        ev.append_add(&Slot::Temp(i + 1), &[Slot::Temp(i), Slot::Param(i)], 0)?;
    }

    ev.append_assign(&Slot::Out(0), &Slot::Temp(num_args))?;
    let f = ev.compile()?.seal()?;

    /* ***************************************************** */

    let mut df = Defuns::new();
    df.add_applet("f", f);

    let mut config = Config::default();
    config.set_complex(false);
    config.set_direct(false);
    config.set_defuns(df);
    config.set_num_args(num_args as u32);

    let mut ev = Translator::new(config);

    ev.set_num_params(0);
    let mut args = Vec::new();

    for i in 0..num_args {
        ev.append_constant(Complex::new((i * i) as f64, 0.0))?;
        ev.append_assign(&Slot::Temp(i), &Slot::Const(i))?;
        args.push(Slot::Temp(i));
    }

    ev.append_fun(&Slot::Out(0), "f", &args, false)?;
    let app = ev.compile()?.seal()?;

    let mut outs = [0.0];
    let expected = (num_args * (num_args - 1) * (2 * num_args - 1) / 6) as f64;

    app.evaluate(&[], &mut outs);

    assert_eq!(outs[0], expected);
    Ok(())
}
