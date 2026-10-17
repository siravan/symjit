mod common;

use common::base_config;
#[cfg(feature = "stable_math")]
use symjit::{Application, Compiler, Complex, Storage};
use symjit::{Composer, Slot, Translator};

#[test]
#[cfg(feature = "stable_math")]
fn complex_negative_integer_powers_preserve_finite_components_and_reload() -> anyhow::Result<()> {
    // Analytic Cartesian powers, including cases whose positive power would
    // overflow even though the inverse power is a representable subnormal.
    let cases: &[(i64, Complex<f64>, Complex<f64>)] = &[
        (-1, Complex::new(1e200, 0.), Complex::new(1e-200, 0.)),
        (-1, Complex::new(1e-200, 0.), Complex::new(1e200, 0.)),
        (-1, Complex::new(0., 1e200), Complex::new(0., -1e-200)),
        (
            -1,
            Complex::new(1e200, 1e200),
            Complex::new(5e-201, -5e-201),
        ),
        (-2, Complex::new(1e155, 0.), Complex::new(1e-310, 0.)),
        (-2, Complex::new(1e-100, 0.), Complex::new(1e200, 0.)),
        (-2, Complex::new(1e155, 1e155), Complex::new(0., -5e-311)),
        (-3, Complex::new(1e103, 0.), Complex::new(1e-309, 0.)),
        (-3, Complex::new(1e-100, 0.), Complex::new(1e300, 0.)),
        (
            -3,
            Complex::new(1e103, 1e103),
            Complex::new(-2.5e-310, -2.5e-310),
        ),
        (-4, Complex::new(1e77, 0.), Complex::new(1e-308, 0.)),
        (-4, Complex::new(1e-75, 0.), Complex::new(1e300, 0.)),
        (-4, Complex::new(1e77, 1e77), Complex::new(-2.5e-309, 0.)),
    ];
    let mut failures = Vec::new();
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(true);
            config.set_fast_complex(fast_complex);
            for &(power, input, expected) in cases {
                let mut translator = Translator::new(config.clone());
                translator.set_num_params(1);
                translator.append_assign(&Slot::Temp(0), &Slot::Param(0))?;
                // In-place destination exercises the ordinary call owner's
                // read-before-write and register allocation boundaries.
                translator.append_pow(&Slot::Temp(0), &Slot::Temp(0), power, false)?;
                translator.append_assign(&Slot::Out(0), &Slot::Temp(0))?;
                let app = translator.compile()?;
                let mut bytes = Vec::new();
                app.save(&mut bytes)?;
                let restored = Application::load(&mut bytes.as_slice(), &config)?;
                let json =
                    format!(r#"[[{{"Pow":[{{"Out":0}},{{"Param":0}},{power},false]}}],0,[]]"#);
                let from_json = Compiler::with_config(config.clone()).translate(json, 1)?;
                for (route, evaluator) in
                    [("fresh", app), ("binary", restored), ("json", from_json)]
                {
                    let actual = evaluator.evaluate_single(&[input]);
                    for (actual, expected) in [actual.re, actual.im]
                        .into_iter()
                        .zip([expected.re, expected.im])
                    {
                        let tolerance =
                            32. * f64::EPSILON * expected.abs() + 2. * f64::from_bits(1);
                        if !actual.is_finite() || (actual - expected).abs() > tolerance {
                            failures.push(format!("{route}, direct={direct}, fast={fast_complex}, {input}^{power}: {actual} != {expected}"));
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn integer_power_rejects_exponents_before_narrowing() -> anyhow::Result<()> {
    for direct in [false, true] {
        for complex in [false, true] {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(complex);
            for power in [i32::MIN as i64, i32::MAX as i64 + 1, i64::MIN, i64::MAX] {
                let mut translator = Translator::new(config.clone());
                translator.set_num_params(1);
                assert!(
                    translator.append_pow(&Slot::Out(0), &Slot::Param(0), power, !complex).is_err(),
                    "direct={direct}, complex={complex}, power={power} must be rejected before narrowing"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "stable_math")]
fn reciprocal_intrinsic_and_real_negative_powers_use_safe_ordering() -> anyhow::Result<()> {
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(true);
            config.set_fast_complex(fast_complex);
            let mut translator = Translator::new(config);
            translator.set_num_params(1);
            translator.append_fun(&Slot::Out(0), "recip", &[Slot::Param(0)], false)?;
            let app = translator.compile()?;
            let actual = app.evaluate_single(&[Complex::new(1e308, 1e308)]);
            assert!((actual.re - 5e-309).abs() <= 2. * f64::from_bits(1));
            assert!((actual.im + 5e-309).abs() <= 2. * f64::from_bits(1));
        }
        for (power, input, expected) in [
            (-2, 1e155_f64, 1e-310_f64),
            (-3, 1e103, 1e-309),
            (i32::MAX as i64, -1., -1.),
            (-(i32::MAX as i64), -1., -1.),
        ] {
            let mut config = base_config();
            config.set_direct(direct);
            let mut translator = Translator::new(config);
            translator.set_num_params(1);
            translator.append_pow(&Slot::Out(0), &Slot::Param(0), power, true)?;
            let app = translator.compile()?;
            let actual = app.evaluate_single(&[input]);
            let tolerance = 32. * f64::EPSILON * expected.abs() + 2. * f64::from_bits(1);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{input}^{power}: {actual} != {expected}"
            );
        }
    }
    Ok(())
}
