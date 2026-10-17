mod common;

use common::base_config;
use symjit::{Compiler, Complex, Composer, Defuns, Expr, Slot, Translator};

#[test]
fn generated_complex_math_does_not_capture_user_callback_names() -> anyhow::Result<()> {
    for direct in [false, true] {
        let mut callbacks = Defuns::new();
        callbacks.add_sliced_func::<Complex<f64>>(
            "stable_divide",
            Box::new(|_| Complex::new(17., 2.)),
        )?;
        callbacks
            .add_sliced_func::<Complex<f64>>("stable_abs", Box::new(|_| Complex::new(19., 3.)))?;
        for name in [
            "__symjit_divide",
            "__symjit_abs",
            "__symjit_root",
            "cplx___symjit_root",
            "__symjit_future",
        ] {
            assert!(callbacks
                .add_sliced_func::<f64>(name, Box::new(|_| 0.))
                .is_err());
        }
        let mut config = base_config();
        config.set_complex(true);
        config.set_direct(direct);
        config.set_defuns(callbacks);
        let mut translator = Translator::new(config);
        translator.set_num_params(2);
        translator.append_div(
            &Slot::Out(0),
            &Slot::Param(0),
            &Slot::Param(1),
            false,
            false,
        )?;
        translator.append_fun(&Slot::Out(1), "abs", &[Slot::Param(0)], false)?;
        translator.append_fun(
            &Slot::Out(2),
            "stable_divide",
            &[Slot::Param(0), Slot::Param(1)],
            false,
        )?;
        translator.append_fun(&Slot::Out(3), "stable_abs", &[Slot::Param(0)], false)?;
        let app = translator.compile()?.seal()?;
        let mut output = [Complex::default(); 4];
        app.evaluate(&[Complex::new(3., 4.), Complex::new(1., -2.)], &mut output);
        assert_eq!(
            output,
            [
                Complex::new(-1., 2.),
                Complex::new(5., 0.),
                Complex::new(17., 2.),
                Complex::new(19., 3.)
            ]
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "stable_math")]
fn complex_root_preserves_range_and_the_small_companion() -> anyhow::Result<()> {
    let tiny_root = f64::from_bits(1).sqrt() * 0.5f64.sqrt();
    for direct in [false, true] {
        for fast in [false, true] {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(true);
            config.set_fast_complex(fast);
            let mut translator = Translator::new(config);
            translator.set_num_params(1);
            translator.append_fun(&Slot::Out(0), "root", &[Slot::Param(0)], false)?;
            let app = translator.compile()?.seal()?;
            for (input, expected) in [
                (Complex::new(3., 4.), Complex::new(2., 1.)),
                (
                    Complex::new(1e-150, 1e-150),
                    Complex::new(
                        1e-75 * ((2.0f64.sqrt() + 1.) / 2.).sqrt(),
                        1e-75 * ((2.0f64.sqrt() - 1.) / 2.).sqrt(),
                    ),
                ),
                (
                    Complex::new(1e-160, 1e-160),
                    Complex::new(
                        1e-80 * ((2.0f64.sqrt() + 1.) / 2.).sqrt(),
                        1e-80 * ((2.0f64.sqrt() - 1.) / 2.).sqrt(),
                    ),
                ),
                (Complex::new(1e-200, 0.), Complex::new(1e-100, 0.)),
                (
                    Complex::new(0., f64::from_bits(1)),
                    Complex::new(tiny_root, tiny_root),
                ),
                (Complex::new(1e308, 1e-100), Complex::new(1e154, 5e-255)),
                (Complex::new(-1e308, 1e-100), Complex::new(5e-255, 1e154)),
                (Complex::new(-1e308, -1e-100), Complex::new(5e-255, -1e154)),
                (
                    Complex::new(1e308, 1e308),
                    Complex::new(
                        1e154 * ((2.0f64.sqrt() + 1.) / 2.).sqrt(),
                        1e154 * ((2.0f64.sqrt() - 1.) / 2.).sqrt(),
                    ),
                ),
                (Complex::new(-4., -0.), Complex::new(0., -2.)),
            ] {
                let mut output = [Complex::default()];
                app.evaluate(&[input], &mut output);
                for (actual, expected) in [output[0].re, output[0].im]
                    .into_iter()
                    .zip([expected.re, expected.im])
                {
                    assert!(
                        (actual - expected).abs() <= 8. * f64::EPSILON * expected.abs(),
                        "direct={direct}, fast={fast}, sqrt({input}): {actual} != {expected}"
                    );
                }
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx") {
                assert!(app.compiled_simd.is_some());
                let mut output = [Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO)];
                app.evaluate(
                    &[Complex::new(
                        wide::f64x4::from([1e-200, 1e308, -1e308, 0.]),
                        wide::f64x4::from([0., 1e-100, 1e-100, f64::from_bits(1)]),
                    )],
                    &mut output,
                );
                for (actual, expected) in output[0]
                    .re
                    .to_array()
                    .into_iter()
                    .chain(output[0].im.to_array())
                    .zip([
                        1e-100, 1e154, 5e-255, tiny_root, 0., 5e-255, 1e154, tiny_root,
                    ])
                {
                    assert!(
                        (actual - expected).abs() <= 8. * f64::EPSILON * expected.abs(),
                        "SIMD direct={direct}, fast={fast}: {actual} != {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn complex_range_calls_preserve_surrounding_expression_values() -> anyhow::Result<()> {
    for fast_complex in [false, true] {
        let mut config = base_config();
        config.set_complex(true);
        config.set_fast_complex(fast_complex);
        let x = Expr::var("x");
        let y = Expr::var("y");
        let expression = &(&(&x * &y) + &(&x + &y)) + &(&(&x / &y) + &y.abs());
        let expression_app = Compiler::with_config(config.clone())
            .compile_params(&[], &[expression], &[x, y])?
            .seal()?;
        for direct in [false, true] {
            config.set_direct(direct);
            let mut translator = Translator::new(config.clone());
            translator.set_num_params(2);
            translator.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Param(1)], 0)?;
            translator.append_add(&Slot::Temp(1), &[Slot::Param(0), Slot::Param(1)], 0)?;
            translator.append_div(
                &Slot::Temp(2),
                &Slot::Param(0),
                &Slot::Param(1),
                false,
                false,
            )?;
            translator.append_fun(&Slot::Temp(3), "abs", &[Slot::Param(1)], false)?;
            translator.append_add(
                &Slot::Out(0),
                &[Slot::Temp(0), Slot::Temp(1), Slot::Temp(2), Slot::Temp(3)],
                0,
            )?;
            let app = translator.compile()?.seal()?;
            for evaluator in [&expression_app, &app] {
                let mut output = [Complex::default()];
                evaluator.evaluate(&[Complex::new(3., 4.), Complex::new(1., -2.)], &mut output);
                assert!((output[0].re - (14. + 5.0f64.sqrt())).abs() < 1e-14);
                assert_eq!(output[0].im, 2.);
            }
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "stable_math")]
fn complex_division_preserves_range_without_reality_hints() -> anyhow::Result<()> {
    let minimum = f64::from_bits(1);
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(true);
            config.set_fast_complex(fast_complex);
            let mut translator = Translator::new(config);
            translator.set_num_params(2);
            translator.append_assign(&Slot::Temp(0), &Slot::Param(0))?;
            translator.append_div(
                &Slot::Temp(0),
                &Slot::Temp(0),
                &Slot::Param(1),
                false,
                false,
            )?;
            translator.append_assign(&Slot::Out(0), &Slot::Temp(0))?;
            let constant = translator.append_constant(Complex::new(1., 0.))?;
            translator.append_div(
                &Slot::Out(1),
                &Slot::Temp(0),
                &Slot::Const(constant),
                false,
                false,
            )?;
            let app = translator.compile()?.seal()?;
            for (numerator, denominator, expected) in [
                (
                    Complex::new(1e-320, -1e-320),
                    Complex::new(1e-320, 0.),
                    Complex::new(1., -1.),
                ),
                (
                    Complex::new(3e-200, 4e-200),
                    Complex::new(1e-200, -2e-200),
                    Complex::new(-1., 2.),
                ),
                (
                    Complex::new(3e200, 4e200),
                    Complex::new(1e200, -2e200),
                    Complex::new(-1., 2.),
                ),
                (
                    Complex::new(1e308, 1e308),
                    Complex::new(1e308, 1e308),
                    Complex::new(1., 0.),
                ),
                (
                    Complex::new(1e308, minimum),
                    Complex::new(1., 0.),
                    Complex::new(1e308, minimum),
                ),
                (
                    Complex::new(1e308, 1e308),
                    Complex::new(1e-320, 1e-320),
                    Complex::new(f64::INFINITY, 0.),
                ),
                (
                    Complex::new(3., 4.),
                    Complex::new(1., -2.),
                    Complex::new(-1., 2.),
                ),
            ] {
                let mut outputs = [Complex::default(); 2];
                app.evaluate(&[numerator, denominator], &mut outputs);
                for actual in outputs {
                    for (actual, expected) in [actual.re, actual.im]
                        .into_iter()
                        .zip([expected.re, expected.im])
                    {
                        if expected == 0. || expected.is_infinite() || expected == minimum {
                            assert_eq!(actual, expected, "direct={direct}, fast_complex={fast_complex}: {numerator}/{denominator}");
                        } else {
                            assert!((actual - expected).abs() <= 8. * f64::EPSILON * expected.abs(), "direct={direct}, fast_complex={fast_complex}: {numerator}/{denominator}: {actual} != {expected}");
                        }
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx") {
                assert!(app.compiled_simd.is_some());
                let mut outputs = [Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO); 2];
                app.evaluate(
                    &[
                        Complex::new(
                            wide::f64x4::from([1e-320, 3e-200, 1e308, 1e308]),
                            wide::f64x4::from([-1e-320, 4e-200, 1e308, minimum]),
                        ),
                        Complex::new(
                            wide::f64x4::from([1e-320, 1e-200, 1e308, 1.]),
                            wide::f64x4::from([0., -2e-200, 1e308, 0.]),
                        ),
                    ],
                    &mut outputs,
                );
                for actual in outputs {
                    for (actual, expected) in
                        actual.re.to_array().into_iter().zip([1., -1., 1., 1e308])
                    {
                        assert!((actual - expected).abs() <= 8. * f64::EPSILON * expected.abs(), "SIMD direct={direct}, fast_complex={fast_complex}: {actual} != {expected}");
                    }
                    for (actual, expected) in
                        actual.im.to_array().into_iter().zip([-1., 2., 0., minimum])
                    {
                        if expected == 0. || expected == minimum {
                            assert_eq!(actual, expected);
                        } else {
                            assert!(
                                (actual - expected).abs() <= 8. * f64::EPSILON * expected.abs()
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "stable_math")]
fn complex_abs_preserves_finite_tiny_and_huge_magnitudes() -> anyhow::Result<()> {
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(true);
            config.set_fast_complex(fast_complex);
            let mut translator = Translator::new(config);
            translator.set_num_params(1);
            translator.append_fun(&Slot::Out(0), "abs", &[Slot::Param(0)], false)?;
            let app = translator.compile()?.seal()?;
            for value in [
                Complex::new(1e-320, 0.),
                Complex::new(3e-200, 4e-200),
                Complex::new(3., 4.),
                Complex::new(3e200, 4e200),
                Complex::new(1e308, 1e308),
                Complex::new(0., -0.),
            ] {
                let mut output = [Complex::default()];
                app.evaluate(&[value], &mut output);
                assert_eq!(output[0].im, 0.);
                assert_eq!(
                    output[0].re,
                    value.re.hypot(value.im),
                    "direct={direct}, fast_complex={fast_complex}: abs({value})"
                );
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx") {
                assert!(app.compiled_simd.is_some());
                let re = [1e-320, 3e-200, 3e200, 1e308];
                let im = [0., 4e-200, 4e200, 1e308];
                let mut output = [Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO)];
                app.evaluate(
                    &[Complex::new(wide::f64x4::from(re), wide::f64x4::from(im))],
                    &mut output,
                );
                assert_eq!(output[0].im.to_array(), [0.; 4]);
                for ((actual, re), im) in output[0].re.to_array().into_iter().zip(re).zip(im) {
                    assert_eq!(
                        actual,
                        re.hypot(im),
                        "SIMD direct={direct}, fast_complex={fast_complex}"
                    );
                }
            }
        }
    }
    Ok(())
}
