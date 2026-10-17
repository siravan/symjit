mod common;

use common::base_config;
use symjit::{Compiler, Complex, Composer, Slot, Translator};

#[test]
fn direct_division_preserves_finite_ratios_and_signed_zero() -> anyhow::Result<()> {
    for direct in [false, true] {
        let mut config = base_config();
        config.set_direct(direct);
        let mut translator = Translator::new(config);
        translator.set_num_params(2);
        translator.append_div(&Slot::Out(0), &Slot::Param(0), &Slot::Param(1), true, true)?;
        let app = translator.compile()?.seal()?;
        for (numerator, denominator) in [
            (1e-320_f64, 1e-320_f64),
            (1e-200, 1e-320),
            (-1e-320, 1e-320),
            (1e308, 1e308),
            (-0., 2.),
            (0., -2.),
            (1., 0.),
            (0., 0.),
        ] {
            let expected = numerator / denominator;
            let actual = app.evaluate_single::<f64>(&[numerator, denominator]);
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "direct={direct}, {numerator}/{denominator}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn division_preserves_operand_order_reality_and_constant_slots() -> anyhow::Result<()> {
    for direct in [false, true] {
        for fast_complex in [false, true] {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(true);
            config.set_fast_complex(fast_complex);
            let mut translator = Translator::new(config);
            translator.set_num_params(2);
            let constant = translator.append_constant(Complex::new(4., -1.))?;
            translator.append_div(
                &Slot::Temp(0),
                &Slot::Param(0),
                &Slot::Param(1),
                false,
                true,
            )?;
            // Read both operands before overwriting the numerator temporary.
            translator.append_div(
                &Slot::Temp(0),
                &Slot::Temp(0),
                &Slot::Const(constant),
                false,
                false,
            )?;
            translator.append_assign(&Slot::Out(0), &Slot::Temp(0))?;
            translator.append_div(&Slot::Out(1), &Slot::Param(1), &Slot::Param(0), true, false)?;
            let app = translator.compile()?.seal()?;
            let numerator = Complex::new(2., -3.);
            let denominator = Complex::new(5., 0.);
            let mut output = [Complex::default(); 2];
            app.evaluate(&[numerator, denominator], &mut output);
            for (actual, expected) in output.into_iter().zip([
                numerator / denominator / Complex::new(4., -1.),
                denominator / numerator,
            ]) {
                assert!(
                    (actual - expected).norm() < 8. * f64::EPSILON * expected.norm(),
                    "direct={direct}, fast_complex={fast_complex}: {actual} != {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn division_instruction_round_trips_through_text_and_json() -> anyhow::Result<()> {
    let sources = [
        "([('div', ('out', 0), ('param', 0), ('param', 1), True, True)],0,[])",
        r#"[[{"Div":[{"Out":0},{"Param":0},{"Param":1},true,true]}],0,[]]"#,
    ];
    for direct in [false, true] {
        for source in sources {
            let mut config = base_config();
            config.set_direct(direct);
            let app = Compiler::with_config(config)
                .translate(source.into(), 2)?
                .seal()?;
            assert_eq!(app.evaluate_single::<f64>(&[1e-320, 1e-320]), 1.);
        }
    }
    Ok(())
}

#[test]
fn select_instruction_accepts_both_wire_formats_without_inactive_nan_leaks() -> anyhow::Result<()> {
    let sources = [
        "([('select', ('out', 0), ('param', 0), ('param', 1), ('param', 2))],0,[])",
        r#"[[{"Select":[{"Out":0},{"Param":0},{"Param":1},{"Param":2}]}],0,[]]"#,
    ];
    for direct in [false, true] {
        for source in sources {
            let mut config = base_config();
            config.set_direct(direct);
            let app = Compiler::with_config(config)
                .translate(source.into(), 3)?
                .seal()?;
            assert_eq!(app.evaluate_single::<f64>(&[0., f64::NAN, 7.]), 7.);
            assert_eq!(app.evaluate_single::<f64>(&[1., 11., f64::NAN]), 11.);
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx") {
                let mut output = [wide::f64x4::ZERO];
                app.evaluate(
                    &[
                        wide::f64x4::from([0., 1., 0., -1.]),
                        wide::f64x4::from([f64::NAN, 11., f64::NAN, 13.]),
                        wide::f64x4::from([7., f64::NAN, 9., f64::NAN]),
                    ],
                    &mut output,
                );
                assert_eq!(output[0].to_array(), [7., 11., 9., 13.]);
            }
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "stable_math")]
fn complex_over_real_division_preserves_subnormal_ratios() -> anyhow::Result<()> {
    for direct in [false, true] {
        for (fast_complex, denominator_is_real) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let mut config = base_config();
            config.set_direct(direct);
            config.set_complex(true);
            config.set_fast_complex(fast_complex);
            let mut translator = Translator::new(config);
            translator.set_num_params(2);
            translator.append_div(
                &Slot::Out(0),
                &Slot::Param(0),
                &Slot::Param(1),
                false,
                denominator_is_real,
            )?;
            let app = translator.compile()?.seal()?;
            for scale in [1e-320, 1e-200, 1e200, 1e308] {
                let numerator = Complex::new(scale, -scale);
                let mut output = [Complex::default()];
                app.evaluate(&[numerator, Complex::new(scale, 0.)], &mut output);
                assert_eq!(
                    output[0],
                    Complex::new(1., -1.),
                    "direct={direct}, fast_complex={fast_complex}, real_denominator={denominator_is_real}, scale={scale}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[cfg(target_arch = "x86_64")]
fn division_preserves_finite_ratios_in_simd_lanes() -> anyhow::Result<()> {
    if !is_x86_feature_detected!("avx") {
        return Ok(());
    }
    for direct in [false, true] {
        let mut config = base_config();
        config.set_direct(direct);
        let mut translator = Translator::new(config);
        translator.set_num_params(2);
        translator.append_div(&Slot::Out(0), &Slot::Param(0), &Slot::Param(1), true, true)?;
        let app = translator.compile()?.seal()?;
        let numerator = [1e-320, 1e308, -3., -0.];
        let denominator = [1e-320, 1e308, 2., 2.];
        let mut output = [wide::f64x4::ZERO];
        app.evaluate(
            &[wide::f64x4::from(numerator), wide::f64x4::from(denominator)],
            &mut output,
        );
        for ((actual, numerator), denominator) in output[0]
            .to_array()
            .into_iter()
            .zip(numerator)
            .zip(denominator)
        {
            assert_eq!(
                actual.to_bits(),
                (numerator / denominator).to_bits(),
                "direct={direct}"
            );
        }
    }
    Ok(())
}
