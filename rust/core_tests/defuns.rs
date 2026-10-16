use crate::defuns::Defuns;
use crate::{
    composer::Composer, config::Config, instruction::Slot, translator::Translator, CompileExt,
};
use num_complex::Complex;

#[test]
fn scalar_complex_callbacks_preserve_all_simd_lanes_and_tails() {
    for direct in [false, true] {
        for threads in [false, true] {
            let mut funcs = Defuns::new();
            funcs
                .add_sliced_func(
                    "probe",
                    Box::new(|args: &[Complex<f64>]| {
                        args[0] * args[0] + Complex::new(2.0, -3.0) * args[1]
                    }),
                )
                .unwrap();
            let mut config = Config::default();
            config.set_complex(true);
            config.set_direct(direct);
            config.set_opt_level(2);
            config.set_defuns(funcs);
            config
                .set_option("use_threads", if threads { "true" } else { "false" })
                .unwrap();
            let mut translator = Translator::new(config);
            translator.set_num_params(2);
            translator
                .append_fun(
                    &Slot::Out(0),
                    "probe",
                    &[Slot::Param(0), Slot::Param(1)],
                    false,
                )
                .unwrap();
            let compiled = translator.compile().unwrap();
            for rows in [1, 2, 3, 4, 5, 8, 17, 255, 256, 257] {
                let args = (0..rows)
                    .flat_map(|i| {
                        [
                            Complex::new(i as f64 + 0.25, 2.0 * i as f64 + 0.75),
                            Complex::new(-3.0 * i as f64 + 1.25, -4.0 * i as f64 - 0.5),
                        ]
                    })
                    .collect::<Vec<_>>();
                let expected = args
                    .chunks_exact(2)
                    .map(|p| p[0] * p[0] + Complex::new(2.0, -3.0) * p[1])
                    .collect::<Vec<_>>();
                let mut result = vec![Complex::new(0.0, 0.0); rows];
                compiled.evaluate_matrix(&args, &mut result, rows);
                assert_eq!(
                    result, expected,
                    "direct={direct} threads={threads} rows={rows}"
                );
            }
        }
    }
}
