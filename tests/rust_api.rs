//! This compiles as an external Rust consumer, independently of the C ABI.
use symjit::{Complex, Composer, Config, Slot, Translator};

#[test]
fn rust_library_exposes_native_composer_and_complex_execution() {
    let mut config = Config::default();
    config.set_complex(true);
    config.set_opt_level(2);
    let mut translator = Translator::new(config);
    translator.set_num_params(1);
    let constant = translator.append_constant(Complex::new(2., -3.)).unwrap();
    translator
        .append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Const(constant)], 0)
        .unwrap();
    let compiled = translator.compile().unwrap();
    let mut result = [Complex::new(0., 0.)];
    compiled.evaluate(&[Complex::new(4., 5.)], &mut result);
    assert_eq!(result, [Complex::new(23., -2.)]);
    assert!(compiled.measure("version") > 0);
}
