#![cfg(target_arch = "aarch64")]

use anyhow::Result;
use symjit::{Application, Compiled, Compiler, CompilerType, Complex, Config, Expr, Storage};
use wide::f64x2;

fn expression() -> (Vec<Expr>, Expr) {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let z = Expr::var("z");
    let mut out = &x * &y;

    for _ in 0..32 {
        out = &(&out * &x) + &(&y * &z);
        out = &out / &(&z + &Expr::from(2));
    }

    (vec![x, y, z], out)
}

fn config(fast_complex: bool, use_simd: bool, compress: bool) -> Config {
    let mut config = Config::new(CompilerType::Arm, 0).unwrap();
    config.set_complex(true);
    config.set_fast_complex(fast_complex);
    config.set_simd(use_simd);
    config.set_compact(true);
    config.set_compress(compress);
    config.set_opt_level(2);
    config
}

fn compile(fast_complex: bool, use_simd: bool, compress: bool) -> Result<symjit::Application> {
    let (params, output) = expression();
    // `Application::evaluate` passes `args` through the parameter pointer.
    // State-backed applications instead require the state/memory execution API.
    let application = Compiler::with_config(config(fast_complex, use_simd, compress))
        .compile_params(&[], &[output], &params)?;
    assert_eq!(application.count_states, 0);
    assert_eq!(application.count_params, 6);
    assert_eq!(application.count_obs, 2);
    Ok(application)
}

#[test]
fn compressed_scalar_and_fast_complex_match_expanded() -> Result<()> {
    let args = [
        Complex::new(1.25, -0.75),
        Complex::new(-0.5, 0.25),
        Complex::new(0.75, 1.5),
    ];

    for fast_complex in [false, true] {
        let expanded = compile(fast_complex, false, false)?;
        let compressed = compile(fast_complex, false, true)?;
        let mut expanded_out = [Complex::default()];
        let mut compressed_out = [Complex::default()];
        expanded.evaluate(&args, &mut expanded_out);
        compressed.evaluate(&args, &mut compressed_out);
        assert_eq!(expanded_out, compressed_out);
        assert_ne!(expanded_out, [Complex::default()]);
        assert!(
            compressed.compiled.as_ref().unwrap().dumps().len()
                < expanded.compiled.as_ref().unwrap().dumps().len()
        );
    }

    Ok(())
}

#[test]
fn compressed_vector_matches_expanded() -> Result<()> {
    let args = [
        Complex::new(f64x2::new([1.25, -0.25]), f64x2::new([-0.75, 0.5])),
        Complex::new(f64x2::new([-0.5, 0.75]), f64x2::new([0.25, -1.0])),
        Complex::new(f64x2::new([0.75, 1.0]), f64x2::new([1.5, -0.5])),
    ];
    let mut expanded = compile(false, true, false)?;
    let mut compressed = compile(false, true, true)?;
    expanded.prepare_simd();
    compressed.prepare_simd();
    let mut expanded_out = [Complex::default()];
    let mut compressed_out = [Complex::default()];
    expanded.evaluate(&args, &mut expanded_out);
    compressed.evaluate(&args, &mut compressed_out);
    assert_eq!(expanded_out, compressed_out);
    assert_ne!(expanded_out, [Complex::default()]);
    assert!(
        compressed.compiled_simd.as_ref().unwrap().dumps().len()
            < expanded.compiled_simd.as_ref().unwrap().dumps().len()
    );

    Ok(())
}

#[test]
fn compressed_emission_is_deterministic() -> Result<()> {
    let first = compile(false, false, true)?;
    let second = compile(false, false, true)?;
    assert_eq!(
        first.compiled.as_ref().unwrap().dumps(),
        second.compiled.as_ref().unwrap().dumps()
    );

    Ok(())
}

#[test]
fn compressed_storage_v3_roundtrip_is_exact_and_stable() -> Result<()> {
    let mut expanded = compile(true, true, false)?;
    let mut compressed = compile(true, true, true)?;
    expanded.prepare_simd();
    compressed.prepare_simd();

    let mut stored = Vec::new();
    compressed.save(&mut stored)?;
    let mut stored_slice = stored.as_slice();
    let loaded = Application::load(&mut stored_slice, &config(true, true, true))?;
    assert!(stored_slice.is_empty());

    let scalar_args = [
        Complex::new(1.25, -0.75),
        Complex::new(-0.5, 0.25),
        Complex::new(0.75, 1.5),
    ];
    let mut expanded_scalar = [Complex::default()];
    let mut loaded_scalar = [Complex::default()];
    expanded.evaluate(&scalar_args, &mut expanded_scalar);
    loaded.evaluate(&scalar_args, &mut loaded_scalar);
    assert_eq!(expanded_scalar, loaded_scalar);
    assert_ne!(expanded_scalar, [Complex::default()]);

    let vector_args = [
        Complex::new(f64x2::new([1.25, -0.25]), f64x2::new([-0.75, 0.5])),
        Complex::new(f64x2::new([-0.5, 0.75]), f64x2::new([0.25, -1.0])),
        Complex::new(f64x2::new([0.75, 1.0]), f64x2::new([1.5, -0.5])),
    ];
    let mut expanded_vector = [Complex::default()];
    let mut loaded_vector = [Complex::default()];
    expanded.evaluate(&vector_args, &mut expanded_vector);
    loaded.evaluate(&vector_args, &mut loaded_vector);
    assert_eq!(expanded_vector, loaded_vector);
    assert_ne!(expanded_vector, [Complex::default()]);

    assert_eq!(
        compressed.compiled.as_ref().unwrap().dumps(),
        loaded.compiled.as_ref().unwrap().dumps()
    );
    assert_eq!(
        compressed.compiled_simd.as_ref().unwrap().dumps(),
        loaded.compiled_simd.as_ref().unwrap().dumps()
    );

    let mut stored_again = Vec::new();
    loaded.save(&mut stored_again)?;
    assert_eq!(stored, stored_again);

    Ok(())
}
