//! `Application::load_for`: loading saved applications for another target (used by sjc).
mod common;
use common::base_config;

use symjit::{var, Application, Compiler, CompilerType, Config, Expr, Storage};

/// sum of sin(k x + y) exp(-z / k), k = 1..39: many live values
fn big() -> (Vec<Expr>, Vec<Expr>) {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let mut s = Expr::from(0.0);
    for k in 1..40 {
        let k = Expr::from(k as f64);
        s = &s + &(&(&(&k * &x) + &y).sin() * &(-&(&z / &k)).exp());
    }
    (vec![x, y, z], vec![s])
}

fn saved(ty: CompilerType, opt_level: u8) -> Vec<u8> {
    let mut config = base_config();
    config.ty = ty;
    config.set_opt_level(opt_level);
    let (states, obs) = big();
    let app = Compiler::with_config(config)
        .compile(&states, &obs)
        .unwrap();
    let mut bytes = Vec::new();
    app.save(&mut bytes).unwrap();
    bytes
}

fn load_for(bytes: &[u8], target: Option<CompilerType>) -> anyhow::Result<Application> {
    Application::load_for(&mut &bytes[..], &base_config(), target)
}

#[test]
fn retargets_follow_the_register_contract() {
    // opt level 3 gives ARM64 more registers than x86-64 has
    let arm3 = saved(CompilerType::Arm, 3);
    let app = load_for(&arm3, None).unwrap();
    assert!(app.config.is_arm64());
    assert!(app.measure("registers") > 14);

    let err = load_for(&arm3, Some(CompilerType::Amd))
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains("registers") && err.contains("x86-64"),
        "{}",
        err
    );
    assert!(load_for(&arm3, Some(CompilerType::ByteCode)).is_err());
    // WebAssembly keeps registers in locals
    #[cfg(feature = "wasm")]
    assert!(load_for(&arm3, Some(CompilerType::Wasm))
        .unwrap()
        .wasm
        .is_some());

    // opt level 2 is portable: the retargeted x86-64 kernel agrees with a direct compilation
    // (up to the order of operations, which follows the saved MIR)
    let arm2 = saved(CompilerType::Arm, 2);
    let mut x86 = load_for(&arm2, Some(CompilerType::Amd)).unwrap();
    assert!(x86.config.is_amd64());
    #[cfg(target_arch = "x86_64")]
    {
        let mut config = base_config();
        config.set_opt_level(2);
        let (states, obs) = big();
        let mut direct = Compiler::with_config(config)
            .compile(&states, &obs)
            .unwrap();
        for args in [[0.5, -1.25, 0.75], [1.5, 0.25, -0.5]] {
            let (a, b) = (x86.call(&args), direct.call(&args));
            assert!((a[0] - b[0]).abs() <= 1e-12 * b[0].abs(), "{:?} {:?}", a, b);
        }
    }
    let _ = &mut x86;

    // `load` itself is unchanged
    let app = Application::load(
        &mut &arm2[..],
        &Config::new(CompilerType::Native, 0).unwrap(),
    );
    assert!(app.is_ok());
}
