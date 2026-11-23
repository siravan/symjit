//! `MachineCode::labels`: the assembler's labels of a compiled kernel (offset, name), sorted by
//! offset, with `@data` where the data area after the code starts (the constants `_const_N_`, the
//! function table). The kernels are compiled for RISC-V, AArch64 and x86-64 on any host (the code is
//! never run), so the labels are checked everywhere.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use std::sync::atomic::{AtomicUsize, Ordering};
use symjit::backend::MachineCode;
use symjit::{Application, Compiled, Compiler, Composer, Config, Expr, Slot, Translator};

/// the test options with `ty` replaced, and complex numbers on or off
fn config_for(ty: &str, complex: bool) -> Config {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let _ = base_config();
    let toml = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/symjit.toml"))
        .unwrap()
        .replace("ty = \"native\"", &format!("ty = \"{}\"", ty));
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("symjit-labels-{}-{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("symjit.toml");
    std::fs::write(&path, toml).unwrap();
    let mut config = Config::from_name(path.to_str().unwrap(), 0).unwrap();
    config.set_complex(complex);
    config
}

/// out0 = p0 ? p0 p1 : p1 + 7.5;  out1 = exp(p0) + 1e6 p1
fn compile_real(config: Config) -> Result<Application> {
    let mut t = Translator::new(config);
    t.set_num_params(2);
    let (p0, p1) = (Slot::Param(0), Slot::Param(1));
    let c75 = t.append_constant(Complex::new(7.5, 0.0))?;
    let c1e6 = t.append_constant(Complex::new(1e6, 0.0))?;

    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(1), &[p0, p1], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_add(&Slot::Temp(2), &[p1, Slot::Const(c75)], 0)?;
    t.append_label(1)?;
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(1), &Slot::Temp(2))?;

    t.append_fun(&Slot::Temp(3), "exp", &[p0], false)?;
    t.append_mul(&Slot::Temp(4), &[p1, Slot::Const(c1e6)], 0)?;
    t.append_add(&Slot::Out(1), &[Slot::Temp(3), Slot::Temp(4)], 0)?;
    t.compile()
}

/// exp(x) + 7.5 y of the states x, y: one output and no parameters, so it has a fast kernel
fn compile_one(mut config: Config) -> Result<Application> {
    config.set_symbolica(false);
    let (x, y) = (Expr::var("x"), Expr::var("y"));
    let f = x.exp().add(&symjit::double(7.5).mul(&y));
    Compiler::with_config(config).compile(&[x, y], &[f])
}

/// out0 = exp(p0 (0.2 + 0.3i)) + p1^-3 + p199
fn compile_complex(config: Config) -> Result<Application> {
    let mut t = Translator::new(config);
    t.set_num_params(200);
    let ca = t.append_constant(Complex::new(0.2, 0.3))?;
    t.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(ca)], 0)?;
    t.append_fun(&Slot::Temp(1), "exp", &[Slot::Temp(0)], false)?;
    t.append_pow(&Slot::Temp(2), &Slot::Param(1), -3, false)?;
    t.append_add(
        &Slot::Out(0),
        &[Slot::Temp(1), Slot::Temp(2), Slot::Param(199)],
        0,
    )?;
    t.compile()
}

/// the offset of the label `name` (once)
fn offset_of(code: &MachineCode<f64>, name: &str) -> usize {
    let at: Vec<usize> = code
        .labels()
        .iter()
        .filter(|(_, n)| n == name)
        .map(|(o, _)| *o)
        .collect();
    assert_eq!(at.len(), 1, "{name}: {:?}", code.labels());
    at[0]
}

/// the double at `ofs` of the kernel's bytes
fn quad(bytes: &[u8], ofs: usize) -> f64 {
    f64::from_le_bytes(bytes[ofs..ofs + 8].try_into().unwrap())
}

/// the checks every kernel passes: sorted, inside the code, one `@data`, the constants after it;
/// returns the values of the constants (`_const_N_`, in order)
fn check(code: &MachineCode<f64>, what: &str) -> Vec<f64> {
    let labels = code.labels();
    let bytes = code.dumps();
    assert_eq!(bytes.len(), code.size, "{what}");
    assert!(!labels.is_empty(), "{what}: no labels");
    assert!(
        labels.windows(2).all(|w| w[0] <= w[1]),
        "{what}: not sorted: {labels:?}"
    );
    assert!(
        labels.iter().all(|(o, _)| *o <= code.size),
        "{what}: {labels:?}"
    );

    let data = offset_of(code, "@data");
    assert!(
        data > 0 && data < code.size,
        "{what}: @data at {data} of {}",
        code.size
    );

    let mut consts = Vec::new();
    for k in 0.. {
        let name = format!("_const_{k}_");
        if !labels.iter().any(|(_, n)| *n == name) {
            break;
        }
        let ofs = offset_of(code, &name);
        assert!(
            ofs >= data,
            "{what}: {name} at {ofs} before @data at {data}"
        );
        consts.push(quad(&bytes, ofs));
    }
    // the first constant opens the data area
    if !consts.is_empty() {
        assert_eq!(offset_of(code, "_const_0_"), data, "{what}: {labels:?}");
    }
    consts
}

fn check_real(ty: &str) {
    let mut app = compile_real(config_for(ty, false)).unwrap();
    let consts = check(app.compiled.as_ref().unwrap(), &format!("{ty} scalar"));
    for c in [7.5, 1e6] {
        assert!(consts.contains(&c), "{ty}: {c} not in {consts:?}");
    }
    // a function table entry for exp, after the constants
    let code = app.compiled.as_ref().unwrap();
    let exp: Vec<_> = code
        .labels()
        .iter()
        .filter(|(_, n)| n.contains("exp"))
        .collect();
    assert!(!exp.is_empty(), "{ty}: no exp entry: {:?}", code.labels());
    assert!(
        exp.iter().all(|(o, _)| *o > offset_of(code, "@data")),
        "{exp:?}"
    );

    if ty != "riscv" {
        app.prepare_simd();
        let consts = check(app.compiled_simd.as_ref().unwrap(), &format!("{ty} simd"));
        assert!(consts.contains(&7.5), "{ty} simd: {consts:?}");
    }
    // one output: the application has a fast kernel too
    let mut app = compile_one(config_for(ty, false)).unwrap();
    assert!(app.get_fast().is_some(), "{ty}: no fast kernel");
    let fast = app.compiled_fast.as_ref().unwrap();
    let consts = check(fast, &format!("{ty} fast"));
    assert!(consts.contains(&7.5), "{ty} fast: {consts:?}");
}

#[test]
fn riscv_labels() {
    check_real("riscv");
}

#[test]
fn arm_labels() {
    check_real("arm");
}

#[test]
fn x86_labels() {
    check_real("amd");
}

#[test]
fn complex_kernels_have_one_data_label() {
    // the complexified path replays MIR through the Complexifier, which must not add `@data`
    for ty in ["riscv", "arm", "amd"] {
        let mut config = config_for(ty, true);
        config.set_fast_complex(false);
        let app = compile_complex(config).unwrap();
        check(
            app.compiled.as_ref().unwrap(),
            &format!("{ty} complexified"),
        );

        let app = compile_complex(config_for(ty, true)).unwrap();
        check(app.compiled.as_ref().unwrap(), &format!("{ty} complex"));
    }
}

#[test]
fn loaded_applications_have_labels() {
    // a JSON application is compiled when it is read, so its kernel has the labels again
    let app = compile_real(config_for("arm", false)).unwrap();
    let text = app.to_json().unwrap();
    let loaded = Application::from_json(&text, &config_for("arm", false)).unwrap();
    let (a, b) = (
        app.compiled.as_ref().unwrap(),
        loaded.compiled.as_ref().unwrap(),
    );
    check(b, "loaded");
    // the same names and the same data area (the function table's order may differ from run to run)
    let names = |c: &MachineCode<f64>| {
        let mut n: Vec<String> = c.labels().iter().map(|(_, n)| n.clone()).collect();
        n.sort();
        n
    };
    assert_eq!(names(a), names(b));
    assert_eq!(offset_of(a, "@data"), offset_of(b, "@data"));
}
