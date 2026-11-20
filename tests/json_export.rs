//! `Application::dump(path, "json")` writes the MIR before complexification (for a real application,
//! its only MIR) and `dump(path, "json-complex")` the complexified MIR of a complex application.
//! `complexify` (which changes the MIR `save` writes) changes neither. Labels are written sorted, so
//! the same model gives the same file. The file names its format and version, holds the
//! application's counts, and writes each constant as its bit pattern (exact for NaN, ±Inf, -0.0).
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use serde_json::Value;
use symjit::{Application, Composer, Slot, Translator};

/// out = exp(p0 × a) + p0
fn compile_complex() -> Result<Application> {
    let mut config = base_config();
    config.set_complex(true);
    let mut t = Translator::new(config);
    t.set_num_params(1);
    let ca = t.append_constant(Complex::new(0.2, 0.3))?;
    t.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(ca)], 0)?;
    t.append_fun(&Slot::Temp(1), "exp", &[Slot::Temp(0)], false)?;
    t.append_add(&Slot::Out(0), &[Slot::Temp(1), Slot::Param(0)], 0)?;
    t.compile()
}

/// two if/else blocks (four labels): out = (p0 ? p1 p2 : p3 p2) + (p1 ? p2 + p3 : p3)
fn compile_real() -> Result<Application> {
    let mut t = Translator::new(base_config());
    t.set_num_params(4);
    let (p0, p1, p2, p3) = (
        Slot::Param(0),
        Slot::Param(1),
        Slot::Param(2),
        Slot::Param(3),
    );
    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(0), &[p1, p2], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_mul(&Slot::Temp(1), &[p3, p2], 0)?;
    t.append_label(1)?;
    t.append_join(&Slot::Temp(2), &p0, &Slot::Temp(0), &Slot::Temp(1))?;
    t.append_if_else(&p1, 2)?;
    t.append_add(&Slot::Temp(3), &[p2, p3], 0)?;
    t.append_goto(3)?;
    t.append_label(2)?;
    t.append_assign(&Slot::Temp(4), &p3)?;
    t.append_label(3)?;
    t.append_join(&Slot::Temp(5), &p1, &Slot::Temp(3), &Slot::Temp(4))?;
    t.append_add(&Slot::Out(0), &[Slot::Temp(2), Slot::Temp(5)], 0)?;
    t.compile()
}

/// the text `dump(.., what)` writes, or None if it refuses
fn dump(app: &mut Application, what: &str, tag: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("symjit-json-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}-{what}.json"));
    let _ = std::fs::remove_file(&path);
    let ok = app.dump(path.to_str().unwrap(), what);
    let text = std::fs::read_to_string(&path).ok();
    let _ = std::fs::remove_file(&path);
    assert_eq!(ok, text.is_some(), "{what}: dump returned {ok}");
    text
}

fn has(json: &str, instruction: &str) -> bool {
    let v: Value = serde_json::from_str(json).unwrap();
    v["code"]
        .as_array()
        .unwrap()
        .iter()
        .any(|ins| ins.get(instruction).is_some())
}

#[test]
fn json_writes_the_real_mir_of_a_real_application() -> Result<()> {
    let mut app = compile_real()?;
    let json = dump(&mut app, "json", "real").expect("json refused for a real application");
    assert!(has(&json, "Bi"));
    assert!(dump(&mut app, "json-complex", "real").is_none());
    Ok(())
}

#[test]
fn json_and_json_complex_survive_complexify() -> Result<()> {
    let mut app = compile_complex()?;
    let real = dump(&mut app, "json", "c1").unwrap();
    let complex = dump(&mut app, "json-complex", "c1").unwrap();
    // the complex MIR has the model's own operations; the complexified MIR works on pairs
    assert!(has(&real, "Bi") && !has(&real, "ComplexBi") && !has(&real, "SaveComplex"));
    assert!(has(&complex, "ComplexBi") && has(&complex, "SaveComplex"));
    assert!(real != complex);

    app.complexify()?;
    assert_eq!(
        dump(&mut app, "json", "c2").unwrap(),
        real,
        "complexify changed `json`"
    );
    assert_eq!(dump(&mut app, "json-complex", "c2").unwrap(), complex);
    Ok(())
}

#[test]
fn json_labels_are_sorted() -> Result<()> {
    let first = dump(&mut compile_real()?, "json", "l1").unwrap();
    let v: Value = serde_json::from_str(&first)?;
    assert!(
        v["labels"].as_object().unwrap().len() >= 2,
        "the model has too few labels: {first}"
    );
    for i in 0..8 {
        let again = dump(&mut compile_real()?, "json", &format!("l{i}")).unwrap();
        assert_eq!(again, first, "the same model gave another file");
    }
    Ok(())
}

/// out = p0 × c2 + c1, with constants that JSON numbers cannot hold (NaN, ±Inf) or easily lose (-0.0)
fn compile_special() -> Result<Application> {
    let mut t = Translator::new(base_config());
    t.set_num_params(1);
    let c1 = t.append_constant(Complex::new(f64::NAN, f64::NEG_INFINITY))?;
    let c2 = t.append_constant(Complex::new(-0.0, f64::INFINITY))?;
    t.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(c2)], 1)?;
    t.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Const(c1)], 1)?;
    t.compile()
}

/// the header fields every file has: the format, its version, symjit's version, and the counts
fn check_header(v: &Value, app: &Application, complexified: bool) {
    assert_eq!(v["format"], "symjit-mir");
    assert_eq!(v["format_version"], 1);
    assert_eq!(v["symjit_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(v["complexified"], complexified);
    assert_eq!(v["count_states"], app.count_states);
    assert_eq!(v["count_params"], app.count_params);
    assert_eq!(v["count_obs"], app.count_obs);
    assert_eq!(v["count_diffs"], app.count_diffs);
    assert!(v["reals"].is_array());
}

#[test]
fn json_has_the_format_and_the_counts() -> Result<()> {
    let mut app = compile_real()?;
    let v: Value = serde_json::from_str(&dump(&mut app, "json", "h1").unwrap())?;
    check_header(&v, &app, false);
    assert_eq!(app.count_params, 4);

    let mut app = compile_complex()?;
    let v: Value = serde_json::from_str(&dump(&mut app, "json", "h2").unwrap())?;
    check_header(&v, &app, false);
    let v: Value = serde_json::from_str(&dump(&mut app, "json-complex", "h2").unwrap())?;
    check_header(&v, &app, true);
    Ok(())
}

#[test]
fn json_constants_are_bit_patterns() -> Result<()> {
    let mut app = compile_special()?;
    let v: Value = serde_json::from_str(&dump(&mut app, "json", "k").unwrap())?;
    let consts: Vec<String> = v["consts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            c.as_str()
                .expect("a constant that is not a string")
                .to_string()
        })
        .collect();
    for c in &consts {
        assert!(
            c.len() == 18 && c.starts_with("0x") && c[2..].chars().all(|h| h.is_ascii_hexdigit()),
            "constant {c}"
        );
    }
    let bits: Vec<u64> = consts
        .iter()
        .map(|c| u64::from_str_radix(&c[2..], 16).unwrap())
        .collect();
    for x in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY, -0.0] {
        assert!(
            bits.contains(&x.to_bits()),
            "{x} ({:#018x}) is not among {consts:?}",
            x.to_bits()
        );
    }
    Ok(())
}
