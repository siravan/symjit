//! `Application::dump(path, "json")` writes the MIR before complexification (for a real application,
//! its only MIR) and `dump(path, "json-complex")` the MIR of real operations: the complexified MIR of
//! a complex application, the same file as `json` for a real one.
//! `complexify` (which changes the MIR `save` writes) changes neither. Labels are written sorted, so
//! the same model gives the same file. The file names its format and version, holds the
//! application's counts, and writes each constant as its bit pattern (exact for NaN, ±Inf, -0.0).
//! `Application::sjb_mir_json` writes the same text from the saved application, without compiling it.
//! The file is the "symjit-application" format (mir/json.rs): `to_json` and `from_json` give back
//! the application, the same `.sjb` bytes.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use serde_json::Value;
use symjit::{Application, Composer, Config, Slot, Storage, Translator};

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
    v["mir"]["code"]
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
    // a real application's one MIR, not complexified, in both modes
    assert_eq!(dump(&mut app, "json-complex", "real").unwrap(), json);
    let v: Value = serde_json::from_str(&json)?;
    assert_eq!(v["mir"]["complexified"], false);
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
        v["mir"]["labels"].as_object().unwrap().len() >= 2,
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

/// the fields every file has: the format, its version, symjit's version, the program's counts
fn check_header(v: &Value, app: &Application, complexified: bool) {
    assert_eq!(v["format"], "symjit-application");
    assert_eq!(v["format_version"], 1);
    assert_eq!(v["symjit_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(v["mir"]["complexified"], complexified);
    let prog = &v["program"];
    assert_eq!(prog["count_states"], app.count_states);
    assert_eq!(prog["count_params"], app.count_params);
    assert_eq!(prog["count_obs"], app.count_obs);
    assert_eq!(prog["count_diffs"], app.count_diffs);
    assert_eq!(
        prog["config"]["options"]["complex"],
        app.config.is_complex()
    );
    assert!(prog["builder"]["consts"].is_array() && prog["builder"]["ft"].is_array());
    assert!(v["prepared"]["compiled"].is_boolean() && v["reals"].is_array());
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
    let consts: Vec<String> = v["mir"]["consts"]
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

/// the saved application's bytes
fn saved(app: &Application) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    app.save(&mut bytes)?;
    Ok(bytes)
}

fn sjb_json(bytes: &[u8], what: &str) -> Result<String> {
    Application::sjb_mir_json(&mut &bytes[..], &base_config(), what)
}

#[test]
fn sjb_mir_json_is_the_dump_of_the_loaded_application() -> Result<()> {
    for (tag, app) in [
        ("sr", compile_real()?),
        ("sc", compile_complex()?),
        ("ss", compile_special()?),
    ] {
        let bytes = saved(&app)?;
        let mut loaded = Application::load(&mut &bytes[..], &base_config())?;
        for what in ["json", "json-complex"] {
            assert_eq!(
                sjb_json(&bytes, what)?,
                dump(&mut loaded, what, tag).unwrap(),
                "{tag} {what}"
            );
        }
    }
    assert!(sjb_json(b"not a saved application", "json").is_err());
    assert!(sjb_json(&saved(&compile_real()?)?, "scalar").is_err());
    Ok(())
}

/// A saved complexified application (complexify, then save) cannot be loaded, but its MIR can be
/// read: "json-complex" is the complexified MIR, and there is no "json".
#[test]
fn sjb_mir_json_of_a_complexified_application() -> Result<()> {
    let mut app = compile_complex()?;
    let complex = dump(&mut app, "json-complex", "cx").unwrap();
    app.complexify()?;
    let bytes = saved(&app)?;
    assert!(Application::load(&mut &bytes[..], &base_config()).is_err());
    assert!(sjb_json(&bytes, "json").is_err());
    let read: Value = serde_json::from_str(&sjb_json(&bytes, "json-complex")?)?;
    let dumped: Value = serde_json::from_str(&complex)?;
    assert_eq!(read["mir"]["complexified"], true);
    for key in ["program", "mir", "reals"] {
        assert_eq!(read[key], dumped[key], "{key}");
    }
    Ok(())
}

/// out = 1 / (p0 + c) with stable_math: the complexified MIR appends its own constants
fn compile_stable_math() -> Result<Application> {
    let mut config = base_config();
    config.set_complex(true);
    config.set_stable_math(true);
    let mut t = Translator::new(config);
    t.set_num_params(1);
    let c = t.append_constant(Complex::new(0.5, -1.5))?;
    t.append_add(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(c)], 0)?;
    t.append_div(&Slot::Out(0), &Slot::Const(c), &Slot::Temp(0), false, false)?;
    t.compile()
}

/// `to_json`, then `from_json`, gives back the application: the same `.sjb` bytes, the same JSON
#[test]
fn json_round_trip_is_the_saved_application() -> Result<()> {
    let mut simd = compile_real()?;
    simd.prepare_simd();
    // the cases cover what they are for: a prepared SIMD kernel, and (stable_math) a builder whose
    // constants are not the saved MIR's (the complexified MIR appends its own)
    // (RISC-V has only the scalar kernel: prepare_simd prepares nothing there)
    let v: Value = serde_json::from_str(&simd.to_json()?)?;
    assert_eq!(v["prepared"]["simd"], cfg!(not(target_arch = "riscv64")));
    let v: Value = serde_json::from_str(&compile_stable_math()?.to_json()?)?;
    assert_ne!(v["program"]["builder"]["consts"], v["mir"]["consts"]);
    for (tag, app) in [
        ("real", compile_real()?),
        ("complex", compile_complex()?),
        ("special", compile_special()?),
        ("stable_math", compile_stable_math()?),
        ("simd", simd),
    ] {
        let json = app.to_json()?;
        let loaded = Application::from_json(&json, &base_config())?;
        assert_eq!(
            saved(&loaded)?,
            saved(&app)?,
            "{tag}: the .sjb bytes differ"
        );
        assert_eq!(loaded.to_json()?, json, "{tag}: the JSON differs");
        // the file `dump(.., "json")` writes is the same (the MIR before complexification)
        let mut again = Application::from_json(&json, &base_config())?;
        assert_eq!(
            dump(&mut again, "json", tag).unwrap(),
            json,
            "{tag}: dump(json)"
        );
    }
    Ok(())
}

/// A loaded application computes what the original computes.
#[test]
fn json_round_trip_evaluates_the_same() -> Result<()> {
    let app = compile_real()?;
    let loaded = Application::from_json(&app.to_json()?, &base_config())?;
    for p in [[1.0, 2.0, 3.0, 4.0], [0.0, -1.5, 2.5, 7.0]] {
        assert_eq!(
            loaded.as_applet().evaluate_single::<f64>(&p).to_bits(),
            app.as_applet().evaluate_single::<f64>(&p).to_bits(),
            "{p:?}"
        );
    }
    Ok(())
}

/// What `from_json` refuses: a complexified MIR (as `load` does), another format or version, a
/// constant that is not a bit pattern, and a `complexified` that the code contradicts.
#[test]
fn from_json_refusals() -> Result<()> {
    let mut app = compile_complex()?;
    app.complexify()?;
    let complexified = app.to_json()?;
    assert!(Application::from_json(&complexified, &base_config()).is_err());

    let json = compile_real()?.to_json()?;
    let edit = |f: &dyn Fn(&mut Value)| -> String {
        let mut v: Value = serde_json::from_str(&json).unwrap();
        f(&mut v);
        v.to_string()
    };
    for (what, text) in [
        ("format", edit(&|v| v["format"] = "symjit-mir".into())),
        ("version", edit(&|v| v["format_version"] = 2.into())),
        (
            "constant",
            edit(&|v| v["program"]["builder"]["consts"] = vec!["1.5"].into()),
        ),
        (
            "complexified",
            edit(&|v| v["mir"]["complexified"] = true.into()),
        ),
        ("ty", edit(&|v| v["program"]["config"]["ty"] = "vax".into())),
        (
            "opt_level",
            edit(&|v| v["program"]["config"]["options"]["opt_level"] = 16.into()),
        ),
    ] {
        assert!(
            Application::from_json(&text, &base_config()).is_err(),
            "{what}"
        );
    }
    assert!(Application::from_json(&json, &base_config()).is_ok());
    Ok(())
}

/// The configuration's JSON is exact: every bit of `opt` (not the getters, which depend on the
/// machine), the compiler type, the stack limit and the argument count come back as they were. A
/// new option bit without a JSON field fails here.
#[test]
fn config_json_is_exact() -> Result<()> {
    let base = base_config();
    let mut opts: Vec<u32> = (0..32).map(|bit| 1u32 << bit).collect();
    opts.extend([0, u32::MAX, base.opt]);
    let mut x: u32 = 0x9e37_79b9;
    for _ in 0..64 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        opts.push(x);
    }

    for ty in [
        "native", "amd", "amd-avx", "amd-sse", "arm", "riscv", "bytecode", "debug",
    ] {
        for &opt in &opts {
            let mut c = Config::from_name(ty, opt)?;
            c.opt = opt; // from_name may adjust it; the JSON must keep any value
            c.stack = 12345;
            c.args = 7;
            let v = c.save_json()?;
            let back = Config::load_json(&v, &base, 1)?;
            assert_eq!(back.opt, c.opt, "{ty} {opt:#010x}");
            assert_eq!(back.ty, c.ty, "{ty} {opt:#010x}");
            assert_eq!((back.stack, back.args), (12345, 7), "{ty} {opt:#010x}");
            assert_eq!(back.save_json()?, v);
        }
    }
    Ok(())
}
