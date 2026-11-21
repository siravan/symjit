//! `Application::dump(path, "scalar-asm")` (also `"simd-asm"`, `"fast-asm"`) writes a kernel as
//! assembly text (rust/disasm). The kernels here are compiled for RISC-V on any host (the code is
//! never run), so the listing is checked everywhere: every instruction before the data area
//! decodes, branches name their labels, and label references show the constant they load.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use std::sync::atomic::{AtomicUsize, Ordering};
use symjit::{Application, Composer, Config, Slot, Translator};

/// the test options with `ty` replaced
fn config_for(ty: &str) -> Config {
    let _ = base_config();
    let toml = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/symjit.toml"))
        .unwrap()
        .replace("ty = \"native\"", &format!("ty = \"{}\"", ty));
    let dir = std::env::temp_dir().join(format!("symjit-disasm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // one file per call: the tests run in parallel
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("{}-{}.toml", ty, n));
    std::fs::write(&path, toml).unwrap();
    Config::from_name(path.to_str().unwrap(), 0).unwrap()
}

/// out0 = p0 ? p0 p1 : p1 + 7.5;  out1 = exp(p0) + 1e6 p1
fn compile(config: Config) -> Result<Application> {
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

/// the text `dump(.., what)` writes, or None if it refuses (`tag` keeps the tests' files apart)
fn dump(app: &mut Application, what: &str, tag: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("symjit-disasm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}-{what}.s"));
    let _ = std::fs::remove_file(&path);
    let ok = app.dump(path.to_str().unwrap(), what);
    let text = std::fs::read_to_string(&path).ok();
    let _ = std::fs::remove_file(&path);
    assert_eq!(ok, text.is_some(), "{what}: dump returned {ok}");
    text
}

/// the lines before the data area
fn code_lines(listing: &str) -> Vec<&str> {
    listing
        .lines()
        .take_while(|l| *l != "@data:")
        .filter(|l| !l.starts_with(';') && !l.ends_with(':'))
        .collect()
}

fn check_listing(listing: &str, kernel: &str) {
    assert!(
        listing.starts_with("; riscv64 kernel"),
        "{kernel}:\n{listing}"
    );
    assert!(listing.contains("\n@self:\n"), "{kernel}:\n{listing}");
    assert!(
        listing.contains("\n@data:\n"),
        "{kernel}: no data area\n{listing}"
    );

    let code = code_lines(listing);
    assert!(code.len() > 20, "{kernel}:\n{listing}");
    for line in &code {
        assert!(
            !line.contains(".word") && !line.contains(".byte"),
            "{kernel}: undecoded instruction\n{line}\n{listing}"
        );
    }
    assert!(
        code.iter().any(|l| l.trim_end().ends_with("ret")),
        "{kernel}:\n{listing}"
    );

    // the data area: the constants under their labels, the function table as addresses
    for label in ["_const_0_:", "_const_1_:", "_func_exp_:"] {
        assert!(
            listing.contains(&format!("\n{label}\n")),
            "{kernel}: no {label}\n{listing}"
        );
    }
    assert!(
        listing.contains("# 7.5\n") && listing.contains("# 1000000.0\n"),
        "{listing}"
    );
    assert!(listing.contains("# address\n"), "{listing}");

    // label references (auipc + fld) name the constant and show its value
    assert!(
        listing.contains("<_const_0_> = 7.5"),
        "{kernel}:\n{listing}"
    );
    assert!(listing.contains("<_func_exp_>"), "{kernel}:\n{listing}");
}

#[test]
fn riscv_scalar_listing() {
    let mut app = compile(config_for("riscv")).unwrap();
    let listing = dump(&mut app, "scalar-asm", "scalar").expect("scalar-asm");
    check_listing(&listing, "scalar");

    // the branches of the if/else name their labels
    assert!(listing.contains("<@main>"), "{listing}");
}

#[test]
fn riscv_fast_listing() {
    let mut app = compile(config_for("riscv")).unwrap();
    // the fast kernel exists for few arguments; its listing is checked like the scalar one
    if let Some(listing) = dump(&mut app, "fast-asm", "fast") {
        check_listing(&listing, "fast");
    }
}

#[test]
fn riscv_has_no_simd_listing() {
    let mut app = compile(config_for("riscv")).unwrap();
    assert!(dump(&mut app, "simd-asm", "simd").is_none());
}

#[test]
fn listing_covers_the_kernel() {
    // the listing is of the bytes the binary dump writes
    let mut app = compile(config_for("riscv")).unwrap();
    let dir = std::env::temp_dir().join(format!("symjit-disasm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("covers-scalar.bin");
    assert!(app.dump(path.to_str().unwrap(), "scalar"));
    let bytes = std::fs::read(&path).unwrap();
    let listing = dump(&mut app, "scalar-asm", "covers").unwrap();
    let size = format!("{} bytes", bytes.len());
    assert!(listing.lines().next().unwrap().contains(&size), "{listing}");
}
