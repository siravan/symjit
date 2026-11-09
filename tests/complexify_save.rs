//! `Application::complexify` keeps the complexified MIR as the MIR `save` writes, for every complex
//! application: compiled with `fast_complex` or not (2.28.4 handled only `fast_complex`, though
//! every complex application's bytecode holds the complexified MIR). A saved complexified MIR is
//! marked as such when read, and `load` refuses it; an ordinary complex MIR is not marked. Its
//! `BinOp::Complex` (re + i im) is a complex model's own instruction, before `Complexifier` as
//! after: 2.28.4's reader took it for complexified MIR, so ordinary complex files did not load.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Application, Composer, Slot, Storage, Translator};

const A: Complex<f64> = Complex::new(0.2, 0.3);

/// out = exp(p0 × a) + p0: a complex product, a call, a sum
fn compile(fast_complex: bool) -> Result<Application> {
    let mut config = base_config();
    config.set_complex(true);
    config.set_fast_complex(fast_complex);
    let mut t = Translator::new(config);
    t.set_num_params(1);
    let ca = t.append_constant(A)?;
    t.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(ca)], 0)?;
    t.append_fun(&Slot::Temp(1), "exp", &[Slot::Temp(0)], false)?;
    t.append_add(&Slot::Out(0), &[Slot::Temp(1), Slot::Param(0)], 0)?;
    t.compile()
}

fn saved(app: &Application) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    app.save(&mut bytes)?;
    Ok(bytes)
}

/// whether the MIR of the saved application `bytes` reads back as complexified
fn complexified(bytes: &[u8]) -> Result<bool> {
    let (_, _, mir) = Application::read_sjb(&mut &bytes[..], &base_config())?;
    Ok(mir.code.is_complexified)
}

#[test]
fn complexify_saves_the_complexified_mir() -> Result<()> {
    for fast_complex in [false, true] {
        let what = format!("fast_complex = {fast_complex}");
        let mut app = compile(fast_complex)?;
        let before = saved(&app)?;
        assert!(!complexified(&before)?, "{what}: the saved complex MIR reads as complexified");
        // an ordinary complex application loads back
        Application::load(&mut &before[..], &base_config())?;

        app.complexify()?;
        assert!(!app.config.fast_complex(), "{what}: fast_complex stays set");
        let after = saved(&app)?;
        assert!(after != before, "{what}: complexify left the saved MIR unchanged");
        assert!(complexified(&after)?, "{what}: the saved MIR does not read as complexified");
        assert!(
            Application::load(&mut &after[..], &base_config()).is_err(),
            "{what}: a complexified MIR loads back"
        );

        // complexify twice: the same file
        app.complexify()?;
        assert!(saved(&app)? == after, "{what}: a second complexify changed the saved MIR");
    }
    Ok(())
}

/// A real application: complexify changes nothing.
#[test]
fn complexify_leaves_a_real_application() -> Result<()> {
    let mut t = Translator::new(base_config());
    t.set_num_params(1);
    t.append_fun(&Slot::Temp(0), "exp", &[Slot::Param(0)], true)?;
    t.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Param(0)], 1)?;
    let mut app = t.compile()?;
    let before = saved(&app)?;
    app.complexify()?;
    assert!(saved(&app)? == before);
    Ok(())
}

/// A loaded application saves the bytes it was loaded from (the real locations are written in
/// order, whatever the HashSet's), so a re-saved file keeps its SHA-256.
#[test]
fn a_loaded_application_saves_its_bytes() -> Result<()> {
    for fast_complex in [false, true] {
        let first = saved(&compile(fast_complex)?)?;
        let again = saved(&Application::load(&mut &first[..], &base_config())?)?;
        assert!(again == first, "fast_complex = {fast_complex}: the re-saved bytes differ");
    }
    Ok(())
}
