//! A compress-mode subroutine call (`LoadArgs` ultra, `Call` of the `_ultra` entry) may pass fewer
//! argument locations than the generator packs; with none at all, the x86-64 `pack_locs` computed its
//! register count as `(0 - 1) / 4 + 1` and panicked with "index out of bounds" (2.29.1). Seen in
//! generated code (pacrunner's split kernels) whose subroutine has every input baked in.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use serde_json::{json, Value};
use symjit::{Application, Composer, Slot, Translator};

/// out = p0 × c, with c computed by a subroutine that takes no arguments
fn application() -> Result<Application> {
    let mut t = Translator::new(base_config());
    t.set_num_params(1);
    let c = t.append_constant(Complex::new(2.5, 0.0))?;
    t.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Const(c)], 0)?;
    let mut v: Value = serde_json::from_str(&t.compile()?.to_json()?)?;
    let opts = &mut v["program"]["config"]["options"];
    opts["compress"] = json!(true);
    opts["ultra"] = json!(true);
    let result = json!({"Stack": 81});
    v["program"]["builder"]["count_stack"] = json!(90);
    v["mir"]["code"] = json!([
        {"LoadArgs": {"locs": [], "complex": false, "ultra": true}},
        {"Call": {"label": "@f_ultra", "num_args": 0}},
        {"Load": {"dst": {"Gen": 0}, "loc": result}},
        {"LoadMath": {"dst": {"Gen": 0}, "loc": {"Param": 0}, "op": "Times", "s1": {"Gen": 0}}},
        {"Save": {"loc": {"Mem": 0}, "src": {"Gen": 0}}},
        {"Branch": {"label": "@success"}},
        {"Label": {"label": "@f_ultra"}},
        {"SaveArgs": {"num_args": 0, "complex": false, "ultra": true}},
        {"Label": {"label": "@f"}},
        {"SaveArgs": {"num_args": 0, "complex": false, "ultra": false}},
        {"LoadConst": {"dst": {"Gen": 0}, "idx": 0}},
        {"Save": {"loc": result, "src": {"Gen": 0}}},
        {"Branch": {"label": ".ret"}},
        "End"
    ]);
    Application::from_json(&v.to_string(), &base_config())
}

#[test]
fn call_with_no_arguments() -> Result<()> {
    let mut app = application()?;
    for p in [2.0, -3.0] {
        assert_eq!(app.as_applet().evaluate_single::<f64>(&[p]), 2.5 * p);
    }
    app.prepare_simd();
    Ok(())
}
