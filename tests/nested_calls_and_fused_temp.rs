//! Two aarch64 bugs that pacrunner's kernels found (2.29.2), here on every architecture:
//!
//! - a subroutine (compress-mode funclet) that calls another: `bl` puts the return address in the
//!   link register, so the inner call overwrote the outer one's and the outer `ret` returned into
//!   itself, forever (arm/mod.rs `call_funclet` now saves the link register around each call; riscv64
//!   had the same bug with `jal` and `ra`, fixed the same way in riscv64/mod.rs);
//! - the SIMD kernel's fused multiply-adds with the destination also a factor (`dst == b`) took a
//!   scratch register among `Temp`, `Gen(2)`, `Gen(3)`, which may hold live values: here `Gen(2)` was
//!   overwritten (arm/vector.rs now uses the scratch register v29).
mod common;
use common::base_config;

use anyhow::Result;
use serde_json::{json, Value};
use symjit::{Application, Compiled, Compiler, Expr};

/// an application of four states and two observables (a Compiler model) with its MIR replaced by
/// `code` (states Mem 0 .. 3, observables Mem 4, 5)
fn application(code: Value, compress: bool) -> Result<Application> {
    let mut config = base_config();
    config.set_symbolica(false);
    let v: Vec<Expr> = ["a", "b", "c", "d"].iter().map(|n| Expr::var(n)).collect();
    let app = Compiler::with_config(config).compile(&v, &[v[0].add(&v[1]), v[2].add(&v[3])])?;
    let mut j: Value = serde_json::from_str(&app.to_json()?)?;
    j["program"]["config"]["options"]["compress"] = json!(compress);
    j["program"]["builder"]["count_stack"] = json!(96);
    j["program"]["builder"]["consts"] = json!(["0x4004000000000000"]); // 2.5
    j["mir"]["consts"] = json!(["0x4004000000000000"]);
    j["mir"]["code"] = code;
    Application::from_json(&j.to_string(), &base_config())
}

type K = fn(*const f64, *const &mut [f64], usize, *const f64) -> i32;

/// the kernel on `points` (each four states), `lanes` points a call: the two observables of each
fn run(f: K, points: &[[f64; 4]], lanes: usize) -> Vec<[f64; 2]> {
    let n = points.len();
    let mut cols: Vec<Vec<f64>> = (0..4)
        .map(|j| points.iter().map(|p| p[j]).collect())
        .collect();
    cols.extend((0..2).map(|_| vec![0.0; n]));
    let slices: Vec<&mut [f64]> = cols.iter_mut().map(|c| &mut c[..]).collect();
    for t in (0..n).step_by(lanes) {
        assert_eq!(f(std::ptr::null(), slices.as_ptr(), t, std::ptr::null()), 0);
    }
    (0..n).map(|i| [cols[4][i], cols[5][i]]).collect()
}

const POINTS: [[f64; 4]; 4] = [
    [1.5, -2.0, 3.0, 0.25],
    [-0.5, 4.0, -1.0, 2.0],
    [2.0, 2.0, 0.5, -3.0],
    [7.0, -1.5, 1.25, 1.0],
];

/// both kernels (scalar, and SIMD where there is one) against `want`
fn check(mut app: Application, want: impl Fn(&[f64; 4]) -> [f64; 2], what: &str) {
    let mut kernels = vec![("scalar", app.compiled.as_ref().unwrap().func(), 1)];
    if !cfg!(target_arch = "riscv64") {
        app.prepare_simd();
        let simd = app.compiled_simd.as_ref().expect("a SIMD kernel");
        kernels.push(("simd", simd.func(), simd.count_lanes()));
    }
    for (kind, f, lanes) in kernels {
        let got = run(f, &POINTS, lanes);
        for (p, g) in POINTS.iter().zip(got) {
            assert_eq!(g, want(p), "{what}, {kind} kernel, point {p:?}");
        }
    }
}

/// a subroutine calling a subroutine: obs0 = a × 2 (2.5 + 2.5) = 10 a, obs1 = 2.5 + b
#[test]
fn a_subroutine_calls_a_subroutine() -> Result<()> {
    let g = |n: u32| json!({"Gen": n});
    let code = json!([
        {"Call": {"label": "@outer", "num_args": 0}},
        {"Load": {"dst": g(0), "loc": {"Stack": 81}}},
        {"Load": {"dst": g(1), "loc": {"Mem": 0}}},
        {"Bi": {"op": "Times", "dst": g(0), "s1": g(0), "s2": g(1)}},
        {"Save": {"loc": {"Mem": 4}, "src": g(0)}},
        {"Load": {"dst": g(0), "loc": {"Stack": 82}}},
        {"Load": {"dst": g(1), "loc": {"Mem": 1}}},
        {"Bi": {"op": "Plus", "dst": g(0), "s1": g(0), "s2": g(1)}},
        {"Save": {"loc": {"Mem": 5}, "src": g(0)}},
        {"Branch": {"label": "@success"}},
        {"Label": {"label": "@outer"}},
        {"Call": {"label": "@inner", "num_args": 0}},
        {"Load": {"dst": g(0), "loc": {"Stack": 82}}},
        {"Bi": {"op": "Plus", "dst": g(0), "s1": g(0), "s2": g(0)}},
        {"Bi": {"op": "Plus", "dst": g(0), "s1": g(0), "s2": g(0)}},
        {"Save": {"loc": {"Stack": 81}, "src": g(0)}},
        {"Branch": {"label": ".ret"}},
        {"Label": {"label": "@inner"}},
        {"LoadConst": {"dst": g(0), "idx": 0}},
        {"Save": {"loc": {"Stack": 82}, "src": g(0)}},
        {"Branch": {"label": ".ret"}},
        "End"
    ]);
    check(
        application(code, true)?,
        |p| [10.0 * p[0], 2.5 + p[1]],
        "nested calls",
    );
    Ok(())
}

/// Temp := c × Temp + d with Temp = b (`dst == b`), while Gen(2) and Gen(3) hold a and d:
/// obs0 = a (Gen(2), live across the fused op), obs1 = b c + d
#[test]
fn a_fused_op_into_its_factor_keeps_the_live_registers() -> Result<()> {
    let g = |n: u32| json!({"Gen": n});
    let load = |d: Value, m: u32| json!({"Load": {"dst": d, "loc": {"Mem": m}}});
    let mut cases = Vec::new();
    for op in ["MulAdd", "MulSub", "NegMulAdd", "NegMulSub"] {
        cases.push(json!([
            load(g(2), 0),
            load(json!("Temp"), 1),
            load(g(0), 2),
            load(g(3), 3),
            {"Fused": {"op": op, "dst": "Temp", "a": g(0), "b": "Temp", "c": g(3)}},
            {"Save": {"loc": {"Mem": 4}, "src": g(2)}},
            {"Save": {"loc": {"Mem": 5}, "src": "Temp"}},
            "End"
        ]));
    }
    let fused: [fn(f64, f64, f64) -> f64; 4] = [
        |a, b, c| a.mul_add(b, c),
        |a, b, c| a.mul_add(b, -c),
        |a, b, c| (-a).mul_add(b, c),
        |a, b, c| (-a).mul_add(b, -c),
    ];
    for (code, f) in cases.into_iter().zip(fused) {
        let what = code[4]["Fused"]["op"].as_str().unwrap().to_string();
        check(
            application(code, false)?,
            move |p| [p[0], f(p[2], p[1], p[3])],
            &what,
        );
    }
    Ok(())
}
