//! Many constants: the aarch64 generators load each constant from the pool through a base register
//! (`x25`, set by the prologues): `ldr d` / `ldr q` with an immediate offset for the first 4,096
//! entries, through a scratch register beyond. 5,000 distinct coefficients reach both paths, in the
//! scalar and the SIMD kernel (every architecture runs it).
mod common;
use common::base_config;

use anyhow::Result;
use symjit::{Compiled, Compiler, Expr};

const N: usize = 5000;

/// the coefficient of the k-th term: distinct, exactly representable products with x = 0.5
fn coef(k: usize) -> f64 {
    1.0 + k as f64 / 1024.0
}

#[test]
fn five_thousand_constants() -> Result<()> {
    let mut config = base_config();
    config.set_symbolica(false);
    let (x, y) = (Expr::var("x"), Expr::var("y"));
    // out0 = sum_k coef(k) x (the sum of the coefficients times x), out1 = the last coefficient + y
    // as a balanced tree (a chain of 5,000 additions recurses too deep)
    let mut terms: Vec<Expr> = (0..N).map(|k| symjit::double(coef(k)).mul(&x)).collect();
    while terms.len() > 1 {
        terms = terms
            .chunks(2)
            .map(|c| {
                if c.len() == 2 {
                    c[0].add(&c[1])
                } else {
                    c[0].clone()
                }
            })
            .collect();
    }
    let sum = terms.pop().unwrap();
    let last = symjit::double(coef(N - 1) + 1.0 / 4096.0).add(&y);
    let mut app = Compiler::with_config(config).compile(&[x, y], &[sum, last])?;

    // the pool really holds more than 4,096 constants (nothing folded them)
    let json: serde_json::Value = serde_json::from_str(&app.to_json()?)?;
    let pool = json["mir"]["consts"].as_array().unwrap().len();
    assert!(pool > 4096, "{pool} constants");

    let points = [[0.5, 1.0], [2.0, -0.25], [-1.0, 3.0], [0.25, 0.5]];
    let total: f64 = (0..N).map(coef).sum();
    let mut kernels = vec![("scalar", app.compiled.as_ref().unwrap().func(), 1)];
    if !cfg!(target_arch = "riscv64") {
        app.prepare_simd();
        let simd = app.compiled_simd.as_ref().expect("a SIMD kernel");
        kernels.push(("simd", simd.func(), simd.count_lanes()));
    }
    for (kind, f, lanes) in kernels {
        let n = points.len();
        let mut cols: Vec<Vec<f64>> = (0..2)
            .map(|j| points.iter().map(|p| p[j]).collect())
            .collect();
        cols.extend((0..2).map(|_| vec![0.0; n]));
        let slices: Vec<&mut [f64]> = cols.iter_mut().map(|c| &mut c[..]).collect();
        for t in (0..n).step_by(lanes) {
            assert_eq!(f(std::ptr::null(), slices.as_ptr(), t, std::ptr::null()), 0);
        }
        for (i, p) in points.iter().enumerate() {
            let want = total * p[0];
            assert!(
                (cols[2][i] - want).abs() <= 1e-9 * want.abs(),
                "{kind}: {} != {want}",
                cols[2][i]
            );
            assert_eq!(
                cols[3][i],
                coef(N - 1) + 1.0 / 4096.0 + p[1],
                "{kind}: the last constant"
            );
        }
    }
    Ok(())
}
