//! f32 kernels (`Application::downgrade_f32`, rust/downgrade.rs): the f64 kernels' machine code
//! rewritten to f32 and run on f32 data, against the f64 kernels on the same inputs.
//!
//! - the scalar kernel computes in the low half of each 8-byte slot (states, observables and
//!   parameters are f32 values in the low four bytes);
//! - the SIMD kernel has twice the lanes and reads packed f32 columns, called with half the index;
//! - kernels that call functions, and SIMD kernels that branch on their lanes, are refused;
//! - riscv64 has only the scalar kernel.
//!
//! The inputs are multiples of 1/128 (exact in f32), so the rounding functions agree exactly.
#![cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
))]
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::backend::MachineCode;
use symjit::{
    Application, Compiled, Compiler, Composer, Config, Expr, F32Kernel, Slot, Translator,
};

const ARCH: &str = if cfg!(target_arch = "x86_64") {
    "x86_64"
} else if cfg!(target_arch = "aarch64") {
    "aarch64"
} else {
    "riscv64"
};

type Kernel = fn(*const f64, *const &mut [f64], usize, *const f64) -> i32;

/// the f32 code as a callable kernel (kept alive by the returned MachineCode)
fn load(code: Vec<u8>, lanes: usize) -> (MachineCode<f64>, Kernel) {
    let m = MachineCode::new(ARCH, code, Vec::new(), false, lanes, false);
    let f = m.func();
    (m, f)
}

/// three states (x, y, z) and the observables the tests check
fn model(config: Config) -> Result<Application> {
    let (x, y, z) = (Expr::var("x"), Expr::var("y"), Expr::var("z"));
    let c = symjit::double;
    let obs = vec![
        // + - * /
        x.add(&y.mul(&z)).sub(&x.div(&y)),
        // sqrt, abs, constants 0.5 and 0.1
        x.abs().sqrt().add(&c(0.5).mul(&y)).sub(&c(0.1).mul(&z)),
        // the rounding functions on exact values
        x.mul(&c(3.0))
            .floor()
            .add(&y.ceil())
            .add(&z.mul(&c(2.5)).round())
            .add(&x.mul(&c(-7.0)).trunc()),
        // min, max
        x.min(&y).add(&y.max(&z)),
        // a select
        x.gt(&y).ifelse(&x.mul(&c(2.0)), &y.sub(&c(1.0))),
        // neg, square, recip, a large constant
        x.neg()
            .add(&x.square().mul(&c(0.25)))
            .add(&z.recip())
            .add(&c(1e10)),
        // cube, an integer power and a division
        y.cube().div(&z.square().add(&c(1e-3))),
    ];
    Compiler::with_config(config).compile(&[x, y, z], &obs)
}

/// the inputs: point i's (x, y, z), multiples of 1/128, y and z away from 0
fn inputs(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            [
                ((i * 37 % 201) as f64 - 100.0) / 64.0,
                ((i * 53 % 173) as f64 - 86.5) / 64.0,
                ((i * 29 % 97) as f64 + 16.0) / 64.0 * if i % 3 == 0 { -1.0 } else { 1.0 },
            ]
        })
        .collect()
}

/// the f64 kernel's outputs (the indirect form: one slice per state and observable)
fn run_f64(f: Kernel, xs: &[[f64; 3]], count_obs: usize, step: usize) -> Vec<Vec<f64>> {
    let n = xs.len();
    let mut cols: Vec<Vec<f64>> = (0..3).map(|j| xs.iter().map(|p| p[j]).collect()).collect();
    cols.extend((0..count_obs).map(|_| vec![0.0; n]));
    let slices: Vec<&mut [f64]> = cols.iter_mut().map(|c| &mut c[..]).collect();
    let params = [0.0f64; 4];
    for t in (0..n).step_by(step) {
        assert_eq!(f(std::ptr::null(), slices.as_ptr(), t, params.as_ptr()), 0);
    }
    cols.split_off(3)
}

/// closeness of an f32 value to the f64 one: a few roundings of the larger terms
#[track_caller]
fn close(v32: f32, v64: f64, scale: f64, what: &str) {
    let tol = 1e-5 * (scale.abs() + v64.abs() + 1.0);
    assert!(
        (v32 as f64 - v64).abs() <= tol,
        "{what}: f32 {v32} vs f64 {v64}"
    );
}

fn configs() -> Vec<(Config, &'static str)> {
    let mut config = base_config();
    config.set_symbolica(false);
    let mut out = vec![(config.clone(), "default")];
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
        let mut c512 = config;
        c512.enable_simd512(true);
        out.push((c512, "avx-512"));
    }
    out
}

#[test]
fn scalar_kernel_in_the_low_half_of_the_slots() -> Result<()> {
    for (config, tag) in configs() {
        let mut app = model(config)?;
        let n = 64;
        let xs = inputs(n);
        let no = 7;
        let f64k = app.compiled.as_ref().unwrap().func();
        let want = run_f64(f64k, &xs, no, 1);

        let (_m, f) = load(app.downgrade_f32(F32Kernel::Scalar)?, 1);
        // 8-byte slots, the f32 bits in the low half (the high half garbage: it must not matter)
        let slot = |v: f64| (0xdead_beefu64 << 32) | (v as f32).to_bits() as u64;
        let mut cols: Vec<Vec<u64>> = (0..3)
            .map(|j| xs.iter().map(|p| slot(p[j])).collect())
            .collect();
        cols.extend((0..no).map(|_| vec![0u64; n]));
        let slices: Vec<&mut [f64]> = cols
            .iter_mut()
            .map(|c| unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f64, n) })
            .collect();
        let params = [0.0f64; 4];
        for t in 0..n {
            assert_eq!(f(std::ptr::null(), slices.as_ptr(), t, params.as_ptr()), 0);
        }
        for k in 0..no {
            for t in 0..n {
                let v = f32::from_bits(cols[3 + k][t] as u32);
                close(
                    v,
                    want[k][t],
                    if k == 5 { 1e10 } else { 1.0 },
                    &format!("{tag} scalar obs {k} point {t}"),
                );
            }
        }
        // the rounding functions: exact
        for t in 0..n {
            assert_eq!(
                f32::from_bits(cols[5][t] as u32) as f64,
                want[2][t],
                "{tag} rounding, point {t}"
            );
        }
    }
    Ok(())
}

#[test]
#[cfg_attr(target_arch = "riscv64", ignore = "RISC-V has only the scalar kernel")]
fn simd_kernel_with_twice_the_lanes() -> Result<()> {
    for (config, tag) in configs() {
        let mut app = model(config)?;
        let code = app.downgrade_f32(F32Kernel::Simd)?;
        let simd = app.compiled_simd.as_ref().expect("a SIMD kernel");
        let lanes = simd.count_lanes();
        let n = 64;
        let xs = inputs(n);
        let no = 7;
        let f64k = simd.func();
        let want = run_f64(f64k, &xs, no, lanes);

        let (_m, f) = load(code, 2 * lanes);
        let mut cols: Vec<Vec<f32>> = (0..3)
            .map(|j| xs.iter().map(|p| p[j] as f32).collect())
            .collect();
        cols.extend((0..no).map(|_| vec![0f32; n]));
        let slices: Vec<&mut [f64]> = cols
            .iter_mut()
            .map(|c| unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f64, n / 2) })
            .collect();
        let params = [0.0f64; 4];
        for i in (0..n).step_by(2 * lanes) {
            // element i of a column at byte 4 i = 8 (i / 2)
            assert_eq!(
                f(std::ptr::null(), slices.as_ptr(), i / 2, params.as_ptr()),
                0
            );
        }
        for k in 0..no {
            for t in 0..n {
                close(
                    cols[3 + k][t],
                    want[k][t],
                    if k == 5 { 1e10 } else { 1.0 },
                    &format!("{tag} simd ({} f32 lanes) obs {k} point {t}", 2 * lanes),
                );
            }
        }
        for t in 0..n {
            assert_eq!(cols[5][t] as f64, want[2][t], "{tag} rounding, point {t}");
        }
    }
    Ok(())
}

/// out = p0 ? p0 p1 : p1 + 0.75, with a branch (Translator if/else, as tests/json_export.rs), in
/// the Symbolica layout (evaluate: the outputs, then the arguments, each an f32 in the low half of
/// an 8-byte slot)
#[test]
fn scalar_kernel_with_a_branch() -> Result<()> {
    let mut t = Translator::new(base_config());
    t.set_num_params(2);
    let c = t.append_constant(Complex::new(0.75, 0.0))?;
    let (p0, p1) = (Slot::Param(0), Slot::Param(1));
    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(0), &[p0, p1], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_add(&Slot::Temp(1), &[p1, Slot::Const(c)], 0)?;
    t.append_label(1)?;
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(0), &Slot::Temp(1))?;
    let mut app = t.compile()?;

    let f64k = app.compiled.as_ref().unwrap().func();
    let (_m, f) = load(app.downgrade_f32(F32Kernel::Scalar)?, 1);
    for (a, b) in [
        (1.5, -2.25),
        (0.0, 4.5),
        (-0.5, 0.5),
        (2.0, 0.0),
        (0.0, -1.0),
    ] {
        let args64 = [a, b];
        let mut out64 = [0.0f64];
        f64k(out64.as_mut_ptr(), std::ptr::null(), 0, args64.as_ptr());
        let args32 = [(a as f32).to_bits() as u64, (b as f32).to_bits() as u64];
        let mut out32 = [0u64];
        f(
            out32.as_mut_ptr() as *const f64,
            std::ptr::null(),
            0,
            args32.as_ptr() as *const f64,
        );
        assert_eq!(
            f32::from_bits(out32[0] as u32) as f64,
            out64[0],
            "({a}, {b})"
        );
    }
    Ok(())
}

/// a complex kernel: on aarch64 (fast complex) (re, im) of each argument and output in the low
/// eight bytes of its 16-byte slot; on riscv64 re and im each in the low half of its own 8-byte
/// slot; on x86-64 its shuffles have no f32 rule and it is refused
#[test]
fn complex_scalar_kernel() -> Result<()> {
    let mut config = base_config();
    config.set_complex(true);
    let mut t = Translator::new(config);
    t.set_num_params(2);
    let c = t.append_constant(Complex::new(0.5, -1.5))?;
    let (p0, p1) = (Slot::Param(0), Slot::Param(1));
    t.append_mul(&Slot::Temp(0), &[p0, p1], 0)?;
    t.append_div(&Slot::Temp(1), &p0, &p1, false, false)?;
    t.append_mul(&Slot::Temp(2), &[p0, Slot::Const(c)], 0)?;
    t.append_add(
        &Slot::Out(0),
        &[Slot::Temp(0), Slot::Temp(1), Slot::Temp(2)],
        0,
    )?;
    let mut app = t.compile()?;

    let r = app.downgrade_f32(F32Kernel::Scalar);
    if cfg!(target_arch = "x86_64") {
        let e = r
            .expect_err("x86-64 complex kernels have no f32 twin")
            .to_string();
        assert!(e.contains("no f32 rewrite") || e.contains("unknown"), "{e}");
        return Ok(());
    }

    let f64k = app.compiled.as_ref().unwrap().func();
    let (_m, f) = load(r?, 1);
    for (a, b) in [
        (Complex::new(1.5, -0.25), Complex::new(0.75, 2.0)),
        (Complex::new(-3.0, 1.0), Complex::new(-0.5, -0.5)),
    ] {
        let args64 = [a.re, a.im, b.re, b.im];
        let mut out64 = [0.0f64; 2];
        f64k(out64.as_mut_ptr(), std::ptr::null(), 0, args64.as_ptr());
        let bits = |x: f64| (x as f32).to_bits() as u64;
        let pair = |z: Complex<f64>| {
            if cfg!(target_arch = "riscv64") {
                [bits(z.re), bits(z.im)]
            } else {
                [bits(z.im) << 32 | bits(z.re), 0u64]
            }
        };
        let args32 = [pair(a), pair(b)].concat();
        let mut out32 = [0u64; 2];
        f(
            out32.as_mut_ptr() as *const f64,
            std::ptr::null(),
            0,
            args32.as_ptr() as *const f64,
        );
        let re = f32::from_bits(out32[0] as u32);
        let im = if cfg!(target_arch = "riscv64") {
            f32::from_bits(out32[1] as u32)
        } else {
            f32::from_bits((out32[0] >> 32) as u32)
        };
        close(re, out64[0], 10.0, &format!("re of ({a}, {b})"));
        close(im, out64[1], 10.0, &format!("im of ({a}, {b})"));
    }
    Ok(())
}

#[test]
fn refusals() -> Result<()> {
    let mut config = base_config();
    config.set_symbolica(false);
    let x = Expr::var("x");

    // a math function: a call
    let mut app = Compiler::with_config(config.clone()).compile(&[x.clone()], &[x.exp()])?;
    for k in [F32Kernel::Scalar, F32Kernel::Simd] {
        let e = app.downgrade_f32(k).unwrap_err().to_string();
        let why = if cfg!(target_arch = "riscv64") && k == F32Kernel::Simd {
            "only the scalar kernel"
        } else {
            "function call"
        };
        assert!(e.contains(why), "{k:?}: {e}");
    }

    // dump: false, and no file
    let path = std::env::temp_dir().join(format!("symjit-f32-{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    assert!(!app.dump(path.to_str().unwrap(), "simd-f32"));
    assert!(!path.exists());
    let mut ok = model(config)?;
    assert_eq!(
        ok.dump(path.to_str().unwrap(), "simd-f32"),
        !cfg!(target_arch = "riscv64")
    );
    assert!(ok.dump(path.to_str().unwrap(), "scalar-f32"));
    let _ = std::fs::remove_file(&path);
    Ok(())
}
