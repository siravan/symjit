#[cfg(feature = "composer")]
use crate::kernel::KernelCompile;
use crate::libm::*;
use crate::mir::Mir;
use crate::CompileExt;

#[test]
fn exp_f64_against_std() {
    // within 1 ulp of the C library's (correctly rounded in practice) exp
    let mut worst = 0u64;
    for i in 0..200001 {
        let x = -745.0 + 1455.0 * i as f64 / 200000.0;
        let (a, b) = (exp(x), x.exp());
        if b.is_finite() && b > 0.0 {
            let d = a.to_bits().abs_diff(b.to_bits());
            assert!(d <= 1, "exp({x:e}) = {a:e}, std: {b:e}");
            worst = worst.max(d);
        } else {
            assert_eq!(a, b, "exp({x:e})");
        }
    }
    for x in [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e300,
        -1e300,
        709.78,
        709.79,
        -745.13,
        -745.14,
    ] {
        assert_eq!(exp(x).to_bits(), x.exp().to_bits(), "exp({x:e})");
    }
    assert!(exp(f64::NAN).is_nan());
    assert!(worst <= 1);
}

fn inputs() -> Vec<f64> {
    let mut xs = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        709.78,
        709.79,
        -745.13,
        -745.14,
        1e300,
        -1e300,
        1e-300,
    ];
    xs.extend((0..3000).map(|i| -745.0 + 1455.0 * (i as f64 * 0.618034).fract()));
    xs.extend((0..1000).map(|i| (i as f64 * 0.7548777).fract() * 2.0 - 1.0));
    xs
}

#[test]
fn library_fits_in_registers() {
    // `inline_calls` and `append_funclets` give the funclets 12 registers
    for fastmath in [false, true] {
        for (name, f) in LIBRARY {
            let mut config = crate::config::Config::default();
            config.set_fastmath(fastmath);
            let fits = |count_regs: u8| {
                let trace = Trace::default();
                let out = f(trace.arg()).id;
                let mut mir = Mir::new(config.clone());
                trace
                    .emit(out, &mut mir, &mut Vec::new(), count_regs)
                    .is_ok()
            };
            let needed = (1..=16).find(|&n| fits(n)).unwrap();
            println!("{name} (fastmath={fastmath}): {needed} registers");
            assert!(needed <= 12, "{name} needs {needed} registers");
        }
    }
    for fastmath in [false, true] {
        for (name, f) in COMPLEX_LIBRARY {
            let mut config = crate::config::Config::default();
            config.set_fastmath(fastmath);
            let fits = |count_regs: u8| {
                let trace = Trace::default();
                let (re, im) = f(trace.arg_k(0), trace.arg_k(1));
                let mut mir = Mir::new(config.clone());
                trace
                    .emit_many(&[re.id, im.id], &mut mir, &mut Vec::new(), count_regs)
                    .is_ok()
            };
            let needed = (1..=16).find(|&n| fits(n)).unwrap();
            println!("complex {name} (fastmath={fastmath}): {needed} registers");
            assert!(needed <= 12, "complex {name} needs {needed} registers");
        }
    }
}

type Variant = (&'static str, fn(&mut Config));

fn variants() -> Vec<Variant> {
    vec![
        ("default", |_| {}),
        ("opt0", |c| c.set_opt_level(0)),
        ("opt1", |c| c.set_opt_level(1)),
        ("opt3", |c| c.set_opt_level(3)),
        ("no-simd", |c| c.set_simd(false)),
        ("simd512", |c| c.enable_simd512(true)),
        ("cse-off", |c| c.set_cse(false)),
    ]
}

use crate::composer::Composer;
use crate::config::Config;
use crate::instruction::Slot;
use crate::translator::Translator;

// f(x) and f(x - 3) + x (no multiply-add that fastmath could fuse in the argument)
fn compile(config: Config, name: &str) -> crate::Application {
    let mut tr = Translator::new(config);
    tr.set_num_params(1);
    let three = tr
        .append_constant(num_complex::Complex::new(-3.0, 0.0))
        .unwrap();
    tr.append_fun(&Slot::Out(0), name, &[Slot::Param(0)], false)
        .unwrap();
    tr.append_add(&Slot::Temp(1), &[Slot::Param(0), Slot::Const(three)], 0)
        .unwrap();
    tr.append_fun(&Slot::Temp(2), name, &[Slot::Temp(1)], false)
        .unwrap();
    tr.append_add(&Slot::Out(1), &[Slot::Temp(2), Slot::Param(0)], 0)
        .unwrap();
    tr.compile().unwrap()
}

/// The C library's version of a function of the library, as the scalar kernels call it
/// (`code.rs`).
fn c_function(name: &str) -> fn(f64) -> f64 {
    match name {
        "exp" => f64::exp,
        "sinh" => f64::sinh,
        "cosh" => f64::cosh,
        "tanh" => f64::tanh,
        "coth" => |x| 1.0 / x.tanh(),
        "sech" => |x| 1.0 / x.cosh(),
        "csch" => |x| 1.0 / x.sinh(),
        "sin" => f64::sin,
        "cos" => f64::cos,
        "ln" => f64::ln,
        "log" => f64::ln,
        "log10" => f64::log10,
        _ => unreachable!(),
    }
}

fn library_function(name: &str) -> fn(f64) -> f64 {
    match name {
        "exp" => exp,
        "sinh" => sinh,
        "cosh" => cosh,
        "tanh" => tanh,
        "coth" => coth,
        "sech" => sech,
        "csch" => csch,
        "sin" => sin,
        "cos" => cos,
        "ln" => ln,
        "log" => ln,
        "log10" => log10,
        _ => unreachable!(),
    }
}

#[test]
fn hyperbolic_f64() {
    // against the C library's (whose 1 / f(x) are themselves an ulp or so off)
    for name in ["sinh", "cosh", "tanh", "coth", "sech", "csch"] {
        let (f, g) = (library_function(name), c_function(name));
        for i in 0..100001 {
            let x = -705.0 + 1410.0 * i as f64 / 100000.0;
            for x in [x, x * 1e-3] {
                let (a, b) = (f(x), g(x));
                if b.is_normal() {
                    let d = a.to_bits().abs_diff(b.to_bits());
                    assert!(d <= 6, "{name}({x:e}) = {a:e}, C: {b:e}");
                }
            }
        }
    }
    let inf = f64::INFINITY;
    let cases: [(&str, f64, f64); 24] = [
        ("sinh", -0.0, -0.0),
        ("sinh", -inf, -inf),
        ("sinh", 710.4, 1.6663642832806494e308),
        ("sinh", 710.5, inf),
        ("cosh", -0.0, 1.0),
        ("cosh", -inf, inf),
        ("cosh", -710.4, 1.6663642832806494e308),
        ("cosh", 711.0, inf),
        ("tanh", -0.0, -0.0),
        ("tanh", -inf, -1.0),
        ("tanh", 1e-300, 1e-300),
        ("tanh", 800.0, 1.0),
        ("coth", -0.0, -inf),
        ("coth", 0.0, inf),
        ("coth", -inf, -1.0),
        ("coth", 1e-300, 1e300),
        ("sech", 0.0, 1.0),
        ("sech", inf, 0.0),
        ("sech", 720.0, 4.06446160484e-313),
        ("sech", -745.0, 9.881312916824931e-324),
        ("csch", -0.0, -inf),
        ("csch", -inf, -0.0),
        ("csch", -720.0, -4.06446160484e-313),
        ("csch", 1e-300, 1e300),
    ];
    for (name, x, want) in cases {
        let got = library_function(name)(x);
        let close = got.to_bits() == want.to_bits()
            || (want.is_normal() && ((got - want) / want).abs() < 4e-16)
            || (want.abs() < 1e-300 && (got - want).abs() <= 1e-323);
        assert!(close, "{name}({x:e}) = {got:e}, not {want:e}");
    }
    for name in ["sinh", "cosh", "tanh", "coth", "sech", "csch"] {
        assert!(library_function(name)(f64::NAN).is_nan(), "{name}(NaN)");
    }
}

/// The library's complex version of `name`, if it has one.
fn complex_library_function(name: &str) -> Option<fn(f64, f64) -> (f64, f64)> {
    match name {
        "sinh" => Some(csinh),
        "cosh" => Some(ccosh),
        "tanh" => Some(ctanh),
        "coth" => Some(ccoth),
        "sech" => Some(csech),
        "csch" => Some(ccsch),
        "exp" => Some(cexp),
        "sin" => Some(csin),
        "cos" => Some(ccos),
        _ => None,
    }
}

#[test]
fn complex_f64() {
    use crate::code::VirtualTable as V;
    use num_complex::Complex;
    type CFunc = extern "C" fn(f64, f64, &mut Complex<f64>);
    let fs: [(&str, CFunc); 9] = [
        ("exp", V::cplx_exp),
        ("sin", V::cplx_sin),
        ("cos", V::cplx_cos),
        ("sinh", V::cplx_sinh),
        ("cosh", V::cplx_cosh),
        ("tanh", V::cplx_tanh),
        ("coth", V::cplx_coth),
        ("sech", V::cplx_sech),
        ("csch", V::cplx_csch),
    ];
    // against the complex C functions (the C path's tanh, `tanh_c`, used the same formula
    // as the library's since 0020)
    for (name, g) in fs {
        let f = complex_library_function(name).unwrap();
        for i in 0..100000 {
            let t = (i as f64 * 0.6180339887).fract();
            let u = (i as f64 * 0.7548776662).fract();
            let (x, y) = match i % 3 {
                0 => (t * 60.0 - 30.0, u * 20.0 - 10.0),
                1 => (t * 1400.0 - 700.0, u * 2e3 - 1e3),
                _ => (t * 4.0 - 2.0, u * 4.0 - 2.0),
            };
            let (re, im) = f(x, y);
            let mut z = Complex::new(0.0, 0.0);
            g(x, y, &mut z);
            let n = z.norm();
            if n.is_finite() && n > 1e-300 {
                let e = (re - z.re).abs().max((im - z.im).abs()) / n;
                assert!(e < 2e-15, "{name}({x:e}, {y:e}) = ({re:e}, {im:e}), C: {z}");
            }
        }
    }
    // real arguments: the real functions (and cosh x sin(+0) = +0)
    for x in [-3.5, -0.25, 0.0, 0.75, 20.0, 600.0] {
        let (re, im) = csinh(x, 0.0);
        assert_eq!(
            (re.to_bits(), im.to_bits()),
            (sinh(x).to_bits(), 0),
            "sinh({x})"
        );
        let (re, _) = ccosh(x, 0.0);
        assert_eq!(re.to_bits(), cosh(x).to_bits(), "cosh({x})");
    }
}

#[test]
fn logarithm_f64() {
    for (name, f, g, ulps) in [
        ("ln", ln as fn(f64) -> f64, f64::ln as fn(f64) -> f64, 1),
        ("log10", log10, f64::log10, 2),
    ] {
        for i in 0..300000 {
            let t = (i as f64 * 0.6180339887).fract();
            for x in [
                t * 4.0,
                (t * 1400.0 - 700.0).exp(),
                1.0 + (t - 0.5) * 1e-6,
                f64::from_bits((t * 4.5e15) as u64 + 1), // subnormals
            ] {
                let (a, b) = (f(x), g(x));
                let d = a.to_bits().abs_diff(b.to_bits());
                assert!(d <= ulps || a == b, "{name}({x:e}) = {a:e}, C: {b:e}");
            }
        }
        for x in [
            0.0,
            -0.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            5e-324,
            f64::MAX,
        ] {
            let (a, b) = (f(x), g(x));
            assert!(
                a == b || (a.is_nan() && b.is_nan()),
                "{name}({x:e}) = {a:e}, C: {b:e}"
            );
        }
        assert!(f(f64::NAN).is_nan());
    }
}

#[test]
fn trigonometric_f64() {
    use std::f64::consts::FRAC_PI_2;
    // against the C library's, also close to multiples of pi/2
    for (name, f, g) in [
        ("sin", sin as fn(f64) -> f64, f64::sin as fn(f64) -> f64),
        ("cos", cos, f64::cos),
    ] {
        for i in 0..400001 {
            let t = i as f64 / 400000.0;
            for x in [
                t * 10.0 - 5.0,
                (t - 0.5) * 2.0 * TRIG_LIMIT,
                (t - 0.5) * 2e3,
                FRAC_PI_2 * (i as f64 - 2e5) * 5.0,
            ] {
                let (a, b) = (f(x), g(x));
                let d = a.to_bits().abs_diff(b.to_bits());
                assert!(
                    d <= 2 || (a - b).abs() < 1e-300,
                    "{name}({x:e}) = {a:e}, C: {b:e}"
                );
            }
        }
    }
    assert_eq!(sin(-0.0f64).to_bits(), (-0.0f64).to_bits());
    assert_eq!(sin(1e-310f64), 1e-310);
    assert_eq!(cos(-0.0f64), 1.0);
    assert_eq!(cos(std::f64::consts::PI), -1.0);
}

// The SIMD kernels run the library's functions; the scalar kernel (one point at a time,
// the rows left after the last full SIMD row, and kernels without SIMD) calls the C
// library's.
#[test]
fn library_inlined_in_simd_kernels() {
    use crate::machine::Compiled;

    let xs = inputs();
    let n = xs.len();
    let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());

    for (name, _) in LIBRARY {
        let (f, g) = (library_function(name), c_function(name));
        // outside a function's domain the kernels call the C function
        let inside = |x: f64| domain_limit(name).is_none_or(|l| x.abs() < l);
        let ours: Vec<[f64; 2]> = xs.iter().map(|&x| [f(x), f(x - 3.0) + x]).collect();
        let c_lib: Vec<[f64; 2]> = xs.iter().map(|&x| [g(x), g(x - 3.0) + x]).collect();
        // the library differs from the C library at some points: equal bits show which
        // one a kernel uses
        assert!(xs.iter().any(|x| f(*x) != g(*x)), "{name}");

        for fastmath in [false, true] {
            for (vname, variant) in variants() {
                let mut config = Config::from_name("native", Config::default().opt).unwrap();
                config.set_simd(true);
                config.set_opt_level(2);
                config.set_fastmath(fastmath);
                config.set_option("inline_math", "true").unwrap();
                variant(&mut config);
                let app = compile(config, name);
                let tag = format!("{name} {vname} fastmath={fastmath}");
                let lanes = app.compiled_simd.as_ref().map_or(0, |c| c.count_lanes());

                // fused multiply-adds change the rounding of some results by an ulp or two
                let close = |a: f64, w: f64| {
                    same(a, w)
                        || (fastmath
                            && a.is_finite()
                            && (a.to_bits().abs_diff(w.to_bits()) <= 2
                                || (a - w).abs() <= 1e-15 * w.abs().max(1.0)))
                };

                for (i, x) in xs.iter().enumerate() {
                    let mut outs = [0.0; 2];
                    app.evaluate(&[*x], &mut outs);
                    assert!(
                        same(outs[0], c_lib[i][0]) && same(outs[1], c_lib[i][1]),
                        "{tag} scalar x = {x:e}: {outs:?} != {:?}",
                        c_lib[i]
                    );
                }

                let mut outs = vec![0.0; 2 * n];
                app.evaluate_matrix(&xs, &mut outs, n);
                let simd_rows = if lanes > 1 { n / lanes * lanes } else { 0 };
                for i in 0..n {
                    let got = [outs[2 * i], outs[2 * i + 1]];
                    // a SIMD row with an argument outside the domain: the scalar kernel
                    let row =
                        i / lanes.max(1) * lanes.max(1)..(i / lanes.max(1) + 1) * lanes.max(1);
                    let in_domain = row
                        .clone()
                        .all(|j| j >= n || (inside(xs[j]) && inside(xs[j] - 3.0)));
                    let ok = if i < simd_rows && in_domain {
                        close(got[0], ours[i][0]) && close(got[1], ours[i][1])
                    } else {
                        same(got[0], c_lib[i][0]) && same(got[1], c_lib[i][1])
                    };
                    assert!(
                        ok,
                        "{tag} matrix row {i} (SIMD rows: {simd_rows}) x = {:e}: {got:?}, \
                         library {:?}, C {:?}",
                        xs[i], ours[i], c_lib[i]
                    );
                }
            }
        }
    }
}

// Complex kernels (`complexify`): the hyperbolic functions have complex funclets, used by
// the SIMD rows whose arguments are all in their domain (real or complex); for exp, sin
// and cos, SIMD rows whose arguments are all real get the real function. The other rows
// (and single points) get the complex C function, as without `inline_math`.
#[test]
fn complex_kernels() {
    use crate::machine::Compiled;
    use num_complex::Complex;

    let n = 1024;
    let re: Vec<f64> = (0..n)
        .map(|i| 12.0 * (i as f64 * 0.618034).fract() - 6.0)
        .collect();
    // groups of 8 points: real, complex, mixed (lanes 0-3 real), complex with half of the
    // real parts outside the domain of the complex funclets, ...
    let zs: Vec<Complex<f64>> = (0..n)
        .map(|i| match (i / 8) % 4 {
            0 => Complex::new(re[i], 0.0),
            1 => Complex::new(re[i], 0.5 + (i % 7) as f64 * 0.25),
            2 if i % 8 < 4 => Complex::new(re[i], 0.0),
            2 => Complex::new(re[i], -1.25),
            _ if i % 8 < 4 => Complex::new(re[i] + 800.0, 0.5),
            _ => Complex::new(re[i], 0.5),
        })
        .collect();
    let in_domain = |name: &str, z: &Complex<f64>| {
        let (lx, ly) = complex_domain(name);
        z.re.abs() < lx && z.im.abs() < ly
    };

    for (name, _) in LIBRARY {
        let f = library_function(name);
        for fast_complex in [true, false] {
            for (vname, variant) in variants() {
                let build = |inline: bool| {
                    let mut config = Config::from_name("native", Config::default().opt).unwrap();
                    config.set_complex(true);
                    config.set_simd(true);
                    config.set_opt_level(2);
                    config.set_fastmath(false);
                    config
                        .set_option("fast_complex", &fast_complex.to_string())
                        .unwrap();
                    config
                        .set_option("inline_math", &inline.to_string())
                        .unwrap();
                    variant(&mut config);
                    let mut tr = Translator::new(config);
                    tr.set_num_params(1);
                    tr.append_fun(&Slot::Out(0), name, &[Slot::Param(0)], false)
                        .unwrap();
                    tr.compile().unwrap()
                };
                let (app, c_app) = (build(true), build(false));
                let tag = format!("{name} {vname} fast_complex={fast_complex}");
                let lanes = app.compiled_simd.as_ref().map_or(0, |c| c.count_lanes());

                let mut got = vec![Complex::new(0.0, 0.0); n];
                let mut c_lib = vec![Complex::new(0.0, 0.0); n];
                app.evaluate_matrix(&zs, &mut got, n);
                c_app.evaluate_matrix(&zs, &mut c_lib, n);
                let same =
                    |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());

                let complex_f = complex_library_function(name);
                for i in 0..n {
                    let full_row = lanes > 1 && (i / lanes + 1) * lanes <= n;
                    // the real function: real arguments (positive for the logarithms)
                    let row_real = full_row
                        && (i / lanes * lanes..(i / lanes + 1) * lanes)
                            .all(|j| zs[j].im == 0.0 && (!positive_domain(name) || zs[j].re > 0.0))
                        && real_in_complex(name);
                    let row_in_domain = full_row
                        && (i / lanes * lanes..(i / lanes + 1) * lanes)
                            .all(|j| in_domain(name, &zs[j]));
                    let (w, ok) = if let (true, Some(cf)) = (row_in_domain, complex_f) {
                        // the complex funclet, also for real arguments
                        let (re, im) = cf(zs[i].re, zs[i].im);
                        (
                            Complex::new(re, im),
                            same(got[i].re, re) && same(got[i].im, im),
                        )
                    } else if row_real {
                        let w = Complex::new(f(zs[i].re), 0.0);
                        (w, same(got[i].re, w.re) && got[i].im.to_bits() == 0)
                    } else {
                        // a real argument gives +0 as the imaginary part (the complex C
                        // function gives a signed zero)
                        let w = c_lib[i];
                        (
                            w,
                            same(got[i].re, w.re) && (same(got[i].im, w.im) || got[i].im == w.im),
                        )
                    };
                    assert!(
                        ok,
                        "{tag} z = {}: {} != {} (real row: {row_real})",
                        zs[i], got[i], w
                    );
                }

                // one point at a time: the complex C function
                for z in zs.iter().step_by(37) {
                    let (a, b) = (app.evaluate_single(&[*z]), c_app.evaluate_single(&[*z]));
                    assert!(
                        same(a.re, b.re) && (same(a.im, b.im) || a.im == b.im),
                        "{tag} single z = {z}: {a} != {b}"
                    );
                }
            }
        }
    }
}

// `libm::exp` on the kernel builder's `Real` builds an exp kernel
#[cfg(feature = "composer")]
#[test]
fn exp_with_the_kernel_builder() {
    use crate::kernel::{KernelBuilder, Real};

    for direct in [false, true] {
        let mut config = Config::from_name("native", Config::default().opt).unwrap();
        config.set_fastmath(false);
        config.set_direct(direct);
        let k = KernelBuilder::new(config);
        let x: Real = k.param();
        let app = k.compile(exp(x)).unwrap().seal().unwrap();
        for x in inputs() {
            let got = app.evaluate_single(&[x]);
            let w = exp(x);
            assert!(
                got.to_bits() == w.to_bits() || (got.is_nan() && w.is_nan()),
                "direct={direct} x = {x:e}: {got} != {w}"
            );
        }
    }
}
