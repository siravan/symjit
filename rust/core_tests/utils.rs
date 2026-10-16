use crate::composer::Composer;
use crate::config::Config;
use crate::instruction::Slot;
use crate::translator::Translator;
use crate::utils::{getexp, getmant, pow2i};
use crate::CompileExt;
use num_complex::Complex;

fn inputs() -> Vec<f64> {
    let mut xs = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        5e-324,
        -2.2e-308,
        f64::MIN_POSITIVE,
        f64::MAX,
        -f64::MAX,
        1e300,
        -1e300,
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        3.0e15,
        -3.0e15,
        1.25e-310,
    ];
    xs.extend((-1100..=1100).map(|k| k as f64));
    xs.extend((0..400).map(|i| (i as f64 * 0.731).sin() * 10f64.powi(i % 600 - 300)));
    xs
}

#[test]
fn exponent_reference_functions() {
    for k in -1022..=1023 {
        assert_eq!(pow2i(k as f64), 2f64.powi(k), "pow2i({k})");
    }
    assert_eq!(pow2i(-1023.0).to_bits(), 0);
    assert_eq!(pow2i(1024.0), f64::INFINITY);
    assert_eq!(pow2i(2.5), 8.0); // x + 1023 is rounded, ties to even
    assert_eq!(pow2i(1.5), 2.0);
    assert_eq!(pow2i(f64::INFINITY).to_bits(), 0);
    assert_eq!(pow2i(f64::NAN).to_bits(), 0);

    for x in inputs() {
        if x.is_normal() {
            let e = getexp(x);
            assert!(
                e.fract() == 0.0 && (-1022.0..=1023.0).contains(&e),
                "getexp({x:e})"
            );
            assert!(
                pow2i(e) <= x.abs() && x.abs() / 2.0 < pow2i(e),
                "getexp({x:e})"
            );
            let m = getmant(x);
            assert!((1.0..2.0).contains(&m.abs()) && m.is_sign_negative() == x.is_sign_negative());
            assert_eq!(m * pow2i(getexp(x)), x, "getmant({x:e})");
        }
    }
    assert_eq!(getexp(0.0), -1023.0);
    assert_eq!(getexp(5e-324), -1023.0);
    assert_eq!(getexp(f64::NAN), 1024.0);
    assert_eq!(getmant(-0.0), -1.0);
    assert_eq!(getmant(f64::INFINITY), 1.0);
}

type Variant = (&'static str, fn(&mut Config));

fn variants() -> Vec<Variant> {
    vec![
        ("default", |_| {}),
        ("opt0", |c| c.set_opt_level(0)),
        ("opt3", |c| c.set_opt_level(3)),
        ("no-fastmath", |c| c.set_fastmath(false)),
        ("no-simd", |c| c.set_simd(false)),
        ("simd512", |c| c.enable_simd512(true)),
        ("simd512-opt3", |c| {
            c.enable_simd512(true);
            c.set_opt_level(3)
        }),
    ]
}

// (the bytecode interpreter, `mir::interpreter`, calls the reference functions; it is not
// reachable through a `Translator`)
fn config(complex: bool, variant: fn(&mut Config)) -> Config {
    // the built-in options (with `symbolica`, the matrix layout of `evaluate_matrix`)
    let mut config = Config::from_name("native", Config::default().opt).unwrap();
    config.set_simd(true);
    config.set_fastmath(true);
    config.set_opt_level(2);
    config.set_complex(complex);
    variant(&mut config);
    config
}

// f(x) and f(x) + x * 0 (a register operand that is not the parameter)
fn compile(op: &str, config: Config) -> crate::Application {
    let mut tr = Translator::new(config);
    tr.set_num_params(1);
    tr.append_fun(&Slot::Out(0), op, &[Slot::Param(0)], false)
        .unwrap();
    tr.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Param(0)], 0)
        .unwrap();
    tr.append_fun(&Slot::Temp(1), op, &[Slot::Temp(0)], false)
        .unwrap();
    tr.append_add(&Slot::Out(1), &[Slot::Temp(1), Slot::Temp(0)], 0)
        .unwrap();
    tr.compile().unwrap()
}

fn reference(op: &str) -> fn(f64) -> f64 {
    match op {
        "pow2i" => pow2i,
        "getexp" => getexp,
        _ => getmant,
    }
}

#[test]
fn exponent_ops_real_kernels() {
    let xs = inputs();
    for op in ["pow2i", "getexp", "getmant"] {
        let f = reference(op);
        let want: Vec<[f64; 2]> = xs.iter().map(|&x| [f(x), f(x * x) + x * x]).collect();
        for (name, variant) in variants() {
            let app = compile(op, config(false, variant));
            let tag = format!("{op} {name}");
            // one point at a time (scalar kernel)
            for (x, w) in xs.iter().zip(&want) {
                let mut outs = [0.0; 2];
                app.evaluate(&[*x], &mut outs);
                assert_eq!(
                    outs.map(f64::to_bits),
                    w.map(f64::to_bits),
                    "{tag} scalar x = {x:e}: {outs:?} != {w:?}"
                );
            }
            // as a matrix (SIMD kernel, with a tail)
            let mut outs = vec![0.0; 2 * xs.len()];
            app.evaluate_matrix(&xs, &mut outs, xs.len());
            for (i, w) in want.iter().enumerate() {
                let got = [outs[2 * i], outs[2 * i + 1]];
                assert_eq!(
                    got.map(f64::to_bits),
                    w.map(f64::to_bits),
                    "{tag} matrix x = {:e}: {got:?} != {w:?}",
                    xs[i]
                );
            }
        }
    }
}

// complex kernels apply the operations to the real part and return a real
#[test]
fn exponent_ops_complex_kernels() {
    let xs: Vec<Complex<f64>> = inputs()
        .iter()
        .enumerate()
        .map(|(i, &x)| Complex::new(x, (i as f64 - 1000.0) * 0.375))
        .collect();
    for op in ["pow2i", "getexp", "getmant"] {
        let f = reference(op);
        for fast_complex in [true, false] {
            for (name, variant) in variants() {
                let mut config = config(true, variant);
                config
                    .set_option("fast_complex", &fast_complex.to_string())
                    .unwrap();
                let mut tr = Translator::new(config);
                tr.set_num_params(1);
                tr.append_fun(&Slot::Out(0), op, &[Slot::Param(0)], false)
                    .unwrap();
                let app = tr.compile().unwrap();
                let tag = format!("{op} {name} fast_complex={fast_complex}");
                let mut outs = vec![Complex::new(0.0, 0.0); xs.len()];
                app.evaluate_matrix(&xs, &mut outs, xs.len());
                for (z, got) in xs.iter().zip(&outs) {
                    let w = f(z.re);
                    assert!(
                        got.re.to_bits() == w.to_bits() && got.im == 0.0,
                        "{tag} z = {z}: {got} != {w}"
                    );
                }
            }
        }
    }
}
