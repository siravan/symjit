//! sjc's Rust source front end against the `#[kernel]` macro: every kernel of
//! `kernels/corpus.rs` is compiled by both (the macro through `include!`, the interpreter from
//! the text), with several options, and must give the same intermediate code (MIR and
//! constants) and the same results.

#[allow(clippy::all, dead_code)]
mod corpus {
    include!("kernels/corpus.rs");
}

#[allow(clippy::all, dead_code)]
mod call_errors {
    include!("kernels/call_errors.rs");
}

use anyhow::Result;
use sjc::source::{Naming, Source};
use symjit::{Application, Compiled, CompilerType, Complex, Config};

const CORPUS: &str = include_str!("kernels/corpus.rs");

type Build = fn(Config) -> Result<Application>;

macro_rules! kernels {
    ($($name:ident),* $(,)?) => {
        vec![$((stringify!($name), corpus::$name::compile as Build)),*]
    };
}

fn all_kernels() -> Vec<(&'static str, Build)> {
    kernels!(
        arith,
        elementary,
        piecewise,
        tree,
        complex_ops,
        area,
        fib,
        several,
        rotate,
        count_inside,
        newton_sqrt,
        mandelbrot,
        mandelbrot_untyped,
        fib_iter,
        collatz,
        escape,
        midpoint_rule,
        masks,
        unrolled,
        mixed,
        bisect,
        shapes,
        real_api,
        orthant,
        step,
        iterate,
        complex_step,
        orbit,
        integrand,
        midpoint_integral,
        pole,
        around_circle,
        ramp,
        ramped,
        square,
        norm2,
        fib_path,
        spiral,
        callbacks_in_loop,
        callback_in_complex,
    )
}

fn configs() -> Vec<(String, Config)> {
    let mut configs = Vec::new();
    for opt in 1..=3 {
        for fastmath in [false, true] {
            let mut config = Config::builtin(CompilerType::Native);
            config.set_opt_level(opt);
            config.set_fastmath(fastmath);
            configs.push((format!("opt level {opt}, fastmath {fastmath}"), config));
        }
    }
    configs
}

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self) -> f64 {
        4.0 * self.next() - 2.0
    }
}

/// Arguments for which the kernel's loops end.
fn arguments(name: &str, n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
    match name {
        "newton_sqrt" => v[0] = v[0].abs() + 0.1,
        "collatz" => v[0] = (v[0].abs() * 10.0).floor() + 1.0,
        "fib" => v[0] = (v[0].abs() * 2.0).floor(),
        "tree" => v[1] = v[1].abs(),
        _ => {}
    }
    v
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

fn compare_results(name: &str, label: &str, a: Application, b: Application) -> Result<()> {
    if name == "area" {
        // random rectangles make it recurse millions of times; the MIR is compared
        return Ok(());
    }
    if matches!(
        name,
        "iterate" | "callbacks_in_loop" | "callback_in_complex"
    ) {
        // sjc compiles callbacks as stand-ins (it does not run the kernels): the MIR is compared
        return Ok(());
    }
    let complex = a.config.is_complex();
    let width = if complex { 2 } else { 1 };
    let (n_in, n_out) = (a.count_params / width, a.count_obs / width);
    let (a, b) = (a.seal()?, b.seal()?);
    let mut rng = Rng(0x5eed ^ name.len() as u64);
    for _ in 0..20 {
        if complex {
            let re = arguments(name, n_in, &mut rng);
            let im = arguments(name, n_in, &mut rng);
            let mut args: Vec<Complex<f64>> = re
                .iter()
                .zip(&im)
                .map(|(&r, &i)| Complex::new(r, i))
                .collect();
            if name == "orthant" {
                // the unit cube, down to boxes of side 1/4 .. 1/2
                let side = 0.25 + 0.25 * rng.next();
                args = vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 1.0),
                    Complex::new(1.0, 1.0),
                    Complex::new(side, 0.0),
                ];
            }
            let mut oa = vec![Complex::default(); n_out];
            let mut ob = vec![Complex::default(); n_out];
            a.evaluate(&args, &mut oa);
            b.evaluate(&args, &mut ob);
            let flat = |o: &[Complex<f64>]| o.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>();
            assert!(
                same(&flat(&oa), &flat(&ob)),
                "{name} ({label}) at {args:?}: {oa:?} != {ob:?}"
            );
        } else {
            let args = arguments(name, n_in, &mut rng);
            let mut oa = vec![0.0; n_out];
            let mut ob = vec![0.0; n_out];
            a.evaluate(&args, &mut oa);
            b.evaluate(&args, &mut ob);
            assert!(
                same(&oa, &ob),
                "{name} ({label}) at {args:?}: {oa:?} != {ob:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn interpreter_builds_the_kernels_of_the_macro() -> Result<()> {
    let src = Source::parse("corpus.rs", CORPUS)?;
    let kernels = all_kernels();
    // every kernel of the file is checked
    let mut listed: Vec<&str> = kernels.iter().map(|(n, _)| *n).collect();
    let mut found = src.kernels();
    listed.sort();
    found.sort();
    assert_eq!(listed, found);

    for (name, build) in kernels {
        for (label, config) in configs() {
            let a = build(config.clone())?;
            // the macro names the kernels and callbacks a kernel calls by their module paths
            let b = src.compile_with(name, config.clone(), &Naming::module("source::corpus"))?;
            // the code, without the list of constants at its end (compared below)
            let mir = |app: &Application| -> String {
                String::from_utf8_lossy(&app.bytecode.dumps())
                    .lines()
                    .filter(|l| !l.starts_with("const["))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            assert_eq!(mir(&a), mir(&b), "{name} ({label}): the MIR differs");
            // the indirect translator adds a random constant to each call of a kernel (to keep
            // CSE from merging calls): those differ between two builds of the macro too
            let again = build(config.clone())?;
            let bits = |app: &Application| -> Vec<Option<u64>> {
                app.prog
                    .builder
                    .consts
                    .iter()
                    .zip(again.prog.builder.consts.iter())
                    .map(|(c, r)| (c.to_bits() == r.to_bits()).then_some(c.to_bits()))
                    .collect()
            };
            assert_eq!(
                a.prog.builder.consts.len(),
                b.prog.builder.consts.len(),
                "{name} ({label}): the number of constants differs"
            );
            assert_eq!(bits(&a), bits(&b), "{name} ({label}): the constants differ");
            assert_eq!(
                (a.count_params, a.count_obs, a.config.is_complex()),
                (b.count_params, b.count_obs, b.config.is_complex()),
                "{name} ({label})"
            );
            compare_results(name, &label, a, b)?;
        }
    }
    Ok(())
}

#[test]
fn call_errors_as_the_macro_reports_them() -> Result<()> {
    let text = include_str!("kernels/call_errors.rs");
    let src = Source::parse("call_errors.rs", text)?;
    let naming = Naming::module("source::call_errors");
    let cases: [(&str, Build, &str); 4] = [
        (
            "real_caller",
            call_errors::real_caller::compile,
            "is called from a real kernel",
        ),
        (
            "calls_two_outputs",
            call_errors::calls_two_outputs::compile,
            "returns one value",
        ),
        ("ping", call_errors::ping::compile, "in a cycle"),
        ("pong", call_errors::pong::compile, "in a cycle"),
    ];
    for (name, build, phrase) in cases {
        let config = Config::builtin(CompilerType::Native);
        let macro_error = match build(config.clone()) {
            Ok(_) => panic!("the macro compiled `{name}`"),
            Err(e) => format!("{e:#}"),
        };
        let sjc_error = match src.compile_with(name, config, &naming) {
            Ok(_) => panic!("sjc compiled `{name}`"),
            Err(e) => format!("{e:#}"),
        };
        assert!(macro_error.contains(phrase), "{name}, macro: {macro_error}");
        assert!(sjc_error.contains(phrase), "{name}, sjc: {sjc_error}");
    }
    Ok(())
}

/// The kernel `name` of `text`, or the error.
fn compile(text: &str, name: &str) -> Result<Application> {
    Source::parse("t.rs", text)?.compile(name, Config::builtin(CompilerType::Native))
}

fn error(text: &str, name: &str) -> String {
    match compile(text, name) {
        Ok(_) => panic!("`{name}` compiled"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn errors_name_the_construct_and_its_position() {
    let e = error("#[kernel] fn f(x: f64) -> f64 { return x; }", "f");
    assert!(
        e.contains("t.rs:1:") && e.contains("cannot `return`"),
        "{e}"
    );

    let e = error(
        "#[kernel(callback)] fn cb(x: i32) -> f64 { 1.0 }\n#[kernel] fn f(x: f64) -> f64 { cb(x) }",
        "f",
    );
    assert!(e.contains("one or two `f64`"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { if x > 0.0 { x; } x }", "f");
    assert!(e.contains("without `else`"), "{e}");

    let e = error(
        "#[kernel] fn f(x: f64) -> f64 { let mut k = 0; while x > 1.0 { k += 1; } x }",
        "f",
    );
    assert!(e.contains("not a kernel value"), "{e}");

    let e = error(
        "#[kernel] fn f(x: f64) -> f64 { println!(\"{}\", 1); x }",
        "f",
    );
    assert!(e.contains("macros"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { x % 2.0 }", "f");
    assert!(e.contains("no `%`"), "{e}");

    let e = error("#[kernel] fn f(x: i32) -> f64 { 1.0 }", "f");
    assert!(e.contains("parameters of a kernel"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { loop {} }", "f");
    assert!(e.contains("did not end"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { x.mul_add(2.0, 1.0) }", "f");
    assert!(e.contains("no method `mul_add`"), "{e}");

    let e = error(
        "#[kernel(inline)] fn h(x: f64) -> f64 { h(x) }\n#[kernel] fn f(x: f64) -> f64 { h(x) }",
        "f",
    );
    assert!(e.contains("cannot call itself"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { x }", "nope");
    assert!(
        e.contains("no `#[kernel]` named `nope`") && e.contains("the kernels are: f"),
        "{e}"
    );

    assert!(Source::parse("t.rs", "fn (").is_err());
}

#[test]
fn if_without_else_that_assigns_is_a_kernel_branch() -> Result<()> {
    // `if c { y = .. }` on a kernel value: the else side keeps `y`
    let app = compile(
        "#[kernel] fn f(x: f64) -> f64 { let mut y = 0.5; if x > 0.0 { y = x * 2.0; } y }",
        "f",
    )?
    .seal()?;
    assert_eq!(app.evaluate_single(&[3.0]), 6.0);
    assert_eq!(app.evaluate_single(&[-3.0]), 0.5);
    Ok(())
}
