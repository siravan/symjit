//! Runs the sjc binary on applications saved here, links the objects into C programs (and
//! runs AArch64 ones under qemu, wasm modules in Node) and compares them with the JIT.
//! The steps that need `cc`, `aarch64-linux-gnu-gcc` + `qemu-aarch64` or `node` are skipped
//! (with a message) when those are missing.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::Result;
use symjit::{var, Application, Compiler, CompilerType, Config, Expr, Storage};

const SJC: &str = env!("CARGO_BIN_EXE_sjc");

// the kernels of the Rust source tests, here for their plain Rust functions
#[allow(clippy::all, dead_code)]
mod corpus {
    include!("kernels/corpus.rs");
}

const POINTS: [[f64; 3]; 4] = [
    [0.5, 2.0, 0.25],
    [-1.25, 0.75, 1.5],
    [1.5, -0.5, 0.125],
    [0.0, 3.0, -0.75],
];

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("sjc")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn have(tool: &str) -> bool {
    let ok = Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipped: `{}` is not available", tool);
    }
    ok
}

fn config(ty: CompilerType, opt_level: u8) -> Config {
    let mut config = Config::new(ty, 0).unwrap();
    config.set_opt_level(opt_level);
    config.set_cse(true);
    config
}

/// sin(x)*y + exp(-x*x), cos(y)/(1 + x*x), x*y - z
fn trig() -> (Vec<Expr>, Vec<Expr>) {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let a = &(&x.sin() * &y) + &(-&(&x * &x)).exp();
    let b = &y.cos() / &(&one + &(&x * &x));
    let c = &(&x * &y) - &z;
    (vec![x, y, z], vec![a, b, c])
}

/// sum of sin(k x + y) exp(-z / k), k = 1..39: register pressure
fn big() -> (Vec<Expr>, Vec<Expr>) {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let mut s = Expr::from(0.0);
    for k in 1..40 {
        let k = Expr::from(k as f64);
        let t = &(&(&k * &x) + &y).sin() * &(-&(&z / &k)).exp();
        s = &s + &t;
    }
    (vec![x, y, z], vec![s])
}

fn compile(model: &(Vec<Expr>, Vec<Expr>), config: Config) -> Application {
    Compiler::with_config(config)
        .compile(&model.0, &model.1)
        .unwrap()
}

fn save(app: &Application, path: &Path) {
    let mut f = std::fs::File::create(path).unwrap();
    app.save(&mut f).unwrap();
}

fn sjc(d: &Path, args: &[&str]) -> Output {
    Command::new(SJC)
        .args(args)
        .current_dir(d)
        .output()
        .unwrap()
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "sjc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The JIT's outputs at `POINTS` (bytecode for models compiled for another machine).
fn reference(model: &(Vec<Expr>, Vec<Expr>), opt_level: u8) -> Vec<Vec<f64>> {
    let mut app = compile(model, config(CompilerType::Native, opt_level));
    POINTS.iter().map(|p| app.call(p)).collect()
}

/// A C program that evaluates kernel `name` (header `name.h`) at `POINTS`, one line of %a
/// outputs per point.
fn driver(d: &Path, name: &str, count_obs: usize) {
    let up = name.to_uppercase();
    let points: Vec<String> = POINTS
        .iter()
        .map(|p| format!("{{{:?}, {:?}, {:?}}}", p[0], p[1], p[2]))
        .collect();
    let c = format!(
        "#include <stdio.h>\n#include \"{name}.h\"\n\
         static const double P[][3] = {{{}}};\n\
         int main(void) {{\n\
         \x20 for (int k = 0; k < {}; k++) {{\n\
         \x20   double mem[{up}_MEM_SIZE] = {{0}};\n\
         \x20   for (int i = 0; i < 3; i++) mem[i] = P[k][i];\n\
         \x20   if ({name}(mem, NULL, 0, NULL) != 0) return 1;\n\
         \x20   for (int j = 0; j < {count_obs}; j++) printf(\"%a \", mem[{up}_COUNT_STATES + j]);\n\
         \x20   printf(\"\\n\");\n\
         \x20 }}\n\
         \x20 return 0;\n}}\n",
        points.join(", "),
        POINTS.len()
    );
    std::fs::write(d.join("main.c"), c).unwrap();
}

fn parse_hex(out: &[u8]) -> Vec<Vec<f64>> {
    String::from_utf8_lossy(out)
        .lines()
        .map(|l| {
            l.split_whitespace()
                .map(|v| hexf(v).unwrap_or_else(|| panic!("bad output `{}`", v)))
                .collect()
        })
        .collect()
}

/// Parses C's %a output (e.g. 0x1.8p+1, -0x0p+0).
fn hexf(s: &str) -> Option<f64> {
    let (neg, s) = match s.strip_prefix('-') {
        Some(t) => (true, t),
        None => (false, s),
    };
    let s = s.strip_prefix("0x")?;
    let (mant, exp) = s.split_once('p')?;
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut v = u64::from_str_radix(int, 16).ok()? as f64;
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        v += c.to_digit(16)? as f64 * scale;
        scale /= 16.0;
    }
    let v = v * 2f64.powi(exp.parse().ok()?);
    Some(if neg { -v } else { v })
}

fn close(a: &[Vec<f64>], b: &[Vec<f64>], rtol: f64) {
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b) {
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(rb) {
            assert!(
                (x - y).abs() <= rtol * y.abs().max(1e-300),
                "{:?} != {:?}",
                a,
                b
            );
        }
    }
}

#[cfg(all(feature = "obj", target_arch = "x86_64", target_os = "linux"))]
#[test]
fn x86_objects_match_the_jit() -> Result<()> {
    if !have("cc") {
        return Ok(());
    }
    for opt in 1..=3 {
        let d = dir(&format!("x86-o{}", opt));
        let model = trig();
        save(
            &compile(&model, config(CompilerType::Native, opt)),
            &d.join("trig.sjb"),
        );
        ok(&sjc(&d, &["trig.sjb", "-o", "kern.o", "--name", "trig"]));
        assert!(d.join("kern.h").exists());
        let h = std::fs::read_to_string(d.join("kern.h"))?;
        assert!(
            h.starts_with("/* kern.h: the kernels in kern.o (x86-64 ELF)"),
            "{}",
            h
        );
        std::fs::rename(d.join("kern.h"), d.join("trig.h"))?;

        driver(&d, "trig", 3);
        let cc = Command::new("cc")
            .args(["main.c", "kern.o", "-lm", "-o", "main"])
            .current_dir(&d)
            .status()?;
        assert!(cc.success());
        let run = Command::new(d.join("main")).output()?;
        assert!(run.status.success());
        // the object holds the JIT's own machine code
        assert_eq!(parse_hex(&run.stdout), reference(&model, opt));
    }
    Ok(())
}

#[cfg(all(feature = "obj", target_arch = "x86_64", target_os = "linux"))]
#[test]
fn aarch64_objects_from_x86_models() -> Result<()> {
    if !have("aarch64-linux-gnu-gcc") || !have("qemu-aarch64") {
        return Ok(());
    }
    for (name, model, opt) in [("trig", trig(), 2), ("big", big(), 3)] {
        let d = dir(&format!("arm-{}", name));
        save(
            &compile(&model, config(CompilerType::Native, opt)),
            &d.join("m.sjb"),
        );
        ok(&sjc(
            &d,
            &[
                "m.sjb",
                "--target",
                "aarch64-unknown-linux-gnu",
                "-o",
                &format!("{}.o", name),
            ],
        ));
        driver(&d, name, model.1.len());
        let cc = Command::new("aarch64-linux-gnu-gcc")
            .args([
                "-static",
                "main.c",
                &format!("{}.o", name),
                "-lm",
                "-o",
                "main",
            ])
            .current_dir(&d)
            .status()?;
        assert!(cc.success());
        let run = Command::new("qemu-aarch64").arg(d.join("main")).output()?;
        assert!(run.status.success());
        // AArch64's libm may differ from x86-64's in the last bit
        close(&parse_hex(&run.stdout), &reference(&model, opt), 1e-13);
    }
    Ok(())
}

#[cfg(feature = "obj")]
#[test]
fn opt_level_3_arm_models_need_arm_registers() {
    let d = dir("registers");
    let model = big();
    let app = compile(&model, config(CompilerType::Arm, 3));
    assert!(app.measure("registers") > 14);
    save(&app, &d.join("big.sjb"));

    let out = sjc(&d, &["big.sjb", "--target", "x86_64", "-o", "big.o"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("registers") && err.contains("x86-64"),
        "{}",
        err
    );
    assert!(!d.join("big.o").exists());

    // the ARM object and the wasm module (no register limit) are fine
    ok(&sjc(
        &d,
        &["big.sjb", "--target", "aarch64-apple-darwin", "-o", "big.o"],
    ));
    assert_eq!(
        &std::fs::read(d.join("big.o")).unwrap()[..4],
        &[0xcf, 0xfa, 0xed, 0xfe]
    );
    #[cfg(feature = "wasm")]
    ok(&sjc(&d, &["big.sjb", "-o", "big.wasm"]));

    // opt level 2 models are portable
    save(
        &compile(&model, config(CompilerType::Arm, 2)),
        &d.join("big2.sjb"),
    );
    ok(&sjc(
        &d,
        &["big2.sjb", "--target", "x86_64", "-o", "big2.o"],
    ));
}

#[cfg(feature = "wasm")]
#[test]
fn wasm_modules_match_the_jit() -> Result<()> {
    let d = dir("wasm");
    let model = trig();
    save(
        &compile(&model, config(CompilerType::Native, 2)),
        &d.join("trig.sjb"),
    );
    ok(&sjc(&d, &["--emit", "wasm", "trig.sjb"]));
    let module = std::fs::read(d.join("trig.wasm"))?;
    assert_eq!(&module[..4], b"\0asm");

    if !have("node") {
        return Ok(());
    }
    let runner = Path::new(env!("CARGO_MANIFEST_DIR")).join("../python/tests/wasm_runner.mjs");
    let points: Vec<String> = POINTS
        .iter()
        .map(|p| format!("[{:?}, {:?}, {:?}]", p[0], p[1], p[2]))
        .collect();
    let req = format!(
        "{{\"module\": {:?}, \"mode\": \"direct\", \"count_states\": 3, \"count_obs\": 3, \
         \"mem_size\": 6, \"points\": [{}], \"params\": []}}",
        d.join("trig.wasm").to_str().unwrap(),
        points.join(", ")
    );
    let mut child = Command::new("node")
        .arg(runner)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    use std::io::Write;
    child.stdin.take().unwrap().write_all(req.as_bytes())?;
    let out = child.wait_with_output()?;
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    // {"obs": [[...], ...], ...}
    let obs = text.split("\"obs\":").nth(1).unwrap();
    let obs = &obs[..obs.find("]]").unwrap() + 2];
    let values: Vec<Vec<f64>> = obs
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split("],")
        .map(|row| {
            row.trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .map(|v| v.trim().parse().unwrap())
                .collect()
        })
        .collect();
    // JavaScript's Math functions are not glibc's
    close(&values, &reference(&model, 2), 1e-14);
    Ok(())
}

#[cfg(feature = "obj")]
#[test]
fn header_only_and_info() {
    let d = dir("info");
    save(
        &compile(&trig(), config(CompilerType::Native, 2)),
        &d.join("trig.sjb"),
    );

    ok(&sjc(&d, &["trig.sjb", "-o", "api.h", "--name", "model"]));
    let h = std::fs::read_to_string(d.join("api.h")).unwrap();
    assert!(h.starts_with("/* api.h: the kernels in model.o"), "{}", h);
    assert!(h.contains("#define MODEL_COUNT_STATES 3"));
    assert!(!d.join("api.o").exists() && !d.join("model.o").exists());

    ok(&sjc(&d, &["trig.sjb", "--no-header"]));
    assert!(d.join("trig.o").exists() && !d.join("trig.h").exists());

    let out = sjc(&d, &["--emit", "info", "trig.sjb"]);
    ok(&out);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("3 states, 0 parameters, 3 outputs"),
        "{}",
        text
    );
    assert!(text.contains("functions:  cos, exp, sin"), "{}", text);
    assert!(text.contains("opt level 2"), "{}", text);
}

#[test]
fn errors() {
    let d = dir("errors");
    let code = |args: &[&str]| sjc(&d, args).status.code();

    assert_eq!(code(&[]), Some(2));
    assert_eq!(code(&["a.sjb", "-O3"]), Some(2));
    assert_eq!(code(&["--help"]), Some(0));
    assert_eq!(code(&["--version"]), Some(0));
    assert_eq!(code(&["missing.sjb"]), Some(1));

    std::fs::write(d.join("junk.sjb"), b"not a model at all").unwrap();
    let out = sjc(&d, &["junk.sjb"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not a saved symjit application"));

    save(
        &compile(&trig(), config(CompilerType::Native, 2)),
        &d.join("t.sjb"),
    );
    assert_eq!(
        code(&["t.sjb", "--emit", "wasm", "--target", "x86_64"]),
        Some(1)
    );
    assert_eq!(code(&["t.sjb", "--target", "wasm32", "-o", "t.o"]), Some(1));
    assert_eq!(code(&["t.sjb", "--target", "riscv64"]), Some(1));
    assert_eq!(code(&["t.sjb", "-o", "t.sjb", "--emit", "obj"]), Some(1));
    // the kernels are named after the output, which must then be a C identifier
    let out = sjc(&d, &["t.sjb", "-o", "my-model.o"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--name"));
}

/// A C program that runs the kernel `name` of a Rust source (its inputs are the parameters,
/// its outputs go to `mem`) at `points`, one line of %a outputs per point.
fn param_driver(d: &Path, name: &str, points: &[Vec<f64>], extra: &str) {
    let up = name.to_uppercase();
    let rows: Vec<String> = points
        .iter()
        .map(|p| {
            format!(
                "{{{}}}",
                p.iter()
                    .map(|x| format!("{:?}", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    let c = format!(
        "#include <math.h>\n#include <stdio.h>\n#include \"{name}.h\"\n{extra}\n\
         static const double P[][{up}_COUNT_PARAMS] = {{{}}};\n\
         int main(void) {{\n\
         \x20 for (int k = 0; k < {}; k++) {{\n\
         \x20   double mem[{up}_MEM_SIZE] = {{0}};\n\
         \x20   if ({name}(mem, NULL, 0, P[k]) != 0) return 1;\n\
         \x20   for (int j = 0; j < {up}_COUNT_OBS; j++) printf(\"%a \", mem[j]);\n\
         \x20   printf(\"\\n\");\n\
         \x20 }}\n\
         \x20 return 0;\n}}\n",
        rows.join(", "),
        points.len()
    );
    std::fs::write(d.join(format!("{name}_main.c")), c).unwrap();
}

fn run_param_driver(d: &Path, name: &str, points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    run_driver_with(d, name, points, "", false)
}

/// The C program of `param_driver`, with `extra` C code (e.g. the callbacks), built for this
/// machine or for AArch64 (run under qemu).
fn run_driver_with(
    d: &Path,
    name: &str,
    points: &[Vec<f64>],
    extra: &str,
    arm: bool,
) -> Vec<Vec<f64>> {
    param_driver(d, name, points, extra);
    let exe = d.join(format!("{name}_main"));
    let mut cc = if arm {
        let mut c = Command::new("aarch64-linux-gnu-gcc");
        c.arg("-static");
        c
    } else {
        Command::new("cc")
    };
    let status = cc
        .args([
            "-Wall",
            "-Werror",
            &format!("{name}_main.c"),
            &format!("{name}.o"),
            "-lm",
            "-o",
        ])
        .arg(&exe)
        .current_dir(d)
        .status()
        .unwrap();
    assert!(status.success());
    let run = if arm {
        Command::new("qemu-aarch64").arg(&exe).output().unwrap()
    } else {
        Command::new(&exe).output().unwrap()
    };
    assert!(run.status.success());
    parse_hex(&run.stdout)
}

/// The C callbacks of the corpus' kernels.
const CALLBACKS_C: &str = "double damp(double x) { return x / (1.0 + fabs(x)); }\n\
                           double length(double x, double y) { return hypot(x, y); }\n";

fn corpus_dir(name: &str) -> PathBuf {
    let d = dir(name);
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/kernels/corpus.rs"),
        d.join("kernels.rs"),
    )
    .unwrap();
    d
}

#[cfg(all(feature = "obj", target_arch = "x86_64", target_os = "linux"))]
#[test]
fn rust_kernels_with_calls_as_objects() {
    if !have("cc") {
        return;
    }
    let d = corpus_dir("rust-calls");
    let compile = |kernel: &str, out: &str| {
        ok(&sjc(
            &d,
            &[
                "kernels.rs",
                "--kernel",
                kernel,
                "-fno-fast-math",
                "-o",
                out,
            ],
        ));
    };

    // a kernel and a callback in a kernel loop: the callee is in the object, the callback is
    // the program's
    compile("iterate", "it.o");
    let h = std::fs::read_to_string(d.join("it.h")).unwrap();
    assert!(
        h.contains("double damp(double x0);") && h.contains("int it__step("),
        "{h}"
    );
    let points: Vec<Vec<f64>> = [(0.3, 5.0), (-1.2, 12.0), (2.5, 3.0)]
        .iter()
        .map(|&(c, n)| vec![c, n])
        .collect();
    let got = run_driver_with(
        &d,
        "it",
        &points,
        "double damp(double x) { return x / (1.0 + fabs(x)); }",
        false,
    );
    for (p, g) in points.iter().zip(&got) {
        let (x, sum) = corpus::iterate(p[0], p[1]);
        assert_eq!(g, &vec![x, sum], "iterate{:?}", p);
    }

    // chains: an inline helper calling a kernel that calls a kernel, and a recursive kernel
    compile("fib_path", "fp.o");
    let points: Vec<Vec<f64>> = [(0.0, 1.0), (5.0, 0.5), (8.0, 0.25)]
        .iter()
        .map(|&(n, s)| vec![n, s])
        .collect();
    let got = run_param_driver(&d, "fp", &points);
    for (p, g) in points.iter().zip(&got) {
        assert_eq!(g, &vec![corpus::fib_path(p[0], p[1])], "fib_path{:?}", p);
    }

    // a complex kernel calling a complex kernel, and a real kernel compiled as a complex one
    compile("orbit", "ob.o");
    let cs = [(0.25, 0.1), (-0.5, 0.3), (0.1, -0.45)];
    let points: Vec<Vec<f64>> = cs.iter().map(|&(re, im)| vec![re, im, 40.0, 0.0]).collect();
    let got = run_param_driver(&d, "ob", &points);
    for (&(re, im), g) in cs.iter().zip(&got) {
        let (n, z) = corpus::orbit(symjit::Complex::new(re, im), 40.0);
        close(&[g.clone()], &[vec![n, 0.0, z.re, z.im]], 1e-12);
    }
    compile("spiral", "sp.o");
    let points = vec![vec![0.5, 0.25, 6.0, 0.0], vec![-1.0, 2.0, 3.0, 0.0]];
    let got = run_param_driver(&d, "sp", &points);
    for (p, g) in points.iter().zip(&got) {
        let w = corpus::spiral(symjit::Complex::new(p[0], p[1]), p[2]);
        close(&[g.clone()], &[vec![w.re, w.im]], 1e-12);
    }

    // a callback of two arguments in a complex kernel (with the real parts)
    compile("callback_in_complex", "cc.o");
    let points = vec![vec![3.0, 4.0], vec![-0.5, 2.0]];
    let got = run_driver_with(&d, "cc", &points, CALLBACKS_C, false);
    for (p, g) in points.iter().zip(&got) {
        let z = corpus::callback_in_complex(symjit::Complex::new(p[0], p[1]));
        close(&[g.clone()], &[vec![z.re, z.im]], 1e-15);
    }

    // a callback named like a C math function would link to it
    std::fs::write(
        d.join("clash.rs"),
        "#[kernel(callback)]\nfn hypot(x: f64, y: f64) -> f64 { x }\n\
         #[kernel]\nfn f(x: f64) -> f64 { hypot(x, x) }\n",
    )
    .unwrap();
    let out = sjc(&d, &["clash.rs", "-o", "clash.o"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be called from an object file"));

    // a kernel with calls cannot be saved (the callees and callbacks are not in the file)
    let out = sjc(&d, &["kernels.rs", "--kernel", "iterate", "--emit", "sjb"]);
    assert_eq!(out.status.code(), Some(1));
}

/// Complex functions (`exp`, `sin`, `ln`, ...) call the C library's C99 functions: within
/// rounding of `num_complex`'s.
#[cfg(all(feature = "obj", target_arch = "x86_64", target_os = "linux"))]
#[test]
fn rust_kernels_with_complex_functions_as_objects() {
    if !have("cc") {
        return;
    }
    let d = corpus_dir("rust-complex-functions");
    ok(&sjc(
        &d,
        &["kernels.rs", "--kernel", "complex_ops", "-o", "co.o"],
    ));
    let h = std::fs::read_to_string(d.join("co.h")).unwrap();
    assert!(
        h.contains("cexp") && h.contains("csin") && h.contains("clog"),
        "{h}"
    );
    let zw = [(0.5, 0.25, -1.0, 0.75), (1.5, -0.5, 0.3, 2.0)];
    let points: Vec<Vec<f64>> = zw.iter().map(|&(a, b, c, e)| vec![a, b, c, e]).collect();
    let got = run_param_driver(&d, "co", &points);
    for (p, g) in points.iter().zip(&got) {
        let (a, b, c) = corpus::complex_ops(
            symjit::Complex::new(p[0], p[1]),
            symjit::Complex::new(p[2], p[3]),
        );
        close(
            &[g.clone()],
            &[vec![a.re, a.im, b.re, b.im, c.re, c.im]],
            1e-13,
        );
    }

    // real functions of a complex kernel's real parameter (`t.cos()`)
    ok(&sjc(
        &d,
        &["kernels.rs", "--kernel", "rotate", "-o", "rot.o"],
    ));
    let points = vec![vec![1.0, 2.0, 0.5, 0.0], vec![-1.0, 0.25, 2.0, 0.0]];
    let got = run_param_driver(&d, "rot", &points);
    for (p, g) in points.iter().zip(&got) {
        let w = corpus::rotate(symjit::Complex::new(p[0], p[1]), p[2]);
        close(&[g.clone()], &[vec![w.re, w.im]], 1e-13);
    }
}

#[cfg(all(feature = "obj", target_arch = "x86_64", target_os = "linux"))]
#[test]
fn rust_kernels_with_calls_as_aarch64_objects() {
    if !have("aarch64-linux-gnu-gcc") || !have("qemu-aarch64") {
        return;
    }
    let d = corpus_dir("rust-calls-arm");
    for (kernel, out) in [("iterate", "ait.o"), ("fib_path", "afp.o")] {
        ok(&sjc(
            &d,
            &[
                "kernels.rs",
                "--kernel",
                kernel,
                "-fno-fast-math",
                "--target",
                "aarch64-linux",
                "-o",
                out,
            ],
        ));
    }
    let points = vec![vec![0.3, 5.0], vec![-1.2, 12.0]];
    let got = run_driver_with(&d, "ait", &points, CALLBACKS_C, true);
    for (p, g) in points.iter().zip(&got) {
        let (x, sum) = corpus::iterate(p[0], p[1]);
        assert_eq!(g, &vec![x, sum], "iterate{:?}", p);
    }
    let points = vec![vec![5.0, 0.5], vec![8.0, 0.25]];
    let got = run_driver_with(&d, "afp", &points, "", true);
    for (p, g) in points.iter().zip(&got) {
        // AArch64's sqrt is exact, as x86-64's
        assert_eq!(g, &vec![corpus::fib_path(p[0], p[1])], "fib_path{:?}", p);
    }
}

#[cfg(all(feature = "obj", target_arch = "x86_64", target_os = "linux"))]
#[test]
fn rust_kernels_as_objects() {
    if !have("cc") {
        return;
    }
    let d = dir("rust");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/kernels/corpus.rs"),
        d.join("kernels.rs"),
    )
    .unwrap();

    // a real kernel with a loop: the plain Rust function's bits (no fused multiply-adds)
    ok(&sjc(
        &d,
        &[
            "kernels.rs",
            "--kernel",
            "newton_sqrt",
            "-fno-fast-math",
            "-o",
            "ns.o",
        ],
    ));
    let points: Vec<Vec<f64>> = [0.5, 2.0, 10.0, 1e-3].iter().map(|&a| vec![a]).collect();
    let got = run_param_driver(&d, "ns", &points);
    for (p, g) in points.iter().zip(&got) {
        let (x, steps) = corpus::newton_sqrt(p[0]);
        assert_eq!(g, &vec![x, steps], "newton_sqrt({})", p[0]);
    }

    // a complex kernel: inputs and outputs are (re, im) pairs
    ok(&sjc(
        &d,
        &[
            "kernels.rs",
            "--kernel",
            "mandelbrot",
            "-fno-fast-math",
            "-o",
            "mb.o",
        ],
    ));
    let cs = [(-0.75, 0.1), (0.3, 0.5), (-2.0, 0.0), (0.25, 0.0)];
    let points: Vec<Vec<f64>> = cs.iter().map(|&(re, im)| vec![re, im]).collect();
    let got = run_param_driver(&d, "mb", &points);
    for (&(re, im), g) in cs.iter().zip(&got) {
        let n = corpus::mandelbrot(symjit::Complex::new(re, im));
        assert_eq!(g, &vec![n, 0.0], "mandelbrot({re}, {im})");
    }

    // through a saved application: the same results
    ok(&sjc(
        &d,
        &[
            "kernels.rs",
            "--kernel",
            "newton_sqrt",
            "-fno-fast-math",
            "--emit",
            "sjb",
        ],
    ));
    assert!(d.join("newton_sqrt.sjb").exists());
    ok(&sjc(&d, &["newton_sqrt.sjb", "-o", "ns2.o"]));
    let points: Vec<Vec<f64>> = [0.5, 2.0, 10.0, 1e-3].iter().map(|&a| vec![a]).collect();
    assert_eq!(
        run_param_driver(&d, "ns2", &points),
        run_param_driver(&d, "ns", &points)
    );

    // the kernel is chosen with --kernel, and named after it by default
    let out = sjc(&d, &["kernels.rs"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("choose one with --kernel"));
    ok(&sjc(&d, &["kernels.rs", "--kernel", "fib_iter"]));
    assert!(d.join("fib_iter.o").exists() && d.join("fib_iter.h").exists());
    // the options are for Rust input only, and errors give the position
    assert_eq!(sjc(&d, &["ns2.sjb", "-O3"]).status.code(), Some(2));
    std::fs::write(
        d.join("bad.rs"),
        "#[kernel]\nfn f(x: f64) -> f64 {\n    x % 2.0\n}\n",
    )
    .unwrap();
    let out = sjc(&d, &["bad.rs"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("bad.rs:3:5:"));
}

#[cfg(feature = "wasm")]
#[test]
fn rust_kernels_as_wasm() {
    let d = dir("rust-wasm");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/kernels/corpus.rs"),
        d.join("kernels.rs"),
    )
    .unwrap();
    ok(&sjc(
        &d,
        &["kernels.rs", "--kernel", "fib_iter", "-o", "fib.wasm"],
    ));
    assert_eq!(&std::fs::read(d.join("fib.wasm")).unwrap()[..4], b"\0asm");
    if !have("node") {
        return;
    }
    let runner = Path::new(env!("CARGO_MANIFEST_DIR")).join("../python/tests/wasm_runner.mjs");
    let req = format!(
        "{{\"module\": {:?}, \"mode\": \"params\", \"count_states\": 0, \"count_obs\": 1, \
         \"mem_size\": 2, \"points\": [[10.0], [20.0], [0.0]], \"params\": []}}",
        d.join("fib.wasm").to_str().unwrap()
    );
    let mut child = Command::new("node")
        .arg(runner)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(req.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("\"obs\": [[55], [6765], [0]]") || text.contains("\"obs\":[[55],[6765],[0]]"),
        "{}",
        text
    );
}
