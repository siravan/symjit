// End-to-end tests of object files holding compiled kernels: models are compiled with
// the Rust API, written as objects in the host's format, linked into a C program with
// `cc ... -lm` and run; the results must match the JIT-compiled kernels. They run on
// Linux and macOS, x86-64 and AArch64, with a C compiler. `arm_kernels_cross_compiled`
// checks AArch64 objects (`ty = "arm"`) on any host.

use super::super::compiler::Compiler;
use super::super::config::{Config, COMPACT, COMPLEX, CSE, FASTMATH, FAST_COMPLEX};
use super::super::expr::Expr;
use super::super::runnable::Application;
use super::{Format, Target};

fn host_ok() -> bool {
    cfg!(all(
        any(target_os = "linux", target_os = "macos"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))
}

// explicit options (Config::default() would read a symjit.toml in the working directory)
fn config(ty: &str, extra: u32) -> Config {
    Config::from_name(ty, CSE | FASTMATH | COMPACT | (2 << 8) | extra).unwrap()
}

struct Model {
    app: Application,
    points: Vec<Vec<f64>>,
    params: Vec<f64>,
}

/// Writes the object for `m`, links a C driver that evaluates every point with the
/// scalar kernel (direct mode) and, if present, the fast kernel, and returns, per point,
/// (status, outputs, fast output). None if there is no C compiler.
fn run_object(m: &mut Model, name: &str) -> Option<Vec<(i32, Vec<f64>, Option<f64>)>> {
    let target = m.app.object_target("").unwrap();
    let obj = m.app.object(name, target).unwrap();
    let fast = obj
        .functions
        .iter()
        .any(|f| f.name == format!("{}_fast", name));

    let dir = std::env::temp_dir().join(format!("symjit-kernel-{}-{}", name, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join(format!("{}.o", name));
    obj.write(obj_path.to_str().unwrap()).unwrap();

    let (cs, co) = (m.app.count_states, m.app.count_obs);
    let mem_size = m.app.prog.mem_size();
    let fmt = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{:.17e}", x))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let mut c = String::new();
    c += "#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n";
    c += &format!(
        "int {}(double *mem, const void *states, size_t idx, const double *params);\n",
        name
    );
    if fast {
        let args = vec!["double"; cs].join(", ");
        c += &format!("double {}_fast({});\n", name, args);
    }
    c += "static unsigned long long bits(double x) { unsigned long long u; memcpy(&u, &x, 8); return u; }\n";
    c += "int main(void) {\n";
    c += &format!(
        "    static const double points[{}][{}] = {{ {} }};\n",
        m.points.len(),
        cs.max(1),
        m.points
            .iter()
            .map(|p| format!("{{ {} }}", fmt(p)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    c += &format!(
        "    static const double params[{}] = {{ {} }};\n",
        m.params.len() + 1,
        fmt(&[m.params.clone(), vec![0.0]].concat())
    );
    c += &format!("    double mem[{}];\n", mem_size);
    c += &format!("    for (int k = 0; k < {}; k++) {{\n", m.points.len());
    c += "        memset(mem, 0, sizeof(mem));\n";
    c += &format!("        memcpy(mem, points[k], {} * sizeof(double));\n", cs);
    c += &format!("        printf(\"%d\", {}(mem, NULL, 0, params));\n", name);
    c += &format!(
        "        for (int i = 0; i < {}; i++) printf(\" %016llx\", bits(mem[{} + i]));\n",
        co, cs
    );
    if fast {
        let args: Vec<String> = (0..cs).map(|i| format!("points[k][{}]", i)).collect();
        c += &format!(
            "        printf(\" %016llx\", bits({}_fast({})));\n",
            name,
            args.join(", ")
        );
    }
    c += "        printf(\"\\n\");\n    }\n    return 0;\n}\n";

    let c_path = dir.join("main.c");
    std::fs::write(&c_path, c).unwrap();
    let exe = dir.join("main");
    let cc = std::process::Command::new("cc")
        .args([c_path.as_os_str(), obj_path.as_os_str()])
        .args(["-lm", "-o"])
        .arg(&exe)
        .output()
        .ok()?;
    assert!(
        cc.status.success() && cc.stderr.is_empty(),
        "cc:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let out = std::process::Command::new(&exe).output().unwrap();
    assert!(out.status.success());
    let _ = std::fs::remove_dir_all(&dir);

    let parse = |w: &str| f64::from_bits(u64::from_str_radix(w, 16).unwrap());
    Some(
        String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|line| {
                let words: Vec<&str> = line.split(' ').collect();
                let obs = words[1..1 + co].iter().map(|w| parse(w)).collect();
                (
                    words[0].parse().unwrap(),
                    obs,
                    fast.then(|| parse(words[1 + co])),
                )
            })
            .collect(),
    )
}

// `rel` relative tolerance: 0 means bit-identical (or both NaN)
fn same(a: f64, b: f64, rel: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= rel * a.abs().max(b.abs())
}

/// Checks the object against the JIT at every point.
fn check(m: &mut Model, name: &str, rel: f64) {
    if !host_ok() {
        return;
    }
    let Some(results) = run_object(m, name) else {
        eprintln!("{}: no C compiler (`cc`), skipped", name);
        return;
    };

    for i in 0..m.params.len() {
        m.app.params[i] = m.params[i];
    }
    for (p, (status, obs, fast)) in m.points.iter().zip(results) {
        let want = m.app.call(p);
        assert_eq!(status, 0);
        for (k, (a, b)) in obs.iter().zip(want.iter()).enumerate() {
            assert!(
                same(*a, *b, rel),
                "{}: output {} at {:?}: object {} JIT {}",
                name,
                k,
                p,
                a,
                b
            );
        }
        if let Some(f) = fast {
            assert!(
                same(f, want[0], rel),
                "{}_fast at {:?}: object {} JIT {}",
                name,
                p,
                f,
                want[0]
            );
        }
    }
}

fn real_model(cfg: Config) -> Model {
    let (x, y) = (Expr::var("x"), Expr::var("y"));
    let one = Expr::from(1.0);
    let two = Expr::from(2.0);
    let obs = vec![
        &(&x * &y) + &x.sin(),
        &(&(&x * &x).neg()).exp() / &(&one + &(&y * &y)),
        x.pow(&y),
        Expr::binary("atan2", &y, &x),
        (&one + &(&x * &x)).ln(),
        (&two + &(&y * &y)).log10(),
        x.lt(&y).ifelse(&x.cos(), &(&y * &two)),
        (&x * &y).tanh(),
    ];
    let app = Compiler::with_config(cfg).compile(&[x, y], &obs).unwrap();
    let points = vec![
        vec![0.5, 2.0],
        vec![1.25, -0.75],
        vec![3.0, 0.5],
        vec![2.0, 3.5],
        vec![0.0, 0.0],
    ];
    Model {
        app,
        points,
        params: vec![],
    }
}

#[test]
fn imports_use_c_library_names() {
    let mut m = real_model(config("native", 0));
    let target = m.app.object_target("").unwrap();
    let obj = m.app.object("model", target).unwrap();
    // ln -> log, log (decimal) -> log10, power -> pow
    assert_eq!(
        obj.undefined_symbols(),
        ["atan2", "cos", "exp", "log", "log10", "pow", "sin", "tanh"]
    );
    assert_eq!(obj.functions.len(), 1); // 8 outputs: no fast kernel
}

#[test]
fn real_model_matches_the_jit() {
    let mut m = real_model(config("native", 0));
    check(&mut m, "real_model", 0.0);
    // the JIT kernels still work after compiling the object kernels
    let mut fresh = real_model(config("native", 0));
    assert_eq!(m.app.call(&[0.5, 2.0]), fresh.app.call(&[0.5, 2.0]));
}

#[test]
fn opt_levels_match_the_jit() {
    for opt in 0..=3u32 {
        let cfg = Config::from_name("native", CSE | FASTMATH | COMPACT | (opt << 8)).unwrap();
        let mut m = real_model(cfg);
        check(&mut m, &format!("opt{}", opt), 0.0);
    }
}

#[test]
fn sse_kernels_match_the_jit() {
    if !cfg!(target_arch = "x86_64") {
        return;
    }
    let mut m = real_model(config("amd-sse", 0));
    check(&mut m, "sse_model", 0.0);
}

#[test]
fn fast_kernel_is_exported() {
    let (x, y) = (Expr::var("x"), Expr::var("y"));
    let obs = vec![&(&x.sin() * &y) + &(&x / &(&Expr::from(1.0) + &(&y * &y)))];
    let app = Compiler::with_config(config("native", 0))
        .compile(&[x, y], &obs)
        .unwrap();
    let mut m = Model {
        app,
        points: vec![vec![0.5, 2.0], vec![-1.0, 0.25], vec![3.0, -1.5]],
        params: vec![],
    };
    let target = m.app.object_target("").unwrap();
    let obj = m.app.object("fast_model", target).unwrap();
    let names: Vec<&str> = obj.functions.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["fast_model", "fast_model_fast"]);
    check(&mut m, "fast_model", 0.0);
}

#[test]
fn parameters() {
    let (x, p) = (Expr::var("x"), Expr::var("p"));
    let obs = vec![&(&x * &p) + &p.sin(), x.exp()];
    let app = Compiler::with_config(config("native", 0))
        .compile_params(&[x], &obs, &[p])
        .unwrap();
    let mut m = Model {
        app,
        points: vec![vec![0.5], vec![-2.0]],
        params: vec![0.75],
    };
    check(&mut m, "param_model", 0.0);
}

#[test]
fn complex_model_matches_the_jit() {
    // complex states are (re, im) pairs; fast complex uses the packed-complex generator
    for (extra, name) in [
        (COMPLEX | FAST_COMPLEX, "complex_fast"),
        (COMPLEX, "complex_slow"),
    ] {
        let (x, y) = (Expr::var("x"), Expr::var("y"));
        let one = Expr::from(1.0);
        let obs = vec![&(&x * &y) + &x, &x / &(&one + &(&y * &y)), &(&x * &x) - &y];
        let app = Compiler::with_config(config("native", extra))
            .compile(&[x, y], &obs)
            .unwrap();
        let points = vec![vec![0.5, 0.25, -1.0, 2.0], vec![1.5, -0.5, 0.75, 0.0]];
        let mut m = Model {
            app,
            points,
            params: vec![],
        };
        check(&mut m, name, 0.0);
    }
}

#[test]
fn functions_outside_libm_are_rejected() {
    let x = Expr::var("x");

    let mut app = Compiler::with_config(config("native", 0))
        .compile(&[x.clone()], &[x.csc()])
        .unwrap();
    let target = app.object_target("").unwrap();
    let err = app.object("f", target).err().unwrap().to_string();
    assert!(
        err.contains("`csc` is not a C math library function"),
        "{}",
        err
    );

    // complex functions
    let mut app = Compiler::with_config(config("native", COMPLEX | FAST_COMPLEX))
        .compile(&[x.clone()], &[x.sin()])
        .unwrap();
    let target = app.object_target("").unwrap();
    let err = app.object("f", target).err().unwrap().to_string();
    assert!(err.contains("complex function"), "{}", err);
}

/// With SYMJIT_LINKER_SAMPLES set to a directory, writes the kernels of `real_model` as
/// model_elf.o and model_macho.o (the host's architecture), and model_arm64_elf.o and
/// model_arm64_macho.o (AArch64), for inspection with external tools.
#[test]
fn write_sample_kernel() {
    let Ok(dir) = std::env::var("SYMJIT_LINKER_SAMPLES") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    for (ty, format, file) in [
        ("native", "elf", "model_elf.o"),
        ("native", "macho", "model_macho.o"),
        ("arm", "elf", "model_arm64_elf.o"),
        ("arm", "macho", "model_arm64_macho.o"),
    ] {
        let mut m = real_model(config(ty, 0));
        let target = m.app.object_target(format).unwrap();
        let obj = m.app.object("model", target).unwrap();
        obj.write(&format!("{}/{}", dir, file)).unwrap();
    }
}

// a header compiles as C and as C++ without warnings
fn check_header(name: &str, write: impl Fn(&str)) {
    if !host_ok() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("symjit-header-{}-{}", name, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join(name);
    write(base.to_str().unwrap());
    let header = std::fs::read_to_string(format!("{}.h", base.display())).unwrap();

    for (compiler, lang) in [("cc", "c"), ("c++", "c++")] {
        let src = dir.join(format!("use.{}", if lang == "c" { "c" } else { "cpp" }));
        std::fs::write(
            &src,
            format!(
                "#include \"{}.h\"\n#include \"{}.h\"\nint f(void);\n",
                name, name
            ),
        )
        .unwrap();
        let out = std::process::Command::new(compiler)
            .args(["-fsyntax-only", "-Wall", "-Wextra", "-Werror", "-x", lang])
            .arg(&src)
            .current_dir(&dir)
            .output();
        let Ok(out) = out else {
            eprintln!("check_header: no `{}`, skipped", compiler);
            continue;
        };
        assert!(
            out.status.success(),
            "{} {}:\n{}\n{}",
            compiler,
            name,
            String::from_utf8_lossy(&out.stderr),
            header
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn headers_compile_cleanly() {
    // a model with a fast kernel, one with parameters and several outputs, and a complex one
    check_header("hfast", |path| {
        let (x, y) = (Expr::var("x"), Expr::var("y"));
        let obs = vec![&x.sin() * &y];
        let mut app = Compiler::with_config(config("native", 0))
            .compile(&[x, y], &obs)
            .unwrap();
        app.write_obj(path).unwrap();
    });
    check_header("hparams", |path| {
        let (x, p) = (Expr::var("x"), Expr::var("p"));
        let obs = vec![&x * &p, x.exp()];
        let mut app = Compiler::with_config(config("native", 0))
            .compile_params(&[x], &obs, &[p])
            .unwrap();
        app.write_obj(path).unwrap();
    });
    check_header("hcomplex", |path| {
        let (x, y) = (Expr::var("x"), Expr::var("y"));
        let obs = vec![&x * &y];
        let mut app = Compiler::with_config(config("native", COMPLEX))
            .compile(&[x, y], &obs)
            .unwrap();
        app.write_obj(path).unwrap();
    });
}

#[test]
fn write_obj_names_and_formats() {
    let dir = std::env::temp_dir().join(format!("symjit-names-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let x = Expr::var("x");
    let mut app = Compiler::with_config(config("native", 0))
        .compile(&[x.clone()], &[x.sin()])
        .unwrap();

    // a path: files in that directory, kernels named after the last component
    let base = dir.join("kernel_one");
    let target = app.object_target("macho").unwrap();
    app.write_obj_for(base.to_str().unwrap(), target).unwrap();
    let bytes = std::fs::read(format!("{}.o", base.display())).unwrap();
    assert_eq!(&bytes[..4], &0xfeedfacfu32.to_le_bytes());
    let header = std::fs::read_to_string(format!("{}.h", base.display())).unwrap();
    let arch = if cfg!(target_arch = "aarch64") {
        "AArch64"
    } else {
        "x86-64"
    };
    assert!(
        header.contains("int kernel_one(double *mem")
            && header.contains(&format!("{} Mach-O", arch))
    );

    assert!(app
        .write_obj(dir.join("not-an-identifier").to_str().unwrap())
        .is_err());
    assert!(app.object_target("coff").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

// AArch64 kernels compiled on any host (`ty = "arm"`): the object calls the C library with
// `bl` + relocations exactly where the JIT kernel calls its own functions (`blr x9`).
#[test]
fn arm_kernels_cross_compiled() {
    let bl = 0x9400_0000u32;
    let blr_x9 = 0xd63f_0120u32;
    let words = |code: &[u8]| -> Vec<u32> {
        code.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };

    let mut m = real_model(config("arm", 0));
    let jit_calls = words(&m.app.dumps())
        .iter()
        .filter(|w| **w == blr_x9)
        .count();
    for format in [Format::Elf, Format::MachO] {
        let obj = m
            .app
            .object("model", Target::new(format, super::Arch::Aarch64))
            .unwrap();
        assert_eq!(
            obj.undefined_symbols(),
            ["atan2", "cos", "exp", "log", "log10", "pow", "sin", "tanh"]
        );
        assert_eq!(obj.relocations.len(), jit_calls);
        for r in obj.relocations.iter() {
            let w = u32::from_le_bytes(obj.code[r.offset..r.offset + 4].try_into().unwrap());
            assert_eq!(w, bl, "relocation against {} at {}", r.symbol, r.offset);
        }
        // object mode emits no `blr x9` through the function table, and no `adrp` (which
        // assumes the code starts at a page boundary)
        let ws = words(&obj.code);
        assert!(!ws.contains(&blr_x9));
        assert!(
            !ws.iter().any(|w| w & 0x9f00_0000 == 0x9000_0000),
            "adrp in object code"
        );

        // constants are PC-relative literal loads (`ldr d, label`) of the model's constants
        let mut loaded = Vec::new();
        for (k, w) in ws.iter().enumerate() {
            if w & 0xff00_0000 == 0x5c00_0000 {
                let imm = (((w >> 5) & 0x7ffff) as i32) << 13 >> 13; // sign-extended imm19
                let target = (4 * k as i64 + 4 * imm as i64) as usize;
                assert!(
                    target % 4 == 0 && target + 8 <= obj.code.len(),
                    "literal at {} -> {}",
                    4 * k,
                    target
                );
                loaded.push(f64::from_le_bytes(
                    obj.code[target..target + 8].try_into().unwrap(),
                ));
            }
        }
        for c in [1.0, 2.0] {
            assert!(loaded.contains(&c), "{} is not loaded ({:?})", c, loaded);
        }
        obj.build().unwrap();
    }

    // the architecture must be the model's
    let err = m
        .app
        .object("model", Target::new(Format::Elf, super::Arch::X86_64))
        .err()
        .unwrap();
    assert!(err.to_string().contains("compiled for Arm"), "{}", err);

    // a fast kernel, and the packed-complex generator
    let (x, y) = (Expr::var("x"), Expr::var("y"));
    let mut app = Compiler::with_config(config("arm", 0))
        .compile(&[x.clone(), y.clone()], &[&x.sin() * &y])
        .unwrap();
    let target = app.object_target("macho").unwrap();
    assert_eq!(target, Target::new(Format::MachO, super::Arch::Aarch64));
    let obj = app.object("f", target).unwrap();
    assert_eq!(
        obj.functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["f", "f_fast"]
    );
    assert_eq!(obj.relocations.len(), 2); // sin, in each kernel

    let mut app = Compiler::with_config(config("arm", COMPLEX | FAST_COMPLEX))
        .compile(&[x.clone(), y.clone()], &[&(&x * &y) + &x])
        .unwrap();
    let obj = app
        .object("c", Target::new(Format::MachO, super::Arch::Aarch64))
        .unwrap();
    assert!(obj.relocations.is_empty());

    // functions outside the C library
    let mut app = Compiler::with_config(config("arm", 0))
        .compile(&[x.clone()], &[x.csc()])
        .unwrap();
    let err = app
        .object("f", Target::new(Format::MachO, super::Arch::Aarch64))
        .err()
        .unwrap();
    assert!(
        err.to_string()
            .contains("`csc` is not a C math library function"),
        "{}",
        err
    );
}
