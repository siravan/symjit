//! `Application::dump(path, "scalar-asm")` (also `"simd-asm"`, `"fast-asm"`) writes a kernel as
//! assembly text that GNU `as` assembles back to the same bytes; `"scalar-asm-hex"` (...) is an
//! objdump-like listing with offsets and bytes (rust/disasm). The kernels here are compiled for
//! RISC-V, AArch64 and x86-64 on any host (the code is never run; x86-64 kernels use the CPU's
//! features: AVX and AVX-512 where the host has them, SSE elsewhere), so the listings are checked
//! everywhere: every instruction before the data area decodes, and labels and label references
//! are named. Where the binutils are installed (`riscv64-linux-gnu-as`, `aarch64-linux-gnu-as`,
//! `x86_64-linux-gnu-as`, or the host's own, with `ld` and `objcopy`), each listing is assembled
//! and linked at address 0, and its bytes must equal the kernel's.
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use symjit::{Application, Composer, Config, Slot, Translator};

/// a directory of its own for each call: the tests run in parallel
fn scratch() -> PathBuf {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("symjit-disasm-{}-{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// the test options with `ty` replaced, and complex numbers on or off
fn config_for(ty: &str, complex: bool) -> Config {
    let _ = base_config();
    let toml = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/symjit.toml"))
        .unwrap()
        .replace("ty = \"native\"", &format!("ty = \"{}\"", ty));
    let path = scratch().join("symjit.toml");
    std::fs::write(&path, toml).unwrap();
    let mut config = Config::from_name(path.to_str().unwrap(), 0).unwrap();
    config.set_complex(complex);
    config
}

/// out0 = p0 ? p0 p1 : p1 + 7.5;  out1 = exp(p0) + 1e6 p1
fn compile_real(config: Config) -> Result<Application> {
    let mut t = Translator::new(config);
    t.set_num_params(2);
    let (p0, p1) = (Slot::Param(0), Slot::Param(1));
    let c75 = t.append_constant(Complex::new(7.5, 0.0))?;
    let c1e6 = t.append_constant(Complex::new(1e6, 0.0))?;

    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(1), &[p0, p1], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_add(&Slot::Temp(2), &[p1, Slot::Const(c75)], 0)?;
    t.append_label(1)?;
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(1), &Slot::Temp(2))?;

    t.append_fun(&Slot::Temp(3), "exp", &[p0], false)?;
    t.append_mul(&Slot::Temp(4), &[p1, Slot::Const(c1e6)], 0)?;
    t.append_add(&Slot::Out(1), &[Slot::Temp(3), Slot::Temp(4)], 0)?;
    t.compile()
}

/// out = exp(p0 (0.2 + 0.3i)) + p1^-3 + p199 (complex; a parameter far from the base)
fn compile_complex(config: Config) -> Result<Application> {
    let mut t = Translator::new(config);
    t.set_num_params(200);
    let ca = t.append_constant(Complex::new(0.2, 0.3))?;
    t.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(ca)], 0)?;
    t.append_fun(&Slot::Temp(1), "exp", &[Slot::Temp(0)], false)?;
    t.append_pow(&Slot::Temp(2), &Slot::Param(1), -3, false)?;
    t.append_add(
        &Slot::Out(0),
        &[Slot::Temp(1), Slot::Temp(2), Slot::Param(199)],
        0,
    )?;
    t.compile()
}

/// the text `dump(.., what)` writes, or None if it refuses
fn dump(app: &mut Application, what: &str) -> Option<String> {
    let path = scratch().join(format!("{what}.s"));
    let ok = app.dump(path.to_str().unwrap(), what);
    let text = std::fs::read_to_string(&path).ok();
    assert_eq!(ok, text.is_some(), "{what}: dump returned {ok}");
    text
}

/// the kernel's bytes (`dump(.., "scalar")` ...)
fn bytes(app: &mut Application, kind: &str) -> Vec<u8> {
    let path = scratch().join(format!("{kind}.bin"));
    assert!(app.dump(path.to_str().unwrap(), kind));
    std::fs::read(&path).unwrap()
}

/// the instruction lines of the assembly form before the data area
fn code_lines(listing: &str) -> Vec<&str> {
    listing
        .lines()
        .take_while(|l| !l.starts_with(".L_data:"))
        .filter(|l| l.starts_with('\t') && !l.trim_start().starts_with('.'))
        .collect()
}

fn check_asm(listing: &str, arch: &str, kind: &str) {
    let ctx = format!("{arch} {kind}:\n{listing}");
    assert!(listing.contains(&format!("kernel, {arch},")), "{ctx}");
    for line in [
        format!("\t.globl symjit_{kind}"),
        format!("symjit_{kind}:"),
        ".L_self:".to_string(),
        ".L_data:".to_string(),
    ] {
        assert!(listing.lines().any(|l| l == line), "no {line}: {ctx}");
    }

    // no offsets or bytes, every instruction decoded (no raw words before the data)
    let code = code_lines(listing);
    assert!(code.len() > 10, "{ctx}");
    for line in listing.lines().take_while(|l| !l.starts_with(".L_data:")) {
        assert!(
            !line.contains(".word") && !line.contains(".inst") && !line.contains(".byte"),
            "undecoded: {line}\n{ctx}"
        );
        assert!(!line.contains(" <"), "an address: {line}\n{ctx}");
    }
    assert!(code.iter().any(|l| l.trim() == "ret"), "{ctx}");
    // the constant pool as .quads under their labels
    assert!(listing.contains("\n.L_const_0_:\n\t.quad 0x"), "{ctx}");
}

/// the binutils of `arch`, if installed: as, ld, objcopy
fn binutils(arch: &str) -> Option<[String; 3]> {
    let triple = match arch {
        "riscv64" => "riscv64-linux-gnu",
        "aarch64" => "aarch64-linux-gnu",
        "x86_64" => "x86_64-linux-gnu",
        _ => return None,
    };
    let native = cfg!(target_os = "linux") && std::env::consts::ARCH == arch;
    for prefix in [format!("{}-", triple), String::new()] {
        if prefix.is_empty() && !native {
            continue;
        }
        let tools = [
            format!("{}as", prefix),
            format!("{}ld", prefix),
            format!("{}objcopy", prefix),
        ];
        let ok = tools.iter().all(|t| {
            Command::new(t)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        });
        if ok {
            return Some(tools);
        }
    }
    None
}

fn run(cmd: &str, args: &[&str], dir: &Path) {
    let out = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{} {:?}:\n{}",
        cmd,
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

/// assembles and links the listing at address 0 and compares its bytes with `expected`;
/// false if the binutils are not installed
fn assembles_to(listing: &str, arch: &str, kind: &str, expected: &[u8]) -> bool {
    let Some([asm, ld, objcopy]) = binutils(arch) else {
        eprintln!("no binutils for {arch}: the listing is not assembled");
        return false;
    };
    let dir = scratch();
    std::fs::write(dir.join("k.s"), listing).unwrap();
    run(&asm, &["-o", "k.o", "k.s"], &dir);
    let entry = format!("symjit_{}", kind);
    run(&ld, &["-Ttext=0", "-e", &entry, "-o", "k.elf", "k.o"], &dir);
    run(
        &objcopy,
        &["-O", "binary", "-j", ".text", "k.elf", "k.bin"],
        &dir,
    );
    let mut got = std::fs::read(dir.join("k.bin")).unwrap();

    // the assembler may pad the section to its alignment (RISC-V with nops) after the kernel
    if got.len() > expected.len() && got.len() - expected.len() < 16 {
        got.truncate(expected.len());
    }

    if got != expected {
        let first = got
            .iter()
            .zip(expected)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(expected.len()));
        let hex = std::fs::read_to_string(dir.join("k.s")).unwrap();
        panic!(
            "{arch} {kind}: {} bytes assembled, {} expected; first difference at 0x{:x}\n{hex}",
            got.len(),
            expected.len(),
            first
        );
    }
    true
}

/// every kernel of `app` that `arch` has: the assembly form decodes, and assembles back
fn check_kernels(mut app: Application, arch: &str, kinds: &[&str]) {
    for kind in kinds {
        let Some(listing) = dump(&mut app, &format!("{kind}-asm")) else {
            continue;
        };
        check_asm(&listing, arch, kind);
        let expected = bytes(&mut app, kind);
        assembles_to(&listing, arch, kind, &expected);

        let hex = dump(&mut app, &format!("{kind}-asm-hex")).unwrap();
        assert!(hex.contains("\n@data:\n"), "{hex}");
        assert!(hex.contains(&format!("{} bytes", expected.len())), "{hex}");
    }
}

#[test]
fn riscv_real_kernels() {
    let mut app = compile_real(config_for("riscv", false)).unwrap();
    // RISC-V has no SIMD kernel
    assert!(dump(&mut app, "simd-asm").is_none());
    let listing = dump(&mut app, "scalar-asm").unwrap();
    // branches name their labels, label references show the constant they load
    assert!(listing.contains(", .L_main\n"), "{listing}");
    assert!(listing.contains("# .L_const_0_ = 7.5"), "{listing}");
    assert!(listing.contains("# .L_func_exp_\n"), "{listing}");
    check_kernels(app, "riscv64", &["scalar", "fast"]);
}

#[test]
fn riscv_complex_kernels() {
    let app = compile_complex(config_for("riscv", true)).unwrap();
    check_kernels(app, "riscv64", &["scalar"]);
}

#[test]
fn arm_real_kernels() {
    let mut app = compile_real(config_for("arm", false)).unwrap();
    let listing = dump(&mut app, "scalar-asm").unwrap();
    assert!(listing.contains("\tadrp x0, .L_const_0_\n"), "{listing}");
    assert!(listing.contains(", #:lo12:.L_const_0_]"), "{listing}");
    assert!(listing.contains("// .L_const_0_ = 7.5"), "{listing}");
    check_kernels(app, "aarch64", &["scalar", "simd", "fast"]);
}

#[test]
fn arm_complex_kernels() {
    let app = compile_complex(config_for("arm", true)).unwrap();
    check_kernels(app, "aarch64", &["scalar", "simd"]);
}

#[test]
fn arm_complexified_kernels() {
    // complex numbers lowered to real pairs (the real generators)
    let mut config = config_for("arm", true);
    config.set_fast_complex(false);
    let app = compile_complex(config).unwrap();
    check_kernels(app, "aarch64", &["scalar", "simd"]);
}

#[test]
fn listing_covers_the_kernel() {
    let mut app = compile_real(config_for("riscv", false)).unwrap();
    let size = bytes(&mut app, "scalar").len();
    let hex = dump(&mut app, "scalar-asm-hex").unwrap();
    let first = hex.lines().next().unwrap();
    assert!(first.contains(&format!("{} bytes", size)), "{hex}");
    // the hex form has an offset and the bytes on each line
    assert!(hex.lines().any(|l| l.starts_with("       0:  ")), "{hex}");
}

#[test]
fn x86_real_kernels() {
    let mut app = compile_real(config_for("amd", false)).unwrap();
    let listing = dump(&mut app, "scalar-asm").unwrap();
    assert!(listing.contains("\t.intel_syntax noprefix\n"), "{listing}");
    assert!(listing.contains("[rip+.L_const_0_]"), "{listing}");
    assert!(listing.contains("# .L_const_0_ = 7.5"), "{listing}");
    check_kernels(app, "x86_64", &["scalar", "simd", "fast"]);
}

#[test]
fn x86_complex_kernels() {
    let app = compile_complex(config_for("amd", true)).unwrap();
    check_kernels(app, "x86_64", &["scalar", "simd"]);

    let mut config = config_for("amd", true);
    config.set_fast_complex(false);
    let app = compile_complex(config).unwrap();
    check_kernels(app, "x86_64", &["scalar", "simd"]);
}

#[test]
fn x86_sse_kernels() {
    let app = compile_real(config_for("amd-sse", false)).unwrap();
    check_kernels(app, "x86_64", &["scalar", "fast"]);
    let app = compile_complex(config_for("amd-sse", true)).unwrap();
    check_kernels(app, "x86_64", &["scalar"]);
}

#[test]
fn x86_avx512_kernels() {
    // AVX-512 kernels where the host has AVX-512 (else the AVX or SSE ones)
    let mut config = config_for("amd", false);
    config.enable_simd512(true);
    check_kernels(compile_real(config).unwrap(), "x86_64", &["scalar", "simd"]);
    let mut config = config_for("amd", true);
    config.enable_simd512(true);
    check_kernels(
        compile_complex(config).unwrap(),
        "x86_64",
        &["scalar", "simd"],
    );
}
