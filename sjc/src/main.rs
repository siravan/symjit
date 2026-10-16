//! sjc, the Symjit compiler driver: compiles a saved symjit application (`.sjb`), or a
//! `#[kernel]` function of a Rust file (`source`), into a relocatable object file and C header
//! (feature `obj`) or a WebAssembly module (feature `wasm`). See `docs/SJC.md`.

// the helpers of the outputs left out of a build without `obj` or `wasm`
#![cfg_attr(
    not(all(feature = "obj", feature = "wasm")),
    allow(dead_code, unused_imports)
)]

mod args;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use sjc::source::{Naming, Source};
use symjit::{Application, CompilerType, Config, Storage};

use args::{Args, Command, Emit, USAGE};

fn main() -> ExitCode {
    match args::parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjc: error: {}", msg);
            eprintln!("Try `sjc --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjc {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(a)) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("sjc: error: {:#}", err);
                ExitCode::from(1)
            }
        },
    }
}

/// The architecture (compiler type) and object format named by `--target`.
fn parse_target(target: &str) -> Result<(CompilerType, Option<&'static str>)> {
    let t = target.to_ascii_lowercase();
    let (ty, rest) = if let Some(rest) = ["x86_64", "x86-64", "amd64", "x64"]
        .iter()
        .find_map(|a| t.strip_prefix(a))
    {
        (CompilerType::Amd, rest)
    } else if let Some(rest) = ["aarch64", "arm64"].iter().find_map(|a| t.strip_prefix(a)) {
        (CompilerType::Arm, rest)
    } else if let Some(rest) = ["wasm32", "wasm"].iter().find_map(|a| t.strip_prefix(a)) {
        (CompilerType::Wasm, rest)
    } else if t == "native" {
        (CompilerType::Native, "")
    } else if t.starts_with("riscv") {
        bail!(
            "object files hold x86-64 or AArch64 kernels, not RISC-V ones (`{}`)",
            target
        )
    } else {
        bail!(
            "unknown target `{}` (use x86_64, aarch64 or wasm32, or a triple such as \
             aarch64-apple-darwin)",
            target
        )
    };

    if !rest.is_empty() && !rest.starts_with('-') {
        bail!("unknown target `{}`", target);
    }

    let format = if ty == CompilerType::Wasm {
        None
    } else if ["darwin", "apple", "macos", "ios"]
        .iter()
        .any(|s| rest.contains(s))
    {
        Some("macho")
    } else if rest.contains("windows") {
        bail!(
            "Windows (COFF) object files are not supported (`{}`)",
            target
        )
    } else if ["linux", "elf", "freebsd", "netbsd", "openbsd", "android"]
        .iter()
        .any(|s| rest.contains(s))
    {
        Some("elf")
    } else {
        None
    };

    Ok((ty, format))
}

/// The output kind: `--emit`, else the extension of `-o`, else from `--target`.
fn emit_kind(a: &Args, ty: Option<CompilerType>) -> Result<Emit> {
    if let Some(emit) = a.emit {
        return Ok(emit);
    }
    if let Some(out) = &a.output {
        match out.extension().and_then(|e| e.to_str()) {
            Some("o") | Some("obj") => return Ok(Emit::Obj),
            Some("wasm") => return Ok(Emit::Wasm),
            Some("h") => return Ok(Emit::Header),
            Some("sjb") => return Ok(Emit::Sjb),
            _ => {}
        }
    }
    Ok(if ty == Some(CompilerType::Wasm) {
        Emit::Wasm
    } else {
        Emit::Obj
    })
}

/// `dir/stem.ext` → `stem`, and `stem.ext` next to the input for a missing output.
fn stem(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("`{}` does not end in a file name", path.display()))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// What sjc compiles: a saved application, or a kernel of a Rust file.
enum Input {
    Saved(Vec<u8>),
    Rust { src: Source, kernel: String },
}

impl Input {
    fn read(a: &Args) -> Result<Input> {
        if !a.rust_input() {
            let bytes = std::fs::read(&a.input)
                .with_context(|| format!("cannot read `{}`", a.input.display()))?;
            return Ok(Input::Saved(bytes));
        }
        let text = std::fs::read_to_string(&a.input)
            .with_context(|| format!("cannot read `{}`", a.input.display()))?;
        let src = Source::parse(&a.input.display().to_string(), &text)?;
        let kernel = match &a.kernel {
            Some(k) => k.clone(),
            None => match src.kernels().as_slice() {
                [k] => k.to_string(),
                [] => bail!("`{}` has no `#[kernel]` function", a.input.display()),
                ks => bail!(
                    "`{}` has several kernels ({}); choose one with --kernel",
                    a.input.display(),
                    ks.join(", ")
                ),
            },
        };
        Ok(Input::Rust { src, kernel })
    }

    /// The stem of the default output: the kernel's name for Rust input.
    fn stem(&self, a: &Args) -> Result<String> {
        match self {
            Input::Saved(_) => stem(&a.input),
            Input::Rust { kernel, .. } => Ok(kernel.clone()),
        }
    }

    /// The application for `ty` (`None`: the saved type; this machine for Rust input).
    fn app(&self, a: &Args, ty: Option<CompilerType>) -> Result<Application> {
        Ok(self.build(a, ty, None)?.0)
    }

    /// The application for `ty`, and for an object file of the kernel `symbol`, the kernels
    /// it calls (Rust input), named after it (`Naming::object`).
    fn build(
        &self,
        a: &Args,
        ty: Option<CompilerType>,
        symbol: Option<&str>,
    ) -> Result<(Application, Vec<(String, Application)>)> {
        let (app, callees) = match self {
            Input::Saved(bytes) => (load(a, bytes, ty)?, Vec::new()),
            Input::Rust { src, kernel } => {
                // the built-in options, not a symjit.toml of the working directory
                let mut config = Config::builtin(ty.unwrap_or(CompilerType::Native));
                if let Some(level) = a.opt_level {
                    config.set_opt_level(level);
                }
                if let Some(fastmath) = a.fastmath {
                    config.set_fastmath(fastmath);
                }
                match symbol {
                    Some(symbol) => {
                        src.compile_for_object(kernel, config, &Naming::object(symbol))?
                    }
                    None => (
                        src.compile_with(kernel, config, &Naming::object(kernel))?,
                        Vec::new(),
                    ),
                }
            }
        };
        if a.verbose {
            match self {
                Input::Saved(_) => eprintln!("sjc: read {}: {}", a.input.display(), describe(&app)),
                Input::Rust { kernel, .. } => eprintln!(
                    "sjc: compiled the kernel `{}` of {}: {}",
                    kernel,
                    a.input.display(),
                    describe(&app)
                ),
            }
        }
        Ok((app, callees))
    }
}

fn load(a: &Args, bytes: &[u8], ty: Option<CompilerType>) -> Result<Application> {
    // the template config only supplies user-defined functions, which saved models
    // cannot carry; no symjit.toml is read
    let config = Config::new(CompilerType::Native, 0)?;
    Application::load_for(&mut &bytes[..], &config, ty).map_err(|err| {
        if err.to_string().starts_with("invalid magic number") {
            anyhow!("`{}` is not a saved symjit application", a.input.display())
        } else {
            err.context(format!("cannot compile `{}`", a.input.display()))
        }
    })
}

fn write(a: &Args, path: &Path, bytes: &[u8]) -> Result<()> {
    if path == a.input {
        bail!("the output `{}` would overwrite the input", path.display());
    }
    std::fs::write(path, bytes).with_context(|| format!("cannot write `{}`", path.display()))?;
    if a.verbose {
        eprintln!("sjc: wrote {} ({} bytes)", path.display(), bytes.len());
    }
    Ok(())
}

fn describe(app: &Application) -> String {
    let c = &app.config;
    let arch = if c.is_amd64() {
        if c.has_avx() {
            "x86-64 (AVX)"
        } else {
            "x86-64 (SSE2)"
        }
    } else if c.is_arm64() {
        "ARM64"
    } else if c.is_riscv64() {
        "RISC-V"
    } else if c.is_wasm() {
        "WebAssembly"
    } else {
        "bytecode"
    };
    format!(
        "{}, opt level {}, {}{} states, {} parameters, {} outputs, {} derivatives",
        arch,
        c.opt_level(),
        if c.is_complex() { "complex, " } else { "" },
        app.count_states,
        app.count_params,
        app.count_obs,
        app.count_diffs
    )
}

fn run(a: &Args) -> Result<()> {
    let (ty, format) = match &a.target {
        Some(t) => {
            let (ty, format) = parse_target(t)?;
            (Some(ty), format)
        }
        None => (None, None),
    };
    let format = a.format.as_deref().or(format);
    let emit = emit_kind(a, ty)?;

    match (emit, ty) {
        (Emit::Wasm, Some(t)) if t != CompilerType::Wasm => {
            bail!(
                "--emit wasm needs a WebAssembly target, not `{}`",
                a.target.as_ref().unwrap()
            )
        }
        (Emit::Obj | Emit::Header, Some(CompilerType::Wasm)) => {
            bail!("WebAssembly is not an object-file target; use --emit wasm or a .wasm output")
        }
        _ => {}
    }
    if format.is_some() && !matches!(emit, Emit::Obj | Emit::Header) {
        bail!("--format applies to object files only");
    }

    let input = Input::read(a)?;
    match emit {
        Emit::Info => info(a, &input),
        Emit::Wasm => emit_wasm(a, &input),
        Emit::Sjb => emit_sjb(a, &input, ty),
        Emit::Obj | Emit::Header => emit_obj(a, &input, emit, ty, format.unwrap_or("")),
    }
}

fn info(a: &Args, input: &Input) -> Result<()> {
    use std::fmt::Write as _;
    use std::io::Write as _;

    let app = input.app(a, None)?;
    let mut funcs: Vec<&str> = app.prog.builder.ft.iter().map(|s| s.as_str()).collect();
    funcs.sort();
    let funcs = if funcs.is_empty() {
        "none".to_string()
    } else {
        funcs.join(", ")
    };

    let mut s = String::new();
    match input {
        Input::Saved(_) => writeln!(s, "{}: a saved symjit application", a.input.display())?,
        Input::Rust { kernel, .. } => {
            writeln!(s, "{}: the kernel `{}`", a.input.display(), kernel)?
        }
    }
    writeln!(s, "  target:     {}", describe(&app))?;
    writeln!(s, "  config:     {:?}", app.config)?;
    writeln!(s, "  memory:     {} doubles", app.prog.mem_size())?;
    writeln!(
        s,
        "  registers:  {} used, {} available",
        app.measure("registers"),
        app.config.count_scratch()
    )?;
    writeln!(s, "  functions:  {}", funcs)?;
    writeln!(
        s,
        "  fast kernel: {}",
        if app.can_fast { "yes" } else { "no" }
    )?;
    writeln!(
        s,
        "  code:       {} MIR instructions, {} bytes of machine code",
        app.measure("mir-size"),
        app.measure("ker-scalar-size")
    )?;
    // a closed pipe (`sjc --emit info m.sjb | head -1`) is not an error
    let _ = std::io::stdout().write_all(s.as_bytes());
    Ok(())
}

/// Saves the kernel of a Rust file as an application (for later sjc runs, or Python's
/// `load_func`).
fn emit_sjb(a: &Args, input: &Input, ty: Option<CompilerType>) -> Result<()> {
    if let Input::Saved(_) = input {
        bail!("`{}` is a saved application already", a.input.display());
    }
    let app = input.app(a, ty)?;
    if app
        .config
        .df
        .as_ref()
        .is_some_and(|df| !df.funcs.is_empty())
    {
        bail!(
            "the kernel calls other kernels or callbacks, which a saved application cannot \
             hold; compile it into an object file instead"
        );
    }
    let mut bytes = Vec::new();
    app.save(&mut bytes)?;
    let out = match &a.output {
        Some(out) => out.clone(),
        None => PathBuf::from(format!("{}.sjb", input.stem(a)?)),
    };
    write(a, &out, &bytes)
}

#[cfg(feature = "wasm")]
fn emit_wasm(a: &Args, input: &Input) -> Result<()> {
    let app = input.app(a, Some(CompilerType::Wasm))?;
    let module = app
        .wasm
        .as_ref()
        .ok_or_else(|| anyhow!("no WebAssembly module was generated"))?;
    let out = match &a.output {
        Some(out) => out.clone(),
        None => PathBuf::from(format!("{}.wasm", input.stem(a)?)),
    };
    write(a, &out, module)
}

#[cfg(not(feature = "wasm"))]
fn emit_wasm(_a: &Args, _input: &Input) -> Result<()> {
    bail!("sjc was built without the `wasm` feature (WebAssembly modules)")
}

#[cfg(feature = "obj")]
fn emit_obj(
    a: &Args,
    input: &Input,
    emit: Emit,
    ty: Option<CompilerType>,
    format: &str,
) -> Result<()> {
    let ext = if emit == Emit::Obj { "o" } else { "h" };
    let out = match &a.output {
        Some(out) => out.clone(),
        None => PathBuf::from(format!("{}.{}", input.stem(a)?, ext)),
    };
    let symbol = match &a.name {
        Some(name) => name.clone(),
        None => stem(&out)?,
    };
    let header = match (emit, &a.header) {
        (Emit::Header, _) => Some(out.clone()),
        (_, Some(h)) => Some(h.clone()),
        _ if a.no_header => None,
        _ => Some(out.with_extension("h")),
    };
    let object_file = if emit == Emit::Obj {
        file_name(&out)
    } else {
        format!("{}.o", symbol)
    };
    let header_file = header
        .as_ref()
        .map_or_else(|| format!("{}.h", symbol), |h| file_name(h));

    let (mut app, mut callees) = input.build(a, ty, Some(&symbol))?;
    if ty.is_none() && !(app.config.is_amd64() || app.config.is_arm64()) {
        // saved as a WebAssembly, bytecode or RISC-V model: compile for this machine
        (app, callees) = input.build(a, Some(CompilerType::Native), Some(&symbol))?;
    }

    let (object, text) = app
        .object_files_with(&symbol, format, &object_file, &header_file, &mut callees)
        .map_err(|err| {
            if a.name.is_none() && err.to_string().contains("C identifier") {
                err.context(format!(
                    "the kernels are named after the output file `{}`; choose a name with --name",
                    out.display()
                ))
            } else {
                err
            }
        })?;

    if emit == Emit::Obj {
        write(a, &out, &object)?;
    }
    if let Some(h) = &header {
        write(a, h, text.as_bytes())?;
    }
    Ok(())
}

#[cfg(not(feature = "obj"))]
fn emit_obj(
    _a: &Args,
    _input: &Input,
    _emit: Emit,
    _ty: Option<CompilerType>,
    _format: &str,
) -> Result<()> {
    bail!("sjc was built without the `obj` feature (object files)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets() {
        let t = |s| parse_target(s).unwrap();
        assert_eq!(t("x86_64"), (CompilerType::Amd, None));
        assert_eq!(
            t("x86_64-unknown-linux-gnu"),
            (CompilerType::Amd, Some("elf"))
        );
        assert_eq!(
            t("aarch64-apple-darwin"),
            (CompilerType::Arm, Some("macho"))
        );
        assert_eq!(t("arm64"), (CompilerType::Arm, None));
        assert_eq!(t("wasm32-unknown-unknown"), (CompilerType::Wasm, None));
        assert_eq!(t("native"), (CompilerType::Native, None));
        assert!(parse_target("riscv64gc-unknown-linux-gnu").is_err());
        assert!(parse_target("x86_64-pc-windows-msvc").is_err());
        assert!(parse_target("sparc").is_err());
        assert!(parse_target("x86_64abc").is_err());
    }

    #[test]
    fn emits() {
        let args = |out: Option<&str>| Args {
            output: out.map(PathBuf::from),
            ..Default::default()
        };
        assert_eq!(emit_kind(&args(Some("m.o")), None).unwrap(), Emit::Obj);
        assert_eq!(emit_kind(&args(Some("m.wasm")), None).unwrap(), Emit::Wasm);
        assert_eq!(emit_kind(&args(Some("m.h")), None).unwrap(), Emit::Header);
        assert_eq!(
            emit_kind(&args(None), Some(CompilerType::Wasm)).unwrap(),
            Emit::Wasm
        );
        assert_eq!(emit_kind(&args(None), None).unwrap(), Emit::Obj);
    }
}
