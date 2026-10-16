//! The command line of sjc.

use std::path::PathBuf;

pub const USAGE: &str = "\
Usage: sjc [OPTIONS] <INPUT.sjb | INPUT.rs>

Compiles a saved symjit application (`f.save(...)` in Python, `Storage::save` in Rust),
or a `#[kernel]` function of a Rust file, into a relocatable object file and C header, or
into a WebAssembly module.

Options:
  -o <FILE>             the output file; its extension selects the output (.o/.obj: object,
                        .wasm: WebAssembly module, .h: header only, .sjb: saved application)
                        unless --emit is given; default <INPUT stem>.o (.wasm for --emit wasm;
                        the kernel's name instead of the stem for Rust input)
  --emit <KIND>         obj | wasm | header | sjb | info (info prints a summary of the model)
  --target <TARGET>     x86_64, aarch64 (arm64) or wasm32, optionally a triple such as
                        aarch64-apple-darwin, whose system picks the object format;
                        default: the architecture the model was saved for (native: this one)
  --format <FORMAT>     elf | macho, the object format (default: from --target, else this
                        system's)
  --name <IDENT>        the name of the kernel functions (default: the output's stem; a C
                        identifier)
  --header <FILE>       the C header (default: the object file with extension .h)
  --no-header           do not write the C header
  -v, --verbose         say what is read and written
  -h, --help            print this help
  -V, --version         print the version

Rust input (`.rs`):
  --kernel <NAME>       the `#[kernel]` function to compile (default: the only one)
  -O0, -O1, -O2, -O3    the opt level (default 2)
  -ffast-math, -fno-fast-math
                        fuse multiplications and additions, or not (default: fuse)

The compilation options of a saved application (opt level, fastmath, complex, ...) are
those it was saved with. Models saved with opt level 3 on ARM64 or RISC-V may use more
registers than x86-64 has; sjc rejects them for x86-64 targets.";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Emit {
    Obj,
    Wasm,
    Header,
    Sjb,
    Info,
}

#[derive(Debug, Default)]
pub struct Args {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub emit: Option<Emit>,
    pub target: Option<String>,
    pub format: Option<String>,
    pub name: Option<String>,
    pub header: Option<PathBuf>,
    pub no_header: bool,
    pub verbose: bool,
    /// Rust input: the kernel, the opt level and fastmath
    pub kernel: Option<String>,
    pub opt_level: Option<u8>,
    pub fastmath: Option<bool>,
}

impl Args {
    /// Whether the input is a Rust file.
    pub fn rust_input(&self) -> bool {
        self.input.extension().and_then(|e| e.to_str()) == Some("rs")
    }
}

/// What the command line asks for.
#[derive(Debug)]
pub enum Command {
    Run(Args),
    Help,
    Version,
}

pub fn parse_emit(s: &str) -> Result<Emit, String> {
    match s {
        "obj" | "o" | "object" => Ok(Emit::Obj),
        "wasm" => Ok(Emit::Wasm),
        "header" | "h" => Ok(Emit::Header),
        "sjb" => Ok(Emit::Sjb),
        "info" => Ok(Emit::Info),
        _ => Err(format!(
            "unknown output kind `{}` (use obj, wasm, header, sjb or info)",
            s
        )),
    }
}

/// Parses the arguments (without the program name).
pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut a = Args::default();
    let mut input: Option<PathBuf> = None;
    let mut it = args.into_iter();

    while let Some(arg) = it.next() {
        // `--opt=value` and `--opt value`
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) if k.starts_with("--") => (k.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            match inline.clone() {
                Some(v) => Ok(v),
                None => it.next().ok_or_else(|| format!("`{}` needs a value", name)),
            }
        };

        match key.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-v" | "--verbose" => a.verbose = true,
            "--no-header" => a.no_header = true,
            "-o" => a.output = Some(PathBuf::from(value("-o")?)),
            "--emit" => a.emit = Some(parse_emit(&value("--emit")?)?),
            "--target" => a.target = Some(value("--target")?),
            "--format" => a.format = Some(value("--format")?),
            "--name" => a.name = Some(value("--name")?),
            "--header" => a.header = Some(PathBuf::from(value("--header")?)),
            "--kernel" => a.kernel = Some(value("--kernel")?),
            "-O0" | "-O1" | "-O2" | "-O3" => a.opt_level = Some(key.as_bytes()[2] - b'0'),
            "-ffast-math" => a.fastmath = Some(true),
            "-fno-fast-math" => a.fastmath = Some(false),
            _ if key.starts_with("-o") && key.len() > 2 => {
                a.output = Some(PathBuf::from(&key[2..]));
            }
            _ if key.starts_with("-O") || key.starts_with("-f") || key.starts_with("-m") => {
                return Err(format!(
                    "unknown option `{}` (sjc takes -O0..-O3, -ffast-math and -fno-fast-math, \
                     for Rust input)",
                    arg
                ));
            }
            _ if key.starts_with('-') && key.len() > 1 => {
                return Err(format!("unknown option `{}`", arg));
            }
            _ => {
                if input.is_some() {
                    return Err(format!(
                        "more than one input (`{}`); sjc compiles one model at a time",
                        arg
                    ));
                }
                input = Some(PathBuf::from(arg));
            }
        }
    }

    a.input = input.ok_or_else(|| "no input file".to_string())?;
    if !a.rust_input() {
        let given = [
            (a.opt_level.is_some(), "-O"),
            (a.fastmath.is_some(), "-ffast-math"),
            (a.kernel.is_some(), "--kernel"),
        ];
        if let Some((_, flag)) = given.iter().find(|(set, _)| *set) {
            return Err(format!(
                "`{}`: the compilation options of a saved application are fixed when it is \
                 compiled and saved; sjc uses the saved ones (options are for Rust input)",
                flag
            ));
        }
    }
    if a.no_header && a.header.is_some() {
        return Err("--no-header and --header exclude each other".to_string());
    }
    Ok(Command::Run(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Result<Args, String> {
        match parse(args.iter().map(|s| s.to_string()))? {
            Command::Run(a) => Ok(a),
            c => panic!("unexpected {:?}", c),
        }
    }

    #[test]
    fn options() {
        let a = run(&[
            "model.sjb",
            "-o",
            "out/m.o",
            "--target=aarch64-apple-darwin",
            "-v",
        ])
        .unwrap();
        assert_eq!(a.input, PathBuf::from("model.sjb"));
        assert_eq!(a.output, Some(PathBuf::from("out/m.o")));
        assert_eq!(a.target.as_deref(), Some("aarch64-apple-darwin"));
        assert!(a.verbose);

        let a = run(&[
            "-om.wasm",
            "--emit",
            "wasm",
            "--name",
            "f",
            "x.sjb",
            "--no-header",
        ])
        .unwrap();
        assert_eq!(a.output, Some(PathBuf::from("m.wasm")));
        assert_eq!(a.emit, Some(Emit::Wasm));
        assert_eq!(a.name.as_deref(), Some("f"));
        assert!(a.no_header);
    }

    #[test]
    fn errors() {
        assert!(run(&[]).unwrap_err().contains("no input"));
        assert!(run(&["a.sjb", "b.sjb"])
            .unwrap_err()
            .contains("more than one input"));
        assert!(run(&["a.sjb", "-O3"])
            .unwrap_err()
            .contains("fixed when it"));
        assert!(run(&["a.rs", "-O4"])
            .unwrap_err()
            .contains("unknown option"));
        let a = run(&["k.rs", "-O3", "-fno-fast-math", "--kernel", "f"]).unwrap();
        assert_eq!((a.opt_level, a.fastmath), (Some(3), Some(false)));
        assert_eq!(a.kernel.as_deref(), Some("f"));
        assert!(a.rust_input());
        assert!(run(&["a.sjb", "--bogus"])
            .unwrap_err()
            .contains("unknown option"));
        assert!(run(&["a.sjb", "-o"]).unwrap_err().contains("needs a value"));
        assert!(run(&["a.sjb", "--emit", "exe"])
            .unwrap_err()
            .contains("unknown output kind"));
        assert!(run(&["a.sjb", "--no-header", "--header", "x.h"]).is_err());
        assert!(matches!(parse(["-h".to_string()]), Ok(Command::Help)));
        assert!(matches!(
            parse(["--version".to_string()]),
            Ok(Command::Version)
        ));
    }
}
