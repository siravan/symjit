//! An application as JSON: the "symjit-application" file, which holds what a saved application
//! (`.sjb`) holds, so that symjit can read it back (`Application::from_json`), and which readers
//! outside symjit can use for the MIR. Each part writes and reads its own JSON (`Storage::save_json`
//! and `load_json`, as `save` and `load_with_version` for the `.sjb`): `Application` the file,
//! `Program` its `program`, `Config` its `config`, `Builder` its `builder`, `Mir` its `mir`.
//!
//! `Application::to_json` (and `save_json`) writes the MIR that `save` writes: the MIR before
//! complexification, or the complexified MIR after `complexify`. `Application::dump(path, "json")`
//! writes the MIR before complexification and `dump(path, "json-complex")` the MIR of real
//! operations (the complexified MIR of a complex application; a real application has one MIR, which
//! both write). `Application::sjb_mir_json` writes the same files from a saved application without
//! compiling it. Like the `.sjb` loader, `from_json` refuses a complexified MIR.
//!
//! Format version 1:
//!   format          "symjit-application"
//!   format_version  1
//!   symjit_version  the version of the symjit that wrote the file, e.g. "2.29.0"
//!   program         the program (`Program`):
//!     config        the configuration as symjit.toml spells it: `ty`, `options` (each option as set,
//!                   with `opt_level`, `stack_limit`, `num_args`, `direct_arena_operation`) and `debug`;
//!                   every bit of the configuration, not only the options in effect on the machine
//!                   that wrote it (`Config::to_exact_toml`)
//!     count_states, count_params, count_obs, count_diffs
//!                   the counts (a complex model counts two f64 values per number)
//!     count_loops   the number of loops
//!     builder       `count_loops`, `count_stack` (the stack slots), `consts` (the program's constant
//!                   pool) and `ft` (the names of the functions it calls, sorted)
//!   prepared        the kernels the application had prepared, prepared again on loading: `compiled`
//!                   (the scalar kernel was machine code), `fast`, `simd`
//!   mir             the MIR (`Mir`):
//!     complexified  whether `code` is a complexified MIR (complex numbers lowered to real pairs)
//!     code          the instructions, as serde writes `Instruction` (externally tagged)
//!     consts        the MIR's constant pool
//!     labels        label → instruction index, sorted by label (for readers: loading recomputes them)
//!   reals           the locations of a complex model that hold real numbers, sorted (Mem, Stack,
//!                   Param; then by index)
//!
//! Every constant is its IEEE 754 bit pattern: "0x" and 16 hex digits, so that every value (NaN, the
//! infinities, -0.0) reads back exactly.
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

use super::super::config::Config;
use super::super::model::Program;
use super::super::symbol::Loc;
use super::super::utils::Storage;
use super::{Instruction, Mir};

pub const FORMAT: &str = "symjit-application";
pub const FORMAT_VERSION: usize = 1;

/// a double as its bit pattern, "0x" and 16 hex digits
pub fn const_bits(x: f64) -> String {
    format!("0x{:016x}", x.to_bits())
}

/// the double of a bit pattern `const_bits` wrote
pub fn parse_bits(s: &str) -> Result<f64> {
    match s.strip_prefix("0x") {
        Some(hex) if hex.len() == 16 => u64::from_str_radix(hex, 16)
            .map(f64::from_bits)
            .map_err(|_| anyhow!("invalid constant {:?}", s)),
        _ => Err(anyhow!(
            "invalid constant {:?} (\"0x\" and 16 hex digits)",
            s
        )),
    }
}

/// `value[name]`, or an error naming the missing field
pub fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| anyhow!("the JSON has no {:?}", name))
}

/// a location's sort key: kind (Mem, Stack, Param), then index (as `save_reals` orders them)
fn loc_key(l: &Loc) -> (u8, u32) {
    match l {
        Loc::Mem(i) => (1, *i),
        Loc::Stack(i) => (2, *i),
        Loc::Param(i) => (3, *i),
    }
}

/// a MIR's JSON (`Mir::save_json`)
pub fn mir_to_json(mir: &Mir) -> Result<Value> {
    let code: Vec<Instruction> = mir.code.iter().collect();
    let consts: Vec<String> = mir.consts.iter().map(|x| const_bits(*x)).collect();
    // sorted: the same MIR gives the same file
    let labels: BTreeMap<&String, &usize> = mir.labels.iter().collect();

    Ok(json!({
        "complexified": mir.code.is_complexified,
        "code": code,
        "consts": consts,
        "labels": labels,
    }))
}

/// the MIR of its JSON (`Mir::load_json`): the instructions and the constants; the labels are
/// recomputed (as `Mir::load_with_version` does)
pub fn mir_from_json(value: &Value, config: &Config) -> Result<Mir> {
    let code: Vec<Instruction> = serde_json::from_value(field(value, "code")?.clone())?;
    let consts: Vec<String> = serde_json::from_value(field(value, "consts")?.clone())?;

    let mut mir = Mir::new(config.clone());
    for ins in code {
        mir.push(ins);
    }
    mir.consts = consts
        .iter()
        .map(|s| parse_bits(s))
        .collect::<Result<_>>()?;
    mir.populate_labels();

    let complexified = field(value, "complexified")?
        .as_bool()
        .ok_or_else(|| anyhow!("mir: complexified is not a boolean"))?;
    if complexified != mir.code.is_complexified {
        return Err(anyhow!(
            "mir: complexified is {}, but the code says {}",
            complexified,
            mir.code.is_complexified
        ));
    }

    Ok(mir)
}

/// the file of an application's parts (`Application::save_json`): its program, the mask of its
/// prepared kernels (as `save` writes it: 1 compiled, 2 fast, 4 simd), a MIR and the reals
pub fn application_to_json(
    prog: &Program,
    mask: usize,
    mir: &Mir,
    reals: &HashSet<Loc>,
) -> Result<Value> {
    let mut reals: Vec<Loc> = reals.iter().copied().collect();
    reals.sort_by_key(loc_key);

    Ok(json!({
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "symjit_version": env!("CARGO_PKG_VERSION"),
        "program": prog.save_json()?,
        "prepared": {
            "compiled": mask & 1 != 0,
            "fast": mask & 2 != 0,
            "simd": mask & 4 != 0,
        },
        "mir": mir.save_json()?,
        "reals": reals,
    }))
}

/// the parts of an application's file (`Application::load_json`): its program, the reals, the MIR
/// and the mask of the kernels to prepare
pub fn application_from_json(
    value: &Value,
    config: &Config,
) -> Result<(Program, HashSet<Loc>, Mir, usize)> {
    if value.get("format").and_then(Value::as_str) != Some(FORMAT) {
        return Err(anyhow!("not a {} file", FORMAT));
    }

    let version = field(value, "format_version")?
        .as_u64()
        .ok_or_else(|| anyhow!("format_version is not a number"))? as usize;

    if version == 0 || version > FORMAT_VERSION {
        return Err(anyhow!(
            "{} version {}: this symjit reads versions 1 to {}",
            FORMAT,
            version,
            FORMAT_VERSION
        ));
    }

    let prog = Program::load_json(field(value, "program")?, config, version)?;
    let mir = Mir::load_json(field(value, "mir")?, prog.config(), version)?;
    let reals: Vec<Loc> = serde_json::from_value(field(value, "reals")?.clone())?;

    let prepared = field(value, "prepared")?;
    let mut mask = 0;
    for (bit, name) in [(1, "compiled"), (2, "fast"), (4, "simd")] {
        if field(prepared, name)?
            .as_bool()
            .ok_or_else(|| anyhow!("prepared: {} is not a boolean", name))?
        {
            mask |= bit;
        }
    }

    Ok((prog, reals.into_iter().collect(), mir, mask))
}
