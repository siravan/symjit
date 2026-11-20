//! The MIR of an application as JSON (`Application::dump(path, "json")` and `"json-complex"`), for
//! readers outside symjit. `json` is the MIR before complexification, `json-complex` the MIR of
//! real operations (the complexified MIR of a complex application); a real application has one
//! MIR, which both write. The file starts with its format (`format`, `format_version`) and the
//! symjit version that wrote it, then the application's counts and real locations, the MIR (`code`,
//! `consts`, `labels`) and its configuration.
//!
//! Format version 1:
//!   format          "symjit-mir"
//!   format_version  1
//!   symjit_version  the crate's version, e.g. "2.29.0"
//!   complexified    whether `code` is the complexified MIR (complex numbers lowered to real pairs);
//!                   false for a real application's (its MIR has real operations only)
//!   count_states, count_params, count_obs, count_diffs
//!                   the application's counts (a complex model counts two f64 values per number)
//!   reals           the locations of a complex model that hold real numbers, sorted (Mem, Stack,
//!                   Param; then by index)
//!   code            the instructions, as serde writes `Instruction` (externally tagged)
//!   consts          the constant pool, each value as its IEEE 754 bit pattern: "0x" and 16 hex
//!                   digits, so that every value (NaN, the infinities, -0.0) reads back exactly
//!   labels          label → instruction index, sorted by label
//!   config          the configuration, as symjit.toml spells it
use anyhow::Result;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::fs;

use super::super::config::ConfigToml;
use super::super::symbol::Loc;
use super::{Instruction, Mir};

pub const FORMAT: &str = "symjit-mir";
pub const FORMAT_VERSION: u32 = 1;

/// what the file says of the application besides its MIR
pub struct MirHeader<'a> {
    pub complexified: bool,
    pub count_states: usize,
    pub count_params: usize,
    pub count_obs: usize,
    pub count_diffs: usize,
    pub reals: &'a HashSet<Loc>,
}

#[derive(Debug, Serialize)]
pub struct ExportMir {
    pub format: &'static str,
    pub format_version: u32,
    pub symjit_version: &'static str,
    pub complexified: bool,
    pub count_states: usize,
    pub count_params: usize,
    pub count_obs: usize,
    pub count_diffs: usize,
    pub reals: Vec<Loc>,
    code: Vec<Instruction>,
    pub consts: Vec<String>,
    pub labels: BTreeMap<String, usize>, // sorted: the same MIR gives the same file
    pub config: ConfigToml,
}

/// a location's sort key: kind (Mem, Stack, Param), then index (as `save_reals` orders them)
fn loc_key(l: &Loc) -> (u8, u32) {
    match l {
        Loc::Mem(i) => (1, *i),
        Loc::Stack(i) => (2, *i),
        Loc::Param(i) => (3, *i),
    }
}

/// a double as its bit pattern, "0x" and 16 hex digits
pub fn const_bits(x: f64) -> String {
    format!("0x{:016x}", x.to_bits())
}

impl ExportMir {
    pub fn new(mir: &Mir, header: &MirHeader) -> ExportMir {
        let code: Vec<Instruction> = mir.code.iter().collect();
        let mut reals: Vec<Loc> = header.reals.iter().copied().collect();
        reals.sort_by_key(loc_key);

        ExportMir {
            format: FORMAT,
            format_version: FORMAT_VERSION,
            symjit_version: env!("CARGO_PKG_VERSION"),
            complexified: header.complexified,
            count_states: header.count_states,
            count_params: header.count_params,
            count_obs: header.count_obs,
            count_diffs: header.count_diffs,
            reals,
            code,
            consts: mir.consts.iter().map(|x| const_bits(*x)).collect(),
            labels: mir.labels.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            config: mir.config.to_config_toml(),
        }
    }

    pub fn save_json(&self, name: &str) -> Result<()> {
        let s = serde_json::to_string(self)?;
        fs::write(name, &s)?;
        Ok(())
    }
}
