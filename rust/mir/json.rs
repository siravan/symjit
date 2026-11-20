use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;

use super::super::config::ConfigToml;
use super::{Instruction, Mir};

#[derive(Debug, Default, Serialize)]
pub struct ExportMir {
    code: Vec<Instruction>,
    pub consts: Vec<f64>,
    pub labels: HashMap<String, usize>,
    pub config: ConfigToml,
}

impl ExportMir {
    pub fn new(mir: &Mir) -> ExportMir {
        let code: Vec<Instruction> = mir.code.iter().collect();

        ExportMir {
            code,
            consts: mir.consts.clone(),
            labels: mir.labels.clone(),
            config: mir.config.to_config_toml(),
        }
    }

    pub fn save_json(&self, name: &str) -> Result<()> {
        let s = serde_json::to_string(self)?;
        fs::write(name, &s)?;
        Ok(())
    }
}
