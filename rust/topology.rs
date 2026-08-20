use anyhow::{anyhow, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;

use super::config::Config;
use super::mir::Mir;
use super::node::Node;
use super::symbol::Symbol;
use super::utils::reg;
use super::utils::Reg;

#[derive(Clone, Debug)]
pub struct Subroutine {
    topology: String,
    num_args: usize,
    body: Option<Node>,
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub counts: HashMap<String, i64>,
    pub subs: HashMap<String, Subroutine>,
    pub config: Config,
    pub salt: Option<String>,
    pub enabled: bool,
    pub args: Vec<Rc<RefCell<Symbol>>>,
}

impl Topology {
    pub fn new(config: Config, args: Vec<Rc<RefCell<Symbol>>>) -> Topology {
        let enabled = config.debug_topology() || config.compress();

        Topology {
            counts: HashMap::new(),
            subs: HashMap::new(),
            config,
            salt: None,
            enabled,
            args,
        }
    }

    pub fn set_salt(&mut self, salt: Option<String>) {
        self.salt = salt.clone()
    }

    pub fn add(&mut self, topo: &str) {
        if let Some(p) = self.counts.get_mut(topo) {
            *p += 1;
        } else {
            self.counts.insert(topo.to_string(), 1);
        }
    }

    pub fn dump(&self, name: &str) -> Result<()> {
        let mut counts: Vec<(&String, &i64)> = self.counts.iter().collect();
        counts.sort_by_key(|&(_, v)| -v);

        let mut fs = std::fs::File::create(name)?;

        for (k, v) in counts.iter() {
            writeln!(fs, "{}\t{}", v, k)?;
        }

        Ok(())
    }

    pub fn prepare(&mut self) {
        for (k, v) in self.counts.iter() {
            let nx = k.chars().filter(|c| *c == 'X').count();

            if (3..=32).contains(&nx) && *v >= 3 {
                self.subs.insert(
                    k.clone(),
                    Subroutine {
                        topology: k.clone(),
                        num_args: nx,
                        body: None,
                    },
                );
            }
        }
    }

    pub fn status(&self, topo: &str) -> Option<(usize, bool)> {
        if self.enabled {
            if let Some(f) = self.subs.get(topo) {
                return Some((f.num_args, f.body.is_some()));
            }
        }
        None
    }

    pub fn define(&mut self, topo: &str, body: Node) {
        let f = self.subs.get_mut(topo).unwrap();
        f.body = Some(body);
    }

    pub fn compile(&self, ir: &mut Mir) -> Result<()> {
        if !self.subs.is_empty() {
            ir.branch("@epilogue");
        }

        for f in self.subs.values() {
            match &f.body {
                Some(body) => {
                    let label = format!("{}_ultra", f.topology);
                    ir.set_label(&label);
                    ir.save_args(f.num_args as u8, true);
                    ir.set_label(&f.topology);
                    ir.save_args(f.num_args as u8, false);
                    let dst = body.compile_tree(ir)?;
                    ir.nop();
                    ir.fmov(Reg::Ret, reg(dst));
                    ir.branch(".ret");
                }
                None => {
                    return Err(anyhow!("subroutine {} does not have a body!", f.topology));
                }
            }
        }

        Ok(())
    }
}
