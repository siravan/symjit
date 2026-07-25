use anyhow::Result;

use super::utils::{is_external_func, reg};
use crate::config::{Config, SPILL_ARENA};
use crate::mir::Mir;
use crate::node::Node;
use crate::symbol::Loc;

#[derive(Debug, Clone)]
pub enum Statement {
    Assign {
        lhs: Node,
        rhs: Node,
    },
    Call {
        op: String,
        lhs: Node,
        arg: Node,
        num_args: usize,
    },
    Label {
        label: String,
    },
    Branch {
        label: String,
    },
    BranchIf {
        cond: Node,
        label: String,
        is_else: bool,
    },
}

impl Statement {
    pub fn assign(lhs: Node, rhs: Node) -> Statement {
        Statement::Assign { lhs, rhs }
    }

    pub fn call(op: &str, lhs: Node, arg: Node, num_args: usize) -> Statement {
        Statement::Call {
            op: op.to_string(),
            lhs,
            arg,
            num_args,
        }
    }

    pub fn compile(&mut self, ir: &mut Mir) -> Result<()> {
        let mut spiller = Spiller::new(ir.config.clone());

        match self {
            Statement::Assign { lhs, rhs } => {
                let r = rhs.compile_tree(ir, &mut spiller)?;
                spiller.reset();
                Self::save(ir, r, lhs);
            }
            Statement::Call {
                op,
                lhs,
                arg,
                num_args,
            } => {
                if is_external_func(op) {
                    let (l, r) = arg.call_external()?;
                    ir.call(op.as_str(), (r - l) as usize)?;
                    Self::save_result(ir, lhs);
                } else {
                    let _ = arg.compile_tree(ir, &mut spiller)?;
                    spiller.reset();
                    ir.call(op.as_str(), *num_args)?;
                    Self::save_result(ir, lhs);
                }
            }
            Statement::Label { label } => {
                ir.set_label(label);
            }
            Statement::Branch { label } => {
                ir.branch(label);
            }
            Statement::BranchIf {
                cond,
                label,
                is_else,
            } => {
                let cond = cond.compile_tree(ir, &mut spiller)?;
                spiller.reset();
                ir.branch_if(reg(cond), label, *is_else);
            }
        };

        Ok(())
    }

    fn save(ir: &mut Mir, r: u8, v: &Node) {
        if let Node::Var { sym, .. } = v {
            match sym.borrow().loc {
                Loc::Stack(idx) => ir.save_stack(reg(r), idx),
                Loc::Mem(idx) => ir.save_mem(reg(r), idx),
                Loc::Param(_) => unreachable!(),
            }
        }
    }

    fn save_result(ir: &mut Mir, v: &Node) {
        if let Node::Var { sym, .. } = v {
            match sym.borrow().loc {
                Loc::Stack(idx) => ir.save_stack_result(idx),
                Loc::Mem(idx) => ir.save_mem_result(idx),
                Loc::Param(_) => unreachable!(),
            }
        }
    }
}

/* Spiller */

#[derive(Debug, Clone)]
pub struct Spiller {
    config: Config,
    pub count_scratch: u8,
    count_temp: u8,
}

impl Spiller {
    pub fn new(config: Config) -> Spiller {
        let count_scratch = config.count_scratch();

        Spiller {
            config,
            count_scratch,
            count_temp: 0,
        }
    }

    pub fn effective(&self, e: u8) -> u8 {
        (e - 1) % (self.count_scratch - 1)
    }

    pub fn push(&mut self) -> u32 {
        assert!(self.count_temp < 32);
        let idx = SPILL_ARENA as u32 + self.count_temp as u32;
        if self.config.is_complex() {
            self.count_temp += 2;
        } else {
            self.count_temp += 1;
        }
        idx
    }

    pub fn pop(&mut self) {
        if self.config.is_complex() {
            self.count_temp -= 2;
        } else {
            self.count_temp -= 1;
        }
    }

    pub fn reset(&mut self) {
        self.count_temp = 0;
    }
}
