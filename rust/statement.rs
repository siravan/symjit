use anyhow::Result;

use super::mir::Mir;
use super::node::Node;
use super::operation::Operation;
use super::symbol::Loc;
use super::topology::Topology;
use super::utils::{is_external_func, reg};

#[derive(Debug, Clone)]
pub enum Statement {
    Assign {
        lhs: Node,
        rhs: Node,
        topo: String,
    },
    Call {
        op: Operation,
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
        Statement::Assign {
            lhs,
            rhs,
            topo: "".into(),
        }
    }

    pub fn call(op: Operation, lhs: Node, arg: Node, num_args: usize) -> Statement {
        Statement::Call {
            op,
            lhs,
            arg,
            num_args,
        }
    }

    pub fn add_topology(&mut self, topology: &mut Topology) {
        if let Statement::Assign { rhs, topo, .. } = self {
            let t = rhs.topology();
            topology.add(&t);
            *topo = t;
        };
    }

    pub fn compile(&mut self, ir: &mut Mir, topology: &mut Topology) -> Result<()> {
        match self {
            Statement::Assign { lhs, rhs, topo } => {
                if let Some((num_args, defined)) = topology.status(topo) {
                    if !defined {
                        let mut n = 0;
                        let body = rhs.subroutine(&topology.args, &mut n);
                        topology.define(topo, body);
                    }

                    let mut locs: Vec<Loc> = Vec::new();
                    rhs.caller(ir, &mut locs);

                    ir.load_args(locs, false);

                    ir.call(topo, 0)?;
                    Self::save_result(ir, lhs);
                    ir.nop();
                } else {
                    let r = rhs.compile_tree(ir)?;
                    Self::save(ir, r, lhs);
                }
            }
            Statement::Call {
                op,
                lhs,
                arg,
                num_args,
            } => {
                if is_external_func(op.as_str()) {
                    let (l, r) = arg.call_external()?;
                    ir.call(op.as_str(), (r - l) as usize)?;
                    Self::save_result(ir, lhs);
                } else {
                    let _ = arg.compile_tree(ir)?;
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
                let cond = cond.compile_tree(ir)?;
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
