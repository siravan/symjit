use anyhow::Result;

use super::utils::reg;
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
        match self {
            Statement::Assign { lhs, rhs } => {
                let r = rhs.compile_tree(ir)?;
                Self::save(ir, r, lhs);
            }
            Statement::Call {
                op,
                lhs,
                arg,
                num_args,
            } => {
                let _ = arg.compile_tree(ir)?;
                ir.call(op.as_str(), *num_args)?;
                Self::save_result(ir, lhs);
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
