use anyhow::Result;

use super::utils::Eval;
use crate::code::VirtualTable;
use crate::generator::Generator;
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

    pub fn compile(&mut self, ir: &mut dyn Generator) -> Result<()> {
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
                let label = format!("_func_{}_", op);
                ir.call(&label, *num_args);
                Self::save(ir, 0, lhs);
            }
        };

        Ok(())
    }

    fn save(ir: &mut dyn Generator, r: u8, v: &Node) {
        if let Node::Var { sym, .. } = v {
            match sym.borrow().loc {
                Loc::Stack(idx) => ir.save_stack(r, idx),
                Loc::Mem(idx) => ir.save_mem(r, idx),
            }
        }
    }
}

impl Eval for Statement {
    fn eval(&self, mem: &mut [f64], stack: &mut [f64]) -> f64 {
        match &self {
            Statement::Assign { lhs, rhs } => {
                let u = rhs.eval(mem, stack);

                if let Node::Var { sym, .. } = lhs {
                    match sym.borrow().loc {
                        Loc::Stack(idx) => stack[idx as usize] = u,
                        Loc::Mem(idx) => mem[idx as usize] = u,
                    }
                }
            }
            Statement::Call { op, lhs, arg, .. } => {
                let _ = arg.eval(mem, stack);
                let x = stack[0];
                let y = stack[1];

                let u = match op.as_str() {
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "tan" => x.tan(),
                    "sinh" => x.sinh(),
                    "cosh" => x.cosh(),
                    "tanh" => x.tanh(),
                    "arcsin" => x.asin(),
                    "arccos" => x.acos(),
                    "arctan" => x.atan(),
                    "arcsinh" => x.asinh(),
                    "arccosh" => x.acosh(),
                    "arctanh" => x.atanh(),
                    "exp" => x.exp(),
                    "ln" => x.ln(),
                    "log" => x.log10(),
                    "power" => x.powf(y),
                    op => {
                        let f = VirtualTable::<f64>::from_str(op)
                            .unwrap_or_else(|_| panic!("operation {} not found.", op));
                        f(x, y)
                    }
                };

                if let Node::Var { sym, .. } = lhs {
                    match sym.borrow().loc {
                        Loc::Stack(idx) => stack[idx as usize] = u,
                        Loc::Mem(idx) => mem[idx as usize] = u,
                    }
                };
            }
        };
        f64::NAN
    }
}
