use super::utils::Eval;
use crate::builder::Builder;
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

    pub fn compile(&self, ir: &mut dyn Generator) {
        match self {
            Statement::Assign { lhs, rhs } => {
                let r = rhs.compile(ir, 0);
                Self::save(ir, r, lhs);
            }
            Statement::Call {
                op,
                lhs,
                arg,
                num_args,
            } => {
                let _ = arg.compile(ir, 0);
                let label = format!("_func_{}_", op);
                ir.call(&label, *num_args);
                Self::save(ir, 0, lhs);
            }
        }
    }

    /*
        fn load(ir: &mut dyn Generator, r: u8, v: &Node) {
            if let Node::Var { loc, .. } = v {
                match loc {
                    Loc::Stack(idx) => ir.load_stack(r, *idx),
                    Loc::Mem(idx) => ir.load_mem(r, *idx),
                }
            }
        }
    */

    fn save(ir: &mut dyn Generator, r: u8, v: &Node) {
        if let Node::Var { loc, .. } = v {
            match loc {
                Loc::Stack(idx) => ir.save_stack(r, *idx),
                Loc::Mem(idx) => ir.save_mem(r, *idx),
            }
        }
    }
}

impl Eval for Statement {
    fn eval(&self, mem: &mut [f64], stack: &mut [f64]) -> f64 {
        match &self {
            Statement::Assign { lhs, rhs } => {
                let u = rhs.eval(mem, stack);

                if let Node::Var { loc, .. } = lhs {
                    match loc {
                        Loc::Stack(idx) => stack[*idx as usize] = u,
                        Loc::Mem(idx) => mem[*idx as usize] = u,
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
                    "rem" => x % y,
                    _ => f64::NAN,
                };

                if let Node::Var { loc, .. } = lhs {
                    match loc {
                        Loc::Stack(idx) => stack[*idx as usize] = u,
                        Loc::Mem(idx) => mem[*idx as usize] = u,
                    }
                };
            }
        };
        f64::NAN
    }
}
