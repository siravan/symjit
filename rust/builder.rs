use anyhow::{anyhow, Result};
use std::collections::HashSet;

use crate::block::Block;
use crate::code::{Func, VirtualTable};
use crate::generator::Generator;
use crate::mir::Mir;
use crate::node::Node;

//****************************************************//

#[derive(Debug, Clone)]
pub struct Builder {
    pub block: Block,
    pub consts: Vec<f64>,
    pub ft: HashSet<String>, // function table (the name of functions),
}

impl Builder {
    pub fn new(cse: bool) -> Builder {
        Builder {
            block: Block::new(cse),
            consts: Vec::new(),
            ft: HashSet::new(),
        }
    }

    pub fn add_assign(&mut self, lhs: Node, rhs: Node) -> Result<Node> {
        self.block.add_assign(lhs.clone(), rhs);
        Ok(lhs)
    }

    pub fn add_call_unary(&mut self, op: &str, arg: Node) -> Result<Node> {
        let lhs = self.block.add_call_unary(op, arg);
        let f = VirtualTable::from_str(op)?; // check to see if op is defined
        if !matches!(f, Func::Unary(_)) {
            return Err(anyhow!("{} is not a unary function", op));
        }
        self.ft.insert(op.to_string());
        Ok(lhs)
    }

    pub fn add_call_binary(&mut self, op: &str, left: Node, right: Node) -> Result<Node> {
        if op == "power" {
            if let Some(val) = right.as_int_const() {
                match val {
                    0 => return self.create_const(1.0),
                    1 => return Ok(left),
                    2 => return self.create_unary("square", left),
                    3 => return self.create_unary("cube", left),
                    -1 => return self.create_unary("recip", left),
                    -2 => {
                        let arg = self.create_unary("square", left)?;
                        return self.create_unary("recip", arg);
                    }
                    -3 => {
                        let arg = self.create_unary("cube", left)?;
                        return self.create_unary("recip", arg);
                    }
                    _ => {
                        return self.create_powi(left, val);
                    }
                }
            };

            if let Some(val) = right.as_const() {
                const ONE_THIRD: f64 = 1.0 / 3.0;

                match val {
                    0.5 => return self.create_unary("root", left),
                    ONE_THIRD => return self.add_call_unary("cbrt", left),
                    1.5 => {
                        let arg = self.create_unary("cube", left)?;
                        return self.create_unary("root", arg);
                    }
                    _ => {}
                }
            }
        }

        let lhs = self.block.add_call_binary(op, left, right);
        let f = VirtualTable::from_str(op)?; // check to see if op is defined
        if !matches!(f, Func::Binary(_)) {
            return Err(anyhow!("{} is not a binary function", op));
        }
        self.ft.insert(op.to_string());
        Ok(lhs)
    }

    pub fn add_ifelse(&mut self, cond: Node, true_val: Node, false_val: Node) -> Result<Node> {
        let tmp = self.block.add_tmp();
        let tmp = self.add_assign(tmp, cond)?;
        let true_val = self.create_binary("select_if", tmp.clone(), true_val)?;
        let false_val = self.create_binary("select_else", tmp, false_val)?;

        self.create_binary("or", true_val, false_val)
    }

    pub fn create_void(&mut self) -> Result<Node> {
        Ok(self.block.create_void())
    }

    pub fn create_const(&mut self, val: f64) -> Result<Node> {
        for (idx, v) in self.consts.iter().enumerate() {
            if *v == val {
                return Ok(Node::Const {
                    val,
                    idx: idx as u32,
                });
            }
        }

        self.consts.push(val);

        Ok(self.block.create_const(val, (self.consts.len() - 1) as u32))
    }

    pub fn create_var(&mut self, name: &str) -> Result<Node> {
        let sym = self
            .block
            .sym_table
            .find_sym(name)
            .ok_or_else(|| anyhow!("variable {} not found", name))?;

        Ok(self.block.create_var(sym))
    }

    pub fn create_unary(&mut self, op: &str, arg: Node) -> Result<Node> {
        /*
        let node = match op {
            _ => self.block.create_unary(op, arg),
        };
        */
        let node = self.block.create_unary(op, arg);
        Ok(node)
    }

    pub fn create_powi(&mut self, arg: Node, power: i32) -> Result<Node> {
        Ok(self.block.create_powi(arg, power))
    }

    pub fn create_binary(&mut self, op: &str, left: Node, right: Node) -> Result<Node> {
        let node = match op {
            "times" if left.is_const(-1.0) => self.block.create_unary("neg", right),
            "times" if right.is_const(-1.0) => self.block.create_unary("neg", left),
            "times" if left.is_const(1.0) => right,
            "times" if right.is_const(1.0) => left,
            "times" if left.is_unary("recip") => {
                self.block
                    .create_binary("divide", right, left.arg().unwrap())
            }
            "times" if right.is_unary("recip") => {
                self.block
                    .create_binary("divide", left, right.arg().unwrap())
            }
            "plus" if left.is_unary("neg") => {
                self.block
                    .create_binary("minus", right, left.arg().unwrap())
            }
            "plus" if right.is_unary("neg") => {
                self.block
                    .create_binary("minus", left, right.arg().unwrap())
            }
            "rem" if left.is_unary("_powi_") => {
                let (arg, power) = left.arg_power().unwrap();
                self.block.create_modular_powi(arg, right, power)
            }
            "min" => {
                let cond = self.create_binary("leq", left.clone(), right.clone())?;
                self.add_ifelse(cond, left, right)?
            }
            "max" => {
                let cond = self.create_binary("geq", left.clone(), right.clone())?;
                self.add_ifelse(cond, left, right)?
            }
            "heaviside" => {
                /*
                 * In sympy, Heaviside is considered a binary operator,
                 * where the second argument is the value at 0 (defaults to 0.5).
                 */
                let zero = self.create_const(0.0)?;
                let one = self.create_const(1.0)?;

                let c0 = self.create_binary("eq", left.clone(), zero.clone())?;
                let x0 = self.add_ifelse(c0, right, one)?;

                let c1 = self.create_binary("geq", left, zero.clone())?;
                self.add_ifelse(c1, x0, zero)?
            }
            _ => self.block.create_binary(op, left, right),
        };

        Ok(node)
    }

    pub fn create_mir(&mut self, fastmath: bool) -> Result<Mir> {
        self.block.eliminate();
        let mut mir = Mir::new(fastmath);
        self.block.compile(&mut mir)?;
        mir.optimize_peephole();
        mir.add_consts(&self.consts);
        Ok(mir)
    }

    pub fn compile_from_mir(
        &mut self,
        mir: &Mir,
        ir: &mut impl Generator,
        count_states: usize,
        count_obs: usize,
    ) -> Result<()> {
        // println!("{:#?}", &self.block.stmts);
        let cap = self.block.sym_table.num_stack as u32;
        ir.prologue_indirect(cap, count_states, count_obs);

        //self.block.compile(ir)?;
        mir.rerun(ir);

        ir.epilogue_indirect(cap, count_states, count_obs);
        self.append_const_section(ir);
        self.append_vt_section(ir);
        ir.seal();
        // println!("{:#?}", &self.block.stmts);
        // println!("{:02x?}", ir.bytes());

        Ok(())
    }

    pub fn compile_fast_from_mir(
        &mut self,
        mir: &Mir,
        ir: &mut impl Generator,
        num_args: u32,
        idx_ret: i32,
    ) -> Result<()> {
        self.block.eliminate();
        // println!("{:#?}", &self.block.stmts);
        let cap = self.block.sym_table.num_stack as u32;
        ir.prologue_fast(cap, num_args);

        // self.block.compile(ir)?;
        mir.rerun(ir);

        ir.epilogue_fast(cap, idx_ret);
        self.append_const_section(ir);
        self.append_vt_section(ir);
        ir.seal();
        // println!("{:#?}", &self.block.stmts);
        // println!("{:02x?}", ir.bytes());

        Ok(())
    }

    fn append_const_section(&self, ir: &mut impl Generator) {
        ir.add_consts(&self.consts);
    }

    fn append_vt_section(&self, ir: &mut impl Generator) {
        for f in self.ft.iter() {
            let p = VirtualTable::from_str(f).expect("func not found");
            ir.add_func(f, p);
        }
    }
}
