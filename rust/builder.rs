use anyhow::{anyhow, Result};
use std::collections::HashSet;

use super::utils::{Compiled, Eval};
use crate::code::VirtualTable;
use crate::generator::Generator;
use crate::node::{Node, VarStatus};
use crate::statement::Statement;
use crate::symbol::SymbolTable;

//****************************************************//

#[derive(Debug, Clone)]
pub struct Builder {
    pub stmts: Vec<Statement>,
    pub consts: Vec<f64>,
    pub sym_table: SymbolTable,
    pub num_tmp: usize,
    pub ft: HashSet<String>, // function table (the name of functions),
    pub intrinsic_unary: Vec<&'static str>,
    pub intrinsic_binary: Vec<&'static str>,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            stmts: Vec::new(),
            consts: Vec::new(),
            sym_table: SymbolTable::new(),
            num_tmp: 0,
            ft: HashSet::new(),
            // the list of intrinsic unary ops, i.e., operations that can be implemented directly in
            // machine code
            intrinsic_unary: vec![
                "abs", "not", "root", "square", "cube", "recip", "round", "floor", "ceiling",
                "trunc",
            ],
            // the list of intrinsic binary ops, i.e., operations that can be implemented directly in
            // machine code
            intrinsic_binary: vec![
                "plus", "minus", "times", "divide", "rem", "gt", "geq", "lt", "leq", "eq", "neq",
                "and", "or", "xor", "if_pos", "if_neg",
            ],
        }
    }

    pub fn add_assign(&mut self, lhs: Node, rhs: Node) -> Result<Node> {
        self.stmts.push(Statement::assign(lhs.clone(), rhs));
        Ok(lhs)
    }

    pub fn add_call_unary(&mut self, op: &str, arg: Node) -> Result<Node> {
        let arg = self.create_unary("_call_", arg)?;
        let lhs = self.add_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 1));
        let _ = VirtualTable::<f64>::from_str(op)?; // check to see if op is defined
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
                match val {
                    0.5 => return self.create_unary("root", left),
                    1.5 => {
                        let arg = self.create_unary("cube", left)?;
                        return self.create_unary("root", arg);
                    }
                    _ => {}
                }
            }
        }

        let arg = self.create_binary("_call_", left, right)?;
        let lhs = self.add_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 2));
        let _ = VirtualTable::<f64>::from_str(op)?; // check to see if op is defined
        self.ft.insert(op.to_string());

        Ok(lhs)
    }

    pub fn add_ifelse(&mut self, cond: Node, true_val: Node, false_val: Node) -> Result<Node> {
        let tmp = self.add_tmp();
        let tmp = self.add_assign(tmp, cond)?;
        let true_val = self.create_binary("select_if", tmp.clone(), true_val)?;
        let false_val = self.create_binary("select_else", tmp, false_val)?;

        self.create_binary("or", true_val, false_val)
    }

    pub fn create_void(&mut self) -> Result<Node> {
        Ok(Node::create_void())
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

        Ok(Node::create_const(val, (self.consts.len() - 1) as u32))
    }

    pub fn create_var(&mut self, name: &str) -> Result<Node> {
        let sym = self
            .sym_table
            .find_sym(name)
            .ok_or_else(|| anyhow!("variable {} not found", name))?;

        Ok(Node::create_var(sym))
    }

    pub fn create_unary(&mut self, op: &str, arg: Node) -> Result<Node> {
        Ok(Node::create_unary(op, arg))
    }

    pub fn create_powi(&mut self, arg: Node, power: i32) -> Result<Node> {
        Ok(Node::create_powi(arg, power))
    }

    pub fn create_binary(&mut self, op: &str, left: Node, right: Node) -> Result<Node> {
        let node = match op {
            "times" if left.is_const(-1.0) => Node::create_unary("neg", right),
            "times" if right.is_const(-1.0) => Node::create_unary("neg", left),
            "times" if left.is_const(1.0) => right,
            "times" if right.is_const(1.0) => left,
            "times" if left.is_unary("recip") => {
                Node::create_binary("divide", right, left.arg().unwrap())
            }
            "times" if right.is_unary("recip") => {
                Node::create_binary("divide", left, right.arg().unwrap())
            }
            "plus" if left.is_unary("neg") => {
                Node::create_binary("minus", right, left.arg().unwrap())
            }
            "plus" if right.is_unary("neg") => {
                Node::create_binary("minus", left, right.arg().unwrap())
            }
            "rem" if left.is_unary("_powi_") => {
                let (arg, power) = left.arg_power().unwrap();
                Node::create_modular_powi(arg, right, power)
            }
            _ => Node::create_binary(op, left, right),
        };

        Ok(node)
    }

    pub fn add_tmp(&mut self) -> Node {
        let name = format!("ψ{}", self.num_tmp);
        self.num_tmp += 1;
        self.sym_table.add_stack(name.as_str());
        let sym = self.sym_table.find_sym(name.as_str()).unwrap();

        Node::Var {
            sym,
            status: VarStatus::Unknown,
        }
    }

    pub fn compile(&mut self, ir: &mut impl Generator) -> Result<()> {
        let cap = self.sym_table.num_stack as u32;
        ir.prologue(cap);

        for stmt in self.stmts.iter_mut() {
            stmt.compile(ir)?;
        }

        ir.epilogue(cap);
        self.append_const_section(ir);
        self.append_vt_section(ir);
        ir.apply_jumps();
        // println!("{:?}", &self.stmts);
        // println!("{:02x?}", ir.bytes());

        Ok(())
    }

    pub fn compile_fast(
        &mut self,
        ir: &mut impl Generator,
        num_args: u32,
        idx_ret: i32,
    ) -> Result<()> {
        let cap = self.sym_table.num_stack as u32;
        ir.prologue_fast(cap, num_args);

        for stmt in self.stmts.iter_mut() {
            stmt.compile(ir)?;
        }

        ir.epilogue_fast(cap, idx_ret);
        self.append_const_section(ir);
        self.append_vt_section(ir);
        ir.apply_jumps();
        // println!("{:?}", &self.stmts);
        // println!("{:02x?}", ir.bytes());

        Ok(())
    }

    fn append_const_section(&self, ir: &mut impl Generator) {
        for (idx, val) in self.consts.iter().enumerate() {
            let label = format!("_const_{}_", idx);
            ir.set_label(label.as_str());
            ir.append_quad((*val).to_bits());
        }
    }

    fn append_vt_section(&self, ir: &mut impl Generator) {
        for f in self.ft.iter() {
            let label = format!("_func_{}_", f);
            ir.set_label(label.as_str());
            let p = VirtualTable::<f64>::from_str(f).expect("func not found");
            ir.append_quad(p as usize as u64);
        }
    }
}

impl Eval for Builder {
    fn eval(&self, mem: &mut [f64], stack: &mut [f64]) -> f64 {
        for stmt in self.stmts.iter() {
            stmt.eval(mem, stack);
        }
        f64::NAN
    }
}

/************************************************/

pub struct ByteCode {
    builder: Builder,
    mem: Vec<f64>,
    stack: Vec<f64>,
}

impl ByteCode {
    pub fn new(builder: Builder, mem: Vec<f64>) -> ByteCode {
        let stack: Vec<f64> = vec![0.0; builder.sym_table.num_stack];

        ByteCode {
            builder,
            mem,
            stack,
        }
    }
}

impl Compiled<f64> for ByteCode {
    fn exec(&mut self) {
        self.builder.eval(&mut self.mem[..], &mut self.stack[..]);
    }

    fn mem(&self) -> &[f64] {
        &self.mem[..]
    }

    fn mem_mut(&mut self) -> &mut [f64] {
        &mut self.mem[..]
    }

    fn dump(&self, _name: &str) {}

    fn func(&self) -> fn(&[f64]) {
        unreachable!()
    }
}
