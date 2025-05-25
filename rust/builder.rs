use std::collections::HashSet;

use super::utils::{Compiled, Eval};
use crate::code::VirtualTable;
use crate::generator::Generator;
use crate::model::Expr;
use crate::node::Node;
use crate::statement::Statement;
use crate::symbol::{Loc, SymbolTable};

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
    const first_shadow: u8 = 2;

    pub fn new() -> Builder {
        Builder {
            stmts: Vec::new(),
            consts: Vec::new(),
            sym_table: SymbolTable::new(),
            num_tmp: 0,
            ft: HashSet::new(),
            // the list of intrinsic unary ops, i.e., operations that can be implemented directly in
            // machine code
            intrinsic_unary: vec!["neq", "abs", "not", "root", "square", "cube", "recip"],
            // the list of intrinsic binary ops, i.e., operations that can be implemented directly in
            // machine code
            intrinsic_binary: vec![
                "plus", "minus", "neg", "times", "divide", "gt", "geq", "lt", "leq", "eq", "neq",
                "and", "or", "xor", "if_pos", "if_neg",
            ],
        }
    }

    pub fn add_assign(&mut self, lhs: Node, rhs: Node) -> Node {
        self.stmts.push(Statement::assign(lhs.clone(), rhs));
        lhs
    }
    
    pub fn add_call_unary(&mut self, op: &str, arg: Node) -> Node {
        let arg = self.create_unary("_call_", arg);
        let lhs = self.add_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 1));
        self.ft.insert(op.to_string());
        
        lhs
    }


    pub fn add_call_binary(&mut self, op: &str, left: Node, right: Node) -> Node {
        if op == "power" {
            if right.is_const(0.0) {
                return self.create_const(1.0);
            } else if right.is_const(1.0) {
                return left;
            } else if right.is_const(2.0) {
                return self.create_unary("square", left);
            } else if right.is_const(3.0) {
                return self.create_unary("cube", left);
            } else if right.is_const(0.5) {
                return self.create_unary("root", left);
            } else if right.is_const(1.5) {
                let arg = self.create_unary("cube", left);
                return self.create_unary("root", arg);
            } else if right.is_const(-1.0) {
                return self.create_unary("recip", left);
            } else if right.is_const(-2.0) {
                let arg = self.create_unary("square", left);
                return self.create_unary("recip", arg);
            } else if right.is_const(-3.0) {
                let arg = self.create_unary("cube", left);
                return self.create_unary("recip", arg);
            }
        }

        let arg = self.create_binary("_call_", left, right);
        let lhs = self.add_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 2));
        self.ft.insert(op.to_string());

        lhs
    }

    pub fn create_void(&mut self) -> Node {
        Node::Void
    }

    pub fn create_const(&mut self, val: f64) -> Node {
        for (idx, v) in self.consts.iter().enumerate() {
            if *v == val {
                return Node::Const {
                    val,
                    idx: idx as u32,
                };
            }
        }

        self.consts.push(val);
        Node::Const {
            val,
            idx: (self.consts.len() - 1) as u32,
        }
    }

    pub fn create_var(&mut self, name: &str) -> Node {
        let loc = self
            .sym_table
            .find(name)
            .expect(&format!("variable {} not found", name));
        Node::Var {
            name: name.to_string(),
            loc,
        }
    }

    pub fn create_unary(&mut self, op: &str, arg: Node) -> Node {
        Node::Unary {
            op: op.to_string(),
            arg: Box::new(arg),
        }
    }

    pub fn create_binary_op(&mut self, op: &str, mut left: Node, mut right: Node) -> Node {
        Node::Binary {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn create_binary(&mut self, op: &str, mut left: Node, mut right: Node) -> Node {
        match op {
            "times" if left.is_const(-1.0) => self.create_unary("neg", right),
            "times" if right.is_const(-1.0) => self.create_unary("neg", left),
            "times" if left.is_unary("recip") => {
                self.create_binary_op("divide", right, left.arg().unwrap())
            },
            "times" if right.is_unary("recip") => {
                self.create_binary_op("divide", left, right.arg().unwrap())
            },            
            "plus" if left.is_unary("neg") => {
                self.create_binary_op("minus", right, left.arg().unwrap())
            }
            "plus" if right.is_unary("neg") => {
                self.create_binary_op("minus", left, right.arg().unwrap())
            }
            _ => self.create_binary_op(op, left, right),
        }
    }

    pub fn add_mem(&mut self, name: &str) {
        self.sym_table.add_mem(name);
    }

    pub fn add_stack(&mut self, name: &str) {
        self.sym_table.add_stack(name);
    }

    pub fn add_tmp(&mut self) -> Node {
        let name = format!("ψ{}", self.num_tmp);
        self.num_tmp += 1;
        let loc = self.sym_table.add_stack(name.as_str());
        let tmp = Node::Var {
            name: name.to_string(),
            loc,
        };

        tmp
    }

    pub fn compile(&mut self, ir: &mut impl Generator) {
        let cap = self.sym_table.num_stack;
        let pad = cap & 1;
        let n: u32 = (cap + pad) as u32;

        ir.prologue(n);

        for stmt in self.stmts.iter() {
            stmt.compile(ir);
        }

        ir.epilogue(n);
        self.append_const_section(ir);
        self.append_vt_section(ir);
        ir.apply_jumps();
        // println!("{:?}", &self.stmts);
        // println!("{:02x?}", ir.bytes());
    }

    fn append_const_section(&self, ir: &mut impl Generator) {
        for (idx, val) in self.consts.iter().enumerate() {
            let label = format!("_const_{}_", idx);
            ir.set_label(label.as_str());
            let u: u64 = unsafe { std::mem::transmute(*val) };
            ir.append_quad(u);
        }
    }

    fn append_vt_section(&self, ir: &mut impl Generator) {
        for f in self.ft.iter() {
            let label = format!("_func_{}_", f);
            ir.set_label(label.as_str());
            let p = VirtualTable::<f64>::from_str(f).expect("func not found");
            let u: u64 = unsafe { std::mem::transmute(p) };
            ir.append_quad(u);
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
    pub builder: Builder,
    pub mem: Vec<f64>,
    pub stack: Vec<f64>,
}

impl ByteCode {
    pub fn new(builder: Builder, mem: Vec<f64>, size: usize) -> ByteCode {
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

    fn dump(&self, name: &str) {}
}
