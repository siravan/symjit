use std::collections::HashMap;
use std::collections::HashSet;

use anyhow::Result;

use crate::amd::{AmdFamily, AmdGenerator};
use crate::code::VirtualTable;
use crate::generator::Generator;
use crate::model::Expr;

#[derive(Debug, Clone, Copy)]
pub enum Loc {
    Stack(u32),
    Mem(u32),
}

#[derive(Debug, Clone)]
pub struct Symbol {
    name: String,
    loc: Loc,
    reg: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    syms: HashMap<String, Symbol>,
    num_stack: usize,
    num_mem: usize,
}

impl SymbolTable {
    const SPILL_AREA: usize = 16;

    pub fn new() -> SymbolTable {
        let mut s = SymbolTable {
            syms: HashMap::new(),
            num_stack: 0,
            num_mem: 0,
        };

        for i in 0..SymbolTable::SPILL_AREA {
            s.add_stack(&format!("μ{}", i));
        }

        s
    }

    pub fn add_mem(&mut self, name: &str) -> Loc {
        match self.find(name) {
            Some(loc) => loc,
            None => {
                let loc = Loc::Mem(self.num_mem as u32);
                self.num_mem += 1;
                let sym = Symbol {
                    name: name.to_string(),
                    loc,
                    reg: None,
                };
                self.syms.insert(name.to_string(), sym);
                loc
            }
        }
    }

    pub fn add_stack(&mut self, name: &str) -> Loc {
        match self.find(name) {
            Some(loc) => loc,
            None => {
                let loc = Loc::Stack(self.num_stack as u32);
                self.num_stack += 1;
                let sym = Symbol {
                    name: name.to_string(),
                    loc,
                    reg: None,
                };
                self.syms.insert(name.to_string(), sym);
                loc
            }
        }
    }

    pub fn find(&self, name: &str) -> Option<Loc> {
        match self.syms.get(name) {
            Some(sym) => Some(sym.loc),
            None => None,
        }
    }
}

//****************************************************//

#[derive(Debug, Clone)]
pub enum Node {
    Void,
    Const {
        val: f64,
        idx: u32,
    },
    Var {
        name: String,
        loc: Loc,
    },
    Unary {
        op: String,
        arg: Box<Node>,
        ershov: usize,
    },
    Binary {
        op: String,
        left: Box<Node>,
        right: Box<Node>,
        ershov: usize,
    },
}

impl Node {
    pub fn ershov_number(&self) -> usize {
        match self {
            Node::Void => 0,
            Node::Const { .. } => 1,
            Node::Var { .. } => 1,
            Node::Unary { ershov, .. } => *ershov,
            Node::Binary { ershov, .. } => *ershov,
        }
    }

    pub fn compile(&self, ir: &mut AmdGenerator, base: u8) -> u8 {
        match self {
            Node::Void => 0,
            Node::Const { .. } => self.compile_const(ir, base),
            Node::Var { .. } => self.compile_var(ir, base),
            Node::Unary { .. } => self.compile_unary(ir, base),
            Node::Binary { .. } => self.compile_binary(ir, base),
        }
    }

    fn compile_const(&self, ir: &mut AmdGenerator, base: u8) -> u8 {
        if let Node::Const { idx, .. } = &self {
            let r = ir.first_shadow() + base;
            let label = format!("_const_{}_", idx);
            ir.load_const(r, &label);
            r
        } else {
            panic!("should not get here!");
        }
    }

    fn compile_var(&self, ir: &mut AmdGenerator, base: u8) -> u8 {
        if let Node::Var { loc, .. } = &self {
            let r = ir.first_shadow() + base;
            match loc {
                Loc::Stack(idx) => ir.load_stack(r, *idx),
                Loc::Mem(idx) => ir.load_mem(r, *idx),
            };
            r
        } else {
            panic!("should not get here!");
        }
    }

    fn compile_unary(&self, ir: &mut AmdGenerator, base: u8) -> u8 {
        if let Node::Unary { op, arg, ershov } = self {
            let r = arg.compile(ir, base);

            match op.as_str() {
                "neg" => ir.neg(r),
                "not" => ir.not(r),
                "abs" => ir.abs(r),
                "root" => ir.root(r),
                "square" => ir.square(r),
                "cube" => ir.cube(r),
                "recip" => ir.recip(r),
                "_call_" => {
                    if r != 0 {
                        ir.fmov(0, r);
                    }
                }
                _ => panic!("unary operation is not recognized"),
            };

            r
        } else {
            panic!("should not get here!");
        }
    }

    fn compile_binary(&self, ir: &mut AmdGenerator, base: u8) -> u8 {
        if let Node::Binary {
            op,
            left,
            right,
            ershov,
        } = self
        {
            let (dst, l, r) = self.alloc(ir, base, left, right, *ershov);

            match op.as_str() {
                "plus" => ir.plus(dst, l, r),
                "minus" => ir.minus(dst, l, r),
                "times" => ir.times(dst, l, r),
                "divide" => ir.divide(dst, l, r),
                "gt" => ir.gt(dst, l, r),
                "geq" => ir.geq(dst, l, r),
                "lt" => ir.lt(dst, l, r),
                "leq" => ir.leq(dst, l, r),
                "eq" => ir.eq(dst, l, r),
                "neq" => ir.neq(dst, l, r),
                "and" => ir.and(dst, l, r),
                "or" => ir.or(dst, l, r),
                "xor" => ir.xor(dst, l, r),
                "select_if" => ir.select_if(dst, l, r),
                "select_else" => ir.select_else(dst, l, r),
                "_call_" => Self::call(ir, l, r),
                _ => panic!("binary operation is not recognized"),
            };

            dst
        } else {
            panic!("should not get here!");
        }
    }

    fn alloc(
        &self,
        ir: &mut AmdGenerator,
        base: u8,
        left: &Node,
        right: &Node,
        ershov: usize,
    ) -> (u8, u8, u8) {
        let mut dst = ir.first_shadow() + base + (ershov as u8) - 1;

        let el = left.ershov_number();
        let er = right.ershov_number();

        let mut l = 0;
        let mut r = 0;

        let last = ir.first_shadow() + ir.count_shadows();

        if dst < last {
            if el == er {
                l = left.compile(ir, base + 1);
                r = right.compile(ir, base);
            } else if el > er {
                l = left.compile(ir, base);
                r = right.compile(ir, base);
            } else {
                r = right.compile(ir, base);
                l = left.compile(ir, base);
            }
        } else {
            let spill: u32 = (dst - last) as u32;

            if er <= el {
                l = left.compile(ir, 0);
                ir.save_stack(l, spill);
                r = right.compile(ir, 0);
                l = 1;
                ir.load_stack(l, spill);
            } else {
                r = right.compile(ir, 0);
                ir.save_stack(r, spill);
                l = left.compile(ir, 0);
                r = 1;
                ir.load_stack(r, spill);
            }

            dst = 0;
        };

        (dst, l, r)
    }

    fn call(ir: &mut AmdGenerator, l: u8, r: u8) {
        if l == 1 && r == 0 {
            ir.fxchg(1, 0);
        } else if r == 0 {
            ir.fmov(1, 0);
            if l != 0 {
                ir.fmov(0, l);
            }
        } else {
            if l != 0 {
                ir.fmov(0, l);
            }
            if r != 0 {
                ir.fmov(1, r);
            }
        }
    }
}

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
    fn assign(lhs: Node, rhs: Node) -> Statement {
        Statement::Assign { lhs, rhs }
    }

    fn call(op: &str, lhs: Node, arg: Node, num_args: usize) -> Statement {
        Statement::Call {
            op: op.to_string(),
            lhs,
            arg,
            num_args,
        }
    }

    pub fn compile(&self, builder: &Builder, ir: &mut AmdGenerator) {
        match &self {
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

    fn load(ir: &mut AmdGenerator, r: u8, v: &Node) {
        if let Node::Var { loc, .. } = v {
            match loc {
                Loc::Stack(idx) => ir.load_stack(r, *idx),
                Loc::Mem(idx) => ir.load_mem(r, *idx),
            }
        }
    }

    fn save(ir: &mut AmdGenerator, r: u8, v: &Node) {
        if let Node::Var { loc, .. } = v {
            match loc {
                Loc::Stack(idx) => ir.save_stack(r, *idx),
                Loc::Mem(idx) => ir.save_mem(r, *idx),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Builder {
    pub stmts: Vec<Statement>,
    pub consts: Vec<f64>,
    pub sym_table: SymbolTable,
    pub num_tmp: usize,
    pub ft: HashSet<String>, // function table (the name of functions)
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
        }
    }

    pub fn add_assign(&mut self, lhs: Node, rhs: Node) {
        self.stmts.push(Statement::assign(lhs, rhs));
    }

    pub fn add_call(&mut self, op: &str, lhs: Node, args: Vec<Node>) {
        let arg = match args.len() {
            1 => self.create_unary("_call_", args[0].clone()),
            2 => self.create_binary("_call_", args[0].clone(), args[1].clone()),
            _ => {
                panic!("more than two arguments are not supported yet!");
            }
        };

        self.stmts.push(Statement::call(op, lhs, arg, args.len()));
        self.ft.insert(op.to_string());
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
        let ershov = arg.ershov_number();
        Node::Unary {
            op: op.to_string(),
            arg: Box::new(arg),
            ershov,
        }
    }

    pub fn create_binary(&mut self, op: &str, left: Node, right: Node) -> Node {
        let l = left.ershov_number();
        let r = right.ershov_number();
        let ershov = if l == r { l + 1 } else { l.max(r) };
        Node::Binary {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
            ershov,
        }
    }

    pub fn add_mem(&mut self, name: &str) {
        self.sym_table.add_mem(name);
    }

    pub fn add_stack(&mut self, name: &str) {
        self.sym_table.add_stack(name);
    }

    pub fn add_tmp(&mut self) -> (Node, String) {
        let name = format!("ψ{}", self.num_tmp);
        self.num_tmp += 1;
        let loc = self.sym_table.add_stack(name.as_str());
        let tmp = Node::Var {
            name: name.to_string(),
            loc,
        };

        (tmp, name.to_string())
    }

    pub fn compile(&mut self, family: AmdFamily) -> Box<dyn Generator> {
        let mut ir = Box::new(AmdGenerator::new(family));

        let cap = self.sym_table.num_stack;
        let pad = cap & 1;
        let n: u32 = (cap + pad) as u32;

        ir.prologue(n);

        for stmt in self.stmts.iter() {
            stmt.compile(&self, &mut ir);
        }

        ir.epilogue(n);
        self.append_const_section(&mut ir);
        self.append_vt_section(&mut ir);
        ir.apply_jumps();
        // println!("{:02x?}", ir.bytes());

        ir
    }

    fn append_const_section(&self, ir: &mut AmdGenerator) {
        for (idx, val) in self.consts.iter().enumerate() {
            let label = format!("_const_{}_", idx);
            ir.set_label(label.as_str());
            let u: u64 = unsafe { std::mem::transmute(*val) };
            ir.append_quad(u);
        }
    }

    fn append_vt_section(&self, ir: &mut AmdGenerator) {
        for f in self.ft.iter() {
            let label = format!("_func_{}_", f);
            ir.set_label(label.as_str());
            let p = VirtualTable::<f64>::from_str(f).expect("func not found");
            let u: u64 = unsafe { std::mem::transmute(p) };
            ir.append_quad(u);
        }
    }
}

pub trait Transformer {
    fn transform(&self, builder: &mut Builder) -> Result<Node>;
}
