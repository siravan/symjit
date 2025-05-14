use std::collections::HashMap;

use crate::amd::AmdCompiler;
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
            None => None
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
            Node::Const {..} => 1,
            Node::Var {..} => 1,
            Node::Unary { ershov, .. } => *ershov,
            Node::Binary { ershov, .. } => *ershov,
        }
    }
    
    pub fn compile(&self, ir: &mut AmdCompiler, base: u8) -> u8 {
        match self {
            Node::Void => 0,
            Node::Const {idx,..} => self.compile_const(ir, base, *idx),
            Node::Var {loc,..} => self.compile_var(ir, base, *loc),
            Node::Unary {op, arg, ershov} => self.compile_unary(ir, base, op.as_str(), arg, *ershov),
            Node::Binary {op, left, right, ershov} => self.compile_binary(ir, base, op.as_str(), left, right, *ershov),
        }
    }
    
    fn compile_const(&self, ir: &mut AmdCompiler, base: u8, idx: u32) -> u8 {
        let r = 2 + base;
        ir.load_const(r, idx);
        r
    }
    
    fn compile_var(&self, ir: &mut AmdCompiler, base: u8, loc: Loc) -> u8 {
        let r = 2 + base;
        match loc {
            Loc::Stack(idx) => ir.load_stack(r, idx),
            Loc::Mem(idx) => ir.load_mem(r, idx),            
        };
        r
    }
    
    fn compile_unary(&self, ir: &mut AmdCompiler, base: u8, op: &str, arg: &Node, ershov: usize) -> u8 {
        let r = arg.compile(ir, base);
        
        match op {
            "neg" => ir.neg(r),
            "not" => ir.not(r),
            "abs" => ir.abs(r),
            "root" => ir.root(r),
            "square" => ir.square(r),
            "cube" => ir.cube(r),
            "recip" => ir.recip(r),
            _ => panic!("unary operation is not recognized")
        };

        r
    }
    
    fn compile_binary(&self, ir: &mut AmdCompiler, base: u8, op: &str, left: &Node, right: &Node, ershov: usize) -> u8 {
        let dst = 2 + base + (ershov as u8) - 1;
        let el = left.ershov_number();
        let er = right.ershov_number();
        
        let mut l = 0;
        let mut r = 0;
        
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
        
        match op {
            "plus" => ir.plus(dst, l, r),
            "minus"  => ir.minus(dst, l, r),
            "times"  => ir.times(dst, l, r),
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
            _ => panic!("binary operation is not recognized")
        };

        dst
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
        args: Vec<Node>,
    },
}

impl Statement {
    pub fn compile(&self, builder: &Builder, ir: &mut AmdCompiler) {
        match &self {
            Statement::Assign{lhs, rhs} => {
                let r = rhs.compile(ir, 0);
                
                if let Node::Var{loc, ..} = lhs {
                    match loc {
                        Loc::Stack(idx) => ir.save_stack(r, *idx),
                        Loc::Mem(idx) => ir.save_mem(r, *idx),
                    }
                }                   
            },
            Statement::Call{..} => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct Builder {
    pub stmts: Vec<Statement>,
    pub consts: Vec<f64>,
    pub sym_table: SymbolTable,
    pub num_tmp: usize,
}

impl Builder {
    const first_shadow : u8 = 2;

    pub fn new() -> Builder {
        Builder {
            stmts: Vec::new(),
            consts: Vec::new(),
            sym_table: SymbolTable::new(),
            num_tmp: 0,
        }
    }

    pub fn add_stmt(&mut self, st: Statement) {
        self.stmts.push(st);
    }

    pub fn create_void(&mut self) -> Node {
        Node::Void
    }

    pub fn create_const(&mut self, val: f64) -> Node {
        for (idx, v) in self.consts.iter().enumerate() {
            if *v == val {
                return Node::Const { val, idx : idx as u32 };
            }
        }

        self.consts.push(val);
        Node::Const {
            val,
            idx: (self.consts.len() - 1) as u32,
        }
    }

    pub fn create_var(&mut self, name: &str) -> Node {
        let loc = self.sym_table.find(name).expect("variable not found");        
        Node::Var {
            name: name.to_string(),
            loc
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
        let ershov = if l == r { l + 1 } else { l };
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
        let tmp = Node::Var { name: name.to_string(), loc };
        
        (tmp, name.to_string())
    }

    pub fn compile(&mut self) -> AmdCompiler {
        let mut ir = AmdCompiler::new();

        let cap = self.sym_table.num_stack;
        let pad = cap & 1;
        let n: u32 = ((cap + pad) * 8) as u32;

        ir.prologue(n);
        
        for stmt in self.stmts.iter() {
            stmt.compile(&self, &mut ir);
        }
        
        ir.epilogue(n);
        
        ir.apply_jumps();
        // println!("{:02x?}", ir.bytes());
        
        ir
    }
    
    pub fn mem(&self) -> Vec<f64> {
        vec![0.0; self.sym_table.num_mem]
    }
}

pub trait Transformer {
    fn transform(&self, builder: &mut Builder) -> Node;
}
