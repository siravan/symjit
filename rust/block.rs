use anyhow::Result;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::utils::Eval;
use crate::generator::Generator;
use crate::node::{Node, VarStatus};
use crate::statement::Statement;
use crate::symbol::{Symbol, SymbolTable};
use crate::COUNT_SCRATCH;

//****************************************************//

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Statement>,
    pub sym_table: SymbolTable,
    pub num_tmp: usize,
    pub cse: bool,
}

impl Block {
    pub fn new(cse: bool) -> Block {
        Block {
            stmts: Vec::new(),
            sym_table: SymbolTable::new(),
            num_tmp: 0,
            cse,
        }
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

    pub fn add_assign(&mut self, lhs: Node, rhs: Node) {
        let rhs = self.process(rhs);
        self.stmts.push(Statement::assign(lhs, rhs));
    }

    pub fn add_call_unary(&mut self, op: &str, arg: Node) -> Node {
        let arg = self.create_unary("_call_", arg);
        let arg = self.process(arg);
        let lhs = self.add_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 1));
        lhs
    }

    pub fn add_call_binary(&mut self, op: &str, left: Node, right: Node) -> Node {
        let arg = self.create_binary("_call_", left, right);
        let arg = self.process(arg);
        let lhs = self.add_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 2));
        lhs
    }

    pub fn compile(&mut self, ir: &mut impl Generator) -> Result<()> {
        for stmt in self.stmts.iter_mut() {
            stmt.compile(ir)?;
        }
        Ok(())
    }

    pub fn create_void(&mut self) -> Node {
        Node::create_void()
    }

    pub fn create_const(&mut self, val: f64, idx: u32) -> Node {
        Node::create_const(val, idx)
    }

    pub fn create_var(&mut self, sym: Rc<RefCell<Symbol>>) -> Node {
        Node::create_var(sym)
    }

    pub fn create_unary(&mut self, op: &str, arg: Node) -> Node {
        Node::create_unary(op, arg, 1)
    }

    pub fn create_binary(&mut self, op: &str, left: Node, right: Node) -> Node {
        Node::create_binary(op, left, right, 1)
    }

    pub fn create_powi(&mut self, arg: Node, power: i32) -> Node {
        Node::create_powi(arg, power)
    }

    pub fn create_modular_powi(&mut self, left: Node, right: Node, power: i32) -> Node {
        Node::create_modular_powi(left, right, power)
    }

    fn process(&mut self, node: Node) -> Node {
        let node = self.trim(node);
        self.eliminate(node)
    }

    fn trim(&mut self, node: Node) -> Node {
        if node.ershov_number() < COUNT_SCRATCH {
            return node;
        }

        // println!("ershov {}", node.ershov_number());

        match node {
            Node::Void => Node::Void,
            Node::Const { val, idx } => Node::Const { val, idx },
            Node::Var { sym, status } => Node::Var { sym, status },
            Node::Unary { op, arg, power, .. } => self.trim_unary(op, arg, power),
            Node::Binary {
                op,
                left,
                right,
                power,
                ..
            } => self.trim_binary(op, left, right, power),
        }
    }

    fn trim_unary(&mut self, op: String, arg: Box<Node>, power: i32) -> Node {
        let arg = self.trim(*arg);
        Node::create_unary(op.as_str(), arg, power)
    }

    fn trim_binary(&mut self, op: String, left: Box<Node>, right: Box<Node>, power: i32) -> Node {
        let left = self.trim(*left);
        let right = self.trim(*right);

        let right = if left.ershov_number() == COUNT_SCRATCH - 1
            && right.ershov_number() == COUNT_SCRATCH - 1
        {
            let lhs = self.add_tmp();
            let right = self.eliminate(right);
            self.stmts.push(Statement::assign(lhs.clone(), right));
            lhs
        } else {
            right
        };

        Node::create_binary(op.as_str(), left, right, power)
    }

    fn eliminate(&mut self, mut node: Node) -> Node {
        if !self.cse {
            return node;
        }

        let mut hs: HashSet<u64> = HashSet::new();
        let mut cs: HashMap<u64, Node> = HashMap::new();
        self.find_common_subexpr(&mut hs, &mut cs, &mut node);

        if cs.is_empty() {
            node
        } else {
            println!("common sub-expression elimination");
            self.subs_common(&cs, node)
        }
    }

    fn find_common_subexpr(
        &mut self,
        hs: &mut HashSet<u64>,
        cs: &mut HashMap<u64, Node>,
        node: &mut Node,
    ) {
        if node.weightof() > 2 {
            let h = node.hashof();

            if hs.contains(&h) {
                if !cs.contains_key(&h) {
                    let lhs = self.add_tmp();
                    self.stmts
                        .push(Statement::assign(lhs.clone(), node.clone()));
                    cs.insert(h, lhs);
                }
            } else {
                hs.insert(h);
            };
        }

        node.first().map(|n| self.find_common_subexpr(hs, cs, n));
        node.second().map(|n| self.find_common_subexpr(hs, cs, n));
    }

    fn subs_common(&mut self, cs: &HashMap<u64, Node>, node: Node) -> Node {
        if node.weightof() < 3 {
            return node;
        }

        match node {
            Node::Void => Node::Void,
            Node::Const { val, idx } => Node::Const { val, idx },
            Node::Var { sym, status } => Node::Var { sym, status },
            Node::Unary {
                op, arg, power, h, ..
            } => self.subs_common_unary(cs, op, arg, power, h),
            Node::Binary {
                op,
                left,
                right,
                power,
                h,
                ..
            } => self.subs_common_binary(cs, op, left, right, power, h),
        }
    }

    fn subs_common_unary(
        &mut self,
        cs: &HashMap<u64, Node>,
        op: String,
        arg: Box<Node>,
        power: i32,
        h: u64,
    ) -> Node {
        if let Some(tmp) = cs.get(&h) {
            return tmp.clone();
        }

        let arg = self.subs_common(cs, *arg);
        Node::create_unary(op.as_str(), arg, power)
    }

    fn subs_common_binary(
        &mut self,
        cs: &HashMap<u64, Node>,
        op: String,
        left: Box<Node>,
        right: Box<Node>,
        power: i32,
        h: u64,
    ) -> Node {
        if let Some(tmp) = cs.get(&h) {
            return tmp.clone();
        }

        let el = left.ershov_number();
        let er = right.ershov_number();

        if el >= er {
            let left = self.subs_common(cs, *left);
            let right = self.subs_common(cs, *right);
            Node::create_binary(op.as_str(), left, right, power)
        } else {
            let right = self.subs_common(cs, *right);
            let left = self.subs_common(cs, *left);
            Node::create_binary(op.as_str(), left, right, power)
        }
    }
}

impl Eval for Block {
    fn eval(&self, mem: &mut [f64], stack: &mut [f64], params: &[f64]) -> f64 {
        for stmt in self.stmts.iter() {
            stmt.eval(mem, stack, params);
        }
        f64::NAN
    }
}
