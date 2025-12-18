use anyhow::Result;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

// use super::utils::Eval;
// use crate::generator::Generator;
use crate::mir::Mir;
use crate::node::{Node, VarStatus};
use crate::statement::Statement;
use crate::symbol::{Loc, Symbol, SymbolTable};
use crate::COUNT_SCRATCH;

use crate::builder::{BINARY, UNARY};

//****************************************************//

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Statement>,
    pub sym_table: SymbolTable,
    pub num_tmp: usize,
    pub cse: bool,
    pub calls: HashMap<(String, u64), Node>,
}

impl Block {
    pub fn new(cse: bool) -> Block {
        Block {
            stmts: Vec::new(),
            sym_table: SymbolTable::new(),
            num_tmp: 0,
            cse,
            calls: HashMap::new(),
        }
    }

    // add_* functions create a new Statement

    pub fn add_label(&mut self, label: &str) {
        self.stmts.push(Statement::Label {
            label: label.to_string(),
        });
    }

    pub fn add_branch(&mut self, cond: Node, label: &str) {
        self.stmts.push(Statement::Branch {
            cond,
            label: label.to_string(),
        })
    }

    pub fn add_assign(&mut self, lhs: Node, rhs: Node) {
        let rhs = self.process(rhs);
        self.stmts.push(Statement::assign(lhs, rhs));
    }

    // **************** Compile the Block! *********************

    pub fn compile(&mut self, ir: &mut Mir) -> Result<()> {
        for stmt in self.stmts.iter_mut() {
            stmt.compile(ir)?;
        }

        Ok(())
    }

    // create_* functions create a new Node

    pub fn create_tmp(&mut self) -> Node {
        let name = format!("ψ{}", self.num_tmp);
        self.num_tmp += 1;
        self.create_tmp_named(&name)
    }

    pub fn create_tmp_named(&mut self, name: &str) -> Node {
        self.sym_table.add_stack(name);
        let sym = self.sym_table.find_sym(name).unwrap();

        Node::Var {
            sym,
            status: VarStatus::Unknown,
        }
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
        Node::create_binary(op, left, right, 1, None)
    }

    pub fn create_powi(&mut self, arg: Node, power: i32) -> Node {
        Node::create_powi(arg, power)
    }

    pub fn create_modular_powi(&mut self, left: Node, right: Node, power: i32) -> Node {
        Node::create_modular_powi(left, right, power)
    }

    pub fn create_ifelse(&mut self, cond: Node, left: Node, right: Node) -> Node {
        let tmp = self.create_tmp();
        self.add_assign(tmp.clone(), cond);
        Node::create_ifelse(&tmp, left, right)
    }

    //******************* Tree Processing ***************************/
    fn process(&mut self, node: Node) -> Node {
        self.trim(node)
    }

    /*
     * trim breaks expressions to assure the ershov_number of the root does not
     * exceed the limit set by COUNT_SCRATCH
     * By default, COUNT_SCRATCH is 14, which is set because of 16 XMM/YMM registers
     * Note that two registers (XMM0 and XMM1) are needed as temporary and for function calls
     */
    fn trim(&mut self, node: Node) -> Node {
        match node {
            Node::Void => Node::Void,
            Node::Const { val, idx } => Node::Const { val, idx },
            Node::Var { sym, status } => Node::Var { sym, status },
            Node::Unary { op, arg, power, .. } => self.trim_unary(&op, *arg, power),
            Node::Binary {
                op,
                left,
                right,
                power,
                cond,
                ..
            } => self.trim_binary(&op, *left, *right, power, cond),
        }
    }

    fn trim_unary(&mut self, op: &str, arg: Node, power: i32) -> Node {
        let arg = self.trim(arg);

        if !UNARY.contains(&op) {
            self.break_call_unary(op, arg)
        } else {
            Node::create_unary(op, arg, power)
        }
    }

    fn break_call_unary(&mut self, op: &str, arg: Node) -> Node {
        let n = (op.to_string(), arg.hashof());

        if self.cse {
            if let Some(lhs) = self.calls.get(&n) {
                return lhs.clone();
            }
        }

        let arg = self.create_unary("_call_", arg);
        let lhs = self.create_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 1));
        self.calls.insert(n, lhs.clone());
        lhs
    }

    fn trim_binary(
        &mut self,
        op: &str,
        left: Node,
        right: Node,
        power: i32,
        cond: Option<Loc>,
    ) -> Node {
        let left = self.trim(left);
        let right = self.trim(right);

        if !BINARY.contains(&op) {
            return self.break_call_binary(op, left, right);
        }

        let right = if left.ershov_number() == COUNT_SCRATCH - 1
            && right.ershov_number() == COUNT_SCRATCH - 1
        {
            let lhs = self.create_tmp();
            self.stmts.push(Statement::assign(lhs.clone(), right));
            lhs
        } else {
            right
        };

        Node::create_binary(op, left, right, power, cond)
    }

    pub fn break_call_binary(&mut self, op: &str, left: Node, right: Node) -> Node {
        let n = (op.to_string(), left.hashof() ^ (right.hashof() + 1));

        if self.cse {
            if let Some(lhs) = self.calls.get(&n) {
                return lhs.clone();
            }
        }

        let left = self.process(left);
        let right = self.process(right);

        let arg = self.create_binary("_call_", left, right);
        let lhs = self.create_tmp();
        self.stmts.push(Statement::call(op, lhs.clone(), arg, 2));
        self.calls.insert(n, lhs.clone());
        lhs
    }

    /*
     * eliminate performs common-subexpression-eliminaton
     * the actual CSE work is done in elimination_pass, which uses
     * a two-pass algorithm.
     * In the first pass, common subexpressions are identified.
     * In the second pass, the right side of statements are rewritten.
     */
    pub fn eliminate(&mut self) {
        for _ in 0..5 {
            if !self.elimination_pass() {
                return;
            }
        }
    }

    pub fn elimination_pass(&mut self) -> bool {
        if !self.cse {
            return false;
        }

        // first-pass
        let mut stmts = std::mem::take(&mut self.stmts);

        let mut hs: HashSet<u64> = HashSet::new(); // hash-value-set to find collision
        let mut cs: HashMap<u64, (Node, Node)> = HashMap::new(); // collision set as (lhs, rhs)

        let mut depth: i32 = 0;

        for s in stmts.iter_mut() {
            match s {
                Statement::Assign { rhs, .. } => {
                    if depth == 0 {
                        self.find_cse(&mut hs, &mut cs, rhs);
                    }
                }
                Statement::Call { arg, .. } => {
                    if depth == 0 {
                        self.find_cse(&mut hs, &mut cs, arg);
                    }
                }
                Statement::Label { .. } => {
                    // The logic here with depth works for Sum/Product
                    // but needs improvement for general multi-block situations
                    depth += 1;
                }
                Statement::Branch { .. } => {
                    assert!(depth > 0);
                    depth -= 1;
                }
            }
        }

        if cs.is_empty() {
            // self.stmts = stmts.drain(..).collect();
            self.stmts = std::mem::take(&mut stmts);
            return false;
        }

        // println!("{} sub-expressions found.", cs.len());

        let mut ls: HashSet<u64> = HashSet::new(); // a set of common subexpression lhs which are added to self.stmts

        for s in stmts {
            match s {
                Statement::Assign { lhs, rhs } => {
                    let rhs = self.rewrite_cse(&cs, &mut ls, rhs);
                    self.stmts.push(Statement::Assign { lhs, rhs });
                }
                Statement::Call {
                    op,
                    lhs,
                    arg,
                    num_args,
                } => {
                    let arg = self.rewrite_cse(&cs, &mut ls, arg);
                    self.stmts.push(Statement::Call {
                        op,
                        lhs,
                        arg,
                        num_args,
                    });
                }
                Statement::Label { label } => {
                    // TODO: Just copying here, may need to change the logic
                    self.stmts.push(Statement::Label { label });
                }
                Statement::Branch { cond, label } => {
                    self.stmts.push(Statement::Branch { cond, label });
                }
            }
        }

        true
    }

    fn find_cse(
        &mut self,
        hs: &mut HashSet<u64>,
        cs: &mut HashMap<u64, (Node, Node)>,
        node: &mut Node,
    ) {
        if node.weightof() >= 5 && !node.is_unary("_call_") && !node.is_binary("_call_") {
            let h = node.hashof();

            if hs.contains(&h) {
                // collision detected!
                cs.entry(h).or_insert_with(|| {
                    let lhs = self.create_tmp();
                    (lhs, node.clone())
                });
            } else {
                hs.insert(h);
            };
        }

        if let Some(n) = node.first() {
            self.find_cse(hs, cs, n)
        };

        if let Some(n) = node.second() {
            self.find_cse(hs, cs, n)
        };
    }

    fn rewrite_cse(
        &mut self,
        cs: &HashMap<u64, (Node, Node)>,
        ls: &mut HashSet<u64>,
        node: Node,
    ) -> Node {
        if node.weightof() < 5 {
            return node;
        }

        match node {
            Node::Void => Node::Void,
            Node::Const { val, idx } => Node::Const { val, idx },
            Node::Var { sym, status } => Node::Var { sym, status },
            Node::Unary {
                op, arg, power, h, ..
            } => self.common_subexpr(cs, ls, h).unwrap_or_else(|| {
                let arg = self.rewrite_cse(cs, ls, *arg);
                Node::create_unary(op.as_str(), arg, power)
            }),
            Node::Binary {
                op,
                left,
                right,
                power,
                cond,
                h,
                ..
            } => self.common_subexpr(cs, ls, h).unwrap_or_else(|| {
                let left = self.rewrite_cse(cs, ls, *left);
                let right = self.rewrite_cse(cs, ls, *right);
                Node::create_binary(op.as_str(), left, right, power, cond)
            }),
        }
    }

    fn common_subexpr(
        &mut self,
        cs: &HashMap<u64, (Node, Node)>,
        ls: &mut HashSet<u64>,
        h: u64,
    ) -> Option<Node> {
        if let Some((lhs, rhs)) = cs.get(&h) {
            let k = &lhs.hashof();

            if !ls.contains(k) {
                self.stmts.push(Statement::assign(lhs.clone(), rhs.clone()));
                ls.insert(*k);
            }

            return Some(lhs.clone());
        }

        None
    }
}
