use anyhow::{anyhow, Result};

use std::cell::RefCell;
use std::rc::Rc;

use crate::generator::Generator;
use crate::symbol::{Loc, Symbol};
use crate::utils::{bool_to_f64, Eval};

#[derive(Debug, Clone)]
pub enum VarStatus {
    Unknown,
    First,
    Mid,
    Last,
    Singular,
}

#[derive(Debug, Clone)]
pub enum Node {
    Void,
    Const {
        val: f64,
        idx: u32,
    },
    Var {
        sym: Rc<RefCell<Symbol>>,
        status: VarStatus,
    },
    Unary {
        op: String,
        arg: Box<Node>,
        ershov: u8,
        power: i32,
    },
    Binary {
        op: String,
        left: Box<Node>,
        right: Box<Node>,
        ershov: u8,
        power: i32,
    },
}

impl Node {
    pub fn create_void() -> Node {
        Node::Void
    }

    pub fn create_const(val: f64, idx: u32) -> Node {
        Node::Const { val, idx }
    }

    pub fn create_var(sym: Rc<RefCell<Symbol>>) -> Node {
        Node::Var {
            sym,
            status: VarStatus::Unknown,
        }
    }

    pub fn create_unary(op: &str, arg: Node) -> Node {
        Node::Unary {
            op: op.to_string(),
            arg: Box::new(arg),
            ershov: 0,
            power: 1,
        }
    }

    pub fn create_binary(op: &str, left: Node, right: Node) -> Node {
        Node::Binary {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
            ershov: 0,
            power: 1,
        }
    }

    pub fn create_powi(arg: Node, power: i32) -> Node {
        Node::Unary {
            op: "_powi_".to_string(),
            arg: Box::new(arg),
            ershov: 0,
            power,
        }
    }

    pub fn create_modular_powi(left: Node, right: Node, power: i32) -> Node {
        Node::Binary {
            op: "_powi_mod_".to_string(),
            left: Box::new(left),
            right: Box::new(right),
            ershov: 0,
            power,
        }
    }

    /// postorder_forward does a forward postorder traversal of the
    /// expression tree and call f at each node.
    /// Nodes are visited in the same order used to generate code.
    /// Note the twist in the middle. The decision to traverse left
    /// or right link depends on ershov number of each link.
    fn postorder_forward(&mut self, f: fn(&mut Node)) {
        match self {
            Node::Void | Node::Const { .. } | Node::Var { .. } => {
                f(self);
            }
            Node::Unary { arg, .. } => {
                arg.postorder_forward(f);
                f(self);
            }
            Node::Binary { left, right, .. } => {
                let el = left.ershov_number();
                let er = right.ershov_number();

                if el >= er {
                    left.postorder_forward(f);
                    right.postorder_forward(f);
                } else {
                    right.postorder_forward(f);
                    left.postorder_forward(f);
                };

                f(self);
            }
        }
    }

    /// postorder_backward does a backward postorder traversal of the
    /// expression tree and call f at each node.
    /// Nodes are visited in the same reverse order used to generate code.
    /// Note the twist in the middle. The decision to traverse left
    /// or right link depends on ershov number of each link.
    fn postorder_backward(&mut self, f: fn(&mut Node)) {
        match self {
            Node::Void | Node::Const { .. } | Node::Var { .. } => {
                f(self);
            }
            Node::Unary { arg, .. } => {
                arg.postorder_backward(f);
                f(self);
            }
            Node::Binary { left, right, .. } => {
                let el = left.ershov_number();
                let er = right.ershov_number();

                if el >= er {
                    right.postorder_backward(f);
                    left.postorder_backward(f);
                } else {
                    left.postorder_backward(f);
                    right.postorder_backward(f);
                };

                f(self);
            }
        }
    }

    /// Ershov number is the number of temporary registers needed to
    /// compile a given node
    pub fn ershov_number(&self) -> u8 {
        match self {
            Node::Void => 0,
            Node::Const { .. } | Node::Var { .. } => 1,
            Node::Unary { ershov, .. } | Node::Binary { ershov, .. } => *ershov,
        }
    }

    fn ershov_func(&mut self) {
        match self {
            Node::Unary { arg, ershov, .. } => {
                *ershov = arg.ershov_number();
            }
            Node::Binary {
                left,
                right,
                ershov,
                ..
            } => {
                let l = left.ershov_number();
                let r = right.ershov_number();
                let e = if l == r { l + 1 } else { l.max(r) };
                *ershov = e;
            }
            _ => {}
        }
    }

    /// Finds and marks the first usage of each Var
    fn mark_first(&mut self) {
        if let Node::Var { sym, status, .. } = self {
            let mut sym = sym.borrow_mut();

            if !sym.visited {
                sym.visited = true;
                *status = VarStatus::First;
            } else {
                *status = VarStatus::Mid;
            }
        }
    }

    /// Finds and marks the last usage of each Var
    fn mark_last(&mut self) {
        if let Node::Var { sym, status, .. } = self {
            let mut sym = sym.borrow_mut();

            if sym.visited {
                sym.visited = false;
                *status = match status {
                    VarStatus::First => VarStatus::Singular,
                    _ => VarStatus::Last,
                }
            }
        }
    }

    /// The main entry point to compile an expression tree
    /// should be called on the root of the expression tree
    pub fn compile_tree(&mut self, ir: &mut dyn Generator) -> Result<u8> {
        self.postorder_forward(Self::ershov_func);
        self.postorder_forward(Self::mark_first);
        self.postorder_backward(Self::mark_last);

        let last = ir.first_shadow() + self.ershov_number();

        // we check ir.three_address() because AmdGenerator::shrink may swap
        // registers when generating code for SSE (two-address code).
        // This check may not be actually necessary, but we need to prove its
        // correctness first.

        let mut pool: Vec<u8> = if ir.three_address() {
            (last..16).rev().collect()
        } else {
            Vec::new()
        };

        // println!("{:#?}", &self);
        self.compile(ir, 0, &mut pool)
    }

    pub fn compile(&self, ir: &mut dyn Generator, base: u8, pool: &mut Vec<u8>) -> Result<u8> {
        match self {
            Node::Void => Ok(0),
            Node::Const { .. } => self.compile_const(ir, base),
            Node::Var { .. } => self.compile_var(ir, base, pool),
            Node::Unary { .. } => self.compile_unary(ir, base, pool),
            Node::Binary { .. } => self.compile_binary(ir, base, pool),
        }
    }

    fn compile_const(&self, ir: &mut dyn Generator, base: u8) -> Result<u8> {
        if let Node::Const { idx, .. } = &self {
            let r = ir.first_shadow() + base;
            let label = format!("_const_{}_", idx);
            ir.load_const(r, &label);
            Ok(r)
        } else {
            unreachable!();
        }
    }

    fn load_var(ir: &mut dyn Generator, dst: u8, loc: &Loc) -> u8 {
        match loc {
            Loc::Stack(idx) => ir.load_stack(dst, *idx),
            Loc::Mem(idx) => ir.load_mem(dst, *idx),
        };

        dst
    }

    /// Loaded and cache variables in Mem and Stack
    /// The basic logic is
    ///     1. At the encounter with a variable, load it into a temporary (cache) register
    ///     2. During the subsequent encounters, use the value in the register
    ///     3. After the last encounter, return the register to the pool of available registers
    fn compile_var(&self, ir: &mut dyn Generator, base: u8, pool: &mut Vec<u8>) -> Result<u8> {
        if let Node::Var { sym, status, .. } = &self {
            let mut sym = sym.borrow_mut();
            let home = ir.first_shadow() + base;

            let dst = match status {
                VarStatus::First => {
                    sym.reg = pool.pop();
                    // if no pool register is available, just use the standard designated register (home)
                    Self::load_var(ir, sym.reg.unwrap_or(home), &sym.loc)
                }
                VarStatus::Mid => {
                    // if no pool register is available, just use the standard designated register (home)
                    // note that this means reloading the variable at each use
                    sym.reg
                        .unwrap_or_else(|| Self::load_var(ir, home, &sym.loc))
                }
                VarStatus::Last => {
                    if let Some(r) = sym.reg {
                        ir.fmov(home, r);
                        pool.push(r);
                        sym.reg = None;
                        home
                    } else {
                        Self::load_var(ir, home, &sym.loc)
                    }
                }
                VarStatus::Singular | VarStatus::Unknown => {
                    // if a variable is Singular, i.e., is used only once, don't
                    // bother with caching
                    Self::load_var(ir, home, &sym.loc)
                }
            };

            Ok(dst)
        } else {
            unreachable!();
        }
    }

    fn compile_unary(&self, ir: &mut dyn Generator, base: u8, pool: &mut Vec<u8>) -> Result<u8> {
        if let Node::Unary { op, arg, power, .. } = self {
            let mut dst = ir.first_shadow() + base + self.ershov_number() - 1;
            let r = arg.compile(ir, base, pool)?;

            match op.as_str() {
                "neg" => ir.neg(dst, r),
                "not" => ir.not(dst, r),
                "abs" => ir.abs(dst, r),
                "root" => ir.root(dst, r),
                "square" => ir.square(dst, r),
                "cube" => ir.cube(dst, r),
                "recip" => ir.recip(dst, r),
                "round" => ir.round(dst, r),
                "floor" => ir.floor(dst, r),
                "ceiling" => ir.ceiling(dst, r),
                "trunc" => ir.trunc(dst, r),
                "_powi_" => ir.powi(dst, r, *power),
                "_call_" => {
                    if r != 0 {
                        ir.fmov(0, r);
                    };
                    dst = 0;
                }
                _ => return Err(anyhow!("unary operator is not recognized")),
            };

            Ok(dst)
        } else {
            unreachable!();
        }
    }

    fn compile_binary(&self, ir: &mut dyn Generator, base: u8, pool: &mut Vec<u8>) -> Result<u8> {
        if let Node::Binary {
            op,
            left,
            right,
            power,
            ..
        } = self
        {
            let (dst, l, r) = self.alloc(ir, base, left, right, pool)?;

            match op.as_str() {
                "plus" => ir.plus(dst, l, r),
                "minus" => ir.minus(dst, l, r),
                "times" => ir.times(dst, l, r),
                "divide" => ir.divide(dst, l, r),
                "rem" => ir.fmod(dst, l, r),
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
                "_powi_mod_" => ir.powi_mod(dst, l, *power, r),
                "_call_" => Self::call(ir, l, r),
                _ => return Err(anyhow!("binary operator is not recognized")),
            };

            Ok(dst)
        } else {
            unreachable!();
        }
    }

    fn alloc(
        &self,
        ir: &mut dyn Generator,
        base: u8,
        left: &Node,
        right: &Node,
        pool: &mut Vec<u8>,
    ) -> Result<(u8, u8, u8)> {
        let el = left.ershov_number();
        let er = right.ershov_number();
        let dst = ir.first_shadow() + base + self.ershov_number() - 1;

        let l;
        let r;

        if dst < 16 {
            if el == er {
                l = left.compile(ir, base + 1, pool)?;
                r = right.compile(ir, base, pool)?;
            } else if el > er {
                l = left.compile(ir, base, pool)?;
                r = right.compile(ir, base, pool)?;
            } else {
                r = right.compile(ir, base, pool)?;
                l = left.compile(ir, base, pool)?;
            }
        } else {
            return Err(anyhow!(
                "the expression is too large (not enough scratch registers)."
            ));
        }

        Ok((dst, l, r))
    }

    fn call(ir: &mut dyn Generator, l: u8, r: u8) {
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

    pub fn is_const(&self, val_: f64) -> bool {
        if let Node::Const { val, .. } = self {
            return *val == val_;
        };
        false
    }

    pub fn as_const(&self) -> Option<f64> {
        if let Node::Const { val, .. } = self {
            Some(*val)
        } else {
            None
        }
    }

    pub fn as_int_const(&self) -> Option<i32> {
        if let Node::Const { val, .. } = self {
            if val.round() == *val && val.abs() < 16384.0 {
                Some(*val as i32)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn is_binary(&self, op_: &str) -> bool {
        if let Node::Binary { op, .. } = self {
            return op == op_;
        };
        false
    }

    pub fn is_unary(&self, op_: &str) -> bool {
        if let Node::Unary { op, .. } = self {
            return op == op_;
        };
        false
    }

    pub fn arg(self) -> Option<Node> {
        if let Node::Unary { arg, .. } = self {
            Some(*arg)
        } else {
            None
        }
    }

    pub fn arg_power(self) -> Option<(Node, i32)> {
        if let Node::Unary { arg, power, .. } = self {
            Some((*arg, power))
        } else {
            None
        }
    }
}

impl Eval for Node {
    fn eval(&self, mem: &mut [f64], stack: &mut [f64]) -> f64 {
        match self {
            Node::Void => 0.0,
            Node::Const { val, .. } => *val,
            Node::Var { sym, .. } => match sym.borrow().loc {
                Loc::Stack(idx) => stack[idx as usize],
                Loc::Mem(idx) => mem[idx as usize],
            },
            Node::Unary { op, arg, power, .. } => {
                let x = arg.eval(mem, stack);

                match op.as_str() {
                    "neg" => -x,
                    "not" => f64::from_bits(!x.to_bits()),
                    "abs" => x.abs(),
                    "root" => x.sqrt(),
                    "square" => x * x,
                    "cube" => x * x * x,
                    "recip" => 1.0 / x,
                    "round" => x.round(),
                    "floor" => x.floor(),
                    "ceiling" => x.ceil(),
                    "trunc" => x.trunc(),
                    "_powi_" => x.powi(*power),
                    "_call_" => {
                        stack[0] = x;
                        x
                    }
                    s => {
                        panic!("op {} not found.", s)
                    } //_ => f64::NAN,
                }
            }
            Node::Binary {
                op,
                left,
                right,
                power,
                ..
            } => {
                let x = left.eval(mem, stack);
                let y = right.eval(mem, stack);

                match op.as_str() {
                    "plus" => x + y,
                    "minus" => x - y,
                    "times" => x * y,
                    "divide" => x / y,
                    "rem" => x % y,
                    "_powi_mod_" => x.powi(*power) % y,
                    "gt" => bool_to_f64(x > y),
                    "geq" => bool_to_f64(x >= y),
                    "lt" => bool_to_f64(x < y),
                    "leq" => bool_to_f64(x <= y),
                    "eq" => bool_to_f64(x == y),
                    "neq" => bool_to_f64(x != y),
                    "and" => f64::from_bits(x.to_bits() & y.to_bits()),
                    "or" => f64::from_bits(x.to_bits() | y.to_bits()),
                    "xor" => f64::from_bits(x.to_bits() ^ y.to_bits()),
                    "select_if" => f64::from_bits(x.to_bits() & y.to_bits()),
                    "select_else" => f64::from_bits(!x.to_bits() & y.to_bits()),
                    "_call_" => {
                        stack[0] = x;
                        stack[1] = y;
                        x
                    }
                    s => {
                        panic!("op {} not found.", s)
                    } //_ => f64::NAN,
                }
            }
        }
    }
}
