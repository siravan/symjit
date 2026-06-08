use anyhow::{anyhow, Result};

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;

// use crate::generator::Generator;
use crate::mir::Mir;
use crate::symbol::{Loc, Symbol};
use crate::utils::reg;

const COMMUTATIVE: &[&str] = &["plus", "times", "eq", "neq", "and", "or", "xor"];

#[derive(Clone)]
pub enum Node {
    Void,
    Const {
        val: f64,
        idx: u32,
    },
    Var {
        sym: Rc<RefCell<Symbol>>,
    },
    Unary {
        op: String,
        arg: Box<Node>,
        power: i32,
        ershov: u8,
        h: u64,
        w: u32,
    },
    Binary {
        op: String,
        left: Box<Node>,
        right: Box<Node>,
        power: i32,
        ershov: u8,
        h: u64,
        w: u32,
        cond: Option<Loc>,
    },
}

impl Node {
    pub fn hashof(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        match self {
            Node::Void => b"void".hash(&mut hasher),
            Node::Const { idx, .. } => {
                b"const".hash(&mut hasher);
                idx.hash(&mut hasher);
            }
            Node::Var { sym, .. } => {
                b"var".hash(&mut hasher);
                sym.borrow().hash(&mut hasher);
            }
            Node::Unary { h, .. } => {
                return *h;
            }
            Node::Binary { h, .. } => {
                return *h;
            }
        };

        hasher.finish()
    }

    pub fn weightof(&self) -> u32 {
        match self {
            Node::Void => 0,
            Node::Const { .. } | Node::Var { .. } => 1,
            Node::Unary { w, .. } => *w,
            Node::Binary { w, .. } => *w,
        }
    }

    pub fn create_void() -> Node {
        Node::Void
    }

    pub fn create_const(val: f64, idx: u32) -> Node {
        Node::Const { val, idx }
    }

    pub fn create_var(sym: Rc<RefCell<Symbol>>) -> Node {
        Node::Var { sym }
    }

    pub fn create_unary(op: &str, arg: Node, power: i32) -> Node {
        let e = arg.ershov_number();

        let mut hasher = DefaultHasher::new();
        op.hash(&mut hasher);
        arg.hashof().hash(&mut hasher);
        power.hash(&mut hasher);

        let w = 1 + arg.weightof();

        Node::Unary {
            op: op.to_string(),
            arg: Box::new(arg),
            ershov: e,
            power,
            h: hasher.finish(),
            w,
        }
    }

    pub fn create_binary(op: &str, left: Node, right: Node, power: i32, cond: Option<Loc>) -> Node {
        let e = Self::calc_ershov(&left, &right);

        let mut hasher = DefaultHasher::new();
        op.hash(&mut hasher);
        power.hash(&mut hasher);

        let mut l = left.hashof();
        let mut r = right.hashof();

        (l, r) = if COMMUTATIVE.contains(&op) && l > r {
            (r, l)
        } else {
            (l, r)
        };

        l.hash(&mut hasher);
        r.hash(&mut hasher);
        cond.hash(&mut hasher);

        let w = 1 + left.weightof() + right.weightof();

        Node::Binary {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
            ershov: e,
            power,
            h: hasher.finish(),
            w,
            cond,
        }
    }

    pub fn create_ifelse(cond: &Node, left: Node, right: Node) -> Node {
        if let Node::Var { sym, .. } = cond {
            let sym = sym.borrow();
            Self::create_binary("_ifelse_", left, right, 0, Some(sym.loc))
        } else {
            unreachable!()
        }
    }

    pub fn create_powi(arg: Node, power: i32) -> Node {
        Self::create_unary("_powi_", arg, power)
    }

    pub fn create_modular_powi(left: Node, right: Node, power: i32) -> Node {
        Self::create_binary("_powi_mod_", left, right, power, None)
    }

    pub fn first(&mut self) -> Option<&mut Node> {
        match self {
            Node::Unary { arg, .. } => Some(arg),
            Node::Binary { left, right, .. } => {
                let el = left.ershov_number();
                let er = right.ershov_number();
                if el >= er {
                    Some(left)
                } else {
                    Some(right)
                }
            }
            _ => None,
        }
    }

    pub fn second(&mut self) -> Option<&mut Node> {
        match self {
            Node::Unary { .. } => None,
            Node::Binary { left, right, .. } => {
                let el = left.ershov_number();
                let er = right.ershov_number();
                if el >= er {
                    Some(right)
                } else {
                    Some(left)
                }
            }
            _ => None,
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

    pub fn calc_ershov(left: &Node, right: &Node) -> u8 {
        let l = left.ershov_number();
        let r = right.ershov_number();

        if l == r {
            l + 1
        } else {
            l.max(r)
        }
    }

    /// The main entry point to compile an expression tree
    /// should be called on the root of the expression tree
    pub fn compile_tree(&mut self, mir: &mut Mir) -> Result<u8> {
        self.compile(mir, 0)
    }

    pub fn compile(&self, mir: &mut Mir, base: u8) -> Result<u8> {
        match self {
            Node::Void => Ok(0),
            Node::Const { .. } => self.compile_const(mir, base),
            Node::Var { .. } => self.compile_var(mir, base),
            Node::Unary { .. } => self.compile_unary(mir, base),
            Node::Binary { .. } => self.compile_binary(mir, base),
        }
    }

    fn compile_const(&self, mir: &mut Mir, base: u8) -> Result<u8> {
        if let Node::Const { idx, .. } = &self {
            mir.load_const(reg(base), *idx);
            Ok(base)
        } else {
            unreachable!();
        }
    }

    fn load_var(mir: &mut Mir, dst: u8, loc: &Loc) -> u8 {
        match loc {
            Loc::Stack(idx) => mir.load_stack(reg(dst), *idx),
            Loc::Mem(idx) => mir.load_mem(reg(dst), *idx),
            Loc::Param(idx) => mir.load_param(reg(dst), *idx),
        };

        dst
    }

    /// Loaded and cache variables in Mem and Stack
    /// The basic logic is
    ///     1. At the encounter with a variable, load it into a temporary (cache) register
    ///     2. During the subsequent encounters, use the value in the register
    ///     3. After the last encounter, return the register to the pool of available registers
    fn compile_var(&self, mir: &mut Mir, base: u8) -> Result<u8> {
        if let Node::Var { sym, .. } = &self {
            let sym = sym.borrow();
            let dst = Self::load_var(mir, base, &sym.loc);

            Ok(dst)
        } else {
            unreachable!();
        }
    }

    fn compile_unary(&self, mir: &mut Mir, base: u8) -> Result<u8> {
        if let Node::Unary { op, arg, power, .. } = self {
            let dst = base + self.ershov_number() - 1;
            let r = arg.compile(mir, base)?;

            match op.as_str() {
                "neg" => mir.neg(reg(dst), reg(r)),
                "not" => mir.not(reg(dst), reg(r)),
                "abs" => mir.abs(reg(dst), reg(r)),
                "root" => mir.root(reg(dst), reg(r)),
                "real_root" => mir.real_root(reg(dst), reg(r)),
                "square" => mir.square(reg(dst), reg(r)),
                "cube" => mir.cube(reg(dst), reg(r)),
                "recip" => mir.recip(reg(dst), reg(r)),
                "round" => mir.round(reg(dst), reg(r)),
                "floor" => mir.floor(reg(dst), reg(r)),
                "ceiling" => mir.ceiling(reg(dst), reg(r)),
                "trunc" => mir.trunc(reg(dst), reg(r)),
                "frac" => mir.frac(reg(dst), reg(r)),
                "_powi_" => mir.powi(reg(dst), reg(r), *power),
                "_call_" => mir.setup_call_unary(reg(r)),
                "real" => mir.real(reg(dst), reg(r)),
                "imaginary" => mir.imaginary(reg(dst), reg(r)),
                "conjugate" => mir.conjugate(reg(dst), reg(r)),
                "iszero" => mir.iszero(reg(dst), reg(r)),
                "isnotzero" => mir.isnotzero(reg(dst), reg(r)),
                _ => return Err(anyhow!("unary operator {:?} is not recognized", op)),
            };

            Ok(dst)
        } else {
            unreachable!();
        }
    }

    fn compile_binary(&self, mir: &mut Mir, base: u8) -> Result<u8> {
        if let Node::Binary {
            op,
            left,
            right,
            power,
            cond,
            ..
        } = self
        {
            if mir.config.is_amd64() {
                if let Ok(dst) = self.load_math(mir, base, op, left, right) {
                    return Ok(dst);
                }

                if let Ok(dst) = self.load_const_math(mir, base, op, left, right) {
                    return Ok(dst);
                }
            }

            let (dst, l, r) = self.alloc(mir, base, left, right)?;

            match op.as_str() {
                "plus" => mir.plus(reg(dst), reg(l), reg(r)),
                "minus" => mir.minus(reg(dst), reg(l), reg(r)),
                "times" => mir.times(reg(dst), reg(l), reg(r)),
                "divide" => mir.divide(reg(dst), reg(l), reg(r)),
                "rem" => mir.fmod(reg(dst), reg(l), reg(r)),
                "gt" => mir.gt(reg(dst), reg(l), reg(r)),
                "geq" => mir.geq(reg(dst), reg(l), reg(r)),
                "lt" => mir.lt(reg(dst), reg(l), reg(r)),
                "leq" => mir.leq(reg(dst), reg(l), reg(r)),
                "eq" => mir.eq(reg(dst), reg(l), reg(r)),
                "neq" => mir.neq(reg(dst), reg(l), reg(r)),
                "and" => mir.and(reg(dst), reg(l), reg(r)),
                "or" => mir.or(reg(dst), reg(l), reg(r)),
                "xor" => mir.xor(reg(dst), reg(l), reg(r)),
                "complex" => mir.complex(reg(dst), reg(l), reg(r)),
                "_ifelse_" => mir.ifelse(reg(dst), reg(l), reg(r), cond.unwrap()),
                "_powi_mod_" => mir.powi_mod(reg(dst), reg(l), *power, reg(r)),
                "_call_" => mir.setup_call_binary(reg(l), reg(r)),
                _ => return Err(anyhow!("binary operator {:?} is not recognized", op)),
            };

            Ok(dst)
        } else {
            unreachable!();
        }
    }

    fn alloc(&self, mir: &mut Mir, base: u8, left: &Node, right: &Node) -> Result<(u8, u8, u8)> {
        let el = left.ershov_number();
        let er = right.ershov_number();
        let dst = base + self.ershov_number() - 1;

        let l;
        let r;

        if dst < mir.config.count_scratch() {
            if el == er {
                l = left.compile(mir, base + 1)?;
                r = right.compile(mir, base)?;
            } else if el > er {
                l = left.compile(mir, base)?;
                r = right.compile(mir, base)?;
            } else {
                r = right.compile(mir, base)?;
                l = left.compile(mir, base)?;
            }
        } else {
            return Err(anyhow!(
                "the expression is too large (not enough scratch registers)."
            ));
        }

        Ok((dst, l, r))
    }

    fn is_leaf_var(&self) -> bool {
        matches!(self, Node::Var { .. })
    }

    fn compile_leaf_var(&self) -> Option<Loc> {
        if let Node::Var { sym } = self {
            Some(sym.borrow().loc)
        } else {
            None
        }
    }

    fn load_math(
        &self,
        mir: &mut Mir,
        base: u8,
        op: &str,
        left: &Node,
        right: &Node,
    ) -> Result<u8> {
        let dst = base + self.ershov_number() - 1;

        if (op == "plus"
            || op == "times"
            || op == "minus"
            || (op == "divide" && !mir.config.is_complex()))    // because complex division uses re(Reg::Temp)
            && right.is_leaf_var()
        {
            let l = left.compile(mir, base)?;
            let t = right.compile_leaf_var().unwrap();

            match op {
                "plus" => mir.plus_load(reg(dst), reg(l), t),
                "minus" => mir.minus_load(reg(dst), reg(l), t),
                "times" => mir.times_load(reg(dst), reg(l), t),
                "divide" => mir.divide_load(reg(dst), reg(l), t),
                _ => unreachable!(),
            }
            return Ok(dst);
        }

        if (op == "plus" || op == "times") && left.is_leaf_var() {
            let r = right.compile(mir, base)?;
            let t = left.compile_leaf_var().unwrap();

            match op {
                "plus" => mir.plus_load(reg(dst), reg(r), t),
                "times" => mir.times_load(reg(dst), reg(r), t),
                _ => unreachable!(),
            }
            return Ok(dst);
        }

        Err(anyhow!("cannot fuse!"))
    }

    pub fn is_leaf_const(&self) -> bool {
        matches!(self, Node::Const { .. })
    }

    fn compile_leaf_const(&self) -> Option<u32> {
        if let Node::Const { idx, .. } = self {
            Some(*idx)
        } else {
            None
        }
    }

    fn load_const_math(
        &self,
        mir: &mut Mir,
        base: u8,
        op: &str,
        left: &Node,
        right: &Node,
    ) -> Result<u8> {
        let dst = base + self.ershov_number() - 1;

        if (op == "plus" || op == "times" || op == "minus" || op == "divide")
            && right.is_leaf_const()
        {
            let l = left.compile(mir, base)?;
            let idx = right.compile_leaf_const().unwrap();

            match op {
                "plus" => mir.plus_load_const(reg(dst), reg(l), idx),
                "minus" => mir.minus_load_const(reg(dst), reg(l), idx),
                "times" => mir.times_load_const(reg(dst), reg(l), idx),
                "divide" => mir.divide_load_const(reg(dst), reg(l), idx),
                _ => unreachable!(),
            }
            return Ok(dst);
        }

        if (op == "plus" || op == "times") && left.is_leaf_const() {
            let r = right.compile(mir, base)?;
            let idx = left.compile_leaf_const().unwrap();

            match op {
                "plus" => mir.plus_load_const(reg(dst), reg(r), idx),
                "times" => mir.times_load_const(reg(dst), reg(r), idx),
                _ => unreachable!(),
            }
            return Ok(dst);
        }

        Err(anyhow!("cannot fuse!"))
    }

    pub fn address(&self) -> Result<usize> {
        if let Node::Var { sym, .. } = &self {
            match sym.borrow().loc {
                Loc::Stack(idx) => Ok(idx as usize),
                _ => Err(anyhow!("only stack locations can be used for slicing.")),
            }
        } else {
            Err(anyhow!("Only stack variables have an address."))
        }
    }

    pub fn call_external(&self) -> Result<(i32, i32)> {
        if let Node::Binary {
            op, left, right, ..
        } = self
        {
            if op != "_call_" {
                return Err(anyhow!("external fun main node should be `_call_`"));
            }

            let l = left.as_int_const().unwrap();
            let r = right.as_int_const().unwrap();
            Ok((l, r))
        } else {
            unreachable!();
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

    pub fn locations(&self) -> HashSet<Loc> {
        match self {
            Node::Void | Node::Const { .. } => HashSet::new(),
            Node::Unary { arg, .. } => arg.locations(),
            Node::Binary {
                left, right, cond, ..
            } => {
                let mut s1 = left.locations();
                let s2 = right.locations();
                s1.extend(s2);
                cond.map(|loc| s1.insert(loc));
                s1
            }
            Node::Var { sym, .. } => {
                let mut s = HashSet::new();
                s.insert(sym.borrow().loc);
                s
            }
        }
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Void => write!(f, "void"),
            Node::Const { val, .. } => write!(f, "const {}", val),
            Node::Var { sym, .. } => write!(f, "var {:?}", sym.borrow()),
            Node::Unary { op, arg, .. } => write!(f, "{}({:?})", op, arg),
            Node::Binary {
                op, left, right, ..
            } => write!(f, "{}({:?}, {:?})", op, left, right),
        }
    }
}
