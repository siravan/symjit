//! The evaluation of a kernel's body. Each construct does what the code the `#[kernel]`
//! macro makes of it does: Rust (numbers, `for` loops, conditions on numbers) runs while the
//! kernel is built, and kernel values add instructions to the `KernelBuilder`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use proc_macro2::Span;
use symjit::kernel::{Cplx, KernelBuilder, Mask, Real, Value};
use symjit::Slot;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{BinOp, Block, Expr, Pat, ReturnType, Stmt, UnOp};

use super::syntax::{
    assigned_in_block, assigned_in_closure, assigned_in_if, is_complex_literal, is_compound,
    is_float_literal, stage_type, ClosureAssigns, Escapes, Reads, Ty,
};
use super::value::{
    binary, cast, coerce, compare, complex, from_slots, lift, logic, method, neg, not, part,
    to_slots, unify, Closure, Cmp, Op, Shape, Slots, Val,
};
use super::{FnDef, FnKind, Naming, Source};

/// How the evaluation of an expression ends, besides with a value.
pub enum Flow<'k> {
    /// an error, with its position
    Error(String),
    Break(Option<Val<'k>>),
    Continue,
}

pub type R<'k, T> = Result<T, Flow<'k>>;

/// The iterations a loop run while the kernel is built may take.
const LOOP_LIMIT: usize = 1 << 24;
/// The depth of calls of inline helpers and closures.
const CALL_LIMIT: usize = 256;

pub fn at(span: Span, msg: impl std::fmt::Display) -> String {
    let p = span.start();
    format!("{}:{}: {}", p.line, p.column + 1, msg)
}

fn fail<'k, T>(span: Span, msg: impl std::fmt::Display) -> R<'k, T> {
    Err(Flow::Error(at(span, msg)))
}

/// Attaches the position of `span` to the error of a value operation.
fn here<'k, T>(span: Span, r: Result<T, String>) -> R<'k, T> {
    r.map_err(|msg| Flow::Error(at(span, msg)))
}

type Scope<'k> = Vec<(String, Val<'k>)>;

pub struct Interp<'k, 's> {
    k: &'k KernelBuilder,
    src: &'s Source,
    naming: &'s Naming,
    /// the function whose body is evaluated: the kernel, an inline helper
    current: Cell<&'s FnDef>,
    env: RefCell<Vec<Scope<'k>>>,
    depth: Cell<usize>,
}

impl<'k, 's> Interp<'k, 's> {
    pub fn new(
        k: &'k KernelBuilder,
        src: &'s Source,
        kernel: &'s FnDef,
        naming: &'s Naming,
    ) -> Self {
        Interp {
            k,
            src,
            naming,
            current: Cell::new(kernel),
            env: RefCell::new(vec![Vec::new()]),
            depth: Cell::new(0),
        }
    }

    // the environment

    fn lookup(&self, name: &str) -> Option<Val<'k>> {
        let env = self.env.borrow();
        env.iter()
            .rev()
            .flat_map(|scope| scope.iter().rev())
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
    }

    fn declare(&self, name: &str, v: Val<'k>) {
        self.env
            .borrow_mut()
            .last_mut()
            .unwrap()
            .push((name.to_string(), v));
    }

    fn set(&self, name: &str, v: Val<'k>) -> bool {
        let mut env = self.env.borrow_mut();
        for scope in env.iter_mut().rev() {
            if let Some(slot) = scope.iter_mut().rev().find(|(n, _)| n == name) {
                slot.1 = v;
                return true;
            }
        }
        false
    }

    fn snapshot(&self) -> Vec<Scope<'k>> {
        self.env.borrow().clone()
    }

    fn restore(&self, env: Vec<Scope<'k>>) {
        *self.env.borrow_mut() = env;
    }

    fn scoped<T>(&self, f: impl FnOnce() -> R<'k, T>) -> R<'k, T> {
        let depth = self.env.borrow().len();
        self.env.borrow_mut().push(Vec::new());
        let r = f();
        self.env.borrow_mut().truncate(depth);
        r
    }

    /// Runs `f` in a fresh environment (a function body): `scope` is its only scope.
    fn in_body<T>(
        &self,
        def: Option<&'s FnDef>,
        scope: Scope<'k>,
        f: impl FnOnce() -> R<'k, T>,
    ) -> R<'k, T> {
        let depth = self.depth.get();
        if depth >= CALL_LIMIT {
            return Err(Flow::Error(format!(
                "inline helpers and closures call each other more than {} deep",
                CALL_LIMIT
            )));
        }
        let saved = std::mem::replace(&mut *self.env.borrow_mut(), vec![scope]);
        let current = self.current.get();
        if let Some(def) = def {
            self.current.set(def);
        }
        self.depth.set(depth + 1);
        let r = f();
        self.depth.set(depth);
        self.current.set(current);
        *self.env.borrow_mut() = saved;
        match r {
            Err(Flow::Break(_)) | Err(Flow::Continue) => {
                Err(Flow::Error("`break` or `continue` outside a loop".into()))
            }
            r => r,
        }
    }

    pub fn stage(&self, ty: &syn::Type) -> Ty {
        stage_type(ty, &|e| self.src.const_usize(e))
    }

    // patterns

    /// Binds the variables of `pat` to the parts of `v`.
    pub fn bind(&self, pat: &Pat, v: Val<'k>) -> R<'k, ()> {
        match pat {
            Pat::Ident(p) => {
                if p.subpat.is_some() {
                    return fail(pat.span(), "`@` patterns are not supported by sjc");
                }
                self.declare(&p.ident.to_string(), v);
                Ok(())
            }
            Pat::Type(pt) => {
                let ty = self.stage(&pt.ty);
                let v = if ty.has_handle() {
                    here(pt.span(), lift(self.k, v, &ty))?
                } else {
                    v
                };
                self.bind(&pt.pat, v)
            }
            Pat::Tuple(t) => match v {
                Val::Tuple(vs) if vs.len() == t.elems.len() => {
                    for (p, v) in t.elems.iter().zip(vs) {
                        self.bind(p, v)?;
                    }
                    Ok(())
                }
                v => fail(
                    pat.span(),
                    format!(
                        "cannot bind {} to a tuple pattern of {} elements",
                        v.describe(),
                        t.elems.len()
                    ),
                ),
            },
            Pat::Slice(s) => match v {
                Val::Array(vs) if vs.len() == s.elems.len() => {
                    for (p, v) in s.elems.iter().zip(vs) {
                        self.bind(p, v)?;
                    }
                    Ok(())
                }
                v => fail(
                    pat.span(),
                    format!(
                        "cannot bind {} to an array pattern of {} elements",
                        v.describe(),
                        s.elems.len()
                    ),
                ),
            },
            Pat::Wild(_) => Ok(()),
            Pat::Paren(p) => self.bind(&p.pat, v),
            Pat::Reference(r) => self.bind(&r.pat, v),
            _ => fail(pat.span(), "this pattern is not supported by sjc"),
        }
    }

    // blocks and statements

    pub fn eval_block(&self, block: &Block) -> R<'k, Val<'k>> {
        self.scoped(|| {
            let mut last = Val::Unit;
            for stmt in block.stmts.iter() {
                last = Val::Unit;
                match stmt {
                    Stmt::Local(local) => self.local(local)?,
                    Stmt::Item(item) => self.item(item)?,
                    Stmt::Expr(e, semi) => {
                        let v = self.eval(e)?;
                        if semi.is_none() {
                            last = v;
                        }
                    }
                    Stmt::Macro(m) => {
                        return fail(
                            m.span(),
                            format!(
                                "macros (`{}!`) are not supported by sjc",
                                path_string(&m.mac.path)
                            ),
                        )
                    }
                }
            }
            Ok(last)
        })
    }

    fn local(&self, local: &syn::Local) -> R<'k, ()> {
        let Some(init) = &local.init else {
            return fail(local.span(), "a `let` needs a value in sjc");
        };
        if init.diverge.is_some() {
            return fail(local.span(), "`let ... else` is not supported by sjc");
        }
        // `let mut s = 0.0` and `let mut z = Complex::new(0.0, 0.0)`: accumulators, made
        // kernel values so that they can take kernel values
        let accumulator = match &local.pat {
            Pat::Ident(p) if p.mutability.is_some() => {
                if is_float_literal(&init.expr) {
                    Some(Ty::Real)
                } else if is_complex_literal(&init.expr) {
                    Some(Ty::Cplx)
                } else {
                    None
                }
            }
            _ => None,
        };
        let mut v = self.eval(&init.expr)?;
        if let Some(ty) = accumulator {
            v = here(init.expr.span(), lift(self.k, v, &ty))?;
        }
        self.bind(&local.pat, v)
    }

    fn item(&self, item: &syn::Item) -> R<'k, ()> {
        match item {
            syn::Item::Const(c) => {
                let v = self.eval(&c.expr)?;
                self.declare(&c.ident.to_string(), v);
                Ok(())
            }
            syn::Item::Use(_) => Ok(()),
            syn::Item::Fn(f) => fail(
                f.sig.ident.span(),
                "functions inside a kernel are not supported by sjc; move it out of the \
                 kernel with `#[kernel(inline)]`",
            ),
            _ => fail(
                item.span(),
                "this item is not supported inside a kernel by sjc",
            ),
        }
    }

    // expressions

    pub fn eval(&self, e: &Expr) -> R<'k, Val<'k>> {
        let span = e.span();
        match e {
            Expr::Lit(l) => self.literal(&l.lit),
            Expr::Path(p) => self.path(p),
            Expr::Paren(p) => self.eval(&p.expr),
            Expr::Group(g) => self.eval(&g.expr),
            Expr::Block(b) => {
                if b.label.is_some() {
                    return fail(span, "labeled blocks are not supported by sjc");
                }
                self.eval_block(&b.block)
            }
            Expr::Unary(u) => {
                let v = self.eval(&u.expr)?;
                match u.op {
                    UnOp::Neg(_) => here(span, neg(v)),
                    UnOp::Not(_) => here(span, not(v)),
                    UnOp::Deref(_) => Ok(v),
                    _ => fail(span, "this operator is not supported by sjc"),
                }
            }
            Expr::Reference(r) => self.eval(&r.expr),
            Expr::Binary(b) => self.binary(b),
            Expr::Assign(a) => {
                let v = self.eval(&a.right)?;
                self.assign(&a.left, v)?;
                Ok(Val::Unit)
            }
            Expr::If(eif) => self.eval_if(eif),
            Expr::While(w) => self.eval_while(w),
            Expr::ForLoop(f) => self.eval_for(f),
            Expr::Loop(l) => self.eval_loop(l),
            Expr::Break(b) => {
                if b.label.is_some() {
                    return fail(span, "labeled `break` is not supported by sjc");
                }
                let v = match &b.expr {
                    Some(e) => Some(self.eval(e)?),
                    None => None,
                };
                Err(Flow::Break(v))
            }
            Expr::Continue(c) => {
                if c.label.is_some() {
                    return fail(span, "labeled `continue` is not supported by sjc");
                }
                Err(Flow::Continue)
            }
            Expr::Call(call) => self.call(call),
            Expr::MethodCall(m) => {
                let recv = self.eval(&m.receiver)?;
                let mut args = m
                    .args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<Vec<_>, _>>()?;
                // `[..].map(|x| ..)`: the closure is applied to each element
                if m.method == "map" && args.len() == 1 && matches!(args[0], Val::Closure(_)) {
                    let items = match recv {
                        Val::Array(vs) => vs,
                        Val::Range(a, b) => (a..b.max(a)).map(Val::Int).collect(),
                        v => return fail(span, format!("{} has no method `map`", v.describe())),
                    };
                    let f = args.pop().unwrap();
                    let mut out = Vec::new();
                    for item in items {
                        out.push(self.call_value(span, f.clone(), vec![item])?);
                    }
                    return Ok(Val::Array(out));
                }
                here(span, method(self.k, recv, &m.method.to_string(), args))
            }
            Expr::Field(f) => {
                let base = self.eval(&f.base)?;
                match &f.member {
                    syn::Member::Named(id) if id == "re" || id == "im" => {
                        here(span, part(base, id == "re"))
                    }
                    syn::Member::Unnamed(i) => match base {
                        Val::Tuple(vs) if (i.index as usize) < vs.len() => {
                            Ok(vs[i.index as usize].clone())
                        }
                        v => fail(span, format!("{} has no field `{}`", v.describe(), i.index)),
                    },
                    syn::Member::Named(id) => {
                        fail(span, format!("{} has no field `{}`", base.describe(), id))
                    }
                }
            }
            Expr::Index(i) => {
                let base = self.eval(&i.expr)?;
                let idx = self.eval(&i.index)?;
                match (base, idx) {
                    (Val::Array(vs), Val::Int(j)) => {
                        match usize::try_from(j).ok().and_then(|j| vs.get(j)) {
                            Some(v) => Ok(v.clone()),
                            None => fail(
                                span,
                                format!("index {} out of bounds (length {})", j, vs.len()),
                            ),
                        }
                    }
                    (b, i) => fail(
                        span,
                        format!("cannot index {} with {}", b.describe(), i.describe()),
                    ),
                }
            }
            Expr::Tuple(t) => {
                if t.elems.is_empty() {
                    return Ok(Val::Unit);
                }
                Ok(Val::Tuple(
                    t.elems
                        .iter()
                        .map(|e| self.eval(e))
                        .collect::<Result<_, _>>()?,
                ))
            }
            Expr::Array(a) => self.array(a),
            Expr::Repeat(r) => {
                let v = self.eval(&r.expr)?;
                match self.eval(&r.len)? {
                    Val::Int(n) if n >= 0 => Ok(Val::Array(vec![v; n as usize])),
                    n => fail(
                        span,
                        format!("the length of an array cannot be {}", n.describe()),
                    ),
                }
            }
            Expr::Cast(c) => {
                let v = self.eval(&c.expr)?;
                let ty = match &*c.ty {
                    syn::Type::Path(p) if p.path.get_ident().is_some() => {
                        p.path.get_ident().unwrap().to_string()
                    }
                    _ => {
                        return fail(
                            span,
                            "sjc casts only to number types (`as f64`, `as i32`, ...)",
                        )
                    }
                };
                here(span, cast(v, &ty))
            }
            Expr::Closure(c) => self.closure(c),
            Expr::Range(r) => {
                let bound = |e: &Option<Box<Expr>>| -> R<'k, i64> {
                    match e {
                        Some(e) => match self.eval(e)? {
                            Val::Int(i) => Ok(i),
                            v => fail(
                                e.span(),
                                format!(
                                    "a range of {} (sjc supports integer ranges)",
                                    v.describe()
                                ),
                            ),
                        },
                        None => fail(span, "open ranges are not supported by sjc"),
                    }
                };
                let (a, b) = (bound(&r.start)?, bound(&r.end)?);
                Ok(match r.limits {
                    syn::RangeLimits::HalfOpen(_) => Val::Range(a, b),
                    syn::RangeLimits::Closed(_) => Val::Range(a, b.saturating_add(1)),
                })
            }
            Expr::Return(_) => fail(
                span,
                "a kernel cannot `return`; its value is the last expression of the body",
            ),
            Expr::Macro(m) => fail(
                span,
                format!(
                    "macros (`{}!`) are not supported by sjc",
                    path_string(&m.mac.path)
                ),
            ),
            Expr::Match(_) => fail(span, "`match` is not supported by sjc; use `if`/`else`"),
            Expr::Let(_) => fail(span, "`if let` and `while let` are not supported by sjc"),
            _ => fail(span, "this expression is not supported by sjc"),
        }
    }

    fn literal(&self, lit: &syn::Lit) -> R<'k, Val<'k>> {
        let span = lit.span();
        match lit {
            syn::Lit::Int(i) => match i.suffix() {
                "f64" => here(
                    span,
                    i.base10_digits()
                        .parse::<f64>()
                        .map(Val::F64)
                        .map_err(|e| e.to_string()),
                ),
                "f32" => fail(span, "`f32` is not supported (kernels compute in `f64`)"),
                _ => here(
                    span,
                    i.base10_parse::<i64>()
                        .map(Val::Int)
                        .map_err(|e| e.to_string()),
                ),
            },
            syn::Lit::Float(f) => match f.suffix() {
                "f32" => fail(span, "`f32` is not supported (kernels compute in `f64`)"),
                _ => here(
                    span,
                    f.base10_parse::<f64>()
                        .map(Val::F64)
                        .map_err(|e| e.to_string()),
                ),
            },
            syn::Lit::Bool(b) => Ok(Val::Bool(b.value)),
            _ => fail(span, "this literal is not supported by sjc"),
        }
    }

    fn path(&self, p: &syn::ExprPath) -> R<'k, Val<'k>> {
        let span = p.span();
        if let Some(id) = p.path.get_ident() {
            let name = id.to_string();
            if let Some(v) = self.lookup(&name) {
                return Ok(v);
            }
            if let Some(expr) = self.src.consts.get(&name) {
                // a constant of the file: evaluated where nothing else is visible
                return self.in_body(None, Vec::new(), || self.eval(expr));
            }
            if let Some(x) = self.src.imported.get(&name) {
                return Ok(Val::F64(*x));
            }
            return fail(span, format!("cannot find `{}`", name));
        }
        let segs: Vec<String> = p
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        if let Some(v) = known_path(&segs) {
            return Ok(v);
        }
        fail(
            span,
            format!(
                "cannot find `{}` (sjc knows the `f64` constants)",
                segs.join("::")
            ),
        )
    }

    fn binary(&self, b: &syn::ExprBinary) -> R<'k, Val<'k>> {
        let span = b.span();
        if is_compound(&b.op) {
            let op = match b.op {
                BinOp::AddAssign(_) => Op::Add,
                BinOp::SubAssign(_) => Op::Sub,
                BinOp::MulAssign(_) => Op::Mul,
                BinOp::DivAssign(_) => Op::Div,
                BinOp::RemAssign(_) => Op::Rem,
                BinOp::BitAndAssign(_) => Op::BitAnd,
                BinOp::BitOrAssign(_) => Op::BitOr,
                BinOp::BitXorAssign(_) => Op::BitXor,
                BinOp::ShlAssign(_) => Op::Shl,
                _ => Op::Shr,
            };
            let old = self.eval(&b.left)?;
            let rhs = self.eval(&b.right)?;
            let new = here(span, binary(op, old, rhs))?;
            self.assign(&b.left, new)?;
            return Ok(Val::Unit);
        }

        // both sides are evaluated, also for `&&` and `||` (`ops::and`, `ops::or`)
        let l = self.eval(&b.left)?;
        let r = self.eval(&b.right)?;
        let cmp = |c| here(span, compare(c, l.clone(), r.clone()));
        match b.op {
            BinOp::And(_) => here(span, logic(true, l, r)),
            BinOp::Or(_) => here(span, logic(false, l, r)),
            BinOp::Lt(_) => cmp(Cmp::Lt),
            BinOp::Le(_) => cmp(Cmp::Le),
            BinOp::Gt(_) => cmp(Cmp::Gt),
            BinOp::Ge(_) => cmp(Cmp::Ge),
            BinOp::Eq(_) => cmp(Cmp::Eq),
            BinOp::Ne(_) => cmp(Cmp::Ne),
            BinOp::Add(_) => here(span, binary(Op::Add, l, r)),
            BinOp::Sub(_) => here(span, binary(Op::Sub, l, r)),
            BinOp::Mul(_) => here(span, binary(Op::Mul, l, r)),
            BinOp::Div(_) => here(span, binary(Op::Div, l, r)),
            BinOp::Rem(_) => here(span, binary(Op::Rem, l, r)),
            BinOp::BitAnd(_) => here(span, binary(Op::BitAnd, l, r)),
            BinOp::BitOr(_) => here(span, binary(Op::BitOr, l, r)),
            BinOp::BitXor(_) => here(span, binary(Op::BitXor, l, r)),
            BinOp::Shl(_) => here(span, binary(Op::Shl, l, r)),
            BinOp::Shr(_) => here(span, binary(Op::Shr, l, r)),
            _ => fail(span, "this operator is not supported by sjc"),
        }
    }

    /// `place = v`: a number assigned to a kernel value becomes a constant (`ops::coerce`).
    fn assign(&self, place: &Expr, v: Val<'k>) -> R<'k, ()> {
        let span = place.span();
        match place {
            Expr::Path(p) if p.path.get_ident().is_some() => {
                let name = p.path.get_ident().unwrap().to_string();
                let Some(old) = self.lookup(&name) else {
                    return fail(span, format!("cannot find `{}`", name));
                };
                let new = here(span, coerce(self.k, &old, v))?;
                self.set(&name, new);
                Ok(())
            }
            Expr::Paren(p) => self.assign(&p.expr, v),
            Expr::Infer(_) => Ok(()),
            Expr::Tuple(t) => match v {
                Val::Tuple(vs) if vs.len() == t.elems.len() => {
                    for (e, v) in t.elems.iter().zip(vs) {
                        self.assign(e, v)?;
                    }
                    Ok(())
                }
                v => fail(span, format!("cannot assign {} to a tuple", v.describe())),
            },
            Expr::Array(a) => match v {
                Val::Array(vs) if vs.len() == a.elems.len() => {
                    for (e, v) in a.elems.iter().zip(vs) {
                        self.assign(e, v)?;
                    }
                    Ok(())
                }
                v => fail(span, format!("cannot assign {} to an array", v.describe())),
            },
            Expr::Index(i) => {
                let base = self.eval(&i.expr)?;
                let idx = self.eval(&i.index)?;
                match (base, idx) {
                    (Val::Array(mut vs), Val::Int(j)) if j >= 0 && (j as usize) < vs.len() => {
                        let j = j as usize;
                        vs[j] = here(span, coerce(self.k, &vs[j], v))?;
                        self.assign(&i.expr, Val::Array(vs))
                    }
                    (b, i) => fail(
                        span,
                        format!(
                            "cannot assign to {} indexed by {}",
                            b.describe(),
                            i.describe()
                        ),
                    ),
                }
            }
            Expr::Field(f) => match &f.member {
                syn::Member::Unnamed(idx) => match self.eval(&f.base)? {
                    Val::Tuple(mut vs) if (idx.index as usize) < vs.len() => {
                        let j = idx.index as usize;
                        vs[j] = here(span, coerce(self.k, &vs[j], v))?;
                        self.assign(&f.base, Val::Tuple(vs))
                    }
                    b => fail(
                        span,
                        format!("{} has no field `{}`", b.describe(), idx.index),
                    ),
                },
                syn::Member::Named(id) => fail(
                    span,
                    format!(
                        "cannot assign to `.{}`; assign a new value to the variable",
                        id
                    ),
                ),
            },
            _ => fail(span, "cannot assign to this expression"),
        }
    }

    /// `[a, b, ..]`: float literals among other values take their type (`[1.0, x]`).
    fn array(&self, a: &syn::ExprArray) -> R<'k, Val<'k>> {
        let mut vals = Vec::new();
        for e in a.elems.iter() {
            vals.push(self.eval(e)?);
        }
        let literal: Vec<bool> = a.elems.iter().map(is_float_literal).collect();
        if literal.iter().any(|l| *l) && !literal.iter().all(|l| *l) {
            let target = vals
                .iter()
                .zip(&literal)
                .find(|(_, l)| !**l)
                .map(|(v, _)| v.clone())
                .unwrap();
            let ty = match target {
                Val::Real(_) => Some(Ty::Real),
                Val::Cplx(_) => Some(Ty::Cplx),
                _ => None,
            };
            if let Some(ty) = ty {
                for (v, l) in vals.iter_mut().zip(&literal) {
                    if *l {
                        *v = here(a.span(), lift(self.k, v.clone(), &ty))?;
                    }
                }
            }
        }
        Ok(Val::Array(vals))
    }

    fn closure(&self, c: &syn::ExprClosure) -> R<'k, Val<'k>> {
        let vars = assigned_in_closure(c);
        if !vars.is_empty() {
            return fail(
                c.span(),
                format!(
                    "sjc's closures cannot assign variables declared outside them ({}); return \
                     the new value from the closure instead",
                    vars.iter()
                        .map(|v| format!("`{v}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }
        let output = match &c.output {
            ReturnType::Type(_, ty) => Some(self.stage(ty)),
            ReturnType::Default => None,
        };
        let env: Vec<(String, Val<'k>)> = self
            .env
            .borrow()
            .iter()
            .flat_map(|scope| scope.iter().cloned())
            .collect();
        Ok(Val::Closure(Rc::new(Closure {
            params: c.inputs.iter().cloned().collect(),
            output,
            body: (*c.body).clone(),
            env,
        })))
    }

    // control flow

    fn eval_if(&self, eif: &syn::ExprIf) -> R<'k, Val<'k>> {
        let span = eif.span();
        // every `if` of a kernel becomes closures in the macro: no `break`/`continue` out of
        // its branches, no closures in them that assign outer variables
        let mut escapes = Escapes::default();
        let mut closures = ClosureAssigns::default();
        escapes.visit_block(&eif.then_branch);
        closures.visit_block(&eif.then_branch);
        if let Some((_, otherwise)) = &eif.else_branch {
            escapes.visit_expr(otherwise);
            closures.visit_expr(otherwise);
        }
        if let Some(s) = escapes.found {
            return fail(
                s,
                "an `if` of a kernel cannot `break` or `continue` an enclosing loop",
            );
        }
        if let Some((s, vars)) = closures.found {
            return fail(
                s,
                format!(
                    "a closure in a kernel `if` or `while` cannot assign {}, declared outside \
                     it; return the new value from the closure instead",
                    vars.iter()
                        .map(|v| format!("`{v}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }

        match self.eval(&eif.cond)? {
            Val::Bool(true) => self.eval_block(&eif.then_branch),
            Val::Bool(false) => match &eif.else_branch {
                Some((_, otherwise)) => self.eval(otherwise),
                None => Ok(Val::Unit),
            },
            Val::Mask(m) => {
                let vars = assigned_in_if(eif);
                if eif.else_branch.is_none() && vars.is_empty() {
                    return fail(
                        span,
                        "an `if` without `else` cannot branch on a kernel value; give it an \
                         `else` branch, and return the values from both branches",
                    );
                }
                self.kernel_branch(span, m, eif, &vars)
            }
            v => fail(
                eif.cond.span(),
                format!(
                    "the condition of an `if` must be a `bool` or a kernel comparison, not {}",
                    v.describe()
                ),
            ),
        }
    }

    /// A branch of the kernel (`KernelBuilder::if_else`): each side runs on the variables as
    /// they are before the branch, and gives its value and the variables `vars` it assigns;
    /// both are joined and the variables updated.
    fn kernel_branch(
        &self,
        span: Span,
        m: Mask<'k>,
        eif: &syn::ExprIf,
        vars: &[String],
    ) -> R<'k, Val<'k>> {
        let before = self.snapshot();
        let shapes: RefCell<Vec<Result<Shape, Flow<'k>>>> = RefCell::new(Vec::new());

        let side = |then: bool| -> Slots<'k> {
            self.restore(before.clone());
            let value = if then {
                self.eval_block(&eif.then_branch)
            } else {
                match &eif.else_branch {
                    Some((_, otherwise)) => self.eval(otherwise),
                    None => Ok(Val::Unit),
                }
            };
            let result = value.and_then(|v| {
                let mut all = vec![v];
                for name in vars {
                    match self.lookup(name) {
                        Some(v) => all.push(v),
                        None => return fail(span, format!("cannot find `{}`", name)),
                    }
                }
                let mut slots = Vec::new();
                let shape = here(span, to_slots(self.k, &Val::Tuple(all), true, &mut slots))?;
                Ok((shape, slots))
            });
            match result {
                Ok((shape, slots)) => {
                    shapes.borrow_mut().push(Ok(shape));
                    Slots::new(slots)
                }
                Err(flow) => {
                    shapes.borrow_mut().push(Err(flow));
                    Slots::new(Vec::new())
                }
            }
        };

        let joined: Slots<'k> = self.k.if_else(m, || side(true), || side(false));
        self.restore(before);

        let mut shapes = shapes.into_inner().into_iter();
        let (t, e) = (shapes.next().unwrap()?, shapes.next().unwrap()?);
        let Some(shape) = unify(&t, &e) else {
            return fail(
                span,
                format!(
                    "the branches of a kernel `if` give {} and {}{}",
                    describe_side(&t, vars),
                    describe_side(&e, vars),
                    "; both must give kernel values of the same shape, and a variable assigned \
                     in them must keep its type"
                ),
            );
        };
        let Val::Tuple(mut vals) = from_slots(self.k, &shape, &mut joined.0.into_iter()) else {
            unreachable!()
        };
        let value = vals.remove(0);
        for (name, v) in vars.iter().zip(vals) {
            let old = self.lookup(name).unwrap();
            let new = here(span, coerce(self.k, &old, v))?;
            self.set(name, new);
        }
        Ok(value)
    }

    fn eval_while(&self, w: &syn::ExprWhile) -> R<'k, Val<'k>> {
        let mut closures = ClosureAssigns::default();
        closures.visit_block(&w.body);
        if let Some((s, vars)) = closures.found {
            return fail(
                s,
                format!(
                    "a closure in a kernel `if` or `while` cannot assign {}, declared outside \
                     it; return the new value from the closure instead",
                    vars.iter()
                        .map(|v| format!("`{v}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }
        if w.label.is_some() {
            return fail(w.span(), "labeled loops are not supported by sjc");
        }
        if matches!(*w.cond, Expr::Let(_)) {
            return fail(w.span(), "`while let` is not supported by sjc");
        }

        // as in the macro: a loop with `break`/`continue`, or that assigns no outer variable,
        // is a Rust loop; otherwise a kernel loop if its condition is a kernel value
        let mut escapes = Escapes::default();
        escapes.visit_block(&w.body);
        let vars = assigned_in_block(&w.body);
        if escapes.found.is_some() || vars.is_empty() || !self.reads_kernel_values(&w.cond) {
            return self.rust_while(w, escapes.found.is_some() || vars.is_empty());
        }
        self.kernel_while(w, &vars)
    }

    /// Whether `e` reads a variable that holds a kernel value.
    fn reads_kernel_values(&self, e: &Expr) -> bool {
        let mut reads = Reads::default();
        reads.visit_expr(e);
        reads
            .0
            .iter()
            .any(|name| self.lookup(name).is_some_and(|v| v.is_kernel()))
    }

    /// A `while` run while the kernel is built.
    fn rust_while(&self, w: &syn::ExprWhile, plain: bool) -> R<'k, Val<'k>> {
        for _ in 0..LOOP_LIMIT {
            match self.eval(&w.cond)? {
                Val::Bool(true) => {}
                Val::Bool(false) => return Ok(Val::Unit),
                Val::Mask(_) => {
                    return fail(
                        w.cond.span(),
                        if plain {
                            "a `while` on a kernel value must assign variables declared outside \
                             it, and cannot `break` or `continue`"
                        } else {
                            "the condition of this loop became a kernel value"
                        },
                    )
                }
                v => {
                    return fail(
                        w.cond.span(),
                        format!("the condition of a `while` cannot be {}", v.describe()),
                    )
                }
            }
            match self.eval_block(&w.body) {
                Ok(_) | Err(Flow::Continue) => {}
                Err(Flow::Break(_)) => return Ok(Val::Unit),
                Err(e) => return Err(e),
            }
        }
        fail(w.span(), too_long())
    }

    /// A loop of the kernel (`KernelBuilder::while_loop`) over the variables `vars`.
    fn kernel_while(&self, w: &syn::ExprWhile, vars: &[String]) -> R<'k, Val<'k>> {
        let span = w.span();
        let mut init = Vec::new();
        for name in vars {
            let Some(v) = self.lookup(name) else {
                return fail(span, format!("cannot find `{}`", name));
            };
            let mut probe = Vec::new();
            if let Err(msg) = to_slots(self.k, &v, false, &mut probe) {
                return fail(
                    span,
                    format!("`{}` is updated in a kernel loop, but {}", name, msg),
                );
            }
            init.push(v);
        }
        let mut slots = Vec::new();
        let shape = here(span, to_slots(self.k, &Val::Tuple(init), false, &mut slots))?;

        let failure: RefCell<Option<Flow<'k>>> = RefCell::new(None);
        let set_state = |state: &Slots<'k>| {
            if let Val::Tuple(vals) = from_slots(self.k, &shape, &mut state.0.iter().copied()) {
                for (name, v) in vars.iter().zip(vals) {
                    self.set(name, v);
                }
            }
        };
        let dummy = || Mask::from_slots(self.k, &mut std::iter::once(Slot::Temp(0)));

        let fin: Slots<'k> = self.k.while_loop(
            Slots::new(slots),
            |state| {
                if failure.borrow().is_some() {
                    return dummy();
                }
                set_state(&state);
                match self.eval(&w.cond) {
                    Ok(Val::Mask(m)) => m,
                    Ok(v) => {
                        *failure.borrow_mut() = Some(Flow::Error(at(
                            w.cond.span(),
                            format!("the condition of a kernel loop cannot be {}", v.describe()),
                        )));
                        dummy()
                    }
                    Err(flow) => {
                        *failure.borrow_mut() = Some(flow);
                        dummy()
                    }
                }
            },
            |state| {
                if failure.borrow().is_some() {
                    return state;
                }
                set_state(&state);
                if let Err(flow) = self.eval_block(&w.body) {
                    *failure.borrow_mut() = Some(flow);
                    return state;
                }
                let vals: Vec<Val<'k>> = vars.iter().map(|n| self.lookup(n).unwrap()).collect();
                let mut out = Vec::new();
                match to_slots(self.k, &Val::Tuple(vals), false, &mut out) {
                    Ok(s) if s == shape => Slots::new(out),
                    Ok(s) => {
                        *failure.borrow_mut() = Some(Flow::Error(at(
                            span,
                            format!(
                                "the variables of a kernel loop must keep their types: ({}) \
                                 become {}",
                                vars.join(", "),
                                s.describe()
                            ),
                        )));
                        state
                    }
                    Err(msg) => {
                        *failure.borrow_mut() = Some(Flow::Error(at(span, msg)));
                        state
                    }
                }
            },
        );
        if let Some(flow) = failure.into_inner() {
            return Err(match flow {
                Flow::Break(_) | Flow::Continue => {
                    Flow::Error(at(span, "a kernel loop cannot `break` or `continue`"))
                }
                f => f,
            });
        }
        set_state(&fin);
        Ok(Val::Unit)
    }

    fn eval_for(&self, f: &syn::ExprForLoop) -> R<'k, Val<'k>> {
        if f.label.is_some() {
            return fail(f.span(), "labeled loops are not supported by sjc");
        }
        let items: Vec<Val<'k>> = match self.eval(&f.expr)? {
            Val::Range(a, b) => {
                if (b - a) as u128 > LOOP_LIMIT as u128 {
                    return fail(f.span(), too_long());
                }
                (a..b).map(Val::Int).collect()
            }
            Val::Array(vs) => vs,
            v => {
                return fail(
                    f.expr.span(),
                    format!(
                        "sjc's `for` loops go over ranges and arrays, not {}",
                        v.describe()
                    ),
                )
            }
        };
        for item in items {
            let r = self.scoped(|| {
                self.bind(&f.pat, item)?;
                self.eval_block(&f.body)
            });
            match r {
                Ok(_) | Err(Flow::Continue) => {}
                Err(Flow::Break(_)) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(Val::Unit)
    }

    fn eval_loop(&self, l: &syn::ExprLoop) -> R<'k, Val<'k>> {
        if l.label.is_some() {
            return fail(l.span(), "labeled loops are not supported by sjc");
        }
        for _ in 0..LOOP_LIMIT {
            match self.eval_block(&l.body) {
                Ok(_) | Err(Flow::Continue) => {}
                Err(Flow::Break(v)) => return Ok(v.unwrap_or(Val::Unit)),
                Err(e) => return Err(e),
            }
        }
        fail(l.span(), too_long())
    }

    // calls

    fn call(&self, call: &syn::ExprCall) -> R<'k, Val<'k>> {
        let span = call.span();
        let args = || -> R<'k, Vec<Val<'k>>> { call.args.iter().map(|a| self.eval(a)).collect() };

        let Expr::Path(p) = &*call.func else {
            let f = self.eval(&call.func)?;
            return self.call_value(span, f, args()?);
        };
        if p.qself.is_some() {
            return fail(
                span,
                "qualified paths (`<T as Trait>::f`) are not supported by sjc",
            );
        }

        if let Some(id) = p.path.get_ident() {
            let name = id.to_string();
            let current = self.current.get();
            if name == current.name {
                return match current.kind {
                    FnKind::Inline => fail(
                        span,
                        "an inline helper cannot call itself (only a kernel can recurse)",
                    ),
                    _ => self.recursion(call, current),
                };
            }
            if let Some(f) = self.lookup(&name) {
                return self.call_value(span, f, args()?);
            }
            if name.starts_with(|c: char| c.is_uppercase()) {
                return fail(
                    span,
                    format!(
                        "sjc cannot call `{}` (tuple structs and enum variants)",
                        name
                    ),
                );
            }
            return match self.src.fns.get(&name) {
                Some(def) => match def.kind {
                    FnKind::Inline => {
                        let args = args()?;
                        self.inline_call(span, def, args)
                    }
                    FnKind::Kernel => {
                        let args = args()?;
                        self.kernel_call(span, def, args)
                    }
                    FnKind::Callback => {
                        let args = args()?;
                        self.callback_call(span, def, args)
                    }
                    FnKind::Plain => fail(
                        span,
                        format!(
                            "`{}` is not a kernel helper: give it `#[kernel(inline)]`",
                            name
                        ),
                    ),
                },
                None => fail(span, format!("cannot find function `{}`", name)),
            };
        }

        let segs: Vec<String> = p
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        if super::syntax::is_complex_new(&call.func) {
            if call.args.len() != 2 {
                return fail(span, "`Complex::new` takes two arguments");
            }
            let mut a = args()?;
            let im = a.pop().unwrap();
            let re = a.pop().unwrap();
            return here(span, complex(self.k, re, im));
        }
        // `f64::max(a, b)` and the like: a method of the first argument
        let f64_path = segs.len() >= 2
            && segs[segs.len() - 2] == "f64"
            && (segs.len() == 2 || matches!(segs[0].as_str(), "std" | "core"));
        if f64_path && !call.args.is_empty() {
            let mut a = args()?;
            let recv = a.remove(0);
            return here(span, method(self.k, recv, segs.last().unwrap(), a));
        }
        fail(span, format!("sjc cannot call `{}`", segs.join("::")))
    }

    fn call_value(&self, span: Span, f: Val<'k>, args: Vec<Val<'k>>) -> R<'k, Val<'k>> {
        let Val::Closure(c) = f else {
            return fail(span, format!("{} is not a function", f.describe()));
        };
        if c.params.len() != args.len() {
            return fail(
                span,
                format!(
                    "the closure takes {} arguments, not {}",
                    c.params.len(),
                    args.len()
                ),
            );
        }
        let v = self.in_body(None, c.env.clone(), || {
            self.scoped(|| {
                for (p, a) in c.params.iter().zip(args) {
                    self.bind(p, a)?;
                }
                self.eval(&c.body)
            })
        })?;
        match &c.output {
            Some(ty) if ty.has_handle() => here(span, lift(self.k, v, ty)),
            _ => Ok(v),
        }
    }

    /// A call of an inline helper: its body is evaluated with the arguments bound (converted
    /// to the types of its parameters).
    fn inline_call(&self, span: Span, def: &'s FnDef, args: Vec<Val<'k>>) -> R<'k, Val<'k>> {
        if def.params.len() != args.len() {
            return fail(
                span,
                format!(
                    "`{}` takes {} arguments, not {}",
                    def.name,
                    def.params.len(),
                    args.len()
                ),
            );
        }
        let mut lifted = Vec::new();
        for ((_, ty), a) in def.params.iter().zip(args) {
            lifted.push(here(span, lift(self.k, a, ty))?);
        }
        let v = self.in_body(Some(def), Vec::new(), || {
            for ((pat, _), a) in def.params.iter().zip(lifted) {
                self.bind(pat, a)?;
            }
            self.eval_block(&def.item.block)
        })?;
        match &def.ret {
            Some(ty) if ty.has_handle() => here(span, lift(self.k, v, ty)),
            _ => Ok(v),
        }
    }

    /// A call of another kernel (`KernelBuilder::call`, the `__call` of its module in the
    /// macro), with the arguments converted to the types of its parameters; it returns one
    /// value, of the type of its result.
    fn kernel_call(&self, span: Span, def: &'s FnDef, args: Vec<Val<'k>>) -> R<'k, Val<'k>> {
        if def.params.len() != args.len() {
            return fail(
                span,
                format!(
                    "`{}` takes {} arguments, not {}",
                    def.name,
                    def.params.len(),
                    args.len()
                ),
            );
        }
        let mut slots = Vec::new();
        for ((_, ty), a) in def.params.iter().zip(args) {
            let v = here(span, lift(self.k, a, ty))?;
            here(span, to_slots(self.k, &v, false, &mut slots))?;
        }
        let name = self.naming.kernel(&def.name);
        let args = Slots::new(slots);
        match &def.ret {
            Some(Ty::Real) => Ok(Val::Real(self.k.call::<_, Real>(&name, args))),
            Some(Ty::Cplx) => Ok(Val::Cplx(self.k.call::<_, Cplx>(&name, args))),
            Some(Ty::Mask) => Ok(Val::Mask(self.k.call::<_, Mask>(&name, args))),
            _ => fail(
                span,
                format!(
                    "a call of the kernel `{}` returns one value, not several",
                    def.name
                ),
            ),
        }
    }

    /// A call of a callback (`KernelBuilder::callback`): one or two real arguments.
    fn callback_call(&self, span: Span, def: &'s FnDef, args: Vec<Val<'k>>) -> R<'k, Val<'k>> {
        let mut reals = Vec::new();
        for a in args {
            match here(span, lift(self.k, a, &Ty::Real))? {
                Val::Real(r) => reals.push(r),
                _ => unreachable!(),
            }
        }
        let name = self.naming.callback(&def.name);
        match reals.as_slice() {
            [x] => Ok(Val::Real(self.k.callback(&name, *x))),
            [x, y] => Ok(Val::Real(self.k.callback(&name, (*x, *y)))),
            _ => fail(
                span,
                format!(
                    "the callback `{}` takes one or two arguments, not {}",
                    def.name,
                    reals.len()
                ),
            ),
        }
    }

    /// A call of the kernel itself (`KernelBuilder::recurse`), with the arguments converted to
    /// the types of its parameters.
    fn recursion(&self, call: &syn::ExprCall, def: &'s FnDef) -> R<'k, Val<'k>> {
        let span = call.span();
        if call.args.len() != def.params.len() {
            return fail(
                span,
                format!("the kernel takes {} arguments", def.params.len()),
            );
        }
        let mut slots = Vec::new();
        for (a, (_, ty)) in call.args.iter().zip(def.params.iter()) {
            let v = self.eval(a)?;
            let v = here(a.span(), lift(self.k, v, ty))?;
            here(a.span(), to_slots(self.k, &v, false, &mut slots))?;
        }
        let args = Slots::new(slots);
        match &def.ret {
            Some(Ty::Real) => Ok(Val::Real(self.k.recurse::<_, Real>(args))),
            Some(Ty::Cplx) => Ok(Val::Cplx(self.k.recurse::<_, Cplx>(args))),
            _ => fail(
                span,
                "a recursive kernel must return one `f64` or `Complex<f64>`",
            ),
        }
    }
}

fn too_long() -> String {
    format!(
        "a loop run while the kernel is built did not end after {} iterations",
        LOOP_LIMIT
    )
}

fn describe_side(shape: &Shape, vars: &[String]) -> String {
    match shape {
        Shape::Tuple(parts) if !vars.is_empty() => {
            let mut s = format!("`{}`", parts[0].describe());
            let assigned: Vec<String> = vars
                .iter()
                .zip(&parts[1..])
                .map(|(v, p)| format!("{v}: {}", p.describe()))
                .collect();
            s += &format!(" (assigning {})", assigned.join(", "));
            s
        }
        Shape::Tuple(parts) => format!("`{}`", parts[0].describe()),
        s => format!("`{}`", s.describe()),
    }
}

pub fn path_string(p: &syn::Path) -> String {
    p.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

/// The value of a path such as `std::f64::consts::PI` or `f64::EPSILON`.
fn known_path<'k>(segs: &[String]) -> Option<Val<'k>> {
    let last = segs.last()?.as_str();
    let prev = segs.get(segs.len().wrapping_sub(2)).map(|s| s.as_str());
    if prev == Some("consts") {
        return float_const(last).map(Val::F64);
    }
    match (prev, last) {
        (Some("f64"), "INFINITY") => Some(Val::F64(f64::INFINITY)),
        (Some("f64"), "NEG_INFINITY") => Some(Val::F64(f64::NEG_INFINITY)),
        (Some("f64"), "NAN") => Some(Val::F64(f64::NAN)),
        (Some("f64"), "EPSILON") => Some(Val::F64(f64::EPSILON)),
        (Some("f64"), "MAX") => Some(Val::F64(f64::MAX)),
        (Some("f64"), "MIN") => Some(Val::F64(f64::MIN)),
        (Some("f64"), "MIN_POSITIVE") => Some(Val::F64(f64::MIN_POSITIVE)),
        (Some("i32"), "MAX") => Some(Val::Int(i32::MAX as i64)),
        (Some("i32"), "MIN") => Some(Val::Int(i32::MIN as i64)),
        (Some("i64"), "MAX") => Some(Val::Int(i64::MAX)),
        (Some("i64"), "MIN") => Some(Val::Int(i64::MIN)),
        _ => None,
    }
}

/// The constants of `std::f64::consts`.
pub fn float_const(name: &str) -> Option<f64> {
    use std::f64::consts::*;
    Some(match name {
        "PI" => PI,
        "TAU" => TAU,
        "E" => E,
        "SQRT_2" => SQRT_2,
        "FRAC_1_SQRT_2" => FRAC_1_SQRT_2,
        "LN_2" => LN_2,
        "LN_10" => LN_10,
        "LOG2_E" => LOG2_E,
        "LOG10_E" => LOG10_E,
        "LOG2_10" => LOG2_10,
        "LOG10_2" => LOG10_2,
        "FRAC_PI_2" => FRAC_PI_2,
        "FRAC_PI_3" => FRAC_PI_3,
        "FRAC_PI_4" => FRAC_PI_4,
        "FRAC_PI_6" => FRAC_PI_6,
        "FRAC_PI_8" => FRAC_PI_8,
        "FRAC_1_PI" => FRAC_1_PI,
        "FRAC_2_PI" => FRAC_2_PI,
        "FRAC_2_SQRT_PI" => FRAC_2_SQRT_PI,
        _ => return None,
    })
}
