//! The syntax rules of `#[kernel]` (`macros/src/lib.rs`), for the interpreter: which
//! variables a branch or loop passes through, which kernels need the direct translator, what
//! the types of parameters and `let`s make of their values.

use std::collections::HashSet;

use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{BinOp, Expr, GenericArgument, PathArguments, Type};

/// The kind of value a type stands for, after `stage_type` in the macro: `f64` is a `Real`,
/// `Complex<f64>` a `Cplx`, `bool` a `Mask` (handles), also inside tuples and arrays; other
/// types are left to Rust (`Other`, e.g. the `i32` parameter of an inline helper).
#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Real,
    Cplx,
    Mask,
    Tuple(Vec<Ty>),
    Array(Box<Ty>, usize),
    Unit,
    Other(String),
}

impl Ty {
    /// Whether the type contains a handle (`staged.handle` in the macro).
    pub fn has_handle(&self) -> bool {
        match self {
            Ty::Real | Ty::Cplx | Ty::Mask => true,
            Ty::Tuple(ts) => ts.iter().any(|t| t.has_handle()),
            Ty::Array(t, _) => t.has_handle(),
            Ty::Unit | Ty::Other(_) => false,
        }
    }

    /// Whether it contains a type that is not a handle (`staged.other`).
    pub fn has_other(&self) -> bool {
        match self {
            Ty::Real | Ty::Cplx | Ty::Mask => false,
            Ty::Tuple(ts) => ts.iter().any(|t| t.has_other()),
            Ty::Array(t, _) => t.has_other(),
            Ty::Unit => false,
            Ty::Other(_) => true,
        }
    }

    pub fn has_complex(&self) -> bool {
        match self {
            Ty::Cplx => true,
            Ty::Tuple(ts) => ts.iter().any(|t| t.has_complex()),
            Ty::Array(t, _) => t.has_complex(),
            _ => false,
        }
    }

    pub fn has_mask(&self) -> bool {
        match self {
            Ty::Mask => true,
            Ty::Tuple(ts) => ts.iter().any(|t| t.has_mask()),
            Ty::Array(t, _) => t.has_mask(),
            _ => false,
        }
    }
}

/// The `Ty` of a Rust type. `len` evaluates the length of an array type (a literal or a
/// constant of the file).
pub fn stage_type(ty: &Type, len: &dyn Fn(&Expr) -> Option<usize>) -> Ty {
    match ty {
        Type::Path(tp) if tp.qself.is_none() => {
            let seg = tp.path.segments.last().unwrap();
            let name = seg.ident.to_string();
            let complex_f64 = match &seg.arguments {
                PathArguments::AngleBracketed(args) => {
                    args.args.len() == 1
                        && matches!(&args.args[0], GenericArgument::Type(Type::Path(p))
                            if p.path.is_ident("f64"))
                }
                _ => false,
            };
            if name == "f64" && seg.arguments.is_none() {
                Ty::Real
            } else if (name == "Complex" && complex_f64) || name == "Complex64" {
                Ty::Cplx
            } else if name == "bool" && seg.arguments.is_none() {
                Ty::Mask
            } else {
                Ty::Other(name)
            }
        }
        Type::Tuple(t) if t.elems.is_empty() => Ty::Unit,
        Type::Tuple(t) => Ty::Tuple(t.elems.iter().map(|e| stage_type(e, len)).collect()),
        Type::Array(a) => match len(&a.len) {
            Some(n) => Ty::Array(Box::new(stage_type(&a.elem, len)), n),
            None => Ty::Other("array".to_string()),
        },
        Type::Paren(p) => stage_type(&p.elem, len),
        Type::Group(g) => stage_type(&g.elem, len),
        _ => Ty::Other("type".to_string()),
    }
}

/// The variables a branch or loop body assigns (`x = ..`, `x += ..`, `x[i] = ..`,
/// `(x, y) = ..`) without declaring them, in order of appearance. Closures are skipped.
#[derive(Default)]
pub struct Assigned {
    names: Vec<String>,
    declared: HashSet<String>,
}

impl Assigned {
    pub fn outer(self) -> Vec<String> {
        let declared = self.declared;
        self.names
            .into_iter()
            .filter(|id| !declared.contains(id))
            .collect()
    }

    fn roots(&mut self, e: &Expr) {
        match e {
            Expr::Path(p) => {
                if let Some(id) = p.path.get_ident() {
                    let id = id.to_string();
                    if !self.names.contains(&id) {
                        self.names.push(id);
                    }
                }
            }
            Expr::Index(i) => self.roots(&i.expr),
            Expr::Field(f) => self.roots(&f.base),
            Expr::Paren(p) => self.roots(&p.expr),
            Expr::Tuple(t) => t.elems.iter().for_each(|e| self.roots(e)),
            Expr::Array(a) => a.elems.iter().for_each(|e| self.roots(e)),
            _ => {}
        }
    }
}

impl<'ast> Visit<'ast> for Assigned {
    fn visit_expr(&mut self, e: &'ast Expr) {
        match e {
            Expr::Assign(a) => self.roots(&a.left),
            Expr::Binary(b) if is_compound(&b.op) => self.roots(&b.left),
            Expr::Closure(_) => return,
            _ => {}
        }
        visit::visit_expr(self, e);
    }

    fn visit_pat_ident(&mut self, p: &'ast syn::PatIdent) {
        self.declared.insert(p.ident.to_string());
        visit::visit_pat_ident(self, p);
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

pub fn is_compound(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_)
    )
}

/// The variables declared outside `block` that it assigns.
pub fn assigned_in_block(block: &syn::Block) -> Vec<String> {
    let mut a = Assigned::default();
    a.visit_block(block);
    a.outer()
}

/// The variables declared outside the branches of an `if` that they assign.
pub fn assigned_in_if(eif: &syn::ExprIf) -> Vec<String> {
    let mut a = Assigned::default();
    a.visit_block(&eif.then_branch);
    if let Some((_, otherwise)) = &eif.else_branch {
        a.visit_expr(otherwise);
    }
    a.outer()
}

/// The variables declared outside a closure that its body assigns.
pub fn assigned_in_closure(c: &syn::ExprClosure) -> Vec<String> {
    let mut a = Assigned::default();
    for input in c.inputs.iter() {
        a.visit_pat(input);
    }
    a.visit_expr(&c.body);
    a.outer()
}

/// Whether the body has a `while` loop: kernel loops need the direct translator.
#[derive(Default)]
pub struct HasWhile(pub bool);

impl<'ast> Visit<'ast> for HasWhile {
    fn visit_expr_while(&mut self, w: &'ast syn::ExprWhile) {
        self.0 = true;
        visit::visit_expr_while(self, w);
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

/// `break` and `continue` that would leave a kernel branch or loop body (not inside a loop
/// of their own): the spans of the first one, if any.
#[derive(Default)]
pub struct Escapes {
    loops: usize,
    pub found: Option<proc_macro2::Span>,
}

impl<'ast> Visit<'ast> for Escapes {
    fn visit_expr(&mut self, e: &'ast Expr) {
        match e {
            Expr::ForLoop(_) | Expr::While(_) | Expr::Loop(_) => {
                self.loops += 1;
                visit::visit_expr(self, e);
                self.loops -= 1;
            }
            Expr::Closure(_) => {}
            Expr::Break(_) | Expr::Continue(_) if self.loops == 0 => {
                self.found.get_or_insert(e.span());
            }
            _ => visit::visit_expr(self, e),
        }
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

/// Closures in a kernel branch or loop that assign variables declared outside them (an
/// error in the macro): the names of the first one's.
#[derive(Default)]
pub struct ClosureAssigns {
    pub found: Option<(proc_macro2::Span, Vec<String>)>,
}

impl<'ast> Visit<'ast> for ClosureAssigns {
    fn visit_expr_closure(&mut self, c: &'ast syn::ExprClosure) {
        let vars = assigned_in_closure(c);
        if !vars.is_empty() && self.found.is_none() {
            self.found = Some((c.span(), vars));
        }
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

/// The identifiers a (condition) expression reads.
#[derive(Default)]
pub struct Reads(pub Vec<String>);

impl<'ast> Visit<'ast> for Reads {
    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        if let Some(id) = p.path.get_ident() {
            self.0.push(id.to_string());
        }
        visit::visit_expr_path(self, p);
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

pub fn is_float_literal(e: &Expr) -> bool {
    match e {
        Expr::Lit(l) => matches!(l.lit, syn::Lit::Float(_)),
        Expr::Unary(u) => matches!(u.op, syn::UnOp::Neg(_)) && is_float_literal(&u.expr),
        Expr::Paren(p) => is_float_literal(&p.expr),
        _ => false,
    }
}

/// `Complex::new(re, im)` (or `Complex64::new`) of float literals.
pub fn is_complex_literal(e: &Expr) -> bool {
    let Expr::Call(call) = e else {
        return false;
    };
    is_complex_new(&call.func) && call.args.len() == 2 && call.args.iter().all(is_float_literal)
}

/// `Complex::new` or `Complex64::new` (with any path before them).
pub fn is_complex_new(func: &Expr) -> bool {
    let Expr::Path(p) = func else {
        return false;
    };
    let segs: Vec<String> = p
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    segs.len() >= 2
        && segs[segs.len() - 1] == "new"
        && (segs[segs.len() - 2] == "Complex" || segs[segs.len() - 2] == "Complex64")
}

/// The names bound in a function (parameters, `let`, closure parameters, `for` patterns,
/// nested functions): calls of these are not calls of kernels or helpers (`Locals` in the
/// macro).
#[derive(Default)]
pub struct Locals(pub HashSet<String>);

impl<'ast> Visit<'ast> for Locals {
    fn visit_pat_ident(&mut self, p: &'ast syn::PatIdent) {
        self.0.insert(p.ident.to_string());
        visit::visit_pat_ident(self, p);
    }

    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        self.0.insert(f.sig.ident.to_string());
    }
}

/// The functions a function calls by a single identifier (kernels, inline helpers,
/// callbacks), in the order in which the macro's rewriter meets them (children first): the
/// `callees` of the macro, which its `compile` registers in this order.
pub fn callees(f: &syn::ItemFn) -> Vec<String> {
    let mut locals = Locals::default();
    for arg in f.sig.inputs.iter() {
        if let syn::FnArg::Typed(pt) = arg {
            locals.visit_pat(&pt.pat);
        }
    }
    locals.visit_block(&f.block);

    struct Calls {
        own: String,
        locals: HashSet<String>,
        found: Vec<String>,
    }
    impl<'ast> Visit<'ast> for Calls {
        fn visit_expr(&mut self, e: &'ast Expr) {
            visit::visit_expr(self, e);
            if let Expr::Call(call) = e {
                if let Expr::Path(p) = &*call.func {
                    if let (None, Some(id)) = (&p.qself, p.path.get_ident()) {
                        let s = id.to_string();
                        if s != self.own
                            && !self.locals.contains(&s)
                            && !s.starts_with(|c: char| c.is_uppercase())
                            && !self.found.contains(&s)
                        {
                            self.found.push(s);
                        }
                    }
                }
            }
        }

        fn visit_item(&mut self, _: &'ast syn::Item) {}
    }

    let mut calls = Calls {
        own: f.sig.ident.to_string(),
        locals: locals.0,
        found: Vec::new(),
    };
    calls.visit_block(&f.block);
    calls.found
}
