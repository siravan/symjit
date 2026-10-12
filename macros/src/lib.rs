//! `#[symjit::kernel]`: compiles a Rust function into a symjit kernel (cargo feature
//! `composer` of `symjit`).
//!
//! ```ignore
//! use num_complex::Complex;
//! use symjit::kernel::kernel;
//!
//! #[kernel]
//! fn norm_clamped(z: Complex<f64>, r: f64) -> f64 {
//!     let n = z.re * z.re + z.im * z.im;
//!     if n < r * r { n.sqrt() } else { r }
//! }
//!
//! let app = norm_clamped::compile(config)?;    // the kernel
//! let v = norm_clamped(Complex::new(3.0, 4.0), 10.0);    // the function itself, as Rust
//! ```
//!
//! The function stays as it is, an ordinary Rust function, and a module of the same name is
//! added with `compile(config: Config) -> Result<Application>`, which builds the kernel with
//! `symjit::kernel::KernelBuilder` (`rust/kernel.rs`): the body runs once, over handles
//! (`Real` for `f64`, `Cplx` for `Complex<f64>`, `Mask` for `bool`), and records the
//! operations. The kernel is complex if a parameter or the result is complex, or with
//! `#[kernel(complex)]`.
//!
//! The macro rewrites only the syntax Rust cannot overload, into the functions of
//! `symjit::kernel::ops`, which work on plain Rust values as well:
//! - comparisons and `&&`/`||` (both sides are evaluated);
//! - `if c { .. } else { .. }`: a kernel branch if `c` is a kernel value, an ordinary `if`
//!   if it is a `bool` (e.g. on a loop counter). The variables declared outside that a
//!   branch assigns are passed through it: each branch works on copies and returns their new
//!   values, which are assigned after the branch. An `if` without `else` on a kernel value
//!   must assign something;
//! - `while c { .. }` that assigns variables declared outside it: a kernel loop if `c` is a
//!   kernel value (the variables must then be kernel values: a counter is `0.0`, not `0`),
//!   an ordinary loop while the kernel is built if it is a `bool`. A kernel with a `while`
//!   is compiled with the direct translator (`compile` sets it), the only one with loops;
//! - `x = value`: a number assigned to a kernel value becomes a constant, and
//!   `let mut x = 1.5` (a float literal) or `let mut z = Complex::new(0.0, 1.0)` (of float
//!   literals) makes `x` a kernel value, so that it can accumulate kernel values;
//! - `z.re`, `z.im` and `Complex::new(re, im)`;
//! - float literals in array literals with other elements (`[1.0, x]`) take their type;
//! - calls of the function itself: recursion (`@self`);
//! - calls of other functions named by a single identifier, which must have the attribute
//!   too (call other functions with a path, e.g. `f64::max(a, b)`, or bind them to a local
//!   variable):
//!   - `#[kernel(inline)]` helpers are inlined;
//!   - `#[kernel]` kernels are called (`KernelBuilder::call`): `compile` compiles each kernel
//!     the kernel calls, also those of its inline helpers, and adds them to the `Defuns` of
//!     its configuration. A called kernel returns one value. A real kernel called from a
//!     complex one is compiled as a complex kernel; a real kernel cannot call a complex one.
//!     Kernels cannot call each other in a cycle (only a kernel itself can recurse);
//!   - `#[kernel(callback)]` functions, ordinary Rust functions of one or two `f64`
//!     returning `f64`, are called at run time (`KernelBuilder::callback`; in complex kernels
//!     with the real parts). A panic in one aborts the process;
//! - typed `let` statements and closure parameters: `f64` and `Complex<f64>` become handles.
//!
//! Everything else is ordinary Rust that runs while the kernel is built: `for` loops are
//! unrolled, and integer and `bool` arithmetic is done once. Not supported: `return`;
//! `break`/`continue` out of a kernel branch (a `while` with them stays a Rust loop);
//! assignments in closures inside kernel branches and loops (an error) and by methods
//! (`v.push(x)`, not passed through). Kernels must be items of a module (not of a function body), since the
//! generated module imports its parent.
use std::collections::HashSet;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{
    parse_quote, parse_quote_spanned, BinOp, Block, Expr, FnArg, GenericArgument, Ident, ItemFn,
    Lifetime, Member, Pat, PathArguments, ReturnType, Token, Type,
};

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = Options::parse(attr.into()).and_then(|opts| {
        let func: ItemFn = syn::parse(item)?;
        expand(opts, func)
    });

    match result {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[derive(Default)]
struct Options {
    /// `#[kernel(inline)]`: a helper inlined into the kernels that call it
    inline: bool,
    /// `#[kernel(complex)]`: a complex kernel even if no parameter or result is complex
    complex: bool,
    /// `#[kernel(callback)]`: a Rust function called by the kernels at run time
    callback: bool,
}

impl Options {
    fn parse(attr: TokenStream2) -> syn::Result<Options> {
        let mut opts = Options::default();
        let idents = Punctuated::<Ident, Token![,]>::parse_terminated.parse2(attr)?;
        for ident in idents {
            match ident.to_string().as_str() {
                "inline" => opts.inline = true,
                "complex" => opts.complex = true,
                "callback" => opts.callback = true,
                _ => {
                    return Err(syn::Error::new(
                        ident.span(),
                        "expected `inline`, `complex` or `callback`",
                    ))
                }
            }
        }
        if opts.callback && (opts.inline || opts.complex) {
            return Err(syn::Error::new(
                Span::call_site(),
                "`callback` cannot be combined with `inline` or `complex`",
            ));
        }
        Ok(opts)
    }
}

/// What a type contains, after `stage_type`.
#[derive(Default)]
struct Staged {
    /// a handle (`Real`, `Cplx` or `Mask`)
    handle: bool,
    /// a `Cplx`
    complex: bool,
    /// a `Mask`
    mask: bool,
    /// a type that is not a handle, beside handles (e.g. `(f64, i32)`)
    other: bool,
}

/// The type of the kernel builder for `ty`: `f64` becomes `Real<'k>`, `Complex<f64>` (or
/// `Complex64`) `Cplx<'k>` and `bool` `Mask<'k>`, also inside tuples and arrays.
fn stage_type(ty: &Type, lt: &Lifetime, staged: &mut Staged) -> Type {
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
                staged.handle = true;
                parse_quote!(::symjit::kernel::Real<#lt>)
            } else if (name == "Complex" && complex_f64) || name == "Complex64" {
                staged.handle = true;
                staged.complex = true;
                parse_quote!(::symjit::kernel::Cplx<#lt>)
            } else if name == "bool" && seg.arguments.is_none() {
                staged.handle = true;
                staged.mask = true;
                parse_quote!(::symjit::kernel::Mask<#lt>)
            } else {
                staged.other = true;
                ty.clone()
            }
        }
        Type::Tuple(t) if !t.elems.is_empty() => {
            let elems = t.elems.iter().map(|e| stage_type(e, lt, staged));
            parse_quote!((#(#elems,)*))
        }
        Type::Array(a) => {
            let elem = stage_type(&a.elem, lt, staged);
            let len = &a.len;
            parse_quote!([#elem; #len])
        }
        Type::Paren(p) => stage_type(&p.elem, lt, staged),
        Type::Group(g) => stage_type(&g.elem, lt, staged),
        _ => {
            staged.other = true;
            ty.clone()
        }
    }
}

struct Param {
    pat: Pat,
    ty: Type,
    staged: Staged,
}

fn expand(opts: Options, func: ItemFn) -> syn::Result<TokenStream2> {
    let sig = &func.sig;
    let name = &sig.ident;
    let vis = &func.vis;
    let lt = Lifetime::new("'k", Span::call_site());

    if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &sig.generics,
            "a kernel cannot be generic",
        ));
    }
    if let Some(a) = &sig.asyncness {
        return Err(syn::Error::new_spanned(a, "a kernel cannot be async"));
    }
    if let Some(v) = &sig.variadic {
        return Err(syn::Error::new_spanned(v, "a kernel cannot be variadic"));
    }

    if opts.callback {
        return expand_callback(&func);
    }

    let mut params = Vec::new();
    for arg in sig.inputs.iter() {
        match arg {
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(r, "a kernel cannot be a method"));
            }
            FnArg::Typed(pt) => {
                let mut staged = Staged::default();
                let ty = stage_type(&pt.ty, &lt, &mut staged);
                params.push(Param {
                    pat: (*pt.pat).clone(),
                    ty,
                    staged,
                });
            }
        }
    }

    let mut ret_staged = Staged::default();
    let ret = match &sig.output {
        ReturnType::Type(_, ty) => Some(stage_type(ty, &lt, &mut ret_staged)),
        ReturnType::Default => None,
    };

    let complex = opts.complex || ret_staged.complex || params.iter().any(|p| p.staged.complex);

    if !opts.inline {
        for (p, arg) in params.iter().zip(sig.inputs.iter()) {
            if !p.staged.handle || p.staged.other || p.staged.mask {
                return Err(syn::Error::new_spanned(
                    arg,
                    "the parameters of a kernel must be `f64` or `Complex<f64>` (or tuples or \
                     arrays of them); use `#[kernel(inline)]` for a helper with other parameters",
                ));
            }
        }
        match &sig.output {
            ReturnType::Type(_, ty) if ret_staged.handle && !ret_staged.other => {
                let _ = ty;
            }
            _ => {
                return Err(syn::Error::new(
                    sig.output
                        .span()
                        .join(sig.ident.span())
                        .unwrap_or(sig.ident.span()),
                    "a kernel must return `f64`, `Complex<f64>` or `bool` (or tuples or arrays \
                     of them)",
                ));
            }
        }
    }

    let mut has_while = HasWhile::default();
    has_while.visit_block(&func.block);
    let has_while = has_while.0;

    // the staged body
    let mut locals = Locals::default();
    for p in params.iter() {
        locals.visit_pat(&p.pat);
    }
    locals.visit_block(&func.block);

    let mut rewriter = Rewriter {
        name: name.clone(),
        inline: opts.inline,
        locals: locals.names,
        params: params.iter().map(|p| p.ty.clone()).collect(),
        ret: ret.clone(),
        lt: lt.clone(),
        closures: 0,
        callees: Vec::new(),
        errors: Vec::new(),
    };
    let mut body: Block = (*func.block).clone();
    rewriter.visit_block_mut(&mut body);
    let callees = rewriter.callees;
    if let Some(err) = rewriter.errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(err);
    }

    let body = match &ret {
        Some(ty) if ret_staged.handle && !ret_staged.other => {
            quote!(::symjit::kernel::ops::lift::<#ty, _>(__k, #body))
        }
        _ => quote!(#body),
    };
    let ret_tokens = match &ret {
        Some(ty) => quote!(-> #ty),
        None => quote!(),
    };

    let doc_fn = format!("[`{name}`] as a symjit kernel (`#[symjit::kernel]`).");
    let allow = quote!(#[allow(
        non_snake_case,
        unused_imports,
        unused_variables,
        unused_mut,
        unused_assignments,
        unused_parens,
        unused_braces,
        dead_code,
        clippy::all
    )]);

    let module = if opts.inline {
        // the arguments are converted to handles (numbers become constants)
        let args = (0..params.len()).map(|i| format_ident!("__a{}", i));
        let arg_decls = params.iter().zip(args.clone()).map(|(p, a)| {
            let ty = &p.ty;
            if p.staged.handle {
                quote!(#a: impl ::symjit::kernel::ops::Lift<#lt, #ty>)
            } else {
                quote!(#a: #ty)
            }
        });
        let lets = params.iter().zip(args).map(|(p, a)| {
            let (pat, ty) = (&p.pat, &p.ty);
            if p.staged.handle {
                quote!(let #pat: #ty = ::symjit::kernel::ops::Lift::lift(#a, __k);)
            } else {
                quote!(let #pat: #ty = #a;)
            }
        });

        quote! {
            #[doc = #doc_fn]
            #allow
            #vis mod #name {
                use super::*;

                /// Inlines the helper into the kernel being built.
                pub fn __call<#lt>(
                    __k: &#lt ::symjit::kernel::KernelBuilder,
                    #(#arg_decls),*
                ) #ret_tokens {
                    #(#lets)*
                    #body
                }

                /// Adds the kernels and callbacks the helper calls to `df`.
                pub fn __register(
                    config: &::symjit::Config,
                    df: &mut ::symjit::Defuns,
                ) -> ::symjit::kernel::Result<()> {
                    #(#callees::__register(config, df)?;)*
                    Ok(())
                }
            }
        }
    } else {
        let anon = Lifetime::new("'_", Span::call_site());
        let param_types = params.iter().map(|p| {
            let mut staged = Staged::default();
            let orig = match &sig.inputs[params.iter().position(|q| std::ptr::eq(q, p)).unwrap()] {
                FnArg::Typed(pt) => (*pt.ty).clone(),
                FnArg::Receiver(_) => unreachable!(),
            };
            stage_type(&orig, &anon, &mut staged)
        });
        let indices = (0..params.len()).map(syn::Index::from);
        let decls = params.iter().map(|p| {
            let (pat, ty) = (&p.pat, &p.ty);
            quote!(#pat: #ty)
        });
        let args = (0..params.len()).map(|i| format_ident!("__a{}", i));
        let arg_decls = params.iter().zip(args.clone()).map(|(p, a)| {
            let ty = &p.ty;
            quote!(#a: impl ::symjit::kernel::ops::Lift<#lt, #ty>)
        });
        let lifted = params.iter().zip(args).map(|(p, a)| {
            let ty = &p.ty;
            quote!(::symjit::kernel::ops::lift::<#ty, _>(__k, #a))
        });

        quote! {
            #[doc = #doc_fn]
            #allow
            #vis mod #name {
                use super::*;

                /// Compiles the kernel; `config` is made complex if the kernel is.
                pub fn compile(
                    config: ::symjit::Config,
                ) -> ::symjit::kernel::Result<::symjit::Application> {
                    __build(config, #complex)
                }

                /// Compiles the kernel, as a complex kernel if `complex`.
                pub fn __build(
                    mut config: ::symjit::Config,
                    complex: bool,
                ) -> ::symjit::kernel::Result<::symjit::Application> {
                    let __building = ::symjit::kernel::ops::building(module_path!())?;
                    config.set_complex(complex);
                    if #has_while {
                        // kernel loops need the direct translator
                        config.set_direct(true);
                    }
                    // the kernels and callbacks it calls
                    let mut __df = ::symjit::Defuns::new();
                    #(#callees::__register(&config, &mut __df)?;)*
                    ::symjit::kernel::add_defuns(&mut config, __df);

                    let __k = ::symjit::kernel::KernelBuilder::new(config);
                    let __p: (#(#param_types,)*) = __k.params();
                    let __out = __stage(&__k, #(__p.#indices),*);
                    __k.compile(__out)
                }

                /// Compiles the kernel for a caller compiled with `config`, and adds it to
                /// `df`.
                pub fn __register(
                    config: &::symjit::Config,
                    df: &mut ::symjit::Defuns,
                ) -> ::symjit::kernel::Result<()> {
                    ::symjit::kernel::ops::add_kernel(config, df, module_path!(), #complex, __build)
                }

                /// Calls the kernel from the kernel being built.
                pub fn __call<#lt>(
                    __k: &#lt ::symjit::kernel::KernelBuilder,
                    #(#arg_decls),*
                ) #ret_tokens {
                    __k.call(module_path!(), (#(#lifted,)*))
                }

                fn __stage<#lt>(
                    __k: &#lt ::symjit::kernel::KernelBuilder,
                    #(#decls),*
                ) #ret_tokens {
                    #body
                }
            }
        }
    };

    Ok(quote! {
        #func
        #module
    })
}

/// `#[kernel(callback)]`: the function stays as it is, and its module registers it (through
/// an `extern "C"` shim) and calls it from kernels.
fn expand_callback(func: &ItemFn) -> syn::Result<TokenStream2> {
    let sig = &func.sig;
    let (name, vis) = (&sig.ident, &func.vis);
    let lt = Lifetime::new("'k", Span::call_site());
    let is_f64 =
        |ty: &Type| matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident("f64"));

    let n = sig.inputs.len();
    let params_ok = (n == 1 || n == 2)
        && sig
            .inputs
            .iter()
            .all(|a| matches!(a, FnArg::Typed(pt) if is_f64(&pt.ty)));
    let ret_ok = matches!(&sig.output, ReturnType::Type(_, ty) if is_f64(ty));
    if !params_ok || !ret_ok || !sig.generics.params.is_empty() || sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            sig,
            "a callback takes one or two `f64` and returns `f64`",
        ));
    }

    let args: Vec<Ident> = (0..n).map(|i| format_ident!("__a{}", i)).collect();
    let add = if n == 1 {
        quote!(add_callback)
    } else {
        quote!(add_callback2)
    };
    let doc_fn =
        format!("[`{name}`] as a callback of symjit kernels (`#[symjit::kernel(callback)]`).");

    Ok(quote! {
        #func

        #[doc = #doc_fn]
        #[allow(non_snake_case, dead_code, clippy::all)]
        #vis mod #name {
            /// The callback with the C ABI of `Defuns`.
            extern "C" fn __shim(#(#args: f64),*) -> f64 {
                super::#name(#(#args),*)
            }

            /// Adds the callback to `df`.
            pub fn __register(
                _config: &::symjit::Config,
                df: &mut ::symjit::Defuns,
            ) -> ::symjit::kernel::Result<()> {
                ::symjit::kernel::ops::#add(df, module_path!(), __shim);
                Ok(())
            }

            /// Calls the callback from the kernel being built.
            pub fn __call<#lt>(
                __k: &#lt ::symjit::kernel::KernelBuilder,
                #(#args: impl ::symjit::kernel::ops::Lift<#lt, ::symjit::kernel::Real<#lt>>),*
            ) -> ::symjit::kernel::Real<#lt> {
                __k.callback(module_path!(), (#(::symjit::kernel::ops::lift(__k, #args),)*))
            }
        }
    })
}

/// The names bound in the function (parameters, `let`, closure parameters, `for` patterns,
/// nested functions): calls of these are left alone.
#[derive(Default)]
struct Locals {
    names: HashSet<String>,
}

impl<'ast> Visit<'ast> for Locals {
    fn visit_pat_ident(&mut self, p: &'ast syn::PatIdent) {
        self.names.insert(p.ident.to_string());
        visit::visit_pat_ident(self, p);
    }

    fn visit_item_fn(&mut self, f: &'ast ItemFn) {
        self.names.insert(f.sig.ident.to_string());
    }
}

/// The variables a branch or loop body assigns (`x = ..`, `x += ..`, `x[i] = ..`,
/// `(x, y) = ..`) without declaring them. Closures are skipped: those made for nested
/// branches assign their own copies, and pass the new values out.
#[derive(Default)]
struct Assigned {
    names: Vec<Ident>,
    declared: HashSet<String>,
}

impl Assigned {
    fn outer(self) -> Vec<Ident> {
        let declared = self.declared;
        self.names
            .into_iter()
            .filter(|id| !declared.contains(&id.to_string()))
            .collect()
    }

    fn roots(&mut self, e: &Expr) {
        match e {
            Expr::Path(p) => {
                if let Some(id) = p.path.get_ident() {
                    if !self.names.iter().any(|n| n == id) {
                        self.names.push(id.clone());
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
            Expr::Binary(b)
                if matches!(
                    b.op,
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
                ) =>
            {
                self.roots(&b.left)
            }
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

fn is_float_literal(e: &Expr) -> bool {
    match e {
        Expr::Lit(l) => matches!(l.lit, syn::Lit::Float(_)),
        Expr::Unary(u) => matches!(u.op, syn::UnOp::Neg(_)) && is_float_literal(&u.expr),
        Expr::Paren(p) => is_float_literal(&p.expr),
        _ => false,
    }
}

/// `Complex::new(re, im)` (or `Complex64::new`) of float literals.
fn is_complex_literal(e: &Expr) -> bool {
    let Expr::Call(call) = e else {
        return false;
    };
    let Expr::Path(p) = &*call.func else {
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
        && call.args.len() == 2
        && call.args.iter().all(is_float_literal)
}

/// Whether the body has a `while` loop: kernel loops need the direct translator.
#[derive(Default)]
struct HasWhile(bool);

impl<'ast> Visit<'ast> for HasWhile {
    fn visit_expr_while(&mut self, w: &'ast syn::ExprWhile) {
        self.0 = true;
        visit::visit_expr_while(self, w);
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

/// Finds `break` and `continue` that would leave a kernel branch (they would leave the
/// closure the branch becomes).
#[derive(Default)]
struct Escapes {
    loops: usize,
    errors: Vec<syn::Error>,
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
                self.errors.push(syn::Error::new_spanned(
                    e,
                    "a kernel `if` cannot `break` or `continue` an enclosing loop",
                ));
            }
            _ => visit::visit_expr(self, e),
        }
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

/// Finds closures in a kernel branch or loop that assign variables declared outside them:
/// the assignments would not be passed through the branch or loop (a kernel branch builds
/// both of its sides, so they would happen whatever the condition). Runs on the code as
/// written, before the closures of nested branches are made.
#[derive(Default)]
struct ClosureAssigns {
    errors: Vec<syn::Error>,
}

impl<'ast> Visit<'ast> for ClosureAssigns {
    fn visit_expr_closure(&mut self, c: &'ast syn::ExprClosure) {
        let mut assigned = Assigned::default();
        for input in c.inputs.iter() {
            assigned.visit_pat(input);
        }
        assigned.visit_expr(&c.body);
        let vars = assigned.outer();
        if !vars.is_empty() {
            let names: Vec<String> = vars.iter().map(|v| format!("`{v}`")).collect();
            self.errors.push(syn::Error::new_spanned(
                c,
                format!(
                    "a closure in a kernel `if` or `while` cannot assign {}, declared outside \
                     it; return the new value from the closure instead",
                    names.join(", ")
                ),
            ));
        }
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

struct Rewriter {
    name: Ident,
    inline: bool,
    locals: HashSet<String>,
    params: Vec<Type>,
    ret: Option<Type>,
    lt: Lifetime,
    closures: usize,
    /// the functions called by a single identifier (kernels, inline helpers, callbacks)
    callees: Vec<Ident>,
    errors: Vec<syn::Error>,
}

impl Rewriter {
    fn stage(&self, ty: &Type) -> (Type, Staged) {
        let mut staged = Staged::default();
        let ty = stage_type(ty, &self.lt, &mut staged);
        (ty, staged)
    }

    fn rewrite(&mut self, e: &mut Expr) {
        // the rewritten expression keeps the span of the original, so that type errors in
        // it (e.g. a kernel `if` without `else`) point at the user's code
        let span = e.span();
        let new: Option<Expr> = match e {
            Expr::Binary(b) => {
                let f = match b.op {
                    BinOp::Lt(_) => "lt",
                    BinOp::Le(_) => "le",
                    BinOp::Gt(_) => "gt",
                    BinOp::Ge(_) => "ge",
                    BinOp::Eq(_) => "eq",
                    BinOp::Ne(_) => "ne",
                    BinOp::And(_) => "and",
                    BinOp::Or(_) => "or",
                    _ => return,
                };
                let f = Ident::new(f, b.op.span());
                let (l, r) = (&b.left, &b.right);
                Some(parse_quote_spanned!(span=> ::symjit::kernel::ops::#f(#l, #r)))
            }
            Expr::If(eif) => {
                if matches!(*eif.cond, Expr::Let(_)) {
                    return;
                }
                let (cond, then) = (&eif.cond, &eif.then_branch);

                // the variables declared outside that the branches assign: each branch
                // returns their new values beside its own value, and they are assigned after
                // the branch (a kernel branch runs both closures while the kernel is built)
                let mut assigned = Assigned::default();
                assigned.visit_block(then);
                if let Some((_, otherwise)) = &eif.else_branch {
                    assigned.visit_expr(otherwise);
                }
                let vars = assigned.outer();

                if vars.is_empty() {
                    match &eif.else_branch {
                        Some((_, otherwise)) => Some(parse_quote_spanned!(span=>
                            ::symjit::kernel::ops::branch(__k, #cond, || #then, || #otherwise)
                        )),
                        None => Some(parse_quote_spanned!(span=>
                            ::symjit::kernel::ops::when(#cond, || #then)
                        )),
                    }
                } else {
                    let carry = quote!((#(#vars,)*));
                    let copies = quote!(#(let mut #vars = #vars;)*);
                    let otherwise = match &eif.else_branch {
                        Some((_, otherwise)) => quote!(#otherwise),
                        None => quote!(()),
                    };
                    Some(parse_quote_spanned!(span=> {
                        let (__v, __c) = ::symjit::kernel::ops::branch(
                            __k,
                            #cond,
                            || { #copies let __v = #then; (__v, #carry) },
                            || { #copies let __v = #otherwise; (__v, #carry) },
                        );
                        #carry = __c;
                        __v
                    }))
                }
            }
            Expr::While(w) => {
                if w.label.is_some() || matches!(*w.cond, Expr::Let(_)) {
                    return;
                }
                // `break`/`continue` cannot leave the closures: such a loop stays as it is
                // (a Rust loop; a kernel condition is then a type error)
                let mut escapes = Escapes::default();
                escapes.visit_block(&w.body);
                if !escapes.errors.is_empty() {
                    return;
                }
                let mut assigned = Assigned::default();
                assigned.visit_block(&w.body);
                let vars = assigned.outer();
                if vars.is_empty() {
                    return;
                }

                let (cond, body) = (&w.cond, &w.body);
                let carry = quote!((#(#vars,)*));
                let pat = quote!((#(mut #vars,)*));
                Some(parse_quote_spanned!(span=> {
                    #carry = ::symjit::kernel::ops::while_loop(
                        __k,
                        #carry,
                        |#pat| #cond,
                        |#pat| { #body; #carry },
                    );
                }))
            }
            Expr::Assign(a) if !matches!(&*a.left, Expr::Tuple(_) | Expr::Infer(_)) => {
                // `x = 1.0` for a kernel value `x`
                let (left, right) = (&a.left, &a.right);
                Some(parse_quote_spanned!(span=>
                    #left = ::symjit::kernel::ops::coerce(__k, &#left, #right)
                ))
            }
            Expr::Array(a)
                if a.elems.iter().any(is_float_literal)
                    && !a.elems.iter().all(is_float_literal) =>
            {
                // `[1.0, x]`: the literals take the type of the other elements
                let elems = a.elems.iter().map(|e| {
                    if is_float_literal(e) {
                        quote!(::symjit::kernel::ops::lift(__k, #e))
                    } else {
                        quote!(#e)
                    }
                });
                Some(parse_quote_spanned!(span=> [#(#elems),*]))
            }
            Expr::Field(f) => match &f.member {
                Member::Named(id) if id == "re" || id == "im" => {
                    let base = &f.base;
                    Some(parse_quote_spanned!(span=> ::symjit::kernel::ops::#id(#base)))
                }
                _ => None,
            },
            Expr::Call(call) => {
                let Expr::Path(p) = &*call.func else {
                    return;
                };
                if p.qself.is_some() {
                    return;
                }
                let segs: Vec<String> = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                let args = &call.args;

                if let Some(id) = p.path.get_ident() {
                    let s = id.to_string();
                    if *id == self.name {
                        self.recursion(call)
                    } else if self.locals.contains(&s) || s.starts_with(|c: char| c.is_uppercase())
                    {
                        None
                    } else {
                        if !self.callees.iter().any(|c| c == id) {
                            self.callees.push(id.clone());
                        }
                        Some(parse_quote_spanned!(span=> #id::__call(__k, #args)))
                    }
                } else if segs.len() >= 2
                    && segs[segs.len() - 1] == "new"
                    && (segs[segs.len() - 2] == "Complex" || segs[segs.len() - 2] == "Complex64")
                {
                    if args.len() != 2 {
                        self.errors.push(syn::Error::new_spanned(
                            &call.args,
                            "`Complex::new` takes two arguments",
                        ));
                        return;
                    }
                    let (re, im) = (&args[0], &args[1]);
                    Some(parse_quote_spanned!(span=> ::symjit::kernel::ops::complex(__k, #re, #im)))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(new) = new {
            *e = new;
        }
    }

    /// A call of the kernel itself: `@self`, with the arguments converted to the types of
    /// the parameters.
    fn recursion(&mut self, call: &syn::ExprCall) -> Option<Expr> {
        if self.inline {
            self.errors.push(syn::Error::new_spanned(
                call,
                "an inline helper cannot call itself (only a kernel can recurse)",
            ));
            return None;
        }
        if call.args.len() != self.params.len() {
            self.errors.push(syn::Error::new_spanned(
                call,
                format!("the kernel takes {} arguments", self.params.len()),
            ));
            return None;
        }
        let args = call
            .args
            .iter()
            .zip(self.params.iter())
            .map(|(a, ty)| quote!(::symjit::kernel::ops::lift::<#ty, _>(__k, #a)));
        let ret = self.ret.as_ref().unwrap();
        Some(parse_quote_spanned!(call.span()=> __k.recurse::<_, #ret>((#(#args,)*))))
    }
}

impl VisitMut for Rewriter {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        match e {
            Expr::Return(r) if self.closures == 0 => {
                self.errors.push(syn::Error::new_spanned(
                    &*r,
                    "a kernel cannot `return`; its value is the last expression of the body",
                ));
                return;
            }
            Expr::If(eif) if !matches!(*eif.cond, Expr::Let(_)) => {
                let mut escapes = Escapes::default();
                let mut closures = ClosureAssigns::default();
                escapes.visit_block(&eif.then_branch);
                closures.visit_block(&eif.then_branch);
                if let Some((_, otherwise)) = &eif.else_branch {
                    escapes.visit_expr(otherwise);
                    closures.visit_expr(otherwise);
                }
                self.errors.extend(escapes.errors);
                self.errors.extend(closures.errors);
            }
            Expr::While(w) => {
                let mut closures = ClosureAssigns::default();
                closures.visit_block(&w.body);
                self.errors.extend(closures.errors);
            }
            _ => {}
        }

        visit_mut::visit_expr_mut(self, e);
        self.rewrite(e);
    }

    fn visit_local_mut(&mut self, local: &mut syn::Local) {
        // `let mut s = 0.0` and `let mut z = Complex::new(0.0, 0.0)`: accumulators, made
        // kernel values so that they can take kernel values (`s += x`)
        let accumulator = match (&local.pat, &local.init) {
            (Pat::Ident(p), Some(init)) if p.mutability.is_some() => {
                if is_float_literal(&init.expr) {
                    Some(quote!(Real))
                } else if is_complex_literal(&init.expr) {
                    Some(quote!(Cplx))
                } else {
                    None
                }
            }
            _ => None,
        };

        visit_mut::visit_local_mut(self, local);

        if let (Some(handle), Some(init)) = (accumulator, &mut local.init) {
            let e = &init.expr;
            let lt = &self.lt;
            *init.expr = parse_quote!(
                ::symjit::kernel::ops::lift::<::symjit::kernel::#handle<#lt>, _>(__k, #e)
            );
        }

        if let Pat::Type(pt) = &mut local.pat {
            let (ty, staged) = self.stage(&pt.ty);
            if staged.handle {
                *pt.ty = ty.clone();
                if let Some(init) = &mut local.init {
                    let e = &init.expr;
                    *init.expr = parse_quote!(::symjit::kernel::ops::lift::<#ty, _>(__k, #e));
                }
            }
        }
    }

    fn visit_expr_closure_mut(&mut self, c: &mut syn::ExprClosure) {
        for input in c.inputs.iter_mut() {
            if let Pat::Type(pt) = input {
                let (ty, staged) = self.stage(&pt.ty);
                if staged.handle {
                    *pt.ty = ty;
                }
            }
        }
        if let ReturnType::Type(_, ty) = &mut c.output {
            let (new, staged) = self.stage(ty);
            if staged.handle {
                **ty = new;
            }
        }

        self.closures += 1;
        visit_mut::visit_expr_closure_mut(self, c);
        self.closures -= 1;
    }

    fn visit_block_mut(&mut self, block: &mut Block) {
        visit_mut::visit_block_mut(self, block);

        // an `if` statement became a call, which needs a `;` unless it is the last one
        let n = block.stmts.len();
        for (i, stmt) in block.stmts.iter_mut().enumerate() {
            if let syn::Stmt::Expr(e @ Expr::Call(_), semi @ None) = stmt {
                if i + 1 < n {
                    *semi = Some(Token![;](e.span()));
                }
            }
        }
    }

    // nested items are ordinary Rust
    fn visit_item_mut(&mut self, _: &mut syn::Item) {}
}
