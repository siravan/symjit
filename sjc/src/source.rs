//! Rust source input: compiles the `#[kernel]` functions of a Rust file (the language of the
//! `composer` feature's macro, `docs/KERNELS.md`) without a Rust compiler. The file is parsed
//! with `syn` and each kernel's body is interpreted (`interp.rs`) against
//! `symjit::kernel::KernelBuilder`, doing what the code the macro generates does, so that the
//! kernel is the one the macro builds.
//!
//! Supported: everything the macro accepts in kernels, `#[kernel(inline)]` helpers, calls of
//! other `#[kernel]` kernels and of `#[kernel(callback)]` functions (which become symbols the
//! program linking the object defines), except `match`, macros, and Rust beyond numbers,
//! tuples, arrays, ranges and simple closures (see `docs/SJC.md`).

mod interp;
mod syntax;
mod value;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use symjit::kernel::{Cplx, KernelBuilder, Real, Value};
use symjit::{Application, Config, Defuns};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Expr, FnArg, Ident, ItemFn, Pat, ReturnType, Token, UseTree};

use interp::{at, float_const, Flow, Interp};
use syntax::{callees, stage_type, HasWhile, Ty};
use value::{lift, to_slots, Slots, Val};

/// What the `#[kernel]` attribute makes of a function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FnKind {
    /// `#[kernel]`: compiled as a kernel
    Kernel,
    /// `#[kernel(inline)]`: inlined into the kernels that call it
    Inline,
    /// `#[kernel(callback)]`: Rust called by kernels at run time
    Callback,
    /// no attribute
    Plain,
}

pub struct FnDef {
    pub name: String,
    pub kind: FnKind,
    /// `#[kernel(complex)]`
    pub complex: bool,
    pub item: ItemFn,
    pub params: Vec<(Pat, Ty)>,
    pub ret: Option<Ty>,
}

/// The names under which the kernels a kernel calls, and its callbacks, are added to the
/// `Defuns` of its configuration: the names of the calls in its code, and of the functions of
/// an object file (`symjit`'s `kernel_symbol` and `user_symbol`).
#[derive(Clone, Debug, Default)]
pub struct Naming {
    /// before the name of a called kernel
    pub kernel_prefix: String,
    /// before the name of a callback (its object symbol is the bare name)
    pub callback_prefix: String,
}

impl Naming {
    /// For the object file of the kernel `symbol`: the kernels it calls are
    /// `<symbol>__<callee>`, the callbacks their own names.
    pub fn object(symbol: &str) -> Naming {
        Naming {
            kernel_prefix: format!("{}__", symbol),
            callback_prefix: String::new(),
        }
    }

    /// As the macro names them: `module_path!()`, e.g. `crate::module::name`.
    pub fn module(path: &str) -> Naming {
        Naming {
            kernel_prefix: format!("{}::", path),
            callback_prefix: format!("{}::", path),
        }
    }

    pub fn kernel(&self, name: &str) -> String {
        format!("{}{}", self.kernel_prefix, name)
    }

    pub fn callback(&self, name: &str) -> String {
        format!("{}{}", self.callback_prefix, name)
    }
}

/// A callback's stand-in while the kernel is compiled: sjc does not run the kernels, and an
/// object file calls the program's function instead.
extern "C" fn callback1(_: f64) -> f64 {
    f64::NAN
}

extern "C" fn callback2(_: f64, _: f64) -> f64 {
    f64::NAN
}

/// The state of a compilation: the kernels being built (to find cycles of calls) and, for
/// an object file, the called kernels collected so far.
struct Build {
    naming: Naming,
    building: RefCell<Vec<String>>,
    /// the called kernels for an object file, by their names in the `Defuns`
    collected: Option<RefCell<Vec<(String, Application)>>>,
}

/// A parsed Rust file.
pub struct Source {
    pub path: String,
    pub fns: HashMap<String, FnDef>,
    /// the functions in the order of the file
    order: Vec<String>,
    /// the constants of the file (`const N: usize = 4;`)
    pub consts: HashMap<String, Expr>,
    /// the names imported from `std::f64::consts` (`use std::f64::consts::PI;`)
    pub imported: HashMap<String, f64>,
}

impl Source {
    pub fn parse(path: &str, text: &str) -> Result<Source> {
        let file = syn::parse_file(text).map_err(|e| {
            let p = e.span().start();
            anyhow!("{}:{}:{}: {}", path, p.line, p.column + 1, e)
        })?;
        let mut src = Source {
            path: path.to_string(),
            fns: HashMap::new(),
            order: Vec::new(),
            consts: HashMap::new(),
            imported: HashMap::new(),
        };

        for item in file.items.iter() {
            match item {
                syn::Item::Const(c) => {
                    src.consts.insert(c.ident.to_string(), (*c.expr).clone());
                }
                syn::Item::Use(u) => src.imports(&u.tree, &mut Vec::new()),
                _ => {}
            }
        }
        for item in file.items.into_iter() {
            if let syn::Item::Fn(f) = item {
                let def = src.fn_def(f)?;
                src.order.push(def.name.clone());
                src.fns.insert(def.name.clone(), def);
            }
        }
        Ok(src)
    }

    /// Records the names a `use` imports from `std::f64::consts`.
    fn imports(&mut self, tree: &UseTree, prefix: &mut Vec<String>) {
        let consts = |p: &[String]| p.last().map(|s| s.as_str()) == Some("consts");
        match tree {
            UseTree::Path(p) => {
                prefix.push(p.ident.to_string());
                self.imports(&p.tree, prefix);
                prefix.pop();
            }
            UseTree::Name(n) if consts(prefix) => {
                let name = n.ident.to_string();
                if let Some(x) = float_const(&name) {
                    self.imported.insert(name, x);
                }
            }
            UseTree::Rename(r) if consts(prefix) => {
                if let Some(x) = float_const(&r.ident.to_string()) {
                    self.imported.insert(r.rename.to_string(), x);
                }
            }
            UseTree::Glob(_) if consts(prefix) => {
                for name in [
                    "PI",
                    "TAU",
                    "E",
                    "SQRT_2",
                    "FRAC_1_SQRT_2",
                    "LN_2",
                    "LN_10",
                    "LOG2_E",
                    "LOG10_E",
                    "LOG2_10",
                    "LOG10_2",
                    "FRAC_PI_2",
                    "FRAC_PI_3",
                    "FRAC_PI_4",
                    "FRAC_PI_6",
                    "FRAC_PI_8",
                    "FRAC_1_PI",
                    "FRAC_2_PI",
                    "FRAC_2_SQRT_PI",
                ] {
                    self.imported
                        .insert(name.to_string(), float_const(name).unwrap());
                }
            }
            UseTree::Group(g) => {
                for t in g.items.iter() {
                    self.imports(t, prefix);
                }
            }
            _ => {}
        }
    }

    fn err(&self, span: proc_macro2::Span, msg: impl std::fmt::Display) -> anyhow::Error {
        anyhow!("{}:{}", self.path, at(span, msg))
    }

    fn fn_def(&self, item: ItemFn) -> Result<FnDef> {
        let name = item.sig.ident.to_string();
        let mut kind = FnKind::Plain;
        let mut complex = false;
        for attr in item.attrs.iter() {
            let last = attr.path().segments.last().map(|s| s.ident.to_string());
            if last.as_deref() != Some("kernel") {
                continue;
            }
            kind = FnKind::Kernel;
            if let syn::Meta::List(_) = &attr.meta {
                let idents = attr
                    .parse_args_with(Punctuated::<Ident, Token![,]>::parse_terminated)
                    .map_err(|e| self.err(e.span(), e))?;
                for id in idents {
                    match id.to_string().as_str() {
                        "inline" => kind = FnKind::Inline,
                        "callback" => kind = FnKind::Callback,
                        "complex" => complex = true,
                        _ => {
                            return Err(
                                self.err(id.span(), "expected `inline`, `complex` or `callback`")
                            )
                        }
                    }
                }
            }
        }

        let mut params = Vec::new();
        for arg in item.sig.inputs.iter() {
            match arg {
                FnArg::Typed(pt) => params.push((
                    (*pt.pat).clone(),
                    stage_type(&pt.ty, &|e| self.const_usize(e)),
                )),
                FnArg::Receiver(r) if kind != FnKind::Plain => {
                    return Err(self.err(r.span(), "a kernel cannot be a method"))
                }
                FnArg::Receiver(_) => {}
            }
        }
        let ret = match &item.sig.output {
            ReturnType::Type(_, ty) => Some(stage_type(ty, &|e| self.const_usize(e))),
            ReturnType::Default => None,
        };
        Ok(FnDef {
            name,
            kind,
            complex,
            item,
            params,
            ret,
        })
    }

    /// The value of an array length: an integer literal or a constant of the file defined by
    /// one.
    pub fn const_usize(&self, e: &Expr) -> Option<usize> {
        match e {
            Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(i),
                ..
            }) => i.base10_parse().ok(),
            Expr::Path(p) => {
                let id = p.path.get_ident()?.to_string();
                self.consts.get(&id).and_then(|e| self.const_usize(e))
            }
            Expr::Paren(p) => self.const_usize(&p.expr),
            Expr::Group(g) => self.const_usize(&g.expr),
            _ => None,
        }
    }

    /// The `#[kernel]` kernels of the file (not helpers or callbacks), in order.
    pub fn kernels(&self) -> Vec<&str> {
        self.order
            .iter()
            .filter(|n| self.fns[*n].kind == FnKind::Kernel)
            .map(|n| n.as_str())
            .collect()
    }

    /// Compiles the kernel `name` with `config` (made complex if the kernel is, and set to the
    /// direct translator if it has a `while`), as `name::compile(config)` of the macro does.
    /// The kernels it calls and its callbacks are named by `Naming::default()`.
    pub fn compile(&self, name: &str, config: Config) -> Result<Application> {
        self.compile_with(name, config, &Naming::default())
    }

    /// `compile`, with the called kernels and callbacks named by `naming`.
    pub fn compile_with(&self, name: &str, config: Config, naming: &Naming) -> Result<Application> {
        let build = Build {
            naming: naming.clone(),
            building: RefCell::new(Vec::new()),
            collected: None,
        };
        self.build(name, config, None, &build)
    }

    /// `compile_with`, and the kernels the kernel calls (also through other kernels), as
    /// applications under their names in the `Defuns`, for
    /// `Application::object_files_with`.
    pub fn compile_for_object(
        &self,
        name: &str,
        config: Config,
        naming: &Naming,
    ) -> Result<(Application, Vec<(String, Application)>)> {
        let build = Build {
            naming: naming.clone(),
            building: RefCell::new(Vec::new()),
            collected: Some(RefCell::new(Vec::new())),
        };
        let app = self.build(name, config, None, &build)?;
        Ok((app, build.collected.unwrap().into_inner()))
    }

    /// Builds the kernel `name` (`__build` of the macro): complex as its signature says, or
    /// as `complex` says for a called kernel (that of its caller).
    fn build(
        &self,
        name: &str,
        mut config: Config,
        complex: Option<bool>,
        build: &Build,
    ) -> Result<Application> {
        let Some(def) = self.fns.get(name).filter(|d| d.kind == FnKind::Kernel) else {
            let kernels = self.kernels();
            return Err(anyhow!(
                "{}: no `#[kernel]` named `{}`{}",
                self.path,
                name,
                if kernels.is_empty() {
                    String::new()
                } else {
                    format!(" (the kernels are: {})", kernels.join(", "))
                }
            ));
        };
        self.check_signature(def)?;

        // kernels that call each other in a cycle (`ops::building`)
        {
            let mut building = build.building.borrow_mut();
            if let Some(i) = building.iter().position(|n| n == name) {
                let mut cycle: Vec<&str> = building[i..].iter().map(|n| n.as_str()).collect();
                cycle.push(name);
                return Err(anyhow!(
                    "{}: the kernels call each other in a cycle ({}); a kernel can only call itself",
                    self.path,
                    cycle.join(" -> ")
                ));
            }
            building.push(name.to_string());
        }
        let result = self.build_kernel(def, &mut config, complex, build);
        build.building.borrow_mut().pop();
        result
    }

    fn build_kernel(
        &self,
        def: &FnDef,
        config: &mut Config,
        complex: Option<bool>,
        build: &Build,
    ) -> Result<Application> {
        let name = def.name.as_str();
        let ret = def.ret.clone().unwrap();
        let complex = complex.unwrap_or_else(|| {
            def.complex || ret.has_complex() || def.params.iter().any(|(_, t)| t.has_complex())
        });
        let mut has_while = HasWhile::default();
        has_while.visit_block(&def.item.block);
        config.set_complex(complex);
        if has_while.0 {
            // kernel loops need the direct translator
            config.set_direct(true);
        }

        // the kernels and callbacks it calls (`__register`)
        let mut df = Defuns::new();
        self.register(def, config, &mut df, build, &mut HashSet::new())?;
        symjit::kernel::add_defuns(config, df);

        let k = KernelBuilder::new(config.clone());
        let interp = Interp::new(&k, self, def, &build.naming);
        let result = (|| -> Result<Vec<symjit::Slot>, Flow> {
            for (pat, ty) in def.params.iter() {
                let v = params(&k, ty);
                interp.bind(pat, v)?;
            }
            let v = interp.eval_block(&def.item.block)?;
            let span = def.item.sig.output.span();
            let out = lift(&k, v, &ret)
                .map_err(|msg| Flow::Error(at(span, format!("the kernel's value: {}", msg))))?;
            let mut slots = Vec::new();
            to_slots(&k, &out, false, &mut slots).map_err(|msg| Flow::Error(at(span, msg)))?;
            Ok(slots)
        })();
        let slots = match result {
            Ok(slots) => slots,
            Err(Flow::Error(msg)) => return Err(anyhow!("{}:{}", self.path, msg)),
            Err(_) => {
                return Err(anyhow!(
                    "{}: `break` or `continue` outside a loop in `{}`",
                    self.path,
                    name
                ))
            }
        };
        k.compile(Slots::new(slots))
            .map_err(|e| anyhow!("{}: the kernel `{}`: {}", self.path, name, e))
    }

    /// Adds the kernels and callbacks `def` calls, also through its inline helpers, to `df`
    /// (the `__register` functions of the macro), compiled for a caller with `config`.
    fn register(
        &self,
        def: &FnDef,
        config: &Config,
        df: &mut Defuns,
        build: &Build,
        inlined: &mut HashSet<String>,
    ) -> Result<()> {
        for callee in callees(&def.item) {
            let Some(c) = self.fns.get(&callee) else {
                continue;
            };
            match c.kind {
                FnKind::Kernel => {
                    let applet = build.naming.kernel(&callee);
                    if df.funcs.contains_key(&applet) {
                        continue;
                    }
                    if config.is_wasm() {
                        return Err(anyhow!(
                            "{}: a WebAssembly module cannot call other kernels yet (`{}` calls \
                             `{}`); make `{}` `#[kernel(inline)]`, or compile an object file",
                            self.path,
                            def.name,
                            callee,
                            callee
                        ));
                    }
                    self.check_signature(c)?;
                    let ret = c.ret.as_ref().unwrap();
                    let c_complex = c.complex
                        || ret.has_complex()
                        || c.params.iter().any(|(_, t)| t.has_complex());
                    if c_complex && !config.is_complex() {
                        return Err(anyhow!(
                            "{}: the complex kernel `{}` is called from a real kernel; make the \
                             caller complex (`#[kernel(complex)]`)",
                            self.path,
                            callee
                        ));
                    }
                    let app =
                        self.build(&callee, config.clone(), Some(config.is_complex()), build)?;
                    if let Some(collected) = &build.collected {
                        if !collected.borrow().iter().any(|(n, _)| *n == applet) {
                            // a second build for the object file (the first one becomes an
                            // applet of the caller's `Defuns`)
                            let again = self.build(
                                &callee,
                                config.clone(),
                                Some(config.is_complex()),
                                build,
                            )?;
                            collected.borrow_mut().push((applet.clone(), again));
                        }
                    }
                    df.add_applet(&applet, app.seal()?);
                }
                FnKind::Callback => {
                    let sig = &c.item.sig;
                    let is_f64 = |ty: &syn::Type| matches!(ty, syn::Type::Path(p) if p.qself.is_none() && p.path.is_ident("f64"));
                    let n = sig.inputs.len();
                    let ok = (n == 1 || n == 2)
                        && sig
                            .inputs
                            .iter()
                            .all(|a| matches!(a, FnArg::Typed(pt) if is_f64(&pt.ty)))
                        && matches!(&sig.output, ReturnType::Type(_, ty) if is_f64(ty))
                        && sig.generics.params.is_empty();
                    if !ok {
                        return Err(self.err(
                            sig.ident.span(),
                            "a callback takes one or two `f64` and returns `f64`",
                        ));
                    }
                    let name = build.naming.callback(&callee);
                    if n == 1 {
                        symjit::kernel::add_callback(df, &name, callback1);
                    } else {
                        symjit::kernel::add_callback2(df, &name, callback2);
                    }
                }
                FnKind::Inline => {
                    if inlined.insert(callee.clone()) {
                        self.register(c, config, df, build, inlined)?;
                    }
                }
                FnKind::Plain => {}
            }
        }
        Ok(())
    }

    /// The checks of the macro on a kernel's signature.
    fn check_signature(&self, def: &FnDef) -> Result<()> {
        let sig = &def.item.sig;
        if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
            return Err(self.err(sig.generics.span(), "a kernel cannot be generic"));
        }
        if sig.asyncness.is_some() {
            return Err(self.err(sig.span(), "a kernel cannot be async"));
        }
        if sig.variadic.is_some() {
            return Err(self.err(sig.span(), "a kernel cannot be variadic"));
        }
        for ((_, ty), arg) in def.params.iter().zip(sig.inputs.iter()) {
            if !ty.has_handle() || ty.has_other() || ty.has_mask() {
                return Err(self.err(
                    arg.span(),
                    "the parameters of a kernel must be `f64` or `Complex<f64>` (or tuples or \
                     arrays of them); use `#[kernel(inline)]` for a helper with other parameters",
                ));
            }
        }
        match &def.ret {
            Some(ty) if ty.has_handle() && !ty.has_other() => Ok(()),
            _ => Err(self.err(
                sig.ident.span(),
                "a kernel must return `f64`, `Complex<f64>` or `bool` (or tuples or arrays of them)",
            )),
        }
    }
}

/// The parameters of type `ty`, declared in order (`Value::from_params`).
fn params<'k>(k: &'k KernelBuilder, ty: &Ty) -> Val<'k> {
    match ty {
        Ty::Real => Val::Real(Real::from_params(k)),
        Ty::Cplx => Val::Cplx(Cplx::from_params(k)),
        Ty::Tuple(ts) => Val::Tuple(ts.iter().map(|t| params(k, t)).collect()),
        Ty::Array(t, n) => Val::Array((0..*n).map(|_| params(k, t)).collect()),
        _ => Val::Unit,
    }
}
