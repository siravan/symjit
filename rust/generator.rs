use anyhow::{anyhow, Result};

use super::code::{Func, VirtualTable};
use super::config::Config;
use super::symbol::Loc;
use super::utils::Reg;

// Object mode (`Config::is_relocatable`): math functions are called directly in the C
// library through relocations, and there is no table of function addresses. User-defined
// functions of one or two arguments (`Defuns::add_unary`/`add_binary`, kernel callbacks) are
// undefined symbols the program linking the object provides, and kernels called by name
// (applets, `Defuns::add_applet`) are called directly too (`object_applet`).

/// The symbol a call to `op` goes to in object mode, or the error to report.
pub fn object_symbol(config: &Config, op: &str) -> Result<String> {
    if config.is_user_func(op) {
        // a user function named like a C math function would link to the C one
        let unary_or_binary = matches!(
            config.df.as_ref().and_then(|df| df.funcs.get(op)),
            Some(Func::Unary(_)) | Some(Func::Binary(_))
        );
        if !unary_or_binary || C_MATH.contains(&user_symbol(op)) {
            return Err(object_external_error(op));
        }
        return Ok(user_symbol(op).to_string());
    }
    VirtualTable::c_name(op)
        .map(|s| s.to_string())
        .ok_or_else(|| {
            anyhow!(
                "`{}` is not a C math library function, so it cannot be called from an object file",
                op
            )
        })
}

/// The functions of C99's `<math.h>` (and the POSIX Bessel functions): a user function with one
/// of these names would link to the C library's.
const C_MATH: &[&str] = &[
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "cbrt",
    "ceil",
    "copysign",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "fdim",
    "floor",
    "fma",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "hypot",
    "ilogb",
    "ldexp",
    "lgamma",
    "llrint",
    "llround",
    "log",
    "log10",
    "log1p",
    "log2",
    "logb",
    "lrint",
    "lround",
    "modf",
    "nan",
    "nearbyint",
    "nextafter",
    "nexttoward",
    "pow",
    "remainder",
    "remquo",
    "rint",
    "round",
    "scalbln",
    "scalbn",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "tgamma",
    "trunc",
    "j0",
    "j1",
    "jn",
    "y0",
    "y1",
    "yn",
];

/// The symbol of the user-defined function `op` in an object file: its name, without the
/// `composer_` prefix of callbacks and the module path of `#[kernel(callback)]` functions
/// (`composer_crate::module::f` is `f`).
pub fn user_symbol(op: &str) -> &str {
    let op = op.strip_prefix("composer_").unwrap_or(op);
    op.rsplit("::").next().unwrap_or(op)
}

/// Whether a call of the kernel `op` (an applet of the configuration's `Defuns`) can be made
/// from an object file: it becomes a direct call of the symbol `kernel_symbol(op)`, which the
/// object (or another one) defines.
pub fn object_applet(config: &Config, op: &str) -> bool {
    matches!(
        config.df.as_ref().and_then(|df| df.funcs.get(op)),
        Some(Func::App(_))
    )
}

/// The symbol of the kernel `op` in an object file: its name with the characters a C
/// identifier cannot have (e.g. the `::` of a module path) replaced by `_`.
pub fn kernel_symbol(op: &str) -> String {
    op.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// User-defined (Python or Rust) functions live in the compiling process.
pub fn object_external_error(op: &str) -> anyhow::Error {
    anyhow!(
        "the user-defined function `{}` cannot be called from an object file",
        op
    )
}

/// The C99 complex function (`<complex.h>`, in the C math library) a call of the complex `op`
/// goes to in object mode. A `double complex` argument and result are in the first two
/// floating-point registers (x86-64 System V and AArch64), which are `Ret` and `Temp`, and the
/// second argument of `cpow` in the next two, `Gen(0)` and `Gen(1)` (x86-64: xmm2 and xmm3,
/// AArch64: d2 and d3); `carg` returns a `double` in `Ret`. The complex functions C99 does not
/// have (`csc`, ..., `log10`, `cbrt`, `sinc`) are composed of these by `complexify`.
pub fn complex_c_name(op: &str) -> Option<&'static str> {
    Some(match op {
        "sin" => "csin",
        "cos" => "ccos",
        "tan" => "ctan",
        "sinh" => "csinh",
        "cosh" => "ccosh",
        "tanh" => "ctanh",
        "arcsin" => "casin",
        "arccos" => "cacos",
        "arctan" => "catan",
        "arcsinh" => "casinh",
        "arccosh" => "cacosh",
        "arctanh" => "catanh",
        "exp" => "cexp",
        "ln" | "log" => "clog",
        "root" => "csqrt",
        "power" => "cpow",
        "arg" => "carg",
        _ => return None,
    })
}

/// The C99 symbol of the complex function `op`, or the error to report.
pub fn object_complex_symbol(op: &str) -> Result<&'static str> {
    complex_c_name(op).ok_or_else(|| object_complex_error(op))
}

pub fn object_complex_error(op: &str) -> anyhow::Error {
    anyhow!(
        "the complex function `{}` cannot be called from an object file",
        op
    )
}

#[derive(Clone, Debug)]
pub struct StackRegions {
    pub cap: u32,
    pub count_states: u32,
    pub count_obs: u32,
    pub count_params: u32,
}

impl StackRegions {
    pub fn new(
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    ) -> StackRegions {
        StackRegions {
            cap: cap as u32,
            count_states: count_states as u32,
            count_obs: count_obs as u32,
            count_params: count_params as u32,
        }
    }
}

pub enum GeneratorType {
    Unknown,
    Complexifier,
    AmdScalar(bool),
    AmdVectorF64x4(bool),
    AmdVectorF64x8(bool),
    AmdComplex,
    AmdSSE(bool),
    ArmScalar(bool),
    ArmVector(bool),
    ArmComplex,
    RiscvScalar(bool),
    #[cfg(feature = "wasm")]
    Wasm,
}

#[allow(dead_code)]
pub trait Generator {
    fn count_shadows(&self) -> u8;
    fn three_address(&self) -> bool;
    fn bytes(&mut self) -> Vec<u8>;
    /// (offset, symbol) of the calls to external functions in `bytes` (object mode)
    fn relocations(&self) -> Vec<(usize, String)> {
        Vec::new()
    }
    fn what(&self) -> GeneratorType;

    fn seal(&mut self);
    fn align(&mut self);
    fn set_label(&mut self, label: &str);
    fn branch(&mut self, label: &str);
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool);

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg);
    fn fxchg(&mut self, dst: Reg, s1: Reg);
    fn load_const(&mut self, dst: Reg, idx: u32);
    fn load_mem(&mut self, dst: Reg, idx: u32);
    fn save_mem(&mut self, dst: Reg, idx: u32);
    fn load_param(&mut self, dst: Reg, idx: u32);
    fn load_stack(&mut self, dst: Reg, idx: u32);
    fn save_stack(&mut self, dst: Reg, idx: u32);

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32);
    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32);
    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32);
    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32);
    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32);

    fn load_args(&mut self, locs: Vec<Loc>, ultra: bool);
    fn save_args(&mut self, num_args: u8, ultra: bool);
    fn load_args_complex(&mut self, locs: Vec<Loc>, ultra: bool);
    fn save_args_complex(&mut self, num_args: u8, ultra: bool);
    /// Called before the `save_args` of a funclet's plain entry (compression mode): `has_calls`
    /// is true if the funclet body calls a function, which may clobber every volatile register.
    fn begin_funclet(&mut self, _has_calls: bool) {}
    /// Whether the kernel runs the funclets of the inlined math library (`libm.rs`, feature
    /// `math`); the others call the C functions (`mir::LIBM_PREFIX`). Only SIMD kernels do: the
    /// library is faster than the C functions when it evaluates all lanes at once.
    fn inlines_math(&self) -> bool {
        false
    }

    fn save_mem_result(&mut self, idx: u32);
    fn save_stack_result(&mut self, idx: u32);

    fn neg(&mut self, dst: Reg, s1: Reg);
    fn sign(&mut self, dst: Reg, s1: Reg);
    fn abs(&mut self, dst: Reg, s1: Reg);
    fn abs2(&mut self, dst: Reg, s1: Reg);
    /// `i * s1` and `-i * s1`: complex kernels only. `Mir::times_i` emits `xor` in a real
    /// kernel, so the real generators leave these empty.
    fn times_i(&mut self, dst: Reg, s1: Reg);
    fn times_neg_i(&mut self, dst: Reg, s1: Reg);
    fn root(&mut self, dst: Reg, s1: Reg);
    fn real_root(&mut self, dst: Reg, s1: Reg);
    fn recip(&mut self, dst: Reg, s1: Reg);
    fn half(&mut self, dst: Reg, s1: Reg);

    fn round(&mut self, dst: Reg, s1: Reg);
    fn floor(&mut self, dst: Reg, s1: Reg);
    fn ceiling(&mut self, dst: Reg, s1: Reg);
    fn trunc(&mut self, dst: Reg, s1: Reg);
    fn frac(&mut self, dst: Reg, s1: Reg);

    // The exponent-field operations: `utils::pow2i`, `getexp` and `getmant` define their
    // results bit for bit. Complex generators apply them to the real part and return a real.
    fn pow2i(&mut self, dst: Reg, s1: Reg);
    fn getexp(&mut self, dst: Reg, s1: Reg);
    fn getmant(&mut self, dst: Reg, s1: Reg);

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg);

    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool;
    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool;

    fn fuse_load_math(&mut self);
    fn support_times2(&self) -> bool;
    fn times2_loc(&mut self, d1: Reg, s1: Reg, l1: Loc, d2: Reg, s2: Reg, l2: Loc);

    fn real(&mut self, dst: Reg, s1: Reg);
    fn imaginary(&mut self, dst: Reg, s1: Reg);
    fn conjugate(&mut self, dst: Reg, s1: Reg);
    fn complex(&mut self, dst: Reg, s1: Reg, s2: Reg);

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg);

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg);
    fn not(&mut self, dst: Reg, s1: Reg);

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg);

    fn add_consts(&mut self, consts: &[f64]);
    fn add_func(&mut self, f: &str, p: Func);
    fn call(&mut self, op: &str, num_args: usize) -> Result<()>;
    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()>;

    fn call_funclet(&mut self, label: &str);
    fn ret(&mut self);

    fn prologue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize);
    fn epilogue_fast(&mut self, cap: usize, count_states: usize, count_obs: usize, idx_ret: i32);

    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    );
    fn epilogue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        count_params: usize,
    );

    fn save_used_registers(&mut self, used: &[Reg]);
    fn load_used_registers(&mut self, used: &[Reg]);

    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32);
}
