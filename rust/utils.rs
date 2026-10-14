use anyhow::Result;
use std::fmt;
use std::io::{Read, Write};

use super::config::Config;
use super::machine::MachineCode;

pub type CompiledFunc<T> = fn(*const T, *const &mut [T], usize, *const T) -> i32;

/// Stable raw descriptor for one column-first P-kernel state plane.
///
/// Unlike `&mut [T]`, this representation can describe duplicate inputs and
/// intentional input/output aliases without creating overlapping Rust
/// references. Callers remain responsible for keeping the allocation alive
/// and for synchronizing every access while a kernel is executing.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PlaneDescriptor<T> {
    pub data: *mut T,
    pub len: usize,
}

impl<T> PlaneDescriptor<T> {
    /// Constructs a descriptor from raw allocation metadata.
    ///
    /// # Safety
    ///
    /// `data` must remain valid and suitably aligned for `len` consecutive
    /// values throughout every kernel call which receives this descriptor.
    pub const unsafe fn from_raw_parts(data: *mut T, len: usize) -> Self {
        Self { data, len }
    }
}

/// Raw-descriptor variant of a compiled kernel entry point.
///
/// The machine-code ABI is identical to [`CompiledFunc`], whose second
/// argument is already consumed as an array of `(data, len)` descriptors.
///
/// Calling the function is unsafe because the generated code dereferences
/// descriptors and parameters without validating their lifetimes, lengths,
/// alignment, alias synchronization, or mutability.
pub type CompiledPlaneFunc<T> =
    unsafe fn(*const T, *const PlaneDescriptor<T>, usize, *const T) -> i32;

pub trait Compiled<T: Sized + Copy + Default> {
    fn exec(&mut self, params: &[T]);
    fn evaluate(&mut self, args: &[T], outs: &mut [T]);
    fn evaluate_single(&mut self, args: &[T]) -> T;
    fn mem(&self) -> &[T];
    fn mem_mut(&mut self) -> &mut [T];
    fn dump(&self, name: &str);
    fn dumps(&self) -> Vec<u8>;
    fn func(&self) -> CompiledFunc<T>;
    fn support_indirect(&self) -> bool;
    fn count_lanes(&self) -> usize;
    fn as_machine(&self) -> Option<&MachineCode<T>>;
}

pub trait Storage: Sized {
    fn save(&self, stream: &mut impl Write) -> Result<()>;
    fn load_with_version(stream: &mut impl Read, config: &Config, version: usize) -> Result<Self>;
}

pub fn bool_to_f64(b: bool) -> f64 {
    const T: f64 = f64::from_bits(!0);
    const F: f64 = f64::from_bits(0);
    if b {
        T
    } else {
        F
    }
}

// The exponent-field operations (`UniOp::Pow2i`, `GetExp`, `GetMant`). They are defined by the
// bit operations below, which every backend reproduces exactly, so they give the same bits on
// every input, including the ones outside their useful range.

/// 1.5 * 2^52 + 1023: adding it rounds `x` to an integer (ties to even) and leaves
/// `x + 1023` in the low bits of the mantissa.
pub const POW2I_MAGIC: u64 = 0x4338_0000_0000_03ff;
/// 2^52
pub const TWO52: u64 = 0x4330_0000_0000_0000;
/// 2^52 + 1023
pub const TWO52_1023: u64 = 0x4330_0000_0000_03ff;
/// the sign and the mantissa of a double
pub const SIGN_MANTISSA: u64 = 0x800f_ffff_ffff_ffff;
/// the exponent field of a double
pub const EXPONENT: u64 = 0x7ff0_0000_0000_0000;

/// 2^x for an integer x in [-1022, 1023]; +0.0 for x = -1023 and +inf for x = 1024. In general,
/// x + 1023 is rounded to an integer (ties to even, so halves of x go to the odd neighbor) and
/// its low 12 bits become the sign and the exponent field of the result, which is +0.0 for
/// infinities and the default NaN.
pub fn pow2i(x: f64) -> f64 {
    f64::from_bits((x + f64::from_bits(POW2I_MAGIC)).to_bits() << 52)
}

/// The exponent field of x minus 1023: floor(log2(|x|)) for normal numbers, -1023 for zeros
/// and subnormals, 1024 for infinities and NaNs.
pub fn getexp(x: f64) -> f64 {
    ((x.to_bits() << 1) >> 53) as f64 - 1023.0
}

/// x with its exponent field set to that of 1.0: the mantissa of x in [1, 2), with the sign of
/// x (+-1.0 for zeros and infinities).
pub fn getmant(x: f64) -> f64 {
    f64::from_bits((x.to_bits() & SIGN_MANTISSA) | 1.0f64.to_bits())
}

/// aligns at a multiple of 32 (to cover different ABIs)
pub fn align_stack(n: u32) -> u32 {
    n + 16 - (n & 15)
}

/*****************************************/

#[derive(PartialEq)]
#[allow(unused)]
pub enum DataType {
    F32,
    F64,
}

#[derive(Copy, Clone, PartialEq, Hash, Eq)]
pub enum Reg {
    Ret,
    Temp,
    Gen(u8),
    Left,
    Right,
    Static(u32),
}

impl fmt::Debug for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::Ret => write!(f, "%$"),
            Reg::Left => write!(f, "%l"),
            Reg::Right => write!(f, "%r"),
            Reg::Temp => write!(f, "%t"),
            Reg::Gen(r) => write!(f, "%{}", r),
            Reg::Static(r) => write!(f, "σ{}", r),
        }
    }
}

pub fn reg(r: u8) -> Reg {
    Reg::Gen(r)
}

#[cfg(test)]
mod exponent_tests {
    use super::{getexp, getmant, pow2i};
    use crate::compiler::Translator;
    use crate::composer::Composer;
    use crate::config::Config;
    use crate::instruction::Slot;
    use num_complex::Complex;

    fn inputs() -> Vec<f64> {
        let mut xs = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            5e-324,
            -2.2e-308,
            f64::MIN_POSITIVE,
            f64::MAX,
            -f64::MAX,
            1e300,
            -1e300,
            0.5,
            1.5,
            2.5,
            -0.5,
            -1.5,
            3.0e15,
            -3.0e15,
            1.25e-310,
        ];
        xs.extend((-1100..=1100).map(|k| k as f64));
        xs.extend((0..400).map(|i| (i as f64 * 0.731).sin() * 10f64.powi(i % 600 - 300)));
        xs
    }

    #[test]
    fn exponent_reference_functions() {
        for k in -1022..=1023 {
            assert_eq!(pow2i(k as f64), 2f64.powi(k), "pow2i({k})");
        }
        assert_eq!(pow2i(-1023.0).to_bits(), 0);
        assert_eq!(pow2i(1024.0), f64::INFINITY);
        assert_eq!(pow2i(2.5), 8.0); // x + 1023 is rounded, ties to even
        assert_eq!(pow2i(1.5), 2.0);
        assert_eq!(pow2i(f64::INFINITY).to_bits(), 0);
        assert_eq!(pow2i(f64::NAN).to_bits(), 0);

        for x in inputs() {
            if x.is_normal() {
                let e = getexp(x);
                assert!(
                    e.fract() == 0.0 && (-1022.0..=1023.0).contains(&e),
                    "getexp({x:e})"
                );
                assert!(
                    pow2i(e) <= x.abs() && x.abs() / 2.0 < pow2i(e),
                    "getexp({x:e})"
                );
                let m = getmant(x);
                assert!(
                    (1.0..2.0).contains(&m.abs()) && m.is_sign_negative() == x.is_sign_negative()
                );
                assert_eq!(m * pow2i(getexp(x)), x, "getmant({x:e})");
            }
        }
        assert_eq!(getexp(0.0), -1023.0);
        assert_eq!(getexp(5e-324), -1023.0);
        assert_eq!(getexp(f64::NAN), 1024.0);
        assert_eq!(getmant(-0.0), -1.0);
        assert_eq!(getmant(f64::INFINITY), 1.0);
    }

    type Variant = (&'static str, fn(&mut Config));

    fn variants() -> Vec<Variant> {
        vec![
            ("default", |_| {}),
            ("opt0", |c| c.set_opt_level(0)),
            ("opt3", |c| c.set_opt_level(3)),
            ("no-fastmath", |c| c.set_fastmath(false)),
            ("no-simd", |c| c.set_simd(false)),
            ("simd512", |c| c.enable_simd512(true)),
            ("simd512-opt3", |c| {
                c.enable_simd512(true);
                c.set_opt_level(3)
            }),
        ]
    }

    // (the bytecode interpreter, `mir::interpreter`, calls the reference functions; it is not
    // reachable through a `Translator`)
    fn config(complex: bool, variant: fn(&mut Config)) -> Config {
        // the built-in options (with `symbolica`, the matrix layout of `evaluate_matrix`)
        let mut config = Config::from_name("native", Config::default().opt).unwrap();
        config.set_simd(true);
        config.set_fastmath(true);
        config.set_opt_level(2);
        config.set_complex(complex);
        variant(&mut config);
        config
    }

    // f(x) and f(x) + x * 0 (a register operand that is not the parameter)
    fn compile(op: &str, config: Config) -> crate::Application {
        let mut tr = Translator::new(config);
        tr.set_num_params(1);
        tr.append_fun(&Slot::Out(0), op, &[Slot::Param(0)], false)
            .unwrap();
        tr.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Param(0)], 0)
            .unwrap();
        tr.append_fun(&Slot::Temp(1), op, &[Slot::Temp(0)], false)
            .unwrap();
        tr.append_add(&Slot::Out(1), &[Slot::Temp(1), Slot::Temp(0)], 0)
            .unwrap();
        tr.compile().unwrap()
    }

    fn reference(op: &str) -> fn(f64) -> f64 {
        match op {
            "pow2i" => pow2i,
            "getexp" => getexp,
            _ => getmant,
        }
    }

    #[test]
    fn exponent_ops_real_kernels() {
        let xs = inputs();
        for op in ["pow2i", "getexp", "getmant"] {
            let f = reference(op);
            let want: Vec<[f64; 2]> = xs.iter().map(|&x| [f(x), f(x * x) + x * x]).collect();
            for (name, variant) in variants() {
                let app = compile(op, config(false, variant));
                let tag = format!("{op} {name}");
                // one point at a time (scalar kernel)
                for (x, w) in xs.iter().zip(&want) {
                    let mut outs = [0.0; 2];
                    app.evaluate(&[*x], &mut outs);
                    assert_eq!(
                        outs.map(f64::to_bits),
                        w.map(f64::to_bits),
                        "{tag} scalar x = {x:e}: {outs:?} != {w:?}"
                    );
                }
                // as a matrix (SIMD kernel, with a tail)
                let mut outs = vec![0.0; 2 * xs.len()];
                app.evaluate_matrix(&xs, &mut outs, xs.len());
                for (i, w) in want.iter().enumerate() {
                    let got = [outs[2 * i], outs[2 * i + 1]];
                    assert_eq!(
                        got.map(f64::to_bits),
                        w.map(f64::to_bits),
                        "{tag} matrix x = {:e}: {got:?} != {w:?}",
                        xs[i]
                    );
                }
            }
        }
    }

    // complex kernels apply the operations to the real part and return a real
    #[test]
    fn exponent_ops_complex_kernels() {
        let xs: Vec<Complex<f64>> = inputs()
            .iter()
            .enumerate()
            .map(|(i, &x)| Complex::new(x, (i as f64 - 1000.0) * 0.375))
            .collect();
        for op in ["pow2i", "getexp", "getmant"] {
            let f = reference(op);
            for fast_complex in [true, false] {
                for (name, variant) in variants() {
                    let mut config = config(true, variant);
                    config
                        .set_option("fast_complex", &fast_complex.to_string())
                        .unwrap();
                    let mut tr = Translator::new(config);
                    tr.set_num_params(1);
                    tr.append_fun(&Slot::Out(0), op, &[Slot::Param(0)], false)
                        .unwrap();
                    let app = tr.compile().unwrap();
                    let tag = format!("{op} {name} fast_complex={fast_complex}");
                    let mut outs = vec![Complex::new(0.0, 0.0); xs.len()];
                    app.evaluate_matrix(&xs, &mut outs, xs.len());
                    for (z, got) in xs.iter().zip(&outs) {
                        let w = f(z.re);
                        assert!(
                            got.re.to_bits() == w.to_bits() && got.im == 0.0,
                            "{tag} z = {z}: {got} != {w}"
                        );
                    }
                }
            }
        }
    }
}
