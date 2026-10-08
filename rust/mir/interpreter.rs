//! The bytecode interpreter of the MIR (`ty="bytecode"`, `debug` mode, and the reference of
//! the tests): `Mir::exec_instruction` runs the code on an array of registers, the model
//! memory, the stack and the parameters.

use num_complex::Complex;
use num_traits::identities::Zero;

use super::super::code::Func;
use super::super::config::SPILL_AREA;
use super::super::symbol::Loc;
use super::super::utils::{bool_to_f64, Reg};
use super::{ArithOp, BinOp, FusedOp, Instruction, Mir, UniOp};

impl Mir {
    fn get(regs: &[f64], r: Reg) -> f64 {
        match r {
            Reg::Ret | Reg::Left => regs[0],
            Reg::Temp | Reg::Right => regs[1],
            Reg::Gen(r) => regs[r as usize + 2],
            Reg::Static(..) => todo!(),
        }
    }

    fn set(regs: &mut [f64], r: Reg, val: f64) {
        match r {
            Reg::Ret | Reg::Left => {
                regs[0] = val;
            }
            Reg::Temp | Reg::Right => {
                regs[1] = val;
            }
            Reg::Gen(r) => {
                regs[r as usize + 2] = val;
            }
            Reg::Static(..) => todo!(),
        }
    }

    fn exec_uniop(regs: &mut [f64], op: UniOp, dst: Reg, s1: Reg) {
        let s1 = Self::get(regs, s1);

        let val = match op {
            UniOp::Neg => -s1,
            UniOp::Not => f64::from_bits(!s1.to_bits()),
            UniOp::Abs => s1.abs(),
            UniOp::Abs2 => s1.abs().powi(2),
            UniOp::Root => s1.sqrt(),
            UniOp::RealRoot => s1.sqrt(),
            UniOp::Recip => 1.0 / s1,
            UniOp::Round => s1.round(),
            UniOp::Floor => s1.floor(),
            UniOp::Ceiling => s1.ceil(),
            UniOp::Trunc => s1.trunc(),
            UniOp::Real => s1,
            UniOp::Imaginary => 0.0,
            UniOp::Conjugate => s1,
            UniOp::Half => s1 / 2.0,
            UniOp::IsZero => bool_to_f64(s1 == 0.0),
            UniOp::IsNotZero => bool_to_f64(s1 != 0.0),
            UniOp::Sign => {
                if s1 < 0.0 {
                    -0.0
                } else {
                    0.0
                }
            }
            UniOp::TimesI | UniOp::TimesNegI => 0.0,
        };

        Self::set(regs, dst, val);
    }

    fn exec_binop(regs: &mut [f64], op: BinOp, dst: Reg, s1: Reg, s2: Reg) {
        let s1 = Self::get(regs, s1);
        let s2 = Self::get(regs, s2);

        let val = match op {
            BinOp::Plus => s1 + s2,
            BinOp::Minus => s1 - s2,
            BinOp::Times => s1 * s2,
            BinOp::Divide => s1 / s2,
            BinOp::GreaterThan => bool_to_f64(s1 > s2),
            BinOp::GreaterThanEqual => bool_to_f64(s1 >= s2),
            BinOp::LittleThan => bool_to_f64(s1 < s2),
            BinOp::LittleThanEqual => bool_to_f64(s1 <= s2),
            BinOp::Equal => bool_to_f64(s1 == s2),
            BinOp::NotEqual => bool_to_f64(s1 != s2),
            BinOp::And => f64::from_bits(s1.to_bits() & s2.to_bits()),
            BinOp::AndNot => f64::from_bits(!s1.to_bits() & s2.to_bits()),
            BinOp::Or => f64::from_bits(s1.to_bits() | s2.to_bits()),
            BinOp::Xor => f64::from_bits(s1.to_bits() ^ s2.to_bits()),
            BinOp::Complex => s1,
        };

        Self::set(regs, dst, val);
    }

    fn exec_fused(regs: &mut [f64], op: FusedOp, dst: Reg, a: Reg, b: Reg, c: Reg) {
        let a = Self::get(regs, a);
        let b = Self::get(regs, b);
        let c = Self::get(regs, c);

        let val = match op {
            FusedOp::MulAdd => a * b + c,
            FusedOp::MulSub => a * b - c,
            FusedOp::NegMulAdd => -a * b + c,
            FusedOp::NegMulSub => -a * b - c,
        };

        Self::set(regs, dst, val);
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_load_math(
        mem: &mut [f64],
        stack: &mut [f64],
        regs: &mut [f64],
        params: &[f64],
        op: ArithOp,
        dst: Reg,
        s1: Reg,
        loc: Loc,
    ) {
        let s1 = Self::get(regs, s1);

        let y = match loc {
            Loc::Mem(idx) => mem[idx as usize],
            Loc::Stack(idx) => stack[idx as usize],
            Loc::Param(idx) => params[idx as usize],
        };

        let val = match op {
            ArithOp::Plus => s1 + y,
            ArithOp::Minus => s1 - y,
            ArithOp::Times => s1 * y,
            ArithOp::Divide => s1 / y,
        };

        Self::set(regs, dst, val);
    }

    fn exec_load_const_math(regs: &mut [f64], op: ArithOp, dst: Reg, s1: Reg, y: f64) {
        let s1 = Self::get(regs, s1);

        let val = match op {
            ArithOp::Plus => s1 + y,
            ArithOp::Minus => s1 - y,
            ArithOp::Times => s1 * y,
            ArithOp::Divide => s1 / y,
        };

        Self::set(regs, dst, val);
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_complex(
        regs: &mut [f64],
        op: ArithOp,
        xd: Reg,
        yd: Reg,
        x1: Reg,
        y1: Reg,
        x2: Reg,
        y2: Reg,
    ) {
        let z1 = Complex::new(Self::get(regs, x1), Self::get(regs, y1));
        let z2 = Complex::new(Self::get(regs, x2), Self::get(regs, y2));

        let val = match op {
            ArithOp::Plus => z1 + z2,
            ArithOp::Minus => z1 - z2,
            ArithOp::Times => z1 * z2,
            ArithOp::Divide => z1 / z2,
        };

        Self::set(regs, xd, val.re);
        Self::set(regs, yd, val.im);
    }

    pub fn exec_instruction(
        &self,
        mem: &mut [f64],
        stack: &mut [f64],
        regs: &mut [f64],
        params: &[f64],
    ) {
        let mut ip: usize = 0;
        let prog: Vec<Instruction> = self.code.iter().collect();
        let n = prog.len();
        // return addresses of subroutine calls (`Call` with no arguments)
        let mut returns: Vec<usize> = Vec::new();

        while ip < n {
            let ins = &prog[ip];

            match ins {
                Instruction::Nop | Instruction::End => {}
                Instruction::Uni { op, dst, s1 } => {
                    Self::exec_uniop(regs, *op, *dst, *s1);
                }
                Instruction::Bi { op, dst, s1, s2 } => {
                    Self::exec_binop(regs, *op, *dst, *s1, *s2);
                }
                Instruction::Mov { dst, s1 } => {
                    let x = Self::get(regs, *s1);
                    Self::set(regs, *dst, x);
                }
                Instruction::Load { dst, loc } => {
                    let val = match loc {
                        Loc::Mem(idx) => mem[*idx as usize],
                        Loc::Stack(idx) => stack[*idx as usize],
                        Loc::Param(idx) => params[*idx as usize],
                    };
                    Self::set(regs, *dst, val);
                }
                Instruction::Save { src, loc } => {
                    let val = Self::get(regs, *src);
                    match loc {
                        Loc::Mem(idx) => {
                            mem[*idx as usize] = val;
                        }
                        Loc::Stack(idx) => {
                            stack[*idx as usize] = val;
                        }
                        Loc::Param(_) => {
                            unreachable!()
                        }
                    };
                }
                Instruction::LoadComplex { xd, yd, loc } => {
                    let (x, y) = match loc {
                        Loc::Mem(idx) => (mem[*idx as usize], mem[1 + *idx as usize]),
                        Loc::Stack(idx) => (stack[*idx as usize], stack[1 + *idx as usize]),
                        Loc::Param(idx) => (params[*idx as usize], params[1 + *idx as usize]),
                    };
                    Self::set(regs, *xd, x);
                    Self::set(regs, *yd, y);
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    let x = Self::get(regs, *xs);
                    let y = Self::get(regs, *ys);
                    match loc {
                        Loc::Mem(idx) => {
                            mem[*idx as usize] = x;
                            mem[1 + *idx as usize] = y;
                        }
                        Loc::Stack(idx) => {
                            stack[*idx as usize] = x;
                            stack[1 + *idx as usize] = y;
                        }
                        Loc::Param(_) => {
                            unreachable!()
                        }
                    };
                }
                Instruction::LoadConst { dst, idx } => {
                    Self::set(regs, *dst, self.consts[*idx as usize]);
                }
                Instruction::LoadArgs { .. } => {
                    unimplemented!()
                }
                Instruction::SaveArgs { .. } => {
                    unimplemented!()
                }
                Instruction::Call { label, num_args: 0 } => {
                    returns.push(ip);
                    ip = *self.labels.get(label).unwrap() - 1;
                }
                Instruction::Call { label, num_args } => {
                    let f = self.find_op(label).unwrap();
                    match &f {
                        Func::Unary(p) => Self::set(regs, Reg::Ret, p(Self::get(regs, Reg::Left))),
                        Func::Binary(p) => Self::set(
                            regs,
                            Reg::Ret,
                            p(Self::get(regs, Reg::Left), Self::get(regs, Reg::Right)),
                        ),
                        Func::UnaryCplx(p) => {
                            let x = Complex::new(
                                Self::get(regs, Reg::Left),
                                Self::get(regs, Reg::Right),
                            );
                            let mut z = Complex::ZERO;
                            p(x.re, x.im, &mut z);
                            Self::set(regs, Reg::Ret, z.re);
                            Self::set(regs, Reg::Temp, z.im);
                        }
                        Func::BinaryCplx(p) => {
                            let x = Complex::new(
                                Self::get(regs, Reg::Left),
                                Self::get(regs, Reg::Right),
                            );
                            let y = Complex::new(
                                Self::get(regs, Reg::Gen(0)),
                                Self::get(regs, Reg::Gen(1)),
                            );
                            let mut z = y;
                            p(x.re, x.im, &mut z);
                            Self::set(regs, Reg::Ret, z.re);
                            Self::set(regs, Reg::Temp, z.im);
                        }
                        Func::PairedUnary(p) => {
                            let pair = p(Self::get(regs, Reg::Ret));
                            Self::set(regs, Reg::Ret, pair.s);
                            Self::set(regs, Reg::Temp, pair.c);
                        }
                        Func::Slice { env, f_scalar, .. } => unsafe {
                            let f: fn(
                                *const std::ffi::c_void,
                                *const f64,
                                usize,
                                *mut f64,
                            ) -> bool = std::mem::transmute(*f_scalar);

                            let mut val: Complex<f64> = Complex::default();
                            f(
                                *env,
                                stack.as_ptr().add(SPILL_AREA),
                                *num_args,
                                &mut val as *mut _ as *mut f64,
                            );

                            Self::set(regs, Reg::Ret, val.re);
                            Self::set(regs, Reg::Temp, val.im);
                        },
                        Func::App(..) | Func::Recursive => unimplemented!(),
                    }
                }
                Instruction::Fused { op, dst, a, b, c } => {
                    Self::exec_fused(regs, *op, *dst, *a, *b, *c);
                }
                Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                } => {
                    let cond = match cond {
                        Loc::Mem(idx) => mem[*idx as usize],
                        Loc::Stack(idx) => stack[*idx as usize],
                        Loc::Param(idx) => params[*idx as usize],
                    };
                    Self::set(
                        regs,
                        *dst,
                        if cond.is_zero() {
                            Self::get(regs, *false_val)
                        } else {
                            Self::get(regs, *true_val)
                        },
                    )
                }
                Instruction::Label { .. } => {}
                Instruction::Branch { label } if label == ".ret" => ip = returns.pop().unwrap(),
                Instruction::Branch { label } => ip = *self.labels.get(label).unwrap() - 1,
                Instruction::BranchIf {
                    cond,
                    label,
                    is_else,
                } => {
                    if (Self::get(regs, *cond) == 0.0) ^ is_else {
                        ip = *self.labels.get(label).unwrap() - 1
                    }
                }
                Instruction::LoadMath { op, dst, s1, loc } => {
                    Self::exec_load_math(mem, stack, regs, params, *op, *dst, *s1, *loc);
                }
                Instruction::LoadConstMath { op, dst, s1, idx } => {
                    Self::exec_load_const_math(regs, *op, *dst, *s1, self.consts[*idx as usize]);
                }
                Instruction::ComplexBi {
                    op,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                } => Self::exec_complex(regs, *op, *xd, *yd, *x1, *y1, *x2, *y2),
            }

            ip += 1;
        }
    }
}
