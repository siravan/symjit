//! The peephole optimizer of the MIR (`Mir::optimize_peephole`, opt level >= 1): it slides a
//! window of up to five instructions over the code and replaces patterns with cheaper
//! sequences (`fuse_*`: op + mov, load/save, reciprocals, zeroing, fused multiply-add,
//! paired multiplications, gotos).

use super::{ArithOp, BinOp, FusedOp, Instruction, Mir, UniOp};
use super::super::serializer::MirWriter;
use super::super::symbol::Loc;
use super::super::utils::Reg;

impl Instruction {
    fn dst(&self) -> Reg {
        match self {
            Instruction::Bi { dst, .. } => *dst,
            Instruction::Uni { dst, .. } => *dst,
            Instruction::Fused { dst, .. } => *dst,
            Instruction::IfElse { dst, .. } => *dst,
            Instruction::Load { dst, .. } => *dst,
            Instruction::LoadConst { dst, .. } => *dst,
            Instruction::LoadConstMath { dst, .. } => *dst,
            Instruction::LoadMath { dst, .. } => *dst,
            Instruction::Mov { dst, .. } => *dst,
            _ => panic!("Instruction {:?} does not have a dst field.", self),
        }
    }

    fn src(&self) -> Reg {
        if let Instruction::Save { src, .. } = self {
            *src
        } else {
            panic!("Instruction {:?} does not have a src field.", self)
        }
    }

    fn s1(&self) -> Reg {
        match self {
            Instruction::Bi { s1, .. } => *s1,
            Instruction::Uni { s1, .. } => *s1,
            Instruction::LoadConstMath { s1, .. } => *s1,
            Instruction::LoadMath { s1, .. } => *s1,
            Instruction::Mov { s1, .. } => *s1,
            _ => panic!("Instruction {:?} does not have an s1 field.", self),
        }
    }

    fn s2(&self) -> Reg {
        if let Instruction::Bi { s2, .. } = self {
            *s2
        } else {
            panic!("Instruction {:?} does not have an s2 field.", self)
        }
    }

    fn loc(&self) -> Loc {
        match self {
            Instruction::Load { loc, .. } => *loc,
            Instruction::LoadMath { loc, .. } => *loc,
            Instruction::Save { loc, .. } => *loc,
            _ => panic!("Instruction {:?} does not have a loc field.", self),
        }
    }

    fn label(&self) -> Option<String> {
        match self {
            Instruction::Label { label } => Some(label.to_string()),
            Instruction::Branch { label } => Some(label.to_string()),
            Instruction::BranchIf { label, .. } => Some(label.to_string()),
            Instruction::Call { label, .. } => Some(label.to_string()),
            _ => None,
        }
    }
}

impl Mir {
    fn fuse_op_mov(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        /*
         * example:
         *      %0 = Root(%2)
         *      %l = %0
         *      call power
         * becomes
         *      %l = Root(%2)
         *      call power
         */
        if let Instruction::Uni { op, .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Uni {
                        op,
                        dst: q1.dst(),
                        s1: q0.s1(),
                    });
                }
            }
        };

        /*
         * example:
         *      %0 = %2 Plus %3
         *      %l = %0
         *      call power
         * becomes
         *      %l = %2 Plus %3
         *      call power
         */
        if let Instruction::Bi { op, .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Bi {
                        op,
                        dst: q1.dst(),
                        s1: q0.s1(),
                        s2: q0.s2(),
                    });
                }
            }
        };

        None
    }

    fn fuse_goto(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Branch { .. } = *q0 {
            if let Instruction::Label { .. } = *q1 {
                if q0.label().unwrap() == q1.label().unwrap() {
                    return Some(q1.clone());
                }
            }
        };

        None
    }

    fn fuse_load(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        /*
         * example
         *      %0 := Stack[2]
         *      %2 := %0
         * becomes
         *      %2 := Stack[2]
         *
         * note that we assume %0 is not needed anymore. This was not true for powi and
         * powi_mod; therefore, we added nop to prevent this rule from firing for those
         * functions.
         */
        if let Instruction::Load { .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Load {
                        dst: q1.dst(),
                        loc: q0.loc(),
                    });
                }
            }
        };

        if let Instruction::LoadConst { idx, .. } = *q0 {
            if let Instruction::Mov { .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::LoadConst { dst: q1.dst(), idx });
                }
            }
        };

        if let Instruction::Load { .. } = *q0 {
            if let Instruction::Save { .. } = *q1 {
                if q0.loc() == q1.loc() && q0.dst() == q1.src() {
                    return Some(Instruction::Nop);
                }
            }
        };

        None
    }

    fn fuse_save(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        // Important: this rule is commented out because of a potential bug,
        // where %0 is needed afterward.
        /*
        /*
        * example
        *      %0 := %1
        *      Mem[4] = %0
        * becomes
        *      Mem[4] := %1
        */
        if let Instruction::Mov { dst, s1 } = *q0 {
            if let Instruction::Save {
                src: dst_q1,
                loc: loc_q1,
            } = *q1
            {
                if dst == dst_q1 {
                    code.push(Instruction::Save {
                        src: s1,
                        loc: loc_q1,
                    });
                    return true;
                }
            }
        }
        */

        /*
         * example
         *      Stack[6] = %2
         *      %0 = Stack[6]
         * becomes
         *      Stack[6] = %2
         *      %0 := %2
         *
         * note that if we know that Stack[6] is not accessed again, we can remove the
         * first instruction, but this is not yet implemented.
         */
        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if q0.loc() == q1.loc() {
                    code.push(q0);
                    return Some(Instruction::Mov {
                        dst: q1.dst(),
                        s1: q0.src(),
                    });
                }
            }
        };

        if let Instruction::Save { .. } = *q0 {
            if let Instruction::LoadMath { op, .. } = *q1 {
                if q0.loc() == q1.loc() {
                    code.push(q0);
                    return Some(Instruction::Bi {
                        op: match op {
                            ArithOp::Plus => BinOp::Plus,
                            ArithOp::Minus => BinOp::Minus,
                            ArithOp::Times => BinOp::Times,
                            ArithOp::Divide => BinOp::Divide,
                        },
                        dst: q1.dst(),
                        s1: q1.s1(),
                        s2: q0.src(),
                    });
                }
            }
        }

        None
    }

    fn fuse_save3(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
        /*
         * this combination happens in return from remote function calls
         * examples:
         *      call sin
         *      Stack[10] = %$
         *      %0 = Stack[10]
         *      Mem[5] = %0
         * becomes
         *      call sin
         *      Mem[5] = %$
         */
        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Save { .. } = *q2 {
                    code.push(q0);
                    if q0.src() == Reg::Ret && q0.loc() == q1.loc() && q1.dst() == q2.src() {
                        return Some(Instruction::Save {
                            src: Reg::Ret,
                            loc: q2.loc(),
                        });
                    }
                }
            }
        };

        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Load { .. } = *q2 {
                    if q0.loc() == q2.loc() && q1.dst() != q2.dst() {
                        code.push(q0);
                        code.push(&Instruction::Mov {
                            dst: q2.dst(),
                            s1: q0.src(),
                        });
                        return Some(q1.clone());
                    }
                }
            }
        };

        if let Instruction::Save { .. } = *q0 {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::LoadMath { op, .. } = *q2 {
                    if q0.loc() == q2.loc() && q2.s1() == q1.dst() && q0.src() != Reg::Ret {
                        code.push(q0);
                        code.push(&Instruction::Load {
                            dst: Reg::Ret,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Bi {
                            op: match op {
                                ArithOp::Plus => BinOp::Plus,
                                ArithOp::Minus => BinOp::Minus,
                                ArithOp::Times => BinOp::Times,
                                ArithOp::Divide => BinOp::Divide,
                            },
                            dst: q2.dst(),
                            s1: Reg::Ret,
                            s2: q0.src(),
                        });
                    }
                }
            }
        }

        None
    }

    fn fuse_recip(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Uni {
            op: UniOp::Recip, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Times, ..
            } = *q1
            {
                if q1.s1() == q1.s2() {
                    // r * r with r = recip(x): both operands are the recip result, so
                    // the fusion below (which discards the recip and divides by its
                    // source instead) would be wrong; leave the pair unfused.
                } else if q0.dst() == q1.s2() {
                    return Some(Instruction::Bi {
                        op: BinOp::Divide,
                        dst: q1.dst(),
                        s1: q1.s1(),
                        s2: q0.s1(),
                    });
                } else if q0.dst() == q1.s1() {
                    return Some(Instruction::Bi {
                        op: BinOp::Divide,
                        dst: q1.dst(),
                        s1: q1.s2(),
                        s2: q0.s1(),
                    });
                }
            }
        }

        if let Instruction::Uni { op: UniOp::Neg, .. } = *q0 {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q1
            {
                if q0.dst() == q1.s2() {
                    return Some(Instruction::Bi {
                        op: BinOp::Minus,
                        dst: q1.dst(),
                        s1: q1.s1(),
                        s2: q0.s1(),
                    });
                } else if q0.dst() == q1.s1() {
                    return Some(Instruction::Bi {
                        op: BinOp::Minus,
                        dst: q1.dst(),
                        s1: q1.s2(),
                        s2: q0.s1(),
                    });
                }
            }
        }

        None
    }

    fn fuse_recip3(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Uni {
            op: UniOp::Recip, ..
        } = *q0
        {
            if !q1.regs().contains(&q0.dst()) && !q1.regs().contains(&q0.s1()) {
                if let Instruction::Bi {
                    op: BinOp::Times, ..
                } = *q2
                {
                    if q0.dst() == q2.s2() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Divide,
                            dst: q2.dst(),
                            s1: q2.s1(),
                            s2: q0.s1(),
                        });
                    } else if q0.dst() == q2.s1() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Divide,
                            dst: q2.dst(),
                            s1: q2.s2(),
                            s2: q0.s1(),
                        });
                    }
                }
            }
        }

        if let Instruction::Uni { op: UniOp::Neg, .. } = *q0 {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q2
            {
                if !q1.regs().contains(&q0.dst()) && !q1.regs().contains(&q0.s1()) {
                    if q0.dst() == q2.s2() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Minus,
                            dst: q2.dst(),
                            s1: q2.s1(),
                            s2: q0.s1(),
                        });
                    } else if q0.dst() == q2.s1() {
                        code.push(q1);
                        return Some(Instruction::Bi {
                            op: BinOp::Minus,
                            dst: q2.dst(),
                            s1: q2.s2(),
                            s2: q0.s1(),
                        });
                    }
                }
            }
        }

        None
    }

    fn fuse_zero(
        &self,
        _code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        if let Instruction::Uni {
            op: UniOp::IsZero, ..
        } = *q0
        {
            if let Instruction::Uni { op: UniOp::Neg, .. } = *q1 {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Uni {
                        op: UniOp::IsNotZero,
                        dst: q1.dst(),
                        s1: q0.s1(),
                    });
                }
            }
        }

        if let Instruction::Uni { op: UniOp::Not, .. } = *q0 {
            if let Instruction::Uni {
                op: UniOp::IsZero, ..
            } = *q1
            {
                if q0.dst() == q1.s1() {
                    return Some(Instruction::Uni {
                        op: UniOp::IsNotZero,
                        dst: q1.dst(),
                        s1: q0.s1(),
                    });
                }
            }
        }

        None
    }

    fn fuse_fma(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
    ) -> Option<Instruction> {
        // TODO: fix the FMA bug for complex noted on `runtests complex`
        if !self.config.fastmath()
        /*|| self.config.is_complex()*/
        {
            return None;
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s1(),
                    });
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Minus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::MulSub,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    return Some(Instruction::Fused {
                        op: FusedOp::NegMulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: q0.s2(),
                        c: q1.s1(),
                    });
                }
            }
        }

        if let Instruction::LoadMath {
            op: ArithOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Plus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::MulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s1(),
                    });
                }
            }
        }

        if let Instruction::LoadMath {
            op: ArithOp::Times, ..
        } = *q0
        {
            if let Instruction::Bi {
                op: BinOp::Minus, ..
            } = *q1
            {
                if q1.s1() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::MulSub,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s2(),
                    });
                }

                if q1.s2() == q0.dst() {
                    code.push(&Instruction::Load {
                        dst: Reg::Ret,
                        loc: q0.loc(),
                    });
                    return Some(Instruction::Fused {
                        op: FusedOp::NegMulAdd,
                        dst: q1.dst(),
                        a: q0.s1(),
                        b: Reg::Ret,
                        c: q1.s1(),
                    });
                }
            }
        }

        None
    }

    fn fuse_fma3(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
        if !self.config.fastmath() {
            return None;
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::LoadConst { idx, .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Plus, ..
                } = *q2
                {
                    if ((q2.s1() == q0.dst() && q2.s2() == q1.dst())
                        || (q2.s1() == q1.dst() && q2.s2() == q0.dst()))
                        && (q0.s1() != Reg::Temp && q0.s2() != Reg::Temp)
                    {
                        code.push(&Instruction::LoadConst {
                            dst: Reg::Temp,
                            idx,
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Plus, ..
                } = *q2
                {
                    if ((q2.s1() == q0.dst() && q2.s2() == q1.dst())
                        || (q2.s1() == q1.dst() && q2.s2() == q0.dst()))
                        && (q0.s1() != Reg::Temp && q0.s2() != Reg::Temp)
                    {
                        code.push(&Instruction::Load {
                            dst: Reg::Temp,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::LoadConst { idx, .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Minus, ..
                } = *q2
                {
                    if q2.s1() == q0.dst()
                        && q2.s2() == q1.dst()
                        && (q0.s1() != Reg::Temp && q0.s2() != Reg::Temp)
                    {
                        code.push(&Instruction::LoadConst {
                            dst: Reg::Temp,
                            idx,
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulSub,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    } else if q2.s1() == q1.dst()
                        && q2.s2() == q0.dst()
                        && (q0.s1() != Reg::Temp && q0.s2() != Reg::Temp)
                    {
                        code.push(&Instruction::LoadConst {
                            dst: Reg::Temp,
                            idx,
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::NegMulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        if let Instruction::Bi {
            op: BinOp::Times, ..
        } = *q0
        {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::Bi {
                    op: BinOp::Minus, ..
                } = *q2
                {
                    if q2.s1() == q0.dst()
                        && q2.s2() == q1.dst()
                        && (q0.s1() != Reg::Temp && q0.s2() != Reg::Temp)
                    {
                        code.push(&Instruction::Load {
                            dst: Reg::Temp,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::MulSub,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    } else if q2.s1() == q1.dst()
                        && q2.s2() == q0.dst()
                        && (q0.s1() != Reg::Temp && q0.s2() != Reg::Temp)
                    {
                        code.push(&Instruction::Load {
                            dst: Reg::Temp,
                            loc: q1.loc(),
                        });
                        return Some(Instruction::Fused {
                            op: FusedOp::NegMulAdd,
                            dst: q2.dst(),
                            a: q0.s1(),
                            b: q0.s2(),
                            c: Reg::Temp,
                        });
                    }
                }
            }
        }

        None
    }

    fn fuse_times2(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
    ) -> Option<Instruction> {
        if !self.config.is_complex() {
            return None;
        }

        if let Instruction::LoadMath {
            op: ArithOp::Times, ..
        } = *q0
        {
            if let Instruction::Load { .. } = *q1 {
                if let Instruction::LoadMath {
                    op: ArithOp::Times, ..
                } = *q2
                {
                    if q1.dst() == q2.s1() && q0.dst() != q1.dst() && q0.s1() != q1.dst() {
                        code.push(q1);
                        code.push(q0);
                        return Some(q2.clone());
                    }
                }
            }
        };

        None
    }

    fn fuse_times2_5(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
        q3: &Instruction,
        q4: &Instruction,
    ) -> Option<Instruction> {
        if !self.config.is_complex() {
            return None;
        }

        if let Instruction::Load { .. } = *q0 {
            if let Instruction::LoadMath {
                op: ArithOp::Times, ..
            } = *q1
            {
                if let Instruction::Save { .. } = *q2 {
                    if let Instruction::Load { .. } = *q3 {
                        if let Instruction::LoadMath {
                            op: ArithOp::Times, ..
                        } = *q4
                        {
                            if q0.dst() == q1.s1()
                                && q1.dst() == q2.src()
                                && q3.dst() == q4.s1()
                                && q2.loc() != q3.loc()
                                && q2.loc() != q4.loc()
                            {
                                code.push(&Instruction::Load {
                                    dst: Reg::Ret,
                                    loc: q0.loc(),
                                });
                                code.push(q3);
                                code.push(&Instruction::LoadMath {
                                    op: ArithOp::Times,
                                    dst: Reg::Ret,
                                    s1: Reg::Ret,
                                    loc: q1.loc(),
                                });
                                code.push(q4);
                                return Some(Instruction::Save {
                                    src: Reg::Ret,
                                    loc: q2.loc(),
                                });
                            }
                        }
                    }
                }
            }
        }

        None
    }

    fn fuse1(
        &self,
        code: &mut MirWriter,
        q0: &Instruction,
        q1: &Instruction,
        q2: &Instruction,
        q3: &Instruction,
        q4: &Instruction,
    ) -> (Instruction, usize) {
        if let Some(v) = self.fuse_times2_5(code, q0, q1, q2, q3, q4) {
            (v, 5)
        } else if let Some(v) = self.fuse_save3(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_fma3(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_times2(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_recip3(code, q0, q1, q2) {
            (v, 3)
        } else if let Some(v) = self.fuse_fma(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_op_mov(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_load(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_save(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_recip(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_zero(code, q0, q1) {
            (v, 2)
        } else if let Some(v) = self.fuse_goto(code, q0, q1) {
            (v, 2)
        } else {
            (Instruction::Nop, 0)
        }
    }

    pub fn optimize_peephole(&mut self, _stage: usize) -> bool {
        let mut success = false;

        let mut code = MirWriter::new();
        let mut iter = self.code.iter_mut();
        let mut q0: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q1: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q2: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q3: Instruction = iter.next().unwrap_or(Instruction::End).clone();
        let mut q4: Instruction = iter.next().unwrap_or(Instruction::End).clone();

        while !matches!(q0, Instruction::End) {
            let (top, num_consumed) = self.fuse1(&mut code, &q0, &q1, &q2, &q3, &q4);
            success |= num_consumed > 1;

            match num_consumed {
                // no matches. move to the next item.
                0 => {
                    code.push(&q0);
                    q0 = q1;
                    q1 = q2;
                    q2 = q3;
                    q3 = q4;
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0 and q1 match.
                2 => {
                    q0 = top;
                    q1 = q2;
                    q2 = q3;
                    q3 = q4;
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0, q1, and q2 match.
                3 => {
                    q0 = top;
                    q1 = q3;
                    q2 = q4;
                    q3 = iter.next().unwrap_or(Instruction::End).clone();
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0, q1, q2, and q3 match.
                4 => {
                    q0 = top;
                    q1 = q4;
                    q2 = iter.next().unwrap_or(Instruction::End).clone();
                    q3 = iter.next().unwrap_or(Instruction::End).clone();
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                // q0, q1, q2, q3, q4 match.
                5 => {
                    q0 = top;
                    q1 = iter.next().unwrap_or(Instruction::End).clone();
                    q2 = iter.next().unwrap_or(Instruction::End).clone();
                    q3 = iter.next().unwrap_or(Instruction::End).clone();
                    q4 = iter.next().unwrap_or(Instruction::End).clone();
                }
                _ => unreachable!(),
            }
        }

        self.code = code;
        success
    }
}
