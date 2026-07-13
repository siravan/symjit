use std::collections::{HashMap, HashSet};
use std::fmt;

use anyhow::Result;

use crate::config::{Config, SLICE_CAP, SPILL_AREA};
use crate::mir::{Instruction, Mir};
use crate::serializer::MirWriter;
use crate::symbol::Loc;
use crate::utils::Reg;

/************************* GreedyAllocator ***************************/

#[derive(Debug, Clone)]
struct Static {
    reg: Reg,
    end: usize,
}

#[derive(Debug, Clone)]
struct Alloc {
    life: usize,
    loc: Option<Loc>,
}

impl Alloc {
    fn new() -> Alloc {
        Alloc {
            life: 0, // number of Static variables assigned to this register (== self.owners.len() is order versions)
            loc: None,
        }
    }
}

// #[derive(Clone)]
pub struct GreedyAllocator {
    pub code: MirWriter,      // the revised mir
    regs: Vec<Option<usize>>, // map of logical registers to static ones
    locs: HashSet<Loc>,       // locs currently in registers
    count_statics: usize,     // number of statis registers
    statics: Vec<Static>,     // the list of static registers
    allocs: Vec<Alloc>,       // allocation for logical registers
    config: Config,
    count_regs: usize,
}

impl fmt::Debug for GreedyAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ins) in self.code.iter().enumerate() {
            writeln!(f, "{:05}\t{:?}", i, ins)?;
        }

        writeln!(f, "...................")?;

        for (i, s) in self.statics.iter().enumerate() {
            writeln!(f, "σ{} := ({:?})", i, &s)?;
        }

        Ok(())
    }
}

impl GreedyAllocator {
    pub fn new(config: Config, count_regs: usize) -> GreedyAllocator {
        GreedyAllocator {
            code: MirWriter::new(),
            regs: vec![None; count_regs],
            count_statics: 0,
            locs: HashSet::new(),
            statics: Vec::new(),
            allocs: vec![Alloc::new(); count_regs],
            config,
            count_regs,
        }
    }

    pub fn optimize(&mut self, mir: &mut Mir) -> Result<()> {
        // create single-static-assignment form
        self.create(mir);

        // allocate registers using a greedy algorithm and
        // replace the static registers with the corresponding
        // logical (colored) registers
        self.color()?;
        self.contract()?;
        // contract the code by removing unnecessary instructions
        mir.code = std::mem::take(&mut self.code);

        Ok(())
    }

    fn push(&mut self, ins: Instruction) {
        self.code.push(&ins);
    }

    // reset for the first pass (logical -> static pass)
    fn reset_regs(&mut self) {
        self.regs = vec![None; self.count_regs];
    }

    // reset for the second pass (static -> logical pass)
    fn reset_allocs(&mut self) {
        self.allocs = vec![Alloc::new(); self.count_regs];
    }

    fn create_static(&mut self, ip: usize, r: Reg) -> usize {
        let idx = self.count_statics;
        self.count_statics += 1;
        self.statics.push(Static { reg: r, end: ip });
        idx
    }

    // consumes a logical register, update the end interval for the corresponding
    // static register and then returns the static register.
    fn consume(&mut self, ip: usize, src: Reg) -> Reg {
        if let Reg::Gen(r) = src {
            let s = self.regs[r as usize]
                .unwrap_or_else(|| panic!("reg {:?} not found (line {})", src, ip));
            self.statics[s].end = ip;
            Reg::Static(s as u32)
        } else {
            src
        }
    }

    // converts a destination logical register to a static one
    fn produce(&mut self, ip: usize, dst: Reg) -> Reg {
        if let Reg::Gen(r) = dst {
            let s = self.create_static(ip, dst);
            self.regs[r as usize] = Some(s);
            Reg::Static(s as u32)
        } else {
            dst
        }
    }

    // helper function to ease producing dst and consuming s1.
    fn unary_op(&mut self, ip: usize, dst: Reg, s1: Reg) -> (Reg, Reg) {
        // note than RHS is consumed before producing LHS
        let s1 = self.consume(ip, s1);
        let dst = self.produce(ip, dst);
        (dst, s1)
    }

    // helper function to ease producing dst and consuming s1 and s2.
    fn binary_op(&mut self, ip: usize, dst: Reg, s1: Reg, s2: Reg) -> (Reg, Reg, Reg) {
        // note than RHS's are consumed before producing LHS
        let s1 = self.consume(ip, s1);
        let s2 = self.consume(ip, s2);
        let dst = self.produce(ip, dst);
        (dst, s1, s2)
    }

    // helper function to ease producing dst and consuming s1, s2, and s3.
    fn ternary_op(
        &mut self,
        ip: usize,
        dst: Reg,
        s1: Reg,
        s2: Reg,
        s3: Reg,
    ) -> (Reg, Reg, Reg, Reg) {
        // note than RHS's are consumed before producing LHS
        let s1 = self.consume(ip, s1);
        let s2 = self.consume(ip, s2);
        let s3 = self.consume(ip, s3);
        let dst = self.produce(ip, dst);
        (dst, s1, s2, s3)
    }

    // The first pass.
    // converts logical to static (SSA-form) registers
    pub fn create(&mut self, mir: &Mir) {
        for (ip, ins) in mir.code.iter().enumerate() {
            match ins {
                Instruction::Nop => self.push(Instruction::Nop),
                Instruction::End => self.push(Instruction::End),
                Instruction::Uni { op, dst, s1 } => {
                    let (dst, s1) = self.unary_op(ip, dst, s1);
                    self.push(Instruction::Uni { op, dst, s1 });
                }
                Instruction::Bi { op, dst, s1, s2 } => {
                    let (dst, s1, s2) = self.binary_op(ip, dst, s1, s2);
                    self.push(Instruction::Bi { op, dst, s1, s2 });
                }
                Instruction::LoadConst { dst, idx } => {
                    let dst = self.produce(ip, dst);
                    self.push(Instruction::LoadConst { dst, idx });
                }
                Instruction::Load { dst, loc } => {
                    let dst = self.produce(ip, dst);
                    self.push(Instruction::Load { dst, loc });
                }
                Instruction::Save { src, loc } => {
                    let src = self.consume(ip, src);
                    self.push(Instruction::Save { src, loc });
                }
                Instruction::LoadComplex { xd, yd, loc } => {
                    let xd = self.produce(ip, xd);
                    let yd = self.produce(ip, yd);
                    self.push(Instruction::LoadComplex { xd, yd, loc });
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    let xs = self.consume(ip, xs);
                    let ys = self.consume(ip, ys);
                    self.push(Instruction::SaveComplex { xs, ys, loc });
                }
                Instruction::Mov { dst, s1 } => {
                    let (dst, s1) = self.unary_op(ip, dst, s1);
                    self.push(Instruction::Mov { dst, s1 });
                }
                Instruction::Fused { op, dst, a, b, c } => {
                    let (dst, a, b, c) = self.ternary_op(ip, dst, a, b, c);
                    self.push(Instruction::Fused { op, dst, a, b, c });
                }
                Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                } => {
                    let (dst, true_val, false_val) = self.binary_op(ip, dst, true_val, false_val);
                    self.push(Instruction::IfElse {
                        dst,
                        true_val,
                        false_val,
                        cond,
                    });
                }
                Instruction::Branch { .. } => {
                    if let Instruction::Branch { label } = ins.clone() {
                        self.push(Instruction::Branch { label });
                    }
                }
                Instruction::BranchIf { .. } => {
                    if let Instruction::BranchIf {
                        cond,
                        label,
                        is_else,
                    } = ins.clone()
                    {
                        let cond = self.consume(ip, cond);
                        self.push(Instruction::BranchIf {
                            cond,
                            label,
                            is_else,
                        });
                    }
                }
                Instruction::Call { .. } | Instruction::Label { .. } => {
                    self.push(ins.clone());
                    self.reset_regs();
                }
                Instruction::LoadMath { op, dst, s1, loc } => {
                    let s1 = self.consume(ip, s1);
                    let dst = self.produce(ip, dst);
                    self.push(Instruction::LoadMath { op, dst, s1, loc });
                }
                Instruction::LoadConstMath { op, dst, s1, idx } => {
                    let s1 = self.consume(ip, s1);
                    let dst = self.produce(ip, dst);
                    self.push(Instruction::LoadConstMath { op, dst, s1, idx });
                }
                Instruction::ComplexBi {
                    op,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                } => {
                    let x1 = self.consume(ip, x1);
                    let y1 = self.consume(ip, y1);
                    let x2 = self.consume(ip, x2);
                    let y2 = self.consume(ip, y2);
                    let xd = self.produce(ip, xd);
                    let yd = self.produce(ip, yd);
                    self.push(Instruction::ComplexBi {
                        op,
                        xd,
                        yd,
                        x1,
                        y1,
                        x2,
                        y2,
                    });
                }
            }
        }
    }

    // returns the logical register corresponding to the static register
    // dst back to the pool.
    fn deallocate(&mut self, dst: Reg) -> Reg {
        if let Reg::Static(s) = dst {
            self.statics[s as usize].reg
        } else {
            dst
        }
    }

    fn assign(&mut self, r: usize, s: usize, loc: Option<Loc>) -> Reg {
        // self.allocs[r].owners.insert(s);
        self.allocs[r].life = self.allocs[r].life.max(self.statics[s].end);
        self.allocs[r].loc = loc;
        let reg = Reg::Gen(r as u8);
        self.statics[s].reg = reg;
        reg
    }

    // allocates a new logical register from the pool and assigns it to
    // the static register dst, optionally with a location.
    fn allocate(&mut self, ip: usize, dst: Reg) -> (Reg, bool) {
        if let Reg::Static(s) = dst {
            let s = s as usize;
            let mut q: Option<usize> = None;

            for (r, alloc) in self.allocs.iter().enumerate() {
                if alloc.life <= ip {
                    if alloc.loc.is_none() {
                        return (self.assign(r, s, None), false);
                    } else if q.is_none() {
                        q = Some(r);
                    }
                }
            }

            return (
                self.assign(q.expect("register pool is empty"), s, None),
                false,
            );
        } else {
            (dst, false)
        }
    }

    fn allocate_loc(&mut self, ip: usize, dst: Reg, loc: Loc) -> (Reg, bool) {
        if let Reg::Static(s) = dst {
            let s = s as usize;
            let mut p: Option<usize> = None;
            let mut q: Option<usize> = None;

            for (r, alloc) in self.allocs.iter().enumerate() {
                if alloc.loc == Some(loc) {
                    return (self.assign(r, s, Some(loc)), true);
                } else if alloc.life <= ip {
                    if alloc.loc.is_none() && p.is_none() {
                        p = Some(r);
                    } else if q.is_none() {
                        q = Some(r);
                    }
                }
            }

            if p.is_some() {
                return (self.assign(p.unwrap(), s, Some(loc)), false);
            } else {
                return (
                    self.assign(q.expect("register pool is empty"), s, Some(loc)),
                    false,
                );
            }
        } else {
            (dst, false)
        }
    }

    // The second pass.
    // Converts static to logical registers.
    fn color(&mut self) -> Result<()> {
        self.reset_regs();
        let code = std::mem::take(&mut self.code);

        // replace all static regs with the corresponding logical ones
        for (ip, ins) in code.iter().enumerate() {
            match ins {
                Instruction::Nop => self.push(Instruction::Nop),
                Instruction::End => self.push(Instruction::End),
                Instruction::Uni { op, dst, s1 } => {
                    let s1 = self.deallocate(s1);
                    let (dst, _) = self.allocate(ip, dst);
                    self.push(Instruction::Uni { op, dst, s1 })
                }
                Instruction::Bi { op, dst, s1, s2 } => {
                    if self.config.is_sse() || self.config.is_arm64() {
                        let (dst, _) = self.allocate(ip, dst);
                        let s1 = self.deallocate(s1);
                        let s2 = self.deallocate(s2);
                        self.push(Instruction::Bi { op, dst, s1, s2 })
                    } else {
                        let s1 = self.deallocate(s1);
                        let s2 = self.deallocate(s2);
                        let (dst, _) = self.allocate(ip, dst);
                        self.push(Instruction::Bi { op, dst, s1, s2 })
                    }
                }
                Instruction::LoadConst { dst, idx } => {
                    let (dst, _) = self.allocate(ip, dst);
                    self.push(Instruction::LoadConst { dst, idx })
                }
                Instruction::Load { dst, loc } => {
                    let (dst, moved) = self.allocate_loc(ip, dst, loc);
                    if !moved {
                        self.push(Instruction::Load { dst, loc });
                        self.locs.insert(loc);
                    }
                }
                Instruction::Save { src, loc } => {
                    let src = self.deallocate(src);
                    self.push(Instruction::Save { src, loc });

                    // this for loop is added due to a bug discovered while
                    // compiling Symbolica expressions (e.g., x^3 + y^3).
                    // A loc should be in only one register (the wrong registers
                    // were used). Therefore, a save invalidates all previous
                    // assignments to that loc.
                    for a in self.allocs.iter_mut() {
                        if Some(loc) == a.loc {
                            a.loc = None;
                        }
                    }

                    if let Reg::Gen(r) = src {
                        self.allocs[r as usize].loc = Some(loc);
                    }
                }
                Instruction::LoadComplex { xd, yd, loc } => {
                    let (xd, moved) = self.allocate_loc(ip, xd, loc);
                    let (yd, _) = self.allocate_loc(ip, yd, loc.imag());
                    if !moved {
                        self.push(Instruction::LoadComplex { xd, yd, loc });
                        self.locs.insert(loc);
                        self.locs.insert(loc.imag());
                    }
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    let xs = self.deallocate(xs);
                    let ys = self.deallocate(ys);
                    self.push(Instruction::SaveComplex { xs, ys, loc });

                    // this for loop is added due to a bug discovered while
                    // compiling Symbolica expressions (e.g., x^3 + y^3).
                    // A loc should be in only one register (the wrong registers
                    // were used). Therefore, a save invalidates all previous
                    // assignments to that loc.
                    for a in self.allocs.iter_mut() {
                        if Some(loc) == a.loc {
                            a.loc = None;
                        }
                        if Some(loc.imag()) == a.loc {
                            a.loc = None;
                        }
                    }

                    if let Reg::Gen(r) = xs {
                        self.allocs[r as usize].loc = Some(loc);
                    }
                    if let Reg::Gen(r) = ys {
                        self.allocs[r as usize].loc = Some(loc.imag());
                    }
                }
                Instruction::Mov { dst, s1 } => {
                    let s1 = self.deallocate(s1);

                    let (dst, _) = if let Reg::Gen(r) = s1 {
                        let loc = self.allocs[r as usize].loc;
                        if loc.is_some() {
                            self.allocate_loc(ip, dst, loc.unwrap())
                        } else {
                            self.allocate(ip, dst)
                        }
                    } else {
                        self.allocate(ip, dst)
                    };

                    self.push(Instruction::Mov { dst, s1 })
                }
                Instruction::Fused { op, dst, a, b, c } => {
                    let a = self.deallocate(a);
                    let b = self.deallocate(b);
                    let c = self.deallocate(c);
                    let (dst, _) = self.allocate(ip, dst);
                    self.push(Instruction::Fused { op, dst, a, b, c });
                }
                Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                } => {
                    if self.config.is_sse() {
                        let (dst, _) = self.allocate(ip, dst);
                        let true_val = self.deallocate(true_val);
                        let false_val = self.deallocate(false_val);
                        self.push(Instruction::IfElse {
                            dst,
                            true_val,
                            false_val,
                            cond,
                        })
                    } else {
                        let true_val = self.deallocate(true_val);
                        let false_val = self.deallocate(false_val);
                        let (dst, _) = self.allocate(ip, dst);
                        self.push(Instruction::IfElse {
                            dst,
                            true_val,
                            false_val,
                            cond,
                        })
                    }
                    self.locs.insert(cond);
                }
                Instruction::Branch { .. } => {
                    if let Instruction::Branch { label } = ins.clone() {
                        self.push(Instruction::Branch { label });
                    }
                    self.reset_allocs(); // needed?
                }
                Instruction::BranchIf { .. } => {
                    if let Instruction::BranchIf {
                        cond,
                        label,
                        is_else,
                    } = ins.clone()
                    {
                        let cond = self.deallocate(cond);
                        self.push(Instruction::BranchIf {
                            cond,
                            label,
                            is_else,
                        });
                        self.reset_allocs();
                    }
                }
                Instruction::Call { .. } | Instruction::Label { .. } => {
                    self.push(ins.clone());
                    self.reset_allocs();
                }
                Instruction::LoadMath { op, dst, s1, loc } => {
                    let s1 = self.deallocate(s1);
                    let (dst, _) = self.allocate(ip, dst);
                    self.push(Instruction::LoadMath { op, dst, s1, loc });
                    self.locs.insert(loc);
                }
                Instruction::LoadConstMath { op, dst, s1, idx } => {
                    let s1 = self.deallocate(s1);
                    let (dst, _) = self.allocate(ip, dst);
                    self.push(Instruction::LoadConstMath { op, dst, s1, idx });
                }
                Instruction::ComplexBi {
                    op,
                    xd,
                    yd,
                    x1,
                    y1,
                    x2,
                    y2,
                } => {
                    if self.config.is_sse() {
                        let (xd, _) = self.allocate(ip, xd);
                        let (yd, _) = self.allocate(ip, yd);
                        let x1 = self.deallocate(x1);
                        let y1 = self.deallocate(y1);
                        let x2 = self.deallocate(x2);
                        let y2 = self.deallocate(y2);
                        self.push(Instruction::ComplexBi {
                            op,
                            xd,
                            yd,
                            x1,
                            y1,
                            x2,
                            y2,
                        })
                    } else {
                        let x1 = self.deallocate(x1);
                        let y1 = self.deallocate(y1);
                        let x2 = self.deallocate(x2);
                        let y2 = self.deallocate(y2);
                        let (xd, _) = self.allocate(ip, xd);
                        let (yd, _) = self.allocate(ip, yd);
                        self.push(Instruction::ComplexBi {
                            op,
                            xd,
                            yd,
                            x1,
                            y1,
                            x2,
                            y2,
                        })
                    }
                }
            }
        }

        Ok(())
    }

    // The third pass.
    // Removes unnessasary instructions.
    fn contract(&mut self) -> Result<()> {
        let code = std::mem::take(&mut self.code);
        let fixed = if self.config.is_complex() {
            (2 * SLICE_CAP + SPILL_AREA) as u32
        } else {
            (SLICE_CAP + SPILL_AREA) as u32
        };

        for ins in code.iter() {
            match ins {
                // This rule is commented out to prevent eliding Args during
                // external calls. If needed, we can restore this rule as long
                // as Args are added to self.locs.
                Instruction::Save { src, loc } => {
                    let keep = if let Loc::Stack(idx) = loc {
                        idx < fixed || self.locs.contains(&loc)
                    } else {
                        true
                    };

                    if keep {
                        self.push(Instruction::Save { src, loc })
                    }
                }
                Instruction::SaveComplex { xs, ys, loc } => {
                    let keep = if let Loc::Stack(idx) = loc {
                        idx < fixed || (self.locs.contains(&loc) && self.locs.contains(&loc.imag()))
                    } else {
                        true
                    };

                    if keep {
                        self.push(Instruction::SaveComplex { xs, ys, loc })
                    }
                }
                Instruction::Mov { dst, s1 } => {
                    if dst != s1 {
                        self.push(Instruction::Mov { dst, s1 });
                    }
                }
                Instruction::Nop => {}
                _ => self.push(ins),
            }
        }

        Ok(())
    }
}
