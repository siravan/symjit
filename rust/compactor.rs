use anyhow::Result;
use std::collections::HashMap;

use crate::config::{Config, SLICE_CAP, SPILL_AREA};
use crate::mir::{Instruction, Mir};
use crate::symbol::Loc;

#[derive(Clone, Debug)]
pub struct Compactor {
    config: Config,
    code: Vec<Instruction>,    // the revised mir
    live: HashMap<Loc, usize>, // the map of live Stack slots and their last used position
    stack: HashMap<Loc, Loc>,  // the map of old Stack slots to new Stack slots
    pool: Vec<Loc>,            // the pool of Stack slots available to reuse
    count_stack: u32,          // next stack id to use if the pool is empty
    depth: isize,              // loop depth
    fixed: u32,
}

impl Compactor {
    pub fn new(config: Config) -> Compactor {
        let fixed = if config.is_complex() {
            (2 * SLICE_CAP + SPILL_AREA) as u32
        } else {
            (SLICE_CAP + SPILL_AREA) as u32
        };

        Compactor {
            config,
            code: Vec::new(),
            live: HashMap::new(),
            stack: HashMap::new(),
            pool: Vec::new(),
            count_stack: fixed,
            depth: 0,
            fixed,
        }
    }

    pub fn compact(&mut self, mir: &mut Mir) -> Result<usize> {
        self.collect_last(mir); // first pass, collect live info
        self.compact_stack(mir); // second pass, rename stack slots
        mir.code = std::mem::take(&mut self.code);
        Ok((self.fixed + self.count_stack) as usize)
    }

    fn push(&mut self, ins: Instruction) {
        self.code.push(ins);
    }

    fn collect_last(&mut self, mir: &Mir) {
        for (ip, ins) in mir.code.iter().enumerate() {
            match *ins {
                Instruction::Load { loc, .. }
                | Instruction::IfElse { cond: loc, .. }
                | Instruction::LoadMath { loc, .. } => {
                    if let Loc::Stack(idx) = loc {
                        if idx >= self.fixed {
                            if let Some(x) = self.live.get_mut(&loc) {
                                *x = ip;
                            }
                        }
                    }
                }
                Instruction::Save { loc, .. } => {
                    if let Loc::Stack(idx) = loc {
                        if idx >= self.fixed {
                            self.live.insert(loc, ip);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn save(&mut self, loc: Loc) -> Loc {
        if let Loc::Stack(idx) = loc {
            if idx < self.fixed {
                loc
            } else if let Some(Loc::Stack(s)) = self.stack.get(&loc) {
                // A stack slot can be assigned more than once in loops (ϕ-constructs)
                Loc::Stack(*s)
            } else {
                let l = match self.pool.pop() {
                    Some(l) => l, // pool is not empty
                    None => {
                        let l = Loc::Stack(self.count_stack);
                        self.count_stack += if self.config.is_complex() { 2 } else { 1 };
                        l
                    }
                };
                self.stack.insert(loc, l);
                l
            }
        } else {
            loc
        }
    }

    fn load(&mut self, loc: Loc, ip: usize) -> Loc {
        if let Loc::Stack(idx) = loc {
            if idx < self.fixed {
                loc
            } else {
                let l = self
                    .stack
                    .get(&loc)
                    .expect(&format!("cannot find {:?}", loc));
                let last = self.live.get(&loc).unwrap();

                // stack slots are not returned to the pool inside loops.
                // We can do better by delaying return to the branch at
                // the end of the loop, but this would add complexity.
                if *last == ip && self.depth == 0 {
                    self.pool.push(*l);
                }
                *l
            }
        } else {
            loc
        }
    }

    fn compact_stack(&mut self, mir: &Mir) {
        for (ip, ins) in mir.code.iter().enumerate() {
            match ins {
                Instruction::Load { dst, loc } => {
                    let l = self.load(*loc, ip);
                    self.push(Instruction::Load { dst: *dst, loc: l });
                }
                Instruction::LoadMath { op, dst, s1, loc } => {
                    let l = self.load(*loc, ip);
                    self.push(Instruction::LoadMath {
                        op: *op,
                        dst: *dst,
                        s1: *s1,
                        loc: l,
                    })
                }
                Instruction::IfElse {
                    dst,
                    true_val,
                    false_val,
                    cond,
                } => {
                    let l = self.load(*cond, ip);
                    self.push(Instruction::IfElse {
                        dst: *dst,
                        true_val: *true_val,
                        false_val: *false_val,
                        cond: l,
                    });
                }
                Instruction::Save { src, loc } => {
                    let l = self.save(*loc);
                    self.push(Instruction::Save { src: *src, loc: l });
                }
                Instruction::Label { label } => {
                    self.depth += 1;
                    self.push(Instruction::Label {
                        label: label.clone(),
                    });
                }
                Instruction::Branch { label } => {
                    self.depth -= 1;
                    self.push(Instruction::Branch {
                        label: label.clone(),
                    });
                }
                Instruction::BranchIf {
                    cond,
                    label,
                    is_else,
                } => {
                    self.depth -= 1;
                    self.push(Instruction::BranchIf {
                        cond: *cond,
                        label: label.clone(),
                        is_else: *is_else,
                    });
                }
                _ => {
                    self.push(ins.clone());
                }
            }
        }
    }
}
