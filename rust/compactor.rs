use anyhow::Result;
use std::collections::HashMap;

use crate::config::Config;
use crate::mir::{Instruction, Mir};
use crate::symbol::Loc;

#[derive(Clone, Debug)]
pub struct Compactor {
    config: Config,
    code: Vec<Instruction>, // the revised mir
    live: HashMap<Loc, usize>,
    stack: HashMap<Loc, Loc>,
    pool: Vec<Loc>,
    count_stack: u32,
    depth: usize,
}

impl Compactor {
    pub fn new(config: Config) -> Compactor {
        Compactor {
            config,
            code: Vec::new(),
            live: HashMap::new(),
            stack: HashMap::new(),
            pool: Vec::new(),
            count_stack: 16,
            depth: 0,
        }
    }

    pub fn compact(&mut self, mir: &mut Mir) -> Result<usize> {
        self.collect_last(mir);
        self.compact_stack(mir);
        mir.code = std::mem::take(&mut self.code);
        Ok(self.count_stack as usize)
    }

    fn push(&mut self, ins: Instruction) {
        self.code.push(ins);
    }

    fn collect_last(&mut self, mir: &Mir) {
        for (ip, ins) in mir.code.iter().enumerate() {
            match *ins {
                Instruction::Load { loc, .. } | Instruction::IfElse { cond: loc, .. } => {
                    if let Loc::Stack(_) = loc {
                        self.live.get_mut(&loc).map(|x| *x = ip);
                    }
                }
                Instruction::Save { loc, .. } => {
                    if let Loc::Stack(idx) = loc {
                        self.live.insert(Loc::Stack(idx), ip);
                    }
                }
                _ => {}
            }
        }
    }

    fn save(&mut self, loc: Loc) -> Loc {
        if let Loc::Stack(_) = loc {
            if let Some(Loc::Stack(s)) = self.stack.get(&loc) {
                Loc::Stack(*s)
            } else {
                let l = match self.pool.pop() {
                    Some(l) => l,
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
        if let Loc::Stack(_) = loc {
            let l = self.stack.get(&loc).unwrap();
            let last = self.live.get(&loc).unwrap();
            if *last == ip && self.depth == 0 {
                self.pool.push(*l);
            }
            *l
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
                    println!("{:?}", ins);
                    self.depth += 1;
                    self.push(Instruction::Label {
                        label: label.clone(),
                    });
                }
                Instruction::Branch { cond, label } => {
                    println!("{:?}", ins);
                    self.depth -= 1;
                    self.push(Instruction::Branch {
                        cond: *cond,
                        label: label.clone(),
                    });
                }
                _ => {
                    self.push(ins.clone());
                }
            }
        }
    }
}
