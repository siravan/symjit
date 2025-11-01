use std::fmt;

use petgraph::algo::coloring::dsatur_coloring;
use petgraph::graph::{NodeIndex, UnGraph};

use crate::mir::{Instruction, Mir};
use crate::symbol::Loc;
use crate::utils::Reg;

use crate::COUNT_SCRATCH;

#[derive(Clone, Debug)]
struct Vertex {
    start: u32,
    end: u32,
    reg: Reg,
}

#[derive(Clone)]
pub struct Allocator {
    pub code: Vec<Instruction>,
    regs: Vec<(Reg, Loc)>,
    count_statics: u32,
    graph: UnGraph<Vertex, ()>,
}

impl fmt::Debug for Allocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ins) in self.code.iter().enumerate() {
            writeln!(f, "{:05}\t{:?}", i, ins)?;
        }

        writeln!(f, "...................")?;
        writeln!(f, "{:#?}", self.graph)?;

        Ok(())
    }
}

impl Allocator {
    pub fn optimize(mir: &mut Mir) {
        let mut allocator = Allocator {
            code: Vec::new(),
            regs: vec![(Reg::Ret, Loc::Nowhere); COUNT_SCRATCH as usize],
            count_statics: 0,
            graph: UnGraph::new_undirected(),
        };

        allocator.create(mir);
        allocator.add_edges();
        allocator.color();

        mir.code = allocator.code;
    }

    fn push(&mut self, ins: Instruction) {
        self.code.push(ins);
    }

    fn ip(&self) -> u32 {
        self.code.len() as u32
    }

    fn reset(&mut self) {
        self.regs = vec![(Reg::Ret, Loc::Nowhere); COUNT_SCRATCH as usize];
    }

    fn overlap(&self, idx1: NodeIndex, idx2: NodeIndex) -> bool {
        let v1 = &self.graph[idx1];
        let v2 = &self.graph[idx2];
        u32::min(v1.end, v2.end) > u32::max(v1.start, v2.start)
    }

    fn add_edges(&mut self) {
        let n = self.graph.node_count();

        for i1 in 0..n {
            let idx1 = NodeIndex::new(i1);
            for i2 in i1 + 1..n {
                let idx2 = NodeIndex::new(i2);

                if self.overlap(idx1, idx2) {
                    self.graph.add_edge(idx1, idx2, ());
                }
            }
        }
    }

    fn create_static(&mut self) -> Reg {
        let s = Reg::Static(self.count_statics);
        self.count_statics += 1;
        self.graph.add_node(Vertex {
            start: self.ip(),
            end: 0,
            reg: Reg::Ret,
        });
        s
    }

    fn consume_static(&mut self, src: Reg) -> Reg {
        if let Reg::Gen(r) = src {
            let (s, _) = self.regs[r as usize];
            if let Reg::Static(k) = s {
                self.graph[NodeIndex::new(k as usize)].end = self.ip();
            };
            s
        } else {
            src
        }
    }

    fn subs_dst(&mut self, dst: Reg) -> Reg {
        if let Reg::Gen(r) = dst {
            let s = self.create_static();
            self.regs[r as usize] = (s, Loc::Nowhere);
            s
        } else {
            dst
        }
    }

    fn subs_uni(&mut self, dst: Reg, s1: Reg) -> (Reg, Reg) {
        let s1 = self.consume_static(s1);
        let dst = self.subs_dst(dst);
        (dst, s1)
    }

    fn subs_bi(&mut self, dst: Reg, s1: Reg, s2: Reg) -> (Reg, Reg, Reg) {
        let s1 = self.consume_static(s1);
        let s2 = self.consume_static(s2);
        let dst = self.subs_dst(dst);
        (dst, s1, s2)
    }

    fn subs_tri(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) -> (Reg, Reg, Reg, Reg) {
        let s1 = self.consume_static(s1);
        let s2 = self.consume_static(s2);
        let s3 = self.consume_static(s3);
        let dst = self.subs_dst(dst);
        (dst, s1, s2, s3)
    }

    fn load(&mut self, dst: Reg, loc: Loc) {
        if let Reg::Gen(r) = dst {
            if let Some(k) = self.regs.iter().position(|(_, l)| *l == loc) {
                self.regs[r as usize] = self.regs[k];
                return;
            } else {
                let dst = self.create_static();
                self.regs[r as usize] = (dst, loc);
                self.push(Instruction::Load { dst, loc });
            };
        } else {
            self.push(Instruction::Load { dst, loc });
        }
    }

    pub fn create(&mut self, mir: &Mir) {
        let u = mir.code.clone();

        for ins in u.iter() {
            match *ins {
                Instruction::Nop => self.push(Instruction::Nop),
                Instruction::Uni { op, dst, s1 } => {
                    let (dst, s1) = self.subs_uni(dst, s1);
                    self.push(Instruction::Uni { op, dst, s1 });
                }
                Instruction::Bi { op, dst, s1, s2 } => {
                    let (dst, s1, s2) = self.subs_bi(dst, s1, s2);
                    self.push(Instruction::Bi { op, dst, s1, s2 });
                }
                Instruction::LoadConst { dst, idx } => {
                    let dst = self.subs_dst(dst);
                    self.push(Instruction::LoadConst { dst, idx });
                }
                Instruction::Load { dst, loc } => {
                    self.load(dst, loc);
                }
                Instruction::Save { src, loc } => {
                    let t = Instruction::Save {
                        src: self.consume_static(src),
                        loc,
                    };
                    self.push(t);
                }
                Instruction::Mov { dst, s1 } => {
                    let (dst, s1) = self.subs_uni(dst, s1);
                    self.push(Instruction::Mov { dst, s1 });
                }
                Instruction::Call { .. } => {
                    self.push(ins.clone());
                    self.reset();
                }
                Instruction::Fused { op, dst, a, b, c } => {
                    let (dst, a, b, c) = self.subs_tri(dst, a, b, c);
                    self.push(Instruction::Fused { op, dst, a, b, c });
                }
            }
        }
    }

    fn alloc(&self, dst: Reg) -> Reg {
        if let Reg::Static(s) = dst {
            let idx = NodeIndex::new(s as usize);
            self.graph[idx].reg
        } else {
            dst
        }
    }

    fn color(&mut self) -> usize {
        let (coloring, count_colors) = dsatur_coloring(&self.graph);

        for (idx, r) in coloring.iter() {
            self.graph[*idx].reg = Reg::Gen(*r as u8);
        }

        let u = self.code.clone();
        self.code.clear();

        for ins in u.iter() {
            match *ins {
                Instruction::Nop => self.push(Instruction::Nop),
                Instruction::Uni { op, dst, s1 } => self.push(Instruction::Uni {
                    op,
                    dst: self.alloc(dst),
                    s1: self.alloc(s1),
                }),
                Instruction::Bi { op, dst, s1, s2 } => self.push(Instruction::Bi {
                    op,
                    dst: self.alloc(dst),
                    s1: self.alloc(s1),
                    s2: self.alloc(s2),
                }),
                Instruction::LoadConst { dst, idx } => self.push(Instruction::LoadConst {
                    dst: self.alloc(dst),
                    idx,
                }),
                Instruction::Load { dst, loc } => self.push(Instruction::Load {
                    dst: self.alloc(dst),
                    loc,
                }),
                Instruction::Save { src, loc } => self.push(Instruction::Save {
                    src: self.alloc(src),
                    loc,
                }),
                Instruction::Mov { dst, s1 } => self.push(Instruction::Mov {
                    dst: self.alloc(dst),
                    s1: self.alloc(s1),
                }),
                Instruction::Call { .. } => {
                    self.push(ins.clone());
                    self.reset();
                }
                Instruction::Fused { op, dst, a, b, c } => {
                    self.push(Instruction::Fused {
                        op,
                        dst: self.alloc(dst),
                        a: self.alloc(a),
                        b: self.alloc(b),
                        c: self.alloc(c),
                    });
                }
            }
        }

        count_colors
    }
}
