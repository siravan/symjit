use std::collections::{HashMap, HashSet};

use super::code::Instruction;
use super::model::Program;
use super::register::Word;

pub enum Event {
    Producer(Word),
    Consumer(Word),
    Caller(String),
}

pub struct Analyzer {
    pub events: Vec<Event>,
}

impl Analyzer {
    pub fn new(prog: &Program) -> Self {
        let mut events: Vec<Event> = Vec::new();

        for c in prog.code.iter() {
            match c {
                Instruction::Unary { op, x, dst, .. } => {
                    events.push(Event::Consumer(*x));
                    events.push(Event::Caller(op.clone()));
                    events.push(Event::Producer(*dst));
                }
                Instruction::Binary { op, x, y, dst, .. } => {
                    events.push(Event::Consumer(*y));
                    events.push(Event::Consumer(*x));
                    events.push(Event::Caller(op.clone()));
                    events.push(Event::Producer(*dst));
                }
                Instruction::IfElse { x1, x2, cond, dst } => {
                    events.push(Event::Consumer(*cond));
                    events.push(Event::Consumer(*x2));
                    events.push(Event::Consumer(*x1));
                    events.push(Event::Caller("select".to_string()));
                    events.push(Event::Producer(*dst));
                }
                _ => {}
            }
        }

        Self { events }
    }

    /*
        A saveable register is produced but is not consumed immediately
        In other words, it cannot be coalesced over consecuative instructions
    */
    pub fn find_saveable(&self) -> HashSet<Word> {
        let mut candidates: Vec<Word> = Vec::new();
        let mut saveable: HashSet<Word> = HashSet::new();

        for l in self.events.iter() {
            match l {
                Event::Producer(p) => {
                    candidates.push(*p);
                }
                Event::Consumer(c) => {
                    let r = candidates.pop();

                    if candidates.contains(c) {
                        saveable.insert(*c);
                    };

                    if let Some(r) = r {
                        candidates.push(r);
                    };
                }
                Event::Caller(_) => {}
            }
        }

        saveable
    }

    pub fn alloc_regs(&self) -> HashMap<Word, u8> {
        let caller = [
            "rem", "power", "exp", "ln", "log", "sin", "cos", "tan", "csc", "sec", "cot", "sinh",
            "cosh", "tanh", "csch", "sech", "coth", "arcsin", "arccos", "arctan", "arcsinh",
            "arccosh", "arctanh",
        ];

        let mut allocs: HashMap<Word, u8> = HashMap::new();
        let mut lives: Vec<Word> = Vec::new();
        let mut depth: usize = 0;

        for l in self.events.iter() {
            match l {
                Event::Producer(p) => {
                    if p.is_temp() {
                        lives.push(*p);
                        depth = depth.max(lives.len());
                    }
                }
                Event::Consumer(c) => {
                    if c.is_temp() {
                        if let Some(r) = lives.pop() {
                            if r != *c {
                                panic!("temps out of stack order");
                            }

                            //allocs.insert(*c, (depth - lives.len() - 1) as u8);
                            allocs.insert(*c, lives.len() as u8);
                        }
                    }
                }
                Event::Caller(op) => {
                    if caller.contains(&op.as_str()) {
                        lives.clear();
                        depth = 0;
                    }
                }
            }
        }

        allocs
    }
}

/*********************************************/

#[derive(Debug)]
pub struct Stack {
    stack: Vec<Word>,
    cap: usize,
}

impl Stack {
    pub fn new() -> Stack {
        Stack {
            stack: Vec::new(),
            cap: 0,
        }
    }

    pub fn push(&mut self, w: &Word) -> usize {
        self.stack.push(*w);
        self.cap = usize::max(self.cap, self.stack.len());
        self.stack.len() - 1
    }

    pub fn pop(&mut self, w: &Word) -> usize {
        let p = self.stack.pop().expect("stack is empty");
        assert!(*w == p);
        self.stack.len()
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }
}
