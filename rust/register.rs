use serde::Serialize;
use std::collections::HashMap;

use crate::utils::f64x4;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Word(pub usize, pub usize); // index, version

impl Word {
    pub fn is_temp(&self) -> bool {
        self.1 != 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
// Adjacency tagging (https://serde.rs/enum-representations.html)
#[serde(tag = "t", content = "c")]
pub enum WordType {
    Const(f64),
    Var(String),
    State(String, f64),
    Diff(String),
    Param(String, f64),
    Obs(String),
    Temp,
}

impl WordType {
    pub fn value(&self) -> Option<f64> {
        match self {
            WordType::State(_, val) => Some(*val),
            WordType::Param(_, val) => Some(*val),
            WordType::Const(val) => Some(*val),
            _ => None,
        }
    }
}

// The register file
#[derive(Debug)]
pub struct Frame {
    pub words: Vec<WordType>,
    pub stack: Vec<WordType>,
    pub named: HashMap<String, usize>,
    pub freed: Vec<Word>,
}

impl Frame {
    pub const ZERO: Word = Word(0, 0);
    pub const ONE: Word = Word(1, 0);
    pub const MINUS_ONE: Word = Word(2, 0);
    pub const MINUS_ZERO: Word = Word(3, 0);

    pub fn new() -> Frame {
        let mut f = Frame {
            words: Vec::new(),
            stack: Vec::new(),
            named: HashMap::new(),
            freed: Vec::new(),
        };

        f.alloc(WordType::Const(0.0));
        f.alloc(WordType::Const(1.0));
        f.alloc(WordType::Const(-1.0));
        f.alloc(WordType::Const(-0.0)); // MSB is 1, all other bits are 0, used for negation by xoring

        f
    }

    fn alloc_temp(&mut self) -> Word {
        if let Some(Word(idx, k)) = self.freed.pop() {
            Word(idx, k + 1) // because temps can share the same memory, version
                             // is increased to differentiate different temps
        } else {
            let idx = self.stack.len();
            self.stack.push(WordType::Temp);
            Word(idx, 1)
        }
    }

    pub fn alloc(&mut self, t: WordType) -> Word {
        let idx = self.words.len();

        match &t {
            WordType::Temp => {
                return self.alloc_temp();
            }
            WordType::Const(val) => {
                // reuse constants
                // note: we use a naive search here instead of a HashMap
                // because f64 is not hashable

                for (i, w) in self.words.iter().enumerate() {
                    if let WordType::Const(x) = w {
                        // note the check with -0.0
                        // this is because 0.0 == -0.0, but we need a separate -0.0
                        // for abs and neg functions
                        if *val == *x && *val != -0.0 {
                            return Word(i, 0);
                        }
                    }
                }
            }
            WordType::Var(s) | WordType::State(s, _) | WordType::Param(s, _) | WordType::Obs(s) => {
                if let Some(_x) = self.named.insert(s.clone(), idx) {
                    panic!("key already exists")
                }
            }
            WordType::Diff(s) => {
                if let Some(_x) = self.named.insert(format!("δ{}", s), idx) {
                    panic!("diff key already exists")
                }
            }
        };

        self.words.push(t);
        Word(idx, 0)
    }

    pub fn free(&mut self, r: Word) {
        // only Temp tegisters can be recycled
        //if let WordType::Temp = self.words[r.0] {
        if r.is_temp() {
            self.freed.push(r);
        };
    }

    pub fn is_state(&self, r: &Word) -> bool {
        !r.is_temp() && matches!(self.words[r.0], WordType::State(_, _))
    }

    pub fn is_param(&self, r: &Word) -> bool {
        !r.is_temp() && matches!(self.words[r.0], WordType::Param(_, _))
    }

    pub fn is_diff(&self, r: &Word) -> bool {
        !r.is_temp() && matches!(self.words[r.0], WordType::Diff(_))
    }

    pub fn is_obs(&self, r: &Word) -> bool {
        !r.is_temp() && matches!(self.words[r.0], WordType::Obs(_))
    }

    pub fn should_save(&self, r: &Word) -> bool {
        self.is_diff(r) || self.is_obs(r)
    }

    pub fn find(&self, s: &str) -> Option<Word> {
        self.named.get(s).map(|idx| Word(*idx, 0))
    }

    pub fn find_diff(&self, s: &str) -> Option<Word> {
        let s = format!("δ{}", s);
        self.find(s.as_str())
    }

    pub fn count_states(&self) -> usize {
        (0..self.words.len())
            .filter(|i| self.is_state(&Word(*i, 0)))
            .count()
    }

    pub fn count_params(&self) -> usize {
        (0..self.words.len())
            .filter(|i| self.is_param(&Word(*i, 0)))
            .count()
    }

    pub fn count_obs(&self) -> usize {
        (0..self.words.len())
            .filter(|i| self.is_obs(&Word(*i, 0)))
            .count()
    }

    pub fn count_diffs(&self) -> usize {
        (0..self.words.len())
            .filter(|i| self.is_diff(&Word(*i, 0)))
            .count()
    }

    pub fn first_state(&self) -> Option<usize> {
        (0..self.words.len()).position(|i| self.is_state(&Word(i, 0)))
    }

    pub fn first_param(&self) -> Option<usize> {
        (0..self.words.len()).position(|i| self.is_param(&Word(i, 0)))
    }

    pub fn first_obs(&self) -> Option<usize> {
        (0..self.words.len()).position(|i| self.is_obs(&Word(i, 0)))
    }

    pub fn first_diff(&self) -> Option<usize> {
        (0..self.words.len()).position(|i| self.is_diff(&Word(i, 0)))
    }

    pub fn mem(&self) -> Vec<f64> {
        self.words
            .iter()
            .map(|x| x.value().unwrap_or(0.0))
            .collect::<Vec<f64>>()
    }

    pub fn mem_simd(&self) -> Vec<f64x4> {
        self.mem()
            .iter()
            .map(|x| f64x4::splat(*x))
            .collect::<Vec<f64x4>>()
    }

    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }
}
