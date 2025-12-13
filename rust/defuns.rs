use crate::code::{BinaryFunc, Func, UnaryFunc};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Defuns {
    pub funcs: HashMap<String, Func>,
}

impl Defuns {
    pub fn new() -> Defuns {
        Defuns {
            funcs: HashMap::new(),
        }
    }

    pub fn add_func(&mut self, name: &str, p: *const usize, num_args: usize) {
        match num_args {
            1 => {
                let f: UnaryFunc = unsafe { std::mem::transmute(p) };
                self.funcs.insert(name.to_string(), Func::Unary(f));
            }
            2 => {
                let f: BinaryFunc = unsafe { std::mem::transmute(p) };
                self.funcs.insert(name.to_string(), Func::Binary(f));
            }
            _ => {
                panic!("only unary and binary functions are supported")
            }
        }
    }

    pub fn add_unary(&mut self, name: &str, f: UnaryFunc) {
        self.funcs.insert(name.to_string(), Func::Unary(f));
    }

    pub fn add_binary(&mut self, name: &str, f: BinaryFunc) {
        self.funcs.insert(name.to_string(), Func::Binary(f));
    }

    pub fn len(&self) -> usize {
        self.funcs.len()
    }
}
