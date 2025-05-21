use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum Loc {
    Stack(u32),
    Mem(u32),
}

#[derive(Debug, Clone)]
struct Symbol {
    name: String,
    loc: Loc,
    reg: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub syms: HashMap<String, Symbol>,
    pub num_stack: usize,
    pub num_mem: usize,
}

impl SymbolTable {
    const SPILL_AREA: usize = 16;

    pub fn new() -> SymbolTable {
        let mut s = SymbolTable {
            syms: HashMap::new(),
            num_stack: 0,
            num_mem: 0,
        };

        for i in 0..SymbolTable::SPILL_AREA {
            s.add_stack(&format!("μ{}", i));
        }

        s
    }

    pub fn add_mem(&mut self, name: &str) -> Loc {
        match self.find(name) {
            Some(loc) => loc,
            None => {
                let loc = Loc::Mem(self.num_mem as u32);
                self.num_mem += 1;
                let sym = Symbol {
                    name: name.to_string(),
                    loc,
                    reg: None,
                };
                self.syms.insert(name.to_string(), sym);
                loc
            }
        }
    }

    pub fn add_stack(&mut self, name: &str) -> Loc {
        match self.find(name) {
            Some(loc) => loc,
            None => {
                let loc = Loc::Stack(self.num_stack as u32);
                self.num_stack += 1;
                let sym = Symbol {
                    name: name.to_string(),
                    loc,
                    reg: None,
                };
                self.syms.insert(name.to_string(), sym);
                loc
            }
        }
    }

    pub fn find(&self, name: &str) -> Option<Loc> {
        match self.syms.get(name) {
            Some(sym) => Some(sym.loc),
            None => None,
        }
    }
}
