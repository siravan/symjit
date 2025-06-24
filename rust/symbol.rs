use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub enum Loc {
    Stack(u32),
    Mem(u32),
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub loc: Loc,
    pub visited: bool,
    pub reg: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub syms: HashMap<String, Rc<RefCell<Symbol>>>,
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

    pub fn add_mem(&mut self, name: &str) {
        if self.find_sym(name).is_none() {
            let loc = Loc::Mem(self.num_mem as u32);
            self.num_mem += 1;
            let sym = Rc::new(RefCell::new(Symbol {
                name: name.to_string(),
                loc,
                visited: false,
                reg: None,
            }));
            self.syms.insert(name.to_string(), sym);
        }
    }

    pub fn add_stack(&mut self, name: &str) {
        if self.find_sym(name).is_none() {
            let loc = Loc::Stack(self.num_stack as u32);
            self.num_stack += 1;
            let sym = Rc::new(RefCell::new(Symbol {
                name: name.to_string(),
                loc,
                visited: false,
                reg: None,
            }));
            self.syms.insert(name.to_string(), sym);
        }
    }

    pub fn find_sym(&self, name: &str) -> Option<Rc<RefCell<Symbol>>> {
        self.syms.get(name).map(Rc::clone)
    }
}
