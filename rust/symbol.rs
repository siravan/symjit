use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Loc {
    Stack(u32),
    Mem(u32),
    Param(u32),
}

impl fmt::Debug for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Loc::Mem(idx) => write!(f, "Mem[{}]", idx),
            Loc::Stack(idx) => write!(f, "Stack[{}]", idx),
            Loc::Param(idx) => write!(f, "Param[{}]", idx),
        }
    }
}

#[derive(Clone)]
pub struct Symbol {
    pub _name: String,
    pub loc: Loc,
    pub visited: bool,
    pub reg: Option<u8>,
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.loc {
            Loc::Stack(idx) => write!(f, "{} in Stack[{}]", self._name, idx),
            Loc::Param(idx) => write!(f, "{} in Param[{}]", self._name, idx),
            Loc::Mem(idx) => write!(f, "{} in Mem[{}]", self._name, idx),
        }
    }
}

impl Hash for Symbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._name.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub syms: HashMap<String, Rc<RefCell<Symbol>>>,
    pub num_stack: usize,
    pub num_mem: usize,
    pub num_param: usize,
}

impl SymbolTable {
    const SPILL_AREA: usize = 16;

    pub fn new() -> SymbolTable {
        let mut s = SymbolTable {
            syms: HashMap::new(),
            num_stack: 0,
            num_mem: 0,
            num_param: 0,
        };

        /*
           The spill area has two functions:

           1, As storage area for fast functions to store arguments
               passed in registers (up to 4 in Windows and 8 otherwise).
           2. To preserve registers XMM6-XMM15 in Windows (if needed).
        */
        for i in 0..SymbolTable::SPILL_AREA {
            s.add_stack(&format!("μ{}", i));
        }

        s
    }

    fn add_sym(&mut self, name: &str, loc: Loc) {
        let sym = Rc::new(RefCell::new(Symbol {
            _name: name.to_string(),
            loc,
            visited: false,
            reg: None,
        }));
        self.syms.insert(name.to_string(), sym);
    }

    pub fn add_mem(&mut self, name: &str) {
        if self.find_sym(name).is_none() {
            let loc = Loc::Mem(self.num_mem as u32);
            self.num_mem += 1;
            self.add_sym(name, loc);
        }
    }

    pub fn add_param(&mut self, name: &str) {
        if self.find_sym(name).is_none() {
            let loc = Loc::Param(self.num_param as u32);
            self.num_param += 1;
            self.add_sym(name, loc);
        }
    }

    pub fn add_stack(&mut self, name: &str) {
        if self.find_sym(name).is_none() {
            let loc = Loc::Stack(self.num_stack as u32);
            self.num_stack += 1;
            self.add_sym(name, loc);
        }
    }

    pub fn find_sym(&self, name: &str) -> Option<Rc<RefCell<Symbol>>> {
        self.syms.get(name).map(Rc::clone)
    }
}
