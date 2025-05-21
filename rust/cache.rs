use crate::generator::Generator;

pub enum Desc {
    Empty,
    Stack(idx: u32),
    Mem(idx: u32),
};

pub struct Cache {
    pub regs: Vec<Desc>
}

impl Cache {
    pub fn new(num_regs: usize) {
        Cache {
            regs: vec![Desc::Empty; num_regs],
        }
    }
    
    pub fn flush(&mut self, ir: &mut dyn Generator, reg: u8) {
        match self.regs[reg] {
            Desc::Stack(k) => {
                ir.save_stack(reg, *k);                       
            },            
            _ => {}
        };
        self.regs[reg] = Desc::Empty;    
    }
    
    pub fn flush_all(&mut self, ir: &mut dyn Generator) {
        for r in 0..self.regs.len() {
            self.flush(ir, r);
        }
    }
    
    pub fn save_stack(&mut self, ir: &mut dyn Generator, reg: u8, idx: u32) {
        self.regs[reg] = Desc::Stack(idx);
    }
    
    pub fn load_stack(&mut self, ir: &mut dyn Generator, reg: u8, idx: u32) {
        if let self.regs[reg] = Desc::Stack(k) && k == idx {
            self.regs[reg] = Desc::Empty;
            return;
        }
    
        self.flush(ir, reg);
        
        for r in 0..self.regs.len() {
            if let self.regs[r] = Desc::Stack(k) && k == idx {
                ir.fmov(reg, r);    // reg != r
                self.regs[r] = Desc::Empty;
                return;
            }
        };
        
        ir.load_stack(reg, idx);
        self.regs[reg] = Desc::Stack(idx);
    }
    
    pub fn save_mem(&mut self, ir: &mut dyn Generator, reg: u8, idx: u32) {        
        ir.save_mem(reg, idx);
        self.regs[reg] = Desc::Mem(idx);
    }
    
    pub fn load_mem(&mut self, ir: &mut dyn Generator, reg: u8, idx: u32) {
        self.flush(ir, reg);
        
        for r in 0..self.regs.len() {
            if let self.regs[r] = Desc::Mem(k) && k == idx {
                if reg != r {
                    ir.fmov(reg, r);                     
                };                
                return;
            }
        };
        
        ir.load_mem(reg, idx);
        self.regs[reg] = Desc::Mem(idx);
    }
}
