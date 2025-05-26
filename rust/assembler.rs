use std::collections::HashMap;

pub struct Assembler {
    buf: Vec<u8>,
    labels: HashMap<String, usize>,
    jumps: Vec<(String, usize, u32)>,
    delta: isize,
    shift: isize,
}

impl Assembler {
    pub fn new(delta: isize, shift: isize) -> Assembler {
        Assembler {
            buf: Vec::new(),
            labels: HashMap::new(),
            jumps: Vec::new(),
            delta,
            shift,
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.buf.clone()
    }

    pub fn append_byte(&mut self, b: u8) {
        self.buf.push(b)
    }

    pub fn append_bytes(&mut self, bs: &[u8]) {
        for b in bs {
            self.append_byte(*b);
        }
    }

    pub fn append_word(&mut self, mut u: u32) {
        // appends u (uint32) as little-endian
        for _ in 0..4 {
            self.append_byte((u & 0xff) as u8);
            u >>= 8;
        }
    }

    pub fn append_quad(&mut self, mut u: u64) {
        // appends u (uint32) as little-endian
        for _ in 0..8 {
            self.append_byte((u & 0xff) as u8);
            u >>= 8;
        }
    }

    pub fn ip(&self) -> usize {
        self.buf.len()
    }

    pub fn set_label(&mut self, label: &str) {
        self.labels.insert(label.to_string(), self.ip());
    }

    pub fn jump(&mut self, label: &str, code: u32) {
        self.jumps.push((label.to_string(), self.ip(), code));
        self.append_word(code);
    }

    pub fn apply_jumps(&mut self) {
        for (label, k, code) in self.jumps.iter() {
            let target = self.labels.get(label).expect("label not found");
            let offset = (*target as isize) - (*k as isize) + self.delta;
            let x = ((offset as u32) << self.shift) | *code;

            self.buf[*k] |= (x & 0xff) as u8;
            self.buf[*k + 1] |= ((x >> 8) & 0xff) as u8;
            self.buf[*k + 2] |= ((x >> 16) & 0xff) as u8;
            self.buf[*k + 3] |= ((x >> 24) & 0xff) as u8;
        }
    }
}
