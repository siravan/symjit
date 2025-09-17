use super::asm::Amd;

#[allow(dead_code)]
impl Amd {
    // sets W bit of vex3 to 1
    // should be called immediately after a vex-prefix instruction
    fn set_w1(&mut self) {
        let b = self.a.buf.pop().unwrap();
        self.a.buf.push(b | 0x80);
    }

    fn vfma(&mut self, reg: u8, vreg: u8, rm: u8, code: u8) {
        self.vex3pd(reg, vreg, rm, 0, 2);
        self.set_w1();
        self.append_byte(code);
        self.modrm_reg(reg, rm);
    }

    // reg = reg * rm + vreg
    pub fn vfmadd132sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0x99);
    }

    // reg = vreg * reg + rm
    pub fn vfmadd213sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xa9);
    }

    // reg = vreg * rm + reg
    pub fn vfmadd231sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xb9);
    }

    // reg = reg * rm - vreg
    pub fn vfmsub132sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0x9b);
    }

    // reg = vreg * reg - rm
    pub fn vfmsub213sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xab);
    }

    // reg = vreg * rm - reg
    pub fn vfmsub231sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xbb);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmadd132sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0x9d);
    }

    // reg = - vreg * reg + rm
    pub fn vfnmadd213sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xad);
    }

    // reg = - vreg * rm + reg
    pub fn vfnmadd231sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xbd);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmsub132sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0x9f);
    }

    // reg = - vreg * reg - rm
    pub fn vfnmsub213sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xaf);
    }

    // reg = - vreg * rm - reg
    pub fn vfnmsub231sd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xbf);
    }

    // reg = reg * rm + vreg
    pub fn vfmadd132pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0x98);
    }

    // reg = vreg * reg + rm
    pub fn vfmadd213pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xa8);
    }

    // reg = vreg * rm + reg
    pub fn vfmadd231pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xb8);
    }

    // reg = reg * rm - vreg
    pub fn vfmsub132pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xa9);
    }

    // reg = vreg * reg - rm
    pub fn vfmsub213pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xaa);
    }

    // reg = vreg * rm - reg
    pub fn vfmsub231pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xba);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmadd132pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0x9c);
    }

    // reg = - vreg * reg + rm
    pub fn vfnmadd213pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xac);
    }

    // reg = - vreg * rm + reg
    pub fn vfnmadd231pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xbc);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmsub132pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0x9e);
    }

    // reg = - vreg * reg - rm
    pub fn vfnmsub213pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xae);
    }

    // reg = - vreg * rm - reg
    pub fn vfnmsub231pd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma(reg, vreg, rm, 0xbe);
    }
}
