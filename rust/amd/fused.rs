use super::asm::Amd;
use super::Prefix;

#[allow(dead_code)]
impl Amd {
    fn vfma(&mut self, reg: u8, vreg: u8, rm: u8, code: u8) {
        self.vex3pd_w1(reg, vreg, rm, 0, 2);
        self.append_byte(code);
        self.modrm_reg(reg, rm);
    }

    fn vfma_dd(&mut self, reg: u8, vreg: u8, rm: u8, code: u8) {
        self.vex3dd_w1(reg, vreg, rm, 0, 2);
        self.append_byte(code);
        self.modrm_reg(reg, rm);
    }

    fn vfma_qd(&mut self, reg: u8, vreg: u8, rm: u8, code: u8) {
        Prefix::new(reg, vreg, rm)
            .set_encoding(2)
            .set_w(1)
            .evex(self);
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
        self.vfma(reg, vreg, rm, 0x9a);
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

    // reg = reg * rm + vreg
    pub fn vfmadd132qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0x98);
    }

    // reg = vreg * reg + rm
    pub fn vfmadd213qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xa8);
    }

    // reg = vreg * rm + reg
    pub fn vfmadd231qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xb8);
    }

    // reg = reg * rm - vreg
    pub fn vfmsub132qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0x9a);
    }

    // reg = vreg * reg - rm
    pub fn vfmsub213qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xaa);
    }

    // reg = vreg * rm - reg
    pub fn vfmsub231qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xba);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmadd132qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0x9c);
    }

    // reg = - vreg * reg + rm
    pub fn vfnmadd213qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xac);
    }

    // reg = - vreg * rm + reg
    pub fn vfnmadd231qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xbc);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmsub132qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0x9e);
    }

    // reg = - vreg * reg - rm
    pub fn vfnmsub213qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xae);
    }

    // reg = - vreg * rm - reg
    pub fn vfnmsub231qd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_qd(reg, vreg, rm, 0xbe);
    }

    // reg = reg * rm + vreg
    pub fn vfmadd132dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0x98);
    }

    // reg = vreg * reg + rm
    pub fn vfmadd213dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xa8);
    }

    // reg = vreg * rm + reg
    pub fn vfmadd231dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xb8);
    }

    // reg = reg * rm - vreg
    pub fn vfmsub132dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0x9a);
    }

    // reg = vreg * reg - rm
    pub fn vfmsub213dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xaa);
    }

    // reg = vreg * rm - reg
    pub fn vfmsub231dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xba);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmadd132dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0x9c);
    }

    // reg = - vreg * reg + rm
    pub fn vfnmadd213dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xac);
    }

    // reg = - vreg * rm + reg
    pub fn vfnmadd231dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xbc);
    }

    // reg = - reg * rm - vreg
    pub fn vfnmsub132dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0x9e);
    }

    // reg = - vreg * reg - rm
    pub fn vfnmsub213dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xae);
    }

    // reg = - vreg * rm - reg
    pub fn vfnmsub231dd(&mut self, reg: u8, vreg: u8, rm: u8) {
        self.vfma_dd(reg, vreg, rm, 0xbe);
    }
}
