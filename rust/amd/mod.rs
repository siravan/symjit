mod asm;

use asm::Amd;

use super::machine::MachineCode;
use super::model::Program;
use super::utils::*;
use crate::code::{BinaryFunc, VirtualTable};

pub struct AmdCompiler {
    amd: Amd,
}

impl Compiler<MachineCode<f64>> for AmdCompiler {
    fn compile(&mut self, prog: &Program) -> MachineCode<f64> {
        MachineCode::new(
            "x86_64",
            self.bytes(),
            VirtualTable::<f64>::from_names(&prog.ft),
            prog.frame.mem(),
        )
    }
}

impl AmdCompiler {
    pub fn new() -> AmdCompiler {
        AmdCompiler { amd: Amd::new() }
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.amd.bytes()
    }

    pub fn fmov(&mut self, dst: u8, r: u8) {
        self.amd.vmovapd(dst, r);
    }
    
    pub fn load_const(&mut self, dst: u8, idx: u32) {
        let label = format!("_const_{}_", idx);
        self.amd.vmovsd_xmm_label(dst, label.as_str());
    }

    pub fn load_mem(&mut self, dst: u8, idx: u32) {
        let offset = 8 * idx as i32;
        self.amd.vmovsd_xmm_mem(dst, Amd::RBP, offset);
    }    

    pub fn save_mem(&mut self, src: u8, idx: u32) {
        let offset = 8 * idx as i32;
        self.amd.vmovsd_mem_xmm(Amd::RBP, offset, src);
    }
    
    pub fn load_stack(&mut self, dst: u8, idx: u32) {
        let offset = 8 * idx as i32;
        self.amd.vmovsd_xmm_mem(dst, Amd::RSP, offset);
    }    

    pub fn save_stack(&mut self, src: u8, idx: u32) {
        let offset = 8 * idx as i32;
        self.amd.vmovsd_mem_xmm(Amd::RSP, offset, src);
    }

    pub fn neg(&mut self, dst: u8) {
        self.amd.vmovsd_xmm_label(1, "_minus_zero_");
        self.amd.vxorpd(dst, dst, 1);
    }

    pub fn abs(&mut self, dst: u8) {
        self.amd.vmovsd_xmm_label(1, "_minus_zero_");
        self.amd.vandnpd(dst, 1, dst);
    }

    pub fn root(&mut self, dst: u8) {
        self.amd.vsqrtsd(dst, dst);
    }

    pub fn square(&mut self, dst: u8) {
        self.amd.vmulsd(dst, dst, dst);
    }

    pub fn cube(&mut self, dst: u8) {
        self.amd.vmulsd(1, dst, dst);
        self.amd.vmulsd(dst, dst, 1);
    }

    pub fn recip(&mut self, dst: u8) {
        self.amd.vmovsd_xmm_label(1, "_one_");
        self.amd.vdivsd(dst, 1, dst);
    }

    pub fn plus(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vaddsd(dst, a, b);
    }

    pub fn minus(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vsubsd(dst, a, b);
    }

    pub fn times(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vmulsd(dst, a, b);
    }

    pub fn divide(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vdivsd(dst, a, b);
    }

    pub fn gt(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vcmpnlesd(dst, a, b);
    }

    pub fn geq(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vcmpnltsd(dst, a, b);
    }

    pub fn lt(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vcmpltsd(dst, a, b);
    }

    pub fn leq(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vcmplesd(dst, a, b);
    }

    pub fn eq(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vcmpeqsd(dst, a, b);
    }

    pub fn neq(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vcmpneqsd(dst, a, b);
    }

    pub fn and(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vandpd(dst, a, b);
    }

    pub fn or(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vorpd(dst, a, b);
    }

    pub fn xor(&mut self, dst: u8, a: u8, b: u8) {
        self.amd.vxorpd(dst, a, b);
    }

    pub fn not(&mut self, dst: u8) {
        self.amd.vxorpd(1, 1, 1);
        self.amd.vcmpeqsd(1, 1, 1);
        self.amd.vxorpd(dst, dst, 1);
    }

    pub fn call_unary(&mut self, dst: u8, idx: u32) {
        self.amd.vzeroupper();

        // Windows 32-byte home area
        #[cfg(target_family = "windows")]
        self.amd.sub_rsp(32);

        let label = format!("_func_{}_", idx);
        self.amd.mov_reg_label(Amd::RAX, label.as_str());
        self.amd.call(Amd::RAX);

        if dst != 0 {
            self.amd.vmovapd(dst, 0);
        }

        #[cfg(target_family = "windows")]
        self.amd.add_rsp(32);
    }

    pub fn call_binary(&mut self, dst: u8, r: u8, idx: u32) {
        self.amd.vzeroupper();

        // Windows 32-byte home area
        #[cfg(target_family = "windows")]
        self.amd.sub_rsp(32);

        if r != 1 {
            self.amd.vmovapd(1, r);
        }

        let label = format!("_func_{}_", idx);
        self.amd.mov_reg_label(Amd::RAX, label.as_str());
        self.amd.call(Amd::RAX);

        if dst != 0 {
            self.amd.vmovapd(dst, 0);
        }

        #[cfg(target_family = "windows")]
        self.amd.add_rsp(32);
    }

    pub fn set_label(&mut self, label: &str) {
        self.amd.a.set_label(label);
    }

    pub fn branch(&mut self, label: &str) {
        self.amd.jmp(label);
    }

    pub fn branch_if(&mut self, cond: u8, true_label: &str) {
        self.amd.vucomisd(cond, cond);
        self.amd.jpe(true_label);
    }

    pub fn branch_if_else(&mut self, cond: u8, true_label: &str, false_label: &str) {
        self.amd.vucomisd(cond, cond);
        self.amd.jpe(true_label);
        self.amd.jmp(false_label);
    }

    pub fn select_if(&mut self, dst: u8, cond: u8, a: u8) {
        self.amd.vandpd(dst, cond, a);
    }

    pub fn select_else(&mut self, dst: u8, cond: u8, a: u8) {
        self.amd.vandnpd(dst, cond, a);
    }

    #[cfg(target_family = "unix")]
    pub fn prologue(&mut self, n: u32) {
        self.amd.push(Amd::RBP);
        self.amd.push(Amd::RBX);
        self.amd.mov(Amd::RBP, Amd::RDI);
        self.amd.mov(Amd::RBX, Amd::RDX);
        self.amd.sub_rsp(n);
    }

    #[cfg(target_family = "windows")]
    pub fn prologue(&mut self, n: u32) {
        self.amd.mov_mem_reg(Amd::RSP, 0x08, Amd::RBP);
        self.amd.mov_mem_reg(Amd::RSP, 0x10, Amd::RBP);
        self.amd.mov(Amd::RBP, Amd::RCX);
        self.amd.mov(Amd::RBX, Amd::R8);
        self.amd.sub_rsp(n);
    }

    #[cfg(target_family = "unix")]
    pub fn epilogue(&mut self, n: u32) {
        self.amd.add_rsp(n);
        self.amd.pop(Amd::RBX);
        self.amd.pop(Amd::RBP);
        self.amd.ret();
    }

    #[cfg(target_family = "windows")]
    pub fn epilogue(&mut self, n: u32) {
        self.amd.add_rsp(n);
        self.amd.mov(Amd::RBX, Amd::RSP, 0x10);
        self.amd.mov(Amd::RBP, Amd::RSP, 0x08);
        self.amd.ret();
    }

    fn align(&mut self) {
        let mut n = self.amd.a.ip();

        while (n & 7) != 0 {
            self.amd.nop();
            n += 1
        }
    }

    pub fn append_const_section(&mut self, p: &[f64]) {
        self.align();

        self.set_label("_minus_zero_");
        let u: u64 = unsafe { std::mem::transmute(-0.0f64) };
        self.amd.quad(u);

        self.set_label("_one_");
        let u: u64 = unsafe { std::mem::transmute(1.0f64) };
        self.amd.quad(u);

        for (idx, val) in p.iter().enumerate() {
            let label = format!("_const_{}_", idx);
            self.set_label(label.as_str());
            let u: u64 = unsafe { std::mem::transmute(val) };
            self.amd.quad(u);
        }
    }

    pub fn append_vt_section(&mut self, vt: &[BinaryFunc<f64>]) {
        self.align();

        for (idx, f) in vt.iter().enumerate() {
            let label = format!("_func_{}_", idx);
            self.set_label(label.as_str());
            let u: u64 = unsafe { std::mem::transmute(f) };
            self.amd.quad(u);
        }
    }
}
