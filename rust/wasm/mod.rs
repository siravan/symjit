// WebAssembly backend: compiles MIR into a self-contained WebAssembly module.
//
// The module is meant to be saved and run by a WebAssembly host (a browser,
// Node, wasmtime, ...), not executed in-process; on the symjit side, calls to a
// `ty = "wasm"` function run on the bytecode interpreter.
//
// Module interface (see docs/WASM.md):
//   (import "symjit" "<op>" (func (param f64 [f64]) (result f64)))  real functions
//   (import "symjit" "cplx_<op>"
//           (func (param f64 f64 [f64 f64]) (result f64 f64)))         complex functions
//   (memory (export "memory") n)
//   (global (export "__heap_base") i32)   first byte the host may use for its data
//   (func (export "run") (param $mem i32) (param $states i32) (param $idx i32)
//                        (param $params i32) (result i32))
//
// `run` has the same two calling modes as the native kernels:
//   direct:   run(mem, 0, 0, params); `mem` points to the model memory
//             (states, then observables, then diffs, as f64).
//   indirect: run(0, states, idx, params); `states` points to an array of
//             (ptr: i32, len: i32) pairs, one per state followed by one per
//             observable (the wasm32 layout of `&mut [f64]`). Element `idx` of
//             each state array is read and element `idx` of each observable
//             array is written.
// It returns 0 on success and 1 if the stack region is exhausted.
//
// Models without parameters, with a single observable and at most 1000 states export
//   (func (export "fast") (param $x0 f64) ... (result f64))
// which evaluates the model at one point (NaN if the stack is exhausted).
//
// Registers are WebAssembly locals; stack slots live in linear memory below
// `__heap_base`, addressed through a mutable stack-pointer global. Labels and jumps
// are structured by flow.rs; recursive calls (`@self`) call `run`.

mod encoder;
mod flow;
#[cfg(test)]
mod generator_tests;
#[cfg(test)]
mod tests;

use anyhow::{anyhow, Result};

use super::code::Func;
use super::config::{Config, ABI_AREA};
use super::generator::{Generator, GeneratorType};
use super::symbol::Loc;
use super::utils::{align_stack, Reg};

use encoder::{
    BlockType, Export, ExportKind, FuncType, Function, Global, Import, MemArg, Module, Op,
    ValType,
};
use flow::Flow;

// parameters of `run`
const L_MEM: u32 = 0;
const L_STATES: u32 = 1;
const L_IDX: u32 = 2;
const L_PARAMS: u32 = 3;
// i32 locals
const L_STACK: u32 = 4;
const L_SAVED_SP: u32 = 5;
const L_I: u32 = 6;
const L_PC: u32 = 7; // block to continue at (control-flow dispatcher)
const L_RET_PC: u32 = 8; // block to return to from a subroutine
const COUNT_I32_LOCALS: u32 = 5;
// f64 locals
const L_RET: u32 = 9;
const L_TEMP: u32 = 10;
const L_GEN0: u32 = 11;

// globals
const G_SP: u32 = 0;
const G_HEAP_BASE: u32 = 1;

const PAGE: u32 = 65536;
// the stack region (below `__heap_base`) is at most this large, unless two frames
// need more; a frame needing more than MAX_FRAME is rejected
const MAX_STACK: u32 = 64 << 20;
const MAX_FRAME: u32 = 512 << 20;
// implementation limits of the WebAssembly JavaScript API, enforced by browsers and
// Node (https://webassembly.github.io/spec/js-api/#limits)
const MAX_FUNCTION_SIZE: usize = 7_654_321;
const MAX_PARAMS: u32 = 1000;
// states/observables copied with a loop instead of unrolled above this count
const UNROLL_LIMIT: usize = 16;

pub const IMPORT_MODULE: &str = "symjit";

enum Item {
    Op(Op),
    /// `f64.const consts[idx]`; constants arrive after the code (`add_consts`)
    Const(u32),
    /// `call` to an imported function; indices are assigned in `seal`
    CallImport(String),
    /// recursive `call` of `run` (`@self`)
    CallSelf,
    Label(String),
    Branch(String),
    BranchIf {
        cond: u32,
        label: String,
        is_else: bool,
    },
    CallFunclet(String),
    Ret,
}

struct ImportedFunc {
    name: String,
    arity: usize,
    complex: bool,
}

pub struct WasmGenerator {
    config: Config,
    body: Vec<Item>,
    consts: Vec<f64>,
    imports: Vec<ImportedFunc>, // in order of first use
    recursive: bool,
    count_gen: u32,
    frame_bytes: u32,
    count_states: u32,
    count_obs: u32,
    mem_size: u32,
    fast_mem_size: Option<u32>,
    unsupported: Vec<String>,
    module: Vec<u8>,
}

impl WasmGenerator {
    pub fn new(config: Config) -> WasmGenerator {
        WasmGenerator {
            config,
            body: Vec::new(),
            consts: Vec::new(),
            imports: Vec::new(),
            recursive: false,
            count_gen: 0,
            frame_bytes: 0,
            count_states: 0,
            count_obs: 0,
            mem_size: 0,
            fast_mem_size: None,
            unsupported: Vec::new(),
            module: Vec::new(),
        }
    }

    /// Fails if the compiled MIR used an operation this backend cannot lower yet.
    pub fn check(&self) -> Result<()> {
        if self.unsupported.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(
                "the WebAssembly backend does not support {} yet",
                self.unsupported.join(", ")
            ))
        }
    }

    fn unsupported(&mut self, what: &str) {
        if !self.unsupported.iter().any(|s| s == what) {
            self.unsupported.push(what.to_string());
        }
        self.op(Op::Unreachable);
    }

    fn op(&mut self, op: Op) {
        self.body.push(Item::Op(op));
    }

    /// Also export `fast(x0, ..., x[n-1]) -> f64`, a wrapper that calls `run` in
    /// direct mode on a model memory of `mem_size` values and returns the first
    /// observable. Only valid for models without parameters.
    pub fn export_fast(&mut self, mem_size: usize) {
        self.fast_mem_size = Some(mem_size as u32);
    }

    /// The size of the model memory in values (states, observables, diffs, ...),
    /// allocated by `run` in indirect mode; at least states + observables.
    pub fn set_mem_size(&mut self, mem_size: usize) {
        self.mem_size = mem_size as u32;
    }

    fn add_import(&mut self, op: &str, arity: usize, complex: bool) -> Result<()> {
        match self.imports.iter().find(|f| f.name == op) {
            Some(f) if f.arity != arity || f.complex != complex => Err(anyhow!(
                "function `{}` called with {} and {} arguments",
                op,
                f.arity,
                arity
            )),
            Some(_) => Ok(()),
            None => {
                self.imports.push(ImportedFunc {
                    name: op.to_string(),
                    arity,
                    complex,
                });
                Ok(())
            }
        }
    }

    // A recursive call, as in the native backends: the arguments are in the stack
    // slots from ABI_AREA on (`__Arg*`), the callee runs in direct mode with its
    // model memory at the start of our stack frame and returns in Stack[0] (and
    // Stack[1] for complex values). A stack overflow in the callee is passed on.
    fn call_self(&mut self) -> Result<()> {
        self.recursive = true;
        self.op(Op::LocalGet(L_STACK));
        self.op(Op::I32Const(0));
        self.op(Op::I32Const(0));
        self.op(Op::LocalGet(L_STACK));
        self.op(Op::I32Const(8 * ABI_AREA as i32));
        self.op(Op::I32Add);
        self.body.push(Item::CallSelf);
        self.op(Op::If(BlockType::Empty));
        self.op(Op::LocalGet(L_SAVED_SP));
        self.op(Op::GlobalSet(G_SP));
        self.op(Op::I32Const(1));
        self.op(Op::Return);
        self.op(Op::End);

        self.load(Reg::Ret, L_STACK, 0);
        if self.config.is_complex() {
            self.load(Reg::Temp, L_STACK, 1);
        }
        Ok(())
    }

    fn get(&mut self, r: Reg) {
        let l = self.local(r);
        self.op(Op::LocalGet(l));
    }

    fn set(&mut self, r: Reg) {
        let l = self.local(r);
        self.op(Op::LocalSet(l));
    }

    // pushes the bits of `r` as an i64
    fn get_bits(&mut self, r: Reg) {
        self.get(r);
        self.op(Op::I64ReinterpretF64);
    }

    fn set_bits(&mut self, r: Reg) {
        self.op(Op::F64ReinterpretI64);
        self.set(r);
    }

    fn unary(&mut self, dst: Reg, s1: Reg, op: Op) {
        self.get(s1);
        self.op(op);
        self.set(dst);
    }

    fn binary(&mut self, dst: Reg, s1: Reg, s2: Reg, op: Op) {
        self.get(s1);
        self.get(s2);
        self.op(op);
        self.set(dst);
    }

    fn set_const(&mut self, dst: Reg, val: f64) {
        self.op(Op::F64Const(val));
        self.set(dst);
    }

    // comparisons produce all-ones (true) or all-zeros (false) masks, as the
    // native backends and the bytecode interpreter do
    fn compare(&mut self, dst: Reg, s1: Reg, s2: Reg, op: Op) {
        self.op(Op::I64Const(-1));
        self.op(Op::I64Const(0));
        self.get(s1);
        self.get(s2);
        self.op(op);
        self.op(Op::Select);
        self.set_bits(dst);
    }

    fn bitwise(&mut self, dst: Reg, s1: Reg, s2: Reg, op: Op) {
        self.get_bits(s1);
        self.get_bits(s2);
        self.op(op);
        self.set_bits(dst);
    }

    // a * b, optionally negated, then `then` (add/sub) with c; two roundings
    // like the bytecode interpreter (core WebAssembly has no fused multiply-add)
    fn mul_then(&mut self, dst: Reg, a: Reg, b: Reg, c: Reg, negate: bool, then: Op) {
        self.get(a);
        self.get(b);
        self.op(Op::F64Mul);
        if negate {
            self.op(Op::F64Neg);
        }
        self.get(c);
        self.op(then);
        self.set(dst);
    }

    fn local(&mut self, r: Reg) -> u32 {
        match r {
            Reg::Ret | Reg::Left => L_RET,
            Reg::Temp | Reg::Right => L_TEMP,
            Reg::Gen(n) => {
                self.count_gen = self.count_gen.max(n as u32 + 1);
                L_GEN0 + n as u32
            }
            Reg::Static(..) => panic!("passing static registers to codegen"),
        }
    }

    fn base(loc: Loc) -> (u32, u32) {
        match loc {
            Loc::Param(idx) => (L_PARAMS, idx),
            Loc::Stack(idx) => (L_STACK, idx),
            Loc::Mem(idx) => (L_MEM, idx),
        }
    }

    fn load(&mut self, dst: Reg, base: u32, idx: u32) {
        let d = self.local(dst);
        self.op(Op::LocalGet(base));
        self.op(Op::F64Load(MemArg::f64(8 * idx)));
        self.op(Op::LocalSet(d));
    }

    fn save(&mut self, src: Reg, base: u32, idx: u32) {
        let s = self.local(src);
        self.op(Op::LocalGet(base));
        self.op(Op::LocalGet(s));
        self.op(Op::F64Store(MemArg::f64(8 * idx)));
    }

    fn save_loc(&mut self, src: Reg, loc: Loc) {
        let (base, idx) = Self::base(loc);
        self.save(src, base, idx);
    }

    /// Pushes the address of element `idx` of the `k`-th array in `states`.
    fn state_elem_addr(&mut self, k: usize) {
        self.op(Op::LocalGet(L_STATES));
        self.op(Op::I32Load(MemArg::i32(8 * k as u32)));
        self.op(Op::LocalGet(L_IDX));
        self.op(Op::I32Add);
    }

    /// Copies `count` values between the state/observable arrays (starting at
    /// array `first`) and the model memory frame. `L_IDX` is already in bytes.
    fn copy_states(&mut self, first: usize, count: usize, to_mem: bool) {
        if count == 0 {
            return;
        }

        if count <= UNROLL_LIMIT {
            for k in first..first + count {
                if to_mem {
                    self.op(Op::LocalGet(L_MEM));
                    self.state_elem_addr(k);
                    self.op(Op::F64Load(MemArg::f64(0)));
                    self.op(Op::F64Store(MemArg::f64(8 * k as u32)));
                } else {
                    self.state_elem_addr(k);
                    self.op(Op::LocalGet(L_MEM));
                    self.op(Op::F64Load(MemArg::f64(8 * k as u32)));
                    self.op(Op::F64Store(MemArg::f64(0)));
                }
            }
            return;
        }

        // both `states` entries and memory slots are 8 bytes apart, so one byte
        // counter (L_I) indexes both
        let start = 8 * first as i32;
        let end = 8 * (first + count) as i32;
        self.op(Op::I32Const(start));
        self.op(Op::LocalSet(L_I));
        self.op(Op::Loop(BlockType::Empty));

        let mem_addr = |g: &mut Self| {
            g.op(Op::LocalGet(L_MEM));
            g.op(Op::LocalGet(L_I));
            g.op(Op::I32Add);
        };
        let elem_addr = |g: &mut Self| {
            g.op(Op::LocalGet(L_STATES));
            g.op(Op::LocalGet(L_I));
            g.op(Op::I32Add);
            g.op(Op::I32Load(MemArg::i32(0)));
            g.op(Op::LocalGet(L_IDX));
            g.op(Op::I32Add);
        };

        if to_mem {
            mem_addr(self);
            elem_addr(self);
        } else {
            elem_addr(self);
            mem_addr(self);
        }
        self.op(Op::F64Load(MemArg::f64(0)));
        self.op(Op::F64Store(MemArg::f64(0)));

        self.op(Op::LocalGet(L_I));
        self.op(Op::I32Const(8));
        self.op(Op::I32Add);
        self.op(Op::LocalTee(L_I));
        self.op(Op::I32Const(end));
        self.op(Op::I32LtU);
        self.op(Op::BrIf(0));
        self.op(Op::End);
    }

    /// `fast(x0, ..., x[n-1]) -> f64`: allocates a model memory of `mem_bytes`
    /// on the stack, stores the arguments as the states, calls `run` in direct
    /// mode and returns the first observable (NaN if the stack is exhausted).
    fn fast_function(&self, m: &mut Module, run_idx: u32, mem_bytes: u32) -> Function {
        let n = self.count_states;
        let ty = m.add_type(FuncType {
            params: vec![ValType::F64; n as usize],
            results: vec![ValType::F64],
        });

        let l_mem = n;
        let l_saved_sp = n + 1;
        let mut b: Vec<Op> = Vec::new();

        b.push(Op::GlobalGet(G_SP));
        b.push(Op::LocalTee(l_saved_sp));
        b.push(Op::I32Const(mem_bytes as i32));
        b.push(Op::I32LtU);
        b.push(Op::If(BlockType::Empty));
        b.push(Op::F64Const(f64::NAN));
        b.push(Op::Return);
        b.push(Op::End);

        b.push(Op::GlobalGet(G_SP));
        b.push(Op::I32Const(mem_bytes as i32));
        b.push(Op::I32Sub);
        b.push(Op::LocalTee(l_mem));
        b.push(Op::GlobalSet(G_SP));

        for i in 0..n {
            b.push(Op::LocalGet(l_mem));
            b.push(Op::LocalGet(i));
            b.push(Op::F64Store(MemArg::f64(8 * i)));
        }

        b.push(Op::LocalGet(l_mem));
        b.push(Op::I32Const(0));
        b.push(Op::I32Const(0));
        b.push(Op::I32Const(0)); // no parameters
        b.push(Op::Call(run_idx));
        b.push(Op::If(BlockType::Empty));
        b.push(Op::LocalGet(l_saved_sp));
        b.push(Op::GlobalSet(G_SP));
        b.push(Op::F64Const(f64::NAN));
        b.push(Op::Return);
        b.push(Op::End);

        b.push(Op::LocalGet(l_mem));
        b.push(Op::F64Load(MemArg::f64(8 * n)));
        b.push(Op::LocalGet(l_saved_sp));
        b.push(Op::GlobalSet(G_SP));
        b.push(Op::End);

        let mut local_names: Vec<String> = (0..n).map(|i| format!("x{}", i)).collect();
        local_names.push("mem".to_string());
        local_names.push("saved_sp".to_string());

        Function {
            type_idx: ty,
            name: "fast".to_string(),
            locals: vec![(2, ValType::I32)],
            local_names,
            body: b,
        }
    }

    /// Lowers the buffered body and builds the module.
    fn build_module(&mut self) -> Vec<u8> {
        let mut m = Module::new();

        for f in self.imports.iter() {
            let width = if f.complex { 2 } else { 1 };
            let ty = m.add_type(FuncType {
                params: vec![ValType::F64; width * f.arity],
                results: vec![ValType::F64; width],
            });
            m.imports.push(Import {
                module: IMPORT_MODULE.to_string(),
                field: f.name.clone(),
                type_idx: ty,
            });
        }
        let run_idx = m.imports.len() as u32;

        let run_ty = m.add_type(FuncType {
            params: vec![ValType::I32; 4],
            results: vec![ValType::I32],
        });

        let mut flow: Vec<Flow> = Vec::with_capacity(self.body.len());

        for item in std::mem::take(&mut self.body) {
            flow.push(match item {
                Item::Op(op) => Flow::Op(op),
                Item::Const(idx) => match self.consts.get(idx as usize) {
                    Some(v) => Flow::Op(Op::F64Const(*v)),
                    None => {
                        self.unsupported(&format!("missing constant {}", idx));
                        Flow::Op(Op::Unreachable)
                    }
                },
                Item::CallImport(name) => {
                    let k = self.imports.iter().position(|f| f.name == name).unwrap();
                    Flow::Op(Op::Call(k as u32))
                }
                Item::CallSelf => Flow::Op(Op::Call(run_idx)),
                Item::Label(l) => Flow::Label(l),
                Item::Branch(l) => Flow::Branch(l),
                Item::BranchIf {
                    cond,
                    label,
                    is_else,
                } => Flow::BranchIf {
                    cond,
                    label,
                    is_else,
                },
                Item::CallFunclet(l) => Flow::CallFunclet(l),
                Item::Ret => Flow::Ret,
            });
        }

        let mut body = match flow::lower(flow, L_PC, L_RET_PC) {
            Ok(body) => body,
            Err(msg) => {
                self.unsupported(&msg);
                vec![Op::Unreachable]
            }
        };
        body.push(Op::End);

        let mut local_names: Vec<String> = [
            "mem", "states", "idx", "params", "stack", "saved_sp", "i", "pc", "ret_pc", "ret", "temp",
        ]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for k in 0..self.count_gen {
            local_names.push(format!("r{}", k));
        }

        let run = Function {
            type_idx: run_ty,
            name: "run".to_string(),
            locals: vec![
                (COUNT_I32_LOCALS, ValType::I32),
                (2 + self.count_gen, ValType::F64),
            ],
            local_names,
            body,
        };
        let size = run.body_size();
        if size > MAX_FUNCTION_SIZE {
            self.unsupported(&format!(
                "a model this large (the code is {} bytes; WebAssembly hosts accept at most {} per function)",
                size, MAX_FUNCTION_SIZE
            ));
        }
        m.funcs.push(run);

        let fast_mem_bytes = self.fast_mem_size.map(|n| align_stack(8 * n.max(1)));

        if let Some(mem_bytes) = fast_mem_bytes {
            // one wasm parameter per state; hosts reject more than MAX_PARAMS
            if self.count_obs > 0 && self.count_states <= MAX_PARAMS {
                let fast = self.fast_function(&mut m, run_idx, mem_bytes);
                m.funcs.push(fast);
            }
        }

        // stack region [0, heap_base): room for several frames (for a deep recursion
        // in recursive models), within MAX_STACK, but always for two frames
        let frame = self.frame_bytes.saturating_add(fast_mem_bytes.unwrap_or(0));
        if frame > MAX_FRAME {
            self.unsupported(&format!("a stack frame of {} bytes (the limit is {})", frame, MAX_FRAME));
        }
        let (frames, min_bytes) = if self.recursive { (1024, 16 * PAGE) } else { (16, PAGE) };
        let stack = frame
            .saturating_mul(frames)
            .clamp(min_bytes, MAX_STACK)
            .max(frame.saturating_mul(2).min(2 * MAX_FRAME));
        let heap_base = stack.div_ceil(PAGE) * PAGE;
        // plus one free page above `__heap_base` for the host's data
        m.memory_pages = Some(heap_base / PAGE + 1);

        m.globals.push(Global {
            ty: ValType::I32,
            mutable: true,
            init: heap_base as i32,
        });
        m.globals.push(Global {
            ty: ValType::I32,
            mutable: false,
            init: heap_base as i32,
        });

        m.exports.push(Export {
            name: "run".to_string(),
            kind: ExportKind::Func,
            idx: run_idx,
        });
        if m.funcs.len() > 1 {
            m.exports.push(Export {
                name: "fast".to_string(),
                kind: ExportKind::Func,
                idx: run_idx + 1,
            });
        }
        m.exports.push(Export {
            name: "memory".to_string(),
            kind: ExportKind::Memory,
            idx: 0,
        });
        m.exports.push(Export {
            name: "__heap_base".to_string(),
            kind: ExportKind::Global,
            idx: G_HEAP_BASE,
        });

        m.encode()
    }
}

impl Generator for WasmGenerator {
    fn count_shadows(&self) -> u8 {
        0
    }

    fn three_address(&self) -> bool {
        true
    }

    fn bytes(&mut self) -> Vec<u8> {
        self.module.clone()
    }

    fn what(&self) -> GeneratorType {
        GeneratorType::Wasm
    }

    fn seal(&mut self) {
        self.module = self.build_module();
    }

    fn align(&mut self) {}

    // jumps are buffered as markers and structured in `seal` (see flow.rs)
    fn set_label(&mut self, label: &str) {
        self.body.push(Item::Label(label.to_string()));
    }

    fn branch(&mut self, label: &str) {
        self.body.push(Item::Branch(label.to_string()));
    }

    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        let cond = self.local(cond);
        self.body.push(Item::BranchIf {
            cond,
            label: label.to_string(),
            is_else,
        });
    }

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        if dst == s1 {
            return;
        }
        let s = self.local(s1);
        let d = self.local(dst);
        self.op(Op::LocalGet(s));
        self.op(Op::LocalSet(d));
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        let a = self.local(s1);
        let b = self.local(s2);
        self.op(Op::LocalGet(a));
        self.op(Op::LocalGet(b));
        self.op(Op::LocalSet(a));
        self.op(Op::LocalSet(b));
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        let d = self.local(dst);
        self.body.push(Item::Const(idx));
        self.op(Op::LocalSet(d));
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.load(dst, L_MEM, idx);
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.save(dst, L_MEM, idx);
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.load(dst, L_PARAMS, idx);
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.load(dst, L_STACK, idx);
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.save(dst, L_STACK, idx);
    }

    // a complex value occupies two consecutive f64 slots (re, im)
    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load(xd, L_MEM, idx);
        self.load(yd, L_MEM, idx + 1);
    }

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save(xs, L_MEM, idx);
        self.save(ys, L_MEM, idx + 1);
    }

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load(xd, L_PARAMS, idx);
        self.load(yd, L_PARAMS, idx + 1);
    }

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load(xd, L_STACK, idx);
        self.load(yd, L_STACK, idx + 1);
    }

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save(xs, L_STACK, idx);
        self.save(ys, L_STACK, idx + 1);
    }

    // used by compression mode, which `Config::compress` disables for wasm
    fn load_args(&mut self, _locs: Vec<Loc>, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn save_args(&mut self, _num_args: u8, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn load_args_complex(&mut self, _locs: Vec<Loc>, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn save_args_complex(&mut self, _num_args: u8, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_loc(Reg::Ret, Loc::Mem(idx));
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_loc(Reg::Ret, Loc::Stack(idx));
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Neg);
    }

    // the sign bit of s1 on a zero: -0.0 if s1 is negative, 0.0 otherwise
    fn sign(&mut self, dst: Reg, s1: Reg) {
        self.op(Op::F64Const(0.0));
        self.get(s1);
        self.op(Op::F64Copysign);
        self.set(dst);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Abs);
    }

    fn abs2(&mut self, dst: Reg, s1: Reg) {
        self.binary(dst, s1, s1, Op::F64Mul);
    }

    // only generated for fast complex, which is off for wasm; the interpreter's value
    fn times_i(&mut self, dst: Reg, _s1: Reg) {
        self.set_const(dst, 0.0);
    }

    fn times_neg_i(&mut self, dst: Reg, _s1: Reg) {
        self.set_const(dst, 0.0);
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Sqrt);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Sqrt);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.op(Op::F64Const(1.0));
        self.get(s1);
        self.op(Op::F64Div);
        self.set(dst);
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.get(s1);
        self.op(Op::F64Const(2.0));
        self.op(Op::F64Div);
        self.set(dst);
    }

    // round half to even, like the aarch64 and riscv64 backends (the bytecode
    // interpreter rounds half away from zero)
    fn round(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Nearest);
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Floor);
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Ceil);
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.unary(dst, s1, Op::F64Trunc);
    }

    // s1 - floor(s1)
    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.get(s1);
        self.get(s1);
        self.op(Op::F64Floor);
        self.op(Op::F64Sub);
        self.set(dst);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(dst, s1, s2, Op::F64Add);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(dst, s1, s2, Op::F64Sub);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(dst, s1, s2, Op::F64Mul);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(dst, s1, s2, Op::F64Div);
    }

    // false: use the generic complex multiplication/division built from the
    // real operations above
    fn times_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        false
    }

    fn divide_complex(
        &mut self,
        _xd: Reg,
        _yd: Reg,
        _x1: Reg,
        _y1: Reg,
        _x2: Reg,
        _y2: Reg,
    ) -> bool {
        false
    }

    fn fuse_load_math(&mut self) {}

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!("support_times2 is false");
    }

    // real-valued MIR (complex values are lowered by complexify)
    fn real(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn imaginary(&mut self, dst: Reg, _s1: Reg) {
        self.set_const(dst, 0.0);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, _s2: Reg) {
        self.fmov(dst, s1);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare(dst, s1, s2, Op::F64Gt);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare(dst, s1, s2, Op::F64Ge);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare(dst, s1, s2, Op::F64Lt);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare(dst, s1, s2, Op::F64Le);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare(dst, s1, s2, Op::F64Eq);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare(dst, s1, s2, Op::F64Ne);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.bitwise(dst, s1, s2, Op::I64And);
    }

    // !s1 & s2
    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.get_bits(s1);
        self.op(Op::I64Const(-1));
        self.op(Op::I64Xor);
        self.get_bits(s2);
        self.op(Op::I64And);
        self.set_bits(dst);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.bitwise(dst, s1, s2, Op::I64Or);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        if s1 == s2 {
            // the zeroing idiom (`IsZero` uses it to make a 0.0 operand)
            self.set_const(dst, 0.0);
        } else {
            self.bitwise(dst, s1, s2, Op::I64Xor);
        }
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.get_bits(s1);
        self.op(Op::I64Const(-1));
        self.op(Op::I64Xor);
        self.set_bits(dst);
    }

    // a * b + c
    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.mul_then(dst, s1, s2, s3, false, Op::F64Add);
    }

    // a * b - c
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.mul_then(dst, s1, s2, s3, false, Op::F64Sub);
    }

    // -a * b + c
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.mul_then(dst, s1, s2, s3, true, Op::F64Add);
    }

    // -a * b - c
    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.mul_then(dst, s1, s2, s3, true, Op::F64Sub);
    }

    fn add_consts(&mut self, consts: &[f64]) {
        self.consts = consts.to_vec();
    }

    fn add_func(&mut self, op: &str, f: Func) {
        match f {
            Func::Unary(_)
            | Func::Binary(_)
            | Func::UnaryCplx(_)
            | Func::BinaryCplx(_)
            | Func::Recursive => {}
            _ => self.unsupported(&format!("user-defined function `{}`", op)),
        }
    }

    // Arguments are in Left (= Ret) and Right (= Temp); the result goes to Ret.
    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        if op == "@self" {
            return self.call_self();
        }

        if self.config.is_external_func(op) || !(1..=2).contains(&num_args) {
            self.unsupported(&format!("user-defined function `{}`", op));
            return Ok(());
        }

        self.add_import(op, num_args, false)?;
        self.op(Op::LocalGet(L_RET));
        if num_args == 2 {
            self.op(Op::LocalGet(L_TEMP));
        }
        self.body.push(Item::CallImport(op.to_string()));
        self.op(Op::LocalSet(L_RET));

        Ok(())
    }

    // The argument is (Ret, Temp), and the second one (Gen(0), Gen(1)); the result
    // goes to (Ret, Temp). Imported as `cplx_<op>`: (re, im[, re, im]) -> (re, im).
    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        if !(1..=2).contains(&num_args) {
            self.unsupported(&format!("complex function `{}`", op));
            return Ok(());
        }

        let name = if op.starts_with("cplx_") {
            op.to_string()
        } else {
            format!("cplx_{}", op)
        };
        self.add_import(&name, num_args, true)?;
        self.op(Op::LocalGet(L_RET));
        self.op(Op::LocalGet(L_TEMP));
        if num_args == 2 {
            self.get(Reg::Gen(0));
            self.get(Reg::Gen(1));
        }
        self.body.push(Item::CallImport(name));
        self.op(Op::LocalSet(L_TEMP));
        self.op(Op::LocalSet(L_RET));

        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        self.body.push(Item::CallFunclet(label.to_string()));
    }

    fn ret(&mut self) {
        self.body.push(Item::Ret);
    }

    // the fast kernel is not generated for wasm (`Config::may_fast`)
    fn prologue_fast(&mut self, _cap: usize, _count_states: usize, _count_obs: usize) {
        self.unsupported("fast kernel");
    }

    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _idx_ret: i32,
    ) {
        self.unsupported("fast kernel");
    }

    fn prologue_indirect(
        &mut self,
        cap: usize,
        count_states: usize,
        count_obs: usize,
        _count_params: usize,
    ) {
        // the whole model memory: an ODE writes its diffs after the observables
        let frame_mem = align_stack(8 * ((count_states + count_obs) as u32).max(self.mem_size));
        let frame_stack = align_stack(8 * cap as u32);
        self.frame_bytes = frame_mem + frame_stack;
        self.count_states = count_states as u32;
        self.count_obs = count_obs as u32;

        // if sp < frame_bytes { return 1 }
        self.op(Op::GlobalGet(G_SP));
        self.op(Op::LocalTee(L_SAVED_SP));
        self.op(Op::I32Const(self.frame_bytes as i32));
        self.op(Op::I32LtU);
        self.op(Op::If(BlockType::Empty));
        self.op(Op::I32Const(1));
        self.op(Op::Return);
        self.op(Op::End);

        // indirect mode: allocate the model memory and gather the states
        self.op(Op::LocalGet(L_STATES));
        self.op(Op::If(BlockType::Empty));
        self.op(Op::GlobalGet(G_SP));
        self.op(Op::I32Const(frame_mem as i32));
        self.op(Op::I32Sub);
        self.op(Op::LocalTee(L_MEM));
        self.op(Op::GlobalSet(G_SP));
        self.op(Op::LocalGet(L_IDX));
        self.op(Op::I32Const(3));
        self.op(Op::I32Shl);
        self.op(Op::LocalSet(L_IDX));
        self.copy_states(0, count_states, true);
        self.op(Op::End);

        self.op(Op::GlobalGet(G_SP));
        self.op(Op::I32Const(frame_stack as i32));
        self.op(Op::I32Sub);
        self.op(Op::LocalTee(L_STACK));
        self.op(Op::GlobalSet(G_SP));
    }

    fn epilogue_indirect(
        &mut self,
        _cap: usize,
        count_states: usize,
        count_obs: usize,
        _count_params: usize,
    ) {
        // indirect mode: scatter the observables
        if count_obs > 0 {
            self.op(Op::LocalGet(L_STATES));
            self.op(Op::If(BlockType::Empty));
            self.copy_states(count_states, count_obs, false);
            self.op(Op::End);
        }

        self.op(Op::LocalGet(L_SAVED_SP));
        self.op(Op::GlobalSet(G_SP));
        self.op(Op::I32Const(0));
    }

    // registers are locals; nothing to preserve
    fn save_used_registers(&mut self, _used: &[Reg]) {}

    fn load_used_registers(&mut self, _used: &[Reg]) {}

    // (true_val & mask) | (false_val & !mask), mask = Stack[idx] (a comparison result)
    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        if true_val == false_val {
            self.fmov(dst, true_val);
            return;
        }

        let mask = |g: &mut Self| {
            g.op(Op::LocalGet(L_STACK));
            g.op(Op::I64Load(MemArg::f64(8 * idx)));
        };

        self.get_bits(true_val);
        mask(self);
        self.op(Op::I64And);
        self.get_bits(false_val);
        mask(self);
        self.op(Op::I64Const(-1));
        self.op(Op::I64Xor);
        self.op(Op::I64And);
        self.op(Op::I64Or);
        self.set_bits(dst);
    }
}
