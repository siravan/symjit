use anyhow::Result;
use std::fmt::Write;
use wasmtime::*;

use crate::code::*;
use crate::model::Program;
use crate::register::Word;
use crate::utils::*;

enum OpType {
    Nop, // not implemented yet!
    Unary(&'static str),
    Binary(&'static str),
    Ternary(&'static str),
}

#[derive(Debug)]
pub struct WasmCompiler {
    buf: String,
    lev: i32,
}

impl WasmCompiler {
    pub fn new() -> WasmCompiler {
        Self {
            buf: String::new(),
            lev: 0,
        }
    }

    fn push(&mut self, s: &str) {
        for _ in 0..4 * self.lev {
            let _ = write!(self.buf, " ");
        }
        let _ = writeln!(self.buf, "{}", s);
        self.lev += s.chars().filter(|x| *x == '(').count() as i32;
        self.lev -= s.chars().filter(|x| *x == ')').count() as i32;
        if self.lev < 0 {
            panic!("more ) than (");
        }
    }

    fn op_code(&mut self, op: &str) -> OpType {
        match op {
            "mov" => OpType::Unary("f64.store"),
            "neg" => OpType::Unary("f64.neg"),
            "sin" => OpType::Unary("call $sin"),
            "cos" => OpType::Unary("call $cos"),
            "tan" => OpType::Unary("call $tan"),
            "csc" => OpType::Unary("call $csc"),
            "sec" => OpType::Unary("call $sec"),
            "cot" => OpType::Unary("call $cot"),
            "arcsin" => OpType::Unary("call $asin"),
            "arccos" => OpType::Unary("call $acos"),
            "arctan" => OpType::Unary("call $atan"),
            "exp" => OpType::Unary("call $exp"),
            "ln" => OpType::Unary("call $ln"),
            "log" => OpType::Unary("call $log"),
            "root" => OpType::Unary("f64.sqrt"),

            "plus" => OpType::Binary("f64.add"),
            "minus" => OpType::Binary("f64.sub"),
            "times" => OpType::Binary("f64.mul"),
            "divide" => OpType::Binary("f64.div"),
            "rem" => OpType::Binary("call $rem"),
            "power" => OpType::Binary("call $power"),
            "gt" => OpType::Binary("f64.gt"),
            "geq" => OpType::Binary("f64.ge"),
            "lt" => OpType::Binary("f64.lt"),
            "leq" => OpType::Binary("f64.le"),
            "eq" => OpType::Binary("f64.eq"),
            "neq" => OpType::Binary("f64.ne"),
            "and" => OpType::Binary("i32.and"),
            "or" => OpType::Binary("i32.or"),
            "xor" => OpType::Binary("i32.xor"),
            "select" => OpType::Ternary("select"),
            _ => {
                let msg = format!("op_code {} not found", op);
                panic!("{}", msg);
            }
        }
    }

    fn imports(&mut self) {
        // unary
        for s in [
            "sin", "cos", "tan", "csc", "sec", "cot", "asin", "acos", "atan", "exp", "ln", "log",
        ] {
            let cmd = format!(
                "(import \"code\" \"{}\" (func ${} (param f64)(result f64)))",
                s, s
            );
            self.push(cmd.as_str());
        }

        // binary
        for s in ["rem", "power"] {
            let cmd = format!(
                "(import \"code\" \"{}\" (func ${} (param f64)(param f64)(result f64)))",
                s, s
            );
            self.push(cmd.as_str());
        }
    }

    fn prologue(&mut self) {
        self.push("(module");
        self.imports();
        self.push("(memory $memory 1)");
        self.push("(export \"memory\" (memory $memory))");
        self.push("(func $run");
    }

    fn epilogue(&mut self) {
        self.push(")"); // (func $run
        self.push("(export \"run\" (func $run))");
        self.push(")"); // (module
    }
}

impl Compiler<WasmCode> for WasmCompiler {
    fn compile(&mut self, prog: &Program) -> WasmCode {
        self.prologue();

        for c in prog.code.iter() {
            match c {
                Instruction::Unary { op, .. } => {
                    if let OpType::Unary(s) = self.op_code(op) {
                        self.push(s);
                    } else {
                        panic!("unkown unary op");
                    }
                }
                Instruction::Binary { op, .. } => {
                    if let OpType::Binary(s) = self.op_code(op) {
                        self.push(s);
                    } else {
                        panic!("unkown binary op");
                    }
                }
                Instruction::IfElse { .. } => {
                    self.push("select");
                }
                Instruction::Eq { dst } => {
                    self.push(format!("i32.const {}", 8 * dst.0).as_str());
                }
                Instruction::Num { val, .. } => self.push(format!("f64.const {}", val).as_str()),
                Instruction::Var { reg, .. } => {
                    self.push(format!("(f64.load (i32.const {}))", 8 * reg.0).as_str())
                }
                _ => {}
            }
        }

        self.epilogue();

        // println!("{}", self.buf);

        WasmCode::new(self.buf.clone(), prog.frame.mem()).unwrap()
    }
}

type HostState = u32;

pub struct WasmCode {
    _mem: Vec<f64>,
    wat: String,
    engine: Engine,
    module: Module,
    store: Store<HostState>,
    linker: Linker<HostState>,
    instance: Instance,
    run: TypedFunc<(), ()>,
    memory: Memory,
}

impl WasmCode {
    fn new(wat: String, _mem: Vec<f64>) -> Result<WasmCode> {
        let engine = Engine::default();
        let module = Module::new(&engine, wat.as_str())?;
        let mut linker = Linker::<HostState>::new(&engine);

        Self::imports(&mut linker).expect("error in importing functions to wasm");

        let mut store: Store<HostState> = Store::new(&engine, 0);
        let instance = linker.instantiate(&mut store, &module)?;
        let run = instance.get_typed_func::<(), ()>(&mut store, "run")?;
        let memory = instance.get_memory(&mut store, "memory").unwrap();

        let p: &mut [f64] = unsafe { std::mem::transmute(memory.data_mut(&mut store)) };
        let _ = p[.._mem.len()].copy_from_slice(&_mem[..]);

        let wasm = WasmCode {
            _mem,
            wat,
            engine,
            module,
            store,
            linker,
            instance,
            run,
            memory,
        };

        Ok(wasm)
    }

    pub fn imports(linker: &mut Linker<HostState>) -> Result<()> {
        linker.func_wrap("code", "sin", |x: f64| -> f64 { x.sin() })?;
        linker.func_wrap("code", "cos", |x: f64| -> f64 { x.cos() })?;
        linker.func_wrap("code", "tan", |x: f64| -> f64 { x.tan() })?;
        linker.func_wrap("code", "csc", |x: f64| -> f64 { 1.0 / x.sin() })?;
        linker.func_wrap("code", "sec", |x: f64| -> f64 { 1.0 / x.cos() })?;
        linker.func_wrap("code", "cot", |x: f64| -> f64 { 1.0 / x.tan() })?;
        linker.func_wrap("code", "asin", |x: f64| -> f64 { x.asin() })?;
        linker.func_wrap("code", "acos", |x: f64| -> f64 { x.acos() })?;
        linker.func_wrap("code", "atan", |x: f64| -> f64 { x.atan() })?;
        linker.func_wrap("code", "exp", |x: f64| -> f64 { x.exp() })?;
        linker.func_wrap("code", "ln", |x: f64| -> f64 { x.ln() })?;
        linker.func_wrap("code", "log", |x: f64| -> f64 { x.log(10.0) })?;
        linker.func_wrap("code", "rem", |x: f64, y: f64| -> f64 { x % y })?;
        linker.func_wrap("code", "power", |x: f64, y: f64| -> f64 { x.powf(y) })?;

        Ok(())
    }
}

impl Compiled for WasmCode {
    fn run(&mut self) {
        self.run.call(&mut self.store, ()).unwrap();
    }

    #[inline]
    fn mem(&self) -> &[f64] {
        let p: &[f64] = unsafe { std::mem::transmute(self.memory.data(&self.store)) };
        p
    }

    #[inline]
    fn mem_mut(&mut self) -> &mut [f64] {
        let p: &mut [f64] = unsafe { std::mem::transmute(self.memory.data_mut(&mut self.store)) };
        p
    }

    fn dump(&self, name: &str) {}
}
