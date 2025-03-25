use anyhow::{anyhow, Result};

use serde::Deserialize;

use crate::code::*;
use crate::register::*;

/// Lowers Expr and its constituents into an intermediate representation
pub trait Lower {
    fn lower(&self, prog: &mut Program) -> Result<Word>;
}

/// Collects instructions and registers
#[derive(Debug)]
pub struct Program {
    pub code: Vec<Instruction>, // the list of instructions
    pub frame: Frame,           // memory (states, registers, constants, ...)
    pub ft: Vec<String>,        // function table (the name of functions)
                                //pub vt: Vec<BinaryFunc<f64>>,    // virtual table (pointer to the functions)
}

impl Program {
    pub fn new(ml: &CellModel) -> Result<Program> {
        let mut frame = Frame::new();

        /*
            this section lays the memory format
            the order of different sections is important!

            the layout is:

            +------------------------+
            | predefined constants   |
            +------------------------+
            | independent variable   |
            +------------------------+
            | state variables        |
            +------------------------+
            | parameters             |
            +------------------------+
            | observables (output)   |
            +------------------------+
            | differentials (output) |
            +------------------------+
            | constants and temps    |
            +------------------------+
        */

        frame.alloc(WordType::Var(ml.iv.name.clone()));

        for v in &ml.states {
            frame.alloc(WordType::State(v.name.clone(), v.val));
        }

        for v in &ml.params {
            frame.alloc(WordType::Param(v.name.clone(), v.val));
        }

        for eq in &ml.obs {
            if let Some(name) = eq.lhs.var() {
                frame.alloc(WordType::Obs(name));
            } else {
                return Err(anyhow!("lhs var not found"));
            }
        }

        for eq in &ml.odes {
            if let Some(name) = eq.lhs.diff_var() {
                frame.alloc(WordType::Diff(name));
            } else {
                return Err(anyhow!("lhs diff var not found"));
            }
        }

        let mut prog = Program {
            code: Vec::new(),
            frame,
            ft: Vec::new(),
        };

        ml.lower(&mut prog)?;
        prog.code.push(Instruction::Nop);

        // we call confirm here to ensure all functions are
        // resolved. If not, it returns an Err. This is to
        // prevent a panic later when virtual table is formed.
        VirtualTable::<f64>::confirm(&prog.ft)?;

        Ok(prog)
    }

    pub fn push(&mut self, s: Instruction) {
        self.code.push(s)
    }

    pub fn pop(&mut self) {
        let _ = self.code.pop();
    }

    fn proc(&mut self, op: &str) -> Proc {
        let p = match self.ft.iter().position(|s| s == op) {
            Some(p) => p,
            None => {
                self.ft.push(op.to_string());
                self.ft.len() - 1
            }
        };
        Proc(p)
    }

    pub fn push_eq(&mut self, dst: Word) {
        self.code.push(Instruction::Eq { dst })
    }

    /// Pushes an Op into code
    pub fn push_unary(&mut self, op: &str, x: Word, dst: Word) {
        let p = self.proc(op);

        self.code.push(Instruction::Unary {
            op: op.to_string(),
            x,
            dst,
            p,
        })
    }

    pub fn push_binary(&mut self, op_: &str, x_: Word, y_: Word, dst_: Word) {
        // optimization by fusing x + (-y) to x - y
        if op_ == "plus" && !self.code.is_empty() {
            let c = self.code.pop().unwrap();
            if let Instruction::Unary { op, x, dst, .. } = c.clone() {
                if op == "neg" && dst == y_ {
                    let p = self.proc("minus");
                    self.code.push(Instruction::Binary {
                        op: "minus".to_string(),
                        x: x_,
                        y: x,
                        dst: dst_,
                        p,
                    });
                    return;
                }
            };
            self.code.push(c);
        }

        let p = self.proc(op_);

        self.code.push(Instruction::Binary {
            op: op_.to_string(),
            x: x_,
            y: y_,
            dst: dst_,
            p,
        })
    }

    pub fn push_ifelse(&mut self, x1: Word, x2: Word, cond: Word, dst: Word) {
        self.code.push(Instruction::IfElse { x1, x2, cond, dst })
    }

    /// Allocates a constant register
    pub fn alloc_const(&mut self, val: f64) -> Word {
        self.frame.alloc(WordType::Const(val))
    }

    /// Allocates a temporary register
    pub fn alloc_temp(&mut self) -> Word {
        self.frame.alloc(WordType::Temp)
    }

    /// Allocates an obeservable register
    pub fn alloc_obs(&mut self, name: &str) -> Word {
        self.frame.alloc(WordType::Obs(name.to_string()))
    }

    /// Frees a word (register) and returns it to a pool
    /// Only temp works actually are freed. Other words are ignored.
    pub fn free(&mut self, r: Word) {
        self.frame.free(r);
    }

    pub fn reg(&self, name: &str) -> Result<Word> {
        match self.frame.find(name) {
            Some(w) => Ok(w),
            None => Err(anyhow!("cannot find reg {} by name", name)),
        }
    }

    pub fn reg_diff(&self, name: &str) -> Result<Word> {
        match self.frame.find_diff(name) {
            Some(w) => Ok(w),
            None => Err(anyhow!("cannot find diff {} by name", name)),
        }
    }
    /*
        pub fn virtual_table(&self) -> Vec<BinaryFunc<f64>> {
            self.vt.clone()
        }
    */
}

/// A defined (state or param) variable
#[derive(Debug, Clone, Deserialize)]
pub struct Variable {
    pub name: String,
    pub val: f64,
}

impl Lower for Variable {
    fn lower(&self, prog: &mut Program) -> Result<Word> {
        prog.reg(&self.name)
    }
}

/// Expr tree
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Expr {
    Tree { op: String, args: Vec<Expr> },
    Const { val: f64 },
    Var { name: String },
}

impl Expr {
    /// Extracts the differentiated variable from the lhs of a diff eq
    pub fn diff_var(&self) -> Option<String> {
        if let Expr::Tree { args, op } = self {
            if op != "Differential" {
                return None;
            }
            if let Expr::Var { name } = &args[0] {
                return Some(name.clone());
            }
        };
        None
    }

    /// Extracts the regular variable from the lhs of an observable eq
    pub fn var(&self) -> Option<String> {
        if let Expr::Var { name } = self {
            return Some(name.clone());
        };
        None
    }

    fn lower_unary(&self, prog: &mut Program, op: &str, args: &[Expr]) -> Result<Word> {
        let x = args[0].lower(prog)?;
        let dst = prog.alloc_temp();
        prog.push_unary(op, x, dst);
        prog.free(x);
        Ok(dst)
    }

    fn lower_binary(&self, prog: &mut Program, op: &str, args: &[Expr]) -> Result<Word> {
        if op == "times" {
            return self.lower_times(prog, args);
        }

        let x = args[0].lower(prog)?;
        let y = args[1].lower(prog)?;
        let dst = prog.alloc_temp();

        prog.push_binary(op, x, y, dst);
        prog.free(y);
        prog.free(x);

        Ok(dst)
    }

    fn lower_times(&self, prog: &mut Program, args: &[Expr]) -> Result<Word> {
        let x = args[0].lower(prog)?;
        let dst = prog.alloc_temp();

        if x == Frame::MINUS_ONE {
            prog.pop();
            let y = args[1].lower(prog)?;
            prog.push_unary("neg", y, dst);
            prog.free(y);
        } else {
            let y = args[1].lower(prog)?;
            if y == Frame::MINUS_ONE {
                prog.pop();
                prog.push_unary("neg", x, dst);
            } else {
                prog.push_binary("times", x, y, dst);
            };
            prog.free(y);
        }

        prog.free(x);

        Ok(dst)
    }

    fn lower_ternary(&self, prog: &mut Program, op: &str, args: &[Expr]) -> Result<Word> {
        if op != "ifelse" {
            return self.lower_poly(prog, op, args);
        }

        // The order of the next three lower calls is important!
        // The order should mirror the order of the arguments in push_ifelse
        let x1 = args[1].lower(prog)?;
        let x2 = args[2].lower(prog)?;
        let cond = args[0].lower(prog)?;
        let dst = prog.alloc_temp();

        prog.push_ifelse(x1, x2, cond, dst);

        prog.free(cond);
        prog.free(x2);
        prog.free(x1);

        Ok(dst)
    }

    fn lower_poly(&self, prog: &mut Program, op: &str, args: &[Expr]) -> Result<Word> {
        if !(op == "plus" || op == "times") {
            return Err(anyhow!("missing poly op: {}", op));
        }

        let mut x = args[0].lower(prog)?;

        for arg in args.iter().skip(1) {
            let y = arg.lower(prog)?;
            let dst = prog.alloc_temp();
            prog.push_binary(op, x, y, dst);
            prog.free(x);
            x = dst;
        }

        Ok(x)
    }
}

impl Lower for Expr {
    fn lower(&self, prog: &mut Program) -> Result<Word> {
        match self {
            Expr::Const { val } => {
                let dst = if *val == 0.0 {
                    Frame::ZERO
                } else if *val == 1.0 {
                    Frame::ONE
                } else if *val == -1.0 {
                    Frame::MINUS_ONE
                } else {
                    prog.alloc_const(*val)
                };
                prog.push(Instruction::Num { val: *val, dst });
                Ok(dst)
            }
            Expr::Var { name } => {
                // Technically, this is not necessary but having Instruction::Var in the code
                // is helpful for debugging
                let dst = prog.reg(name)?;
                prog.push(Instruction::Var {
                    name: name.clone(),
                    reg: dst,
                });
                Ok(dst)
            }
            Expr::Tree { op, args } => match args.len() {
                1 => self.lower_unary(prog, op, args),
                2 => self.lower_binary(prog, op, args),
                3 => self.lower_ternary(prog, op, args),
                _ => self.lower_poly(prog, op, args),
            },
        }
    }
}

/// Represents lhs ~ rhs
#[derive(Debug, Clone, Deserialize)]
pub struct Equation {
    pub lhs: Expr,
    pub rhs: Expr,
}

impl Lower for Equation {
    fn lower(&self, prog: &mut Program) -> Result<Word> {
        let dst = if let Some(var) = self.lhs.diff_var() {
            prog.reg_diff(&var)?
        } else if let Some(var) = self.lhs.var() {
            prog.reg(&var)?
        } else {
            return Err(anyhow!("undefined diff variable"));
        };

        prog.push_eq(dst);

        let src = self.rhs.lower(prog)?;

        prog.push_unary("mov", src, dst);
        Ok(Frame::ZERO)
    }
}

/// Loads a model from a JSON file
/// Historically from a CellML source; hence the name.
#[derive(Debug, Clone, Deserialize)]
pub struct CellModel {
    pub iv: Variable,
    pub params: Vec<Variable>,
    pub states: Vec<Variable>,
    #[allow(dead_code)]
    pub algs: Vec<Equation>,
    pub odes: Vec<Equation>,
    pub obs: Vec<Equation>,
}

impl CellModel {
    pub fn load(text: &str) -> Result<CellModel> {
        Ok(serde_json::from_str(text)?)
    }
}

impl Lower for CellModel {
    fn lower(&self, prog: &mut Program) -> Result<Word> {
        for eq in &self.obs {
            eq.lower(prog)?;
        }

        for eq in &self.odes {
            eq.lower(prog)?;
        }

        Ok(Frame::ZERO)
    }
}
