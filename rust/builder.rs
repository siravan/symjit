use anyhow::{anyhow, Result};
use std::collections::HashSet;

use crate::allocator::Allocator;
use crate::block::Block;
use crate::defuns::Defuns;
use crate::generator::Generator;
use crate::mir::Mir;
use crate::node::Node;
use crate::symbol::SymbolTable;
use crate::COUNT_SCRATCH;

// the list of intrinsic unary ops, i.e., operations that can be implemented directly in
// machine code
pub const UNARY: &[&str] = &[
    "abs", "not", "neg", "root", "square", "cube", "recip", "round", "floor", "ceiling", "trunc",
    "frac", "_powi_", "_call_",
];
// the list of intrinsic binary ops, i.e., operations that can be implemented directly in
// machine code
pub const BINARY: &[&str] = &[
    "plus",
    "minus",
    "times",
    "divide",
    "rem",
    "gt",
    "geq",
    "lt",
    "leq",
    "eq",
    "neq",
    "and",
    "or",
    "xor",
    "_ifelse_",
    "_powi_mod_",
    "_call_",
    "min",
    "max",
    "heaviside",
];

//****************************************************//

#[derive(Debug, Clone)]
pub struct Builder {
    pub primary_block: Block,
    pub consts: Vec<f64>,
    pub ft: HashSet<String>, // function table (the name of functions),
    pub count_loops: usize,
}

impl Builder {
    pub fn new(cse: bool) -> Builder {
        Builder {
            primary_block: Block::new(cse),
            consts: Vec::new(),
            ft: HashSet::new(),
            count_loops: 0,
        }
    }

    pub fn symbol_table(&mut self) -> &mut SymbolTable {
        &mut self.block().sym_table
    }

    pub fn block(&mut self) -> &mut Block {
        &mut self.primary_block
    }

    pub fn block_shared(&self) -> &Block {
        &self.primary_block
    }

    pub fn has_loop(&self) -> bool {
        self.count_loops > 0
    }

    pub fn add_assign(&mut self, lhs: Node, rhs: Node) -> Result<Node> {
        self.block().add_assign(lhs.clone(), rhs);
        Ok(lhs)
    }

    pub fn add_unary(&mut self, op: &str, arg: Node) -> Result<Node> {
        if !UNARY.contains(&op) {
            // let f = VirtualTable::from_str(op)?; // check to see if op is defined
            // if !matches!(f, Func::Unary(_)) {
            //     return Err(anyhow!("{} is not a unary function", op));
            // }
            self.ft.insert(op.to_string());
        }

        self.create_unary(op, arg)
    }

    pub fn add_binary(&mut self, op: &str, left: Node, right: Node) -> Result<Node> {
        if op == "power" {
            if let Some(val) = right.as_int_const() {
                match val {
                    0 => return self.create_const(1.0),
                    1 => return Ok(left),
                    2 => return self.create_unary("square", left),
                    3 => return self.create_unary("cube", left),
                    -1 => return self.create_unary("recip", left),
                    -2 => {
                        let arg = self.create_unary("square", left)?;
                        return self.create_unary("recip", arg);
                    }
                    -3 => {
                        let arg = self.create_unary("cube", left)?;
                        return self.create_unary("recip", arg);
                    }
                    _ => {
                        return self.create_powi(left, val);
                    }
                }
            };

            if let Some(val) = right.as_const() {
                const ONE_THIRD: f64 = 1.0 / 3.0;

                match val {
                    0.5 => return self.create_unary("root", left),
                    ONE_THIRD => return self.add_unary("cbrt", left),
                    1.5 => {
                        let arg = self.create_unary("cube", left)?;
                        return self.create_unary("root", arg);
                    }
                    _ => {}
                }
            }
        }

        if !BINARY.contains(&op) {
            // let f = VirtualTable::from_str(op)?; // check to see if op is defined
            // if !matches!(f, Func::Binary(_)) {
            //     return Err(anyhow!("{} is not a binary function", op));
            // }
            self.ft.insert(op.to_string());
        }

        self.create_binary(op, left, right)
    }

    pub fn add_loop_prefix(&mut self, op: &str, var: Node, start: Node) -> Result<(Node, usize)> {
        assert!(op == "Sum" || op == "Product");

        let accum_var = self.block().create_tmp();

        self.block().add_assign(var, start);
        let init = self.create_const(if op == "Sum" { 0.0 } else { 1.0 })?;
        self.block().add_assign(accum_var.clone(), init);

        let label = format!(".L{}", self.count_loops);
        self.count_loops += 1;
        self.block().add_label(&label);

        Ok((accum_var, self.count_loops - 1))
    }

    pub fn add_loop_body(
        &mut self,
        op: &str,
        eq: Node,
        var: Node,
        end: Node,
        accum_var: Node,
        loop_id: usize,
    ) -> Result<Node> {
        let p = if op == "Sum" {
            self.create_binary("plus", accum_var.clone(), eq)?
        } else {
            self.create_binary("times", accum_var.clone(), eq)?
        };

        self.add_assign(accum_var.clone(), p)?;
        let one = self.create_const(1.0)?;
        let q = self.create_binary("plus", var.clone(), one)?;
        self.add_assign(var.clone(), q)?;
        let cond = self.create_binary("leq", var, end)?;

        let label = format!(".L{}", loop_id);
        self.block().add_branch(cond, &label);

        Ok(accum_var)
    }

    pub fn create_ifelse(&mut self, cond: Node, true_val: Node, false_val: Node) -> Result<Node> {
        Ok(self.block().create_ifelse(cond, true_val, false_val))
    }

    pub fn create_void(&mut self) -> Result<Node> {
        Ok(self.block().create_void())
    }

    pub fn create_const(&mut self, val: f64) -> Result<Node> {
        for (idx, v) in self.consts.iter().enumerate() {
            if *v == val {
                return Ok(Node::Const {
                    val,
                    idx: idx as u32,
                });
            }
        }

        self.consts.push(val);

        let n = self.consts.len();
        Ok(self.block().create_const(val, (n - 1) as u32))
    }

    pub fn create_var(&mut self, name: &str) -> Result<Node> {
        let sym = self
            .symbol_table()
            .find_sym(name)
            .ok_or_else(|| anyhow!("variable {} not found", name))?;

        Ok(self.block().create_var(sym))
    }

    pub fn create_unary(&mut self, op: &str, arg: Node) -> Result<Node> {
        Ok(self.block().create_unary(op, arg))
    }

    pub fn create_powi(&mut self, arg: Node, power: i32) -> Result<Node> {
        Ok(self.block().create_powi(arg, power))
    }

    pub fn create_modular_powi(&mut self, left: Node, right: Node, power: i32) -> Result<Node> {
        Ok(self.block().create_modular_powi(left, right, power))
    }

    pub fn create_binary(&mut self, op: &str, left: Node, right: Node) -> Result<Node> {
        let node = match op {
            "times" if left.is_const(-1.0) => self.create_unary("neg", right)?,
            "times" if right.is_const(-1.0) => self.create_unary("neg", left)?,
            "times" if left.is_const(1.0) => right,
            "times" if right.is_const(1.0) => left,
            "times" if left.is_unary("recip") => {
                self.create_binary("divide", right, left.arg().unwrap())?
            }
            "times" if right.is_unary("recip") => {
                self.create_binary("divide", left, right.arg().unwrap())?
            }
            "plus" if left.is_unary("neg") => {
                self.create_binary("minus", right, left.arg().unwrap())?
            }
            "plus" if right.is_unary("neg") => {
                self.create_binary("minus", left, right.arg().unwrap())?
            }
            "rem" if left.is_unary("_powi_") => {
                let (arg, power) = left.arg_power().unwrap();
                self.create_modular_powi(arg, right, power)?
            }
            "min" => {
                let cond = self.create_binary("leq", left.clone(), right.clone())?;
                self.create_ifelse(cond, left, right)?
            }
            "max" => {
                let cond = self.create_binary("geq", left.clone(), right.clone())?;
                self.create_ifelse(cond, left, right)?
            }
            "heaviside" => {
                /*
                 * In sympy, Heaviside is considered a binary operator,
                 * where the second argument is the value at 0 (defaults to 0.5).
                 */
                let zero = self.create_const(0.0)?;
                let one = self.create_const(1.0)?;

                let c0 = self.create_binary("eq", left.clone(), zero.clone())?;
                let x0 = self.create_ifelse(c0, right, one)?;

                let c1 = self.create_binary("geq", left, zero.clone())?;
                self.create_ifelse(c1, x0, zero)?
            }
            // note: block() is needed here to prevent a infinite loop
            _ => self.block().create_binary(op, left, right),
        };

        Ok(node)
    }

    pub fn compile_mir(&mut self, fastmath: bool, opt_level: u8, df: &Defuns) -> Result<Mir> {
        let mut mir = Mir::new(opt_level, fastmath, df);

        self.block().eliminate();
        self.block().compile(&mut mir)?;

        if opt_level >= 1 {
            mir.optimize_peephole();
        }

        if opt_level >= 2 {
            Allocator::optimize(&mut mir);
        }

        mir.add_consts(&self.consts);
        mir.populate_labels();

        Ok(mir)
    }

    fn save_registers(mir: &Mir, ir: &mut impl Generator) {
        if ir.count_shadows() < COUNT_SCRATCH {
            let used = mir.used_registers();
            ir.save_used_registers(&used);
        }
    }

    fn restore_registers(mir: &Mir, ir: &mut impl Generator) {
        if ir.count_shadows() < COUNT_SCRATCH {
            let used = mir.used_registers();
            ir.load_used_registers(&used);
        }
    }

    pub fn compile_from_mir(
        &mut self,
        mir: &Mir,
        ir: &mut impl Generator,
        count_states: usize,
        count_obs: usize,
    ) -> Result<()> {
        // println!("{:#?}", mir.used_registers());
        let cap = self.symbol_table().num_stack as u32;
        ir.prologue_indirect(cap, count_states, count_obs);

        Self::save_registers(mir, ir);
        mir.rerun(ir)?;
        Self::restore_registers(mir, ir);

        ir.epilogue_indirect(cap, count_states, count_obs);
        ir.align();
        self.append_const_section(ir);
        self.append_vt_section(mir, ir);
        ir.seal();
        // println!("{:#?}", &self.block().stmts);
        // println!("{:02x?}", ir.bytes());

        Ok(())
    }

    pub fn compile_fast_from_mir(
        &mut self,
        mir: &Mir,
        ir: &mut impl Generator,
        num_args: u32,
        idx_ret: i32,
    ) -> Result<()> {
        self.block().eliminate();
        // println!("{:#?}", &self.block().stmts);
        let cap = self.symbol_table().num_stack as u32;
        ir.prologue_fast(cap, num_args);

        Self::save_registers(mir, ir);
        mir.rerun(ir)?;
        Self::restore_registers(mir, ir);

        ir.epilogue_fast(cap, idx_ret);
        ir.align();
        self.append_const_section(ir);
        self.append_vt_section(mir, ir);
        ir.seal();
        // println!("{:#?}", &self.block().stmts);
        // println!("{:02x?}", ir.bytes());

        Ok(())
    }

    fn append_const_section(&self, ir: &mut impl Generator) {
        ir.add_consts(&self.consts);
    }

    fn append_vt_section(&self, mir: &Mir, ir: &mut impl Generator) {
        for op in self.ft.iter() {
            let p = mir.find_op(op).expect("func not found");
            ir.add_func(op, p);
        }
    }
}
