use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::io::{Read, Write};

use super::allocator::GreedyAllocator;
use super::block::Block;
use super::compactor::Compactor;
use super::config::Config;
use super::generator::Generator;
use super::mir::Mir;
use super::node::Node;
use super::operation::Operation;
use super::symbol::SymbolTable;
use super::utils::Storage;

#[derive(Debug, Clone)]
pub struct Builder {
    pub primary_block: Block,
    pub consts: Vec<f64>,
    pub ft: HashSet<String>, // function table (the name of functions),
    pub count_loops: usize,
    pub config: Config,
    pub count_stack: Option<usize>,
    pub salt: Option<String>,
}

impl Default for Builder {
    fn default() -> Self {
        let config = Config::default();

        Builder {
            primary_block: Block::new(config.clone()),
            consts: Vec::new(),
            ft: HashSet::new(),
            count_loops: 0,
            config,
            count_stack: None,
            salt: None,
        }
    }
}

impl Builder {
    const MAGIC: usize = 0x12f21e25abe627bc;

    pub fn new(config: Config) -> Builder {
        Builder {
            primary_block: Block::new(config.clone()),
            consts: Vec::new(),
            ft: HashSet::new(),
            count_loops: 0,
            config,
            count_stack: None,
            salt: None,
        }
    }

    pub fn set_salt(&mut self, salt: String) {
        self.salt = Some(salt);
    }

    pub fn symbol_table(&mut self) -> &mut SymbolTable {
        &mut self.block().sym_table
    }

    pub fn stack_size(&self) -> usize {
        match self.count_stack {
            Some(size) => size,
            None => self.primary_block.sym_table.num_stack,
        }
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

    pub fn add_unary(&mut self, op: Operation, arg: Node) -> Result<Node> {
        if !self.config.is_intrinsic_unary(&op) {
            self.ft.insert(op.to_string());
        }

        self.create_unary(op, arg)
    }

    pub fn add_binary(&mut self, op: Operation, left: Node, right: Node) -> Result<Node> {
        if op.as_str() == "power" {
            if let Some(val) = right.as_int_const() {
                if let Some(left) = left.as_const() {
                    return self.create_const(left.powi(val));
                }

                match val {
                    0 => return self.create_const(1.0),
                    1 => return Ok(left),
                    2 => return self.create_unary(Operation::new("square"), left),
                    3 => return self.create_unary(Operation::new("cube"), left),
                    -1 => return self.create_unary(Operation::new("recip"), left),
                    -2 => {
                        let arg = self.create_unary(Operation::new("square"), left)?;
                        return self.create_unary(Operation::new("recip"), arg);
                    }
                    -3 => {
                        let arg = self.create_unary(Operation::new("cube"), left)?;
                        return self.create_unary(Operation::new("recip"), arg);
                    }
                    _ => {
                        return self.create_powi(left, val);
                    }
                }
            };

            if let Some(val) = right.as_const() {
                const ONE_THIRD: f64 = 1.0 / 3.0;

                match val {
                    0.5 => return self.create_unary(Operation::new("root"), left),
                    ONE_THIRD => return self.add_unary(Operation::new("cbrt"), left),
                    1.5 => {
                        let arg = self.create_unary(Operation::new("cube"), left)?;
                        return self.create_unary(Operation::new("root"), arg);
                    }
                    _ => {}
                }
            }
        }

        if !self.config.is_intrinsic_binary(&op) {
            self.ft.insert(op.to_string());
        }

        self.create_binary(op, left, right)
    }

    pub fn add_loop_prefix(
        &mut self,
        op: Operation,
        var: Node,
        start: Node,
    ) -> Result<(Node, usize)> {
        assert!(op.as_str() == "Sum" || op.as_str() == "Product");

        let accum_var = self.block().create_tmp();

        self.block().add_assign(var, start);
        let init = self.create_const(if op.as_str() == "Sum" { 0.0 } else { 1.0 })?;
        self.block().add_assign(accum_var.clone(), init);

        let label = format!(".L{}", self.count_loops);
        self.count_loops += 1;
        self.block().add_label(&label);

        Ok((accum_var, self.count_loops - 1))
    }

    pub fn add_loop_body(
        &mut self,
        op: Operation,
        eq: Node,
        var: Node,
        end: Node,
        accum_var: Node,
        loop_id: usize,
    ) -> Result<Node> {
        let p = if op.as_str() == "Sum" {
            self.create_binary(Operation::Plus, accum_var.clone(), eq)?
        } else {
            self.create_binary(Operation::Times, accum_var.clone(), eq)?
        };

        self.add_assign(accum_var.clone(), p)?;
        let one = self.create_const(1.0)?;
        let q = self.create_binary(Operation::Plus, var.clone(), one.clone())?;
        self.add_assign(var.clone(), q)?;
        let cond = self.create_binary(Operation::new("leq"), var, end)?;
        let label = format!(".L{}", loop_id);
        self.block().add_branch_if(cond, &label, true);

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

    pub fn load_args(&mut self, args: Vec<Node>) -> Result<()> {
        self.block().load_args(args);
        Ok(())
    }

    pub fn create_unary(&mut self, op: Operation, arg: Node) -> Result<Node> {
        Ok(self.block().create_unary(op, arg))
    }

    pub fn create_powi(&mut self, arg: Node, power: i32) -> Result<Node> {
        Ok(self.block().create_powi(arg, power))
    }

    pub fn create_modular_powi(&mut self, left: Node, right: Node, power: i32) -> Result<Node> {
        Ok(self.block().create_modular_powi(left, right, power))
    }

    pub fn create_binary(&mut self, op: Operation, left: Node, right: Node) -> Result<Node> {
        let node = match &op {
            Operation::Times if left.is_const(-1.0) => {
                self.create_unary(Operation::new("neg"), right)?
            }
            Operation::Times if right.is_const(-1.0) => {
                self.create_unary(Operation::new("neg"), left)?
            }
            Operation::Times if left.is_const(1.0) && !right.is_leaf_const() => right,
            Operation::Times if left.is_const(1.0) && right.is_leaf_const() => {
                self.create_unary(Operation::new("real"), right)?
            }
            Operation::Times if right.is_const(1.0) && !left.is_leaf_const() => left,
            Operation::Times if right.is_const(1.0) && left.is_leaf_const() => {
                self.create_unary(Operation::new("real"), left)?
            }
            Operation::Times if left.is_unary("recip") => {
                self.create_binary(Operation::Divide, right, left.arg().unwrap())?
            }
            Operation::Times if right.is_unary("recip") => {
                self.create_binary(Operation::Divide, left, right.arg().unwrap())?
            }
            Operation::Divide if right.is_unary("recip") => {
                self.create_binary(Operation::Times, left, right.arg().unwrap())?
            }
            Operation::Plus if left.is_unary("neg") => {
                self.create_binary(Operation::Minus, right, left.arg().unwrap())?
            }
            Operation::Plus if right.is_unary("neg") => {
                self.create_binary(Operation::Minus, left, right.arg().unwrap())?
            }
            Operation::Op(s) => match s.as_str() {
                "rem" if left.is_unary("_powi_") && !self.config.is_complex() => {
                    let (arg, power) = left.arg_power().unwrap();
                    self.create_modular_powi(arg, right, power)?
                }
                "min" => {
                    let cond =
                        self.create_binary(Operation::new("leq"), left.clone(), right.clone())?;
                    self.create_ifelse(cond, left, right)?
                }
                "max" => {
                    let cond =
                        self.create_binary(Operation::new("geq"), left.clone(), right.clone())?;
                    self.create_ifelse(cond, left, right)?
                }
                "heaviside" => {
                    /*
                     * In sympy, Heaviside is considered a binary operator,
                     * where the second argument is the value at 0 (defaults to 0.5).
                     */
                    let zero = self.create_const(0.0)?;
                    let one = self.create_const(1.0)?;

                    let c0 =
                        self.create_binary(Operation::new("eq"), left.clone(), zero.clone())?;
                    let x0 = self.create_ifelse(c0, right, one)?;

                    let c1 = self.create_binary(Operation::new("geq"), left, zero.clone())?;
                    self.create_ifelse(c1, x0, zero)?
                }
                // note: block() is needed here to prevent a infinite loop
                _ => self.block().create_binary(op, left, right),
            },
            _ => self.block().create_binary(op, left, right),
        };

        Ok(node)
    }

    pub fn compile_mir(&mut self, mir: &mut Mir) -> Result<()> {
        self.block().eliminate();
        let salt = self.salt.clone();
        self.block().compile(mir, salt)?;
        self.block().compile_subroutines(mir)
    }

    pub fn optimize_mir(&mut self, mir: &mut Mir) -> Result<()> {
        let opt_level = self.config.opt_level();

        if opt_level >= 1
        /*&& !self.config.compress()*/
        {
            mir.optimize_peephole(1);

            /*
                combining sin and cos into sin_cos
                currently inactive because it is in fact slower!
            */
            // mir.optimize_peephole(2);
        }

        if opt_level >= 2 {
            GreedyAllocator::new(self.config.clone(), self.config.count_scratch() as usize)
                .optimize(mir)?;
        }

        if opt_level >= 3 {
            // ColoringAllocator::new(self.config.clone()).optimize(mir)?;
        }

        if self.config.compact() {
            self.count_stack = Compactor::new(self.config.clone()).compact(mir).ok();
        }

        mir.add_consts(&self.consts);
        mir.populate_labels();

        Ok(())
    }

    fn save_registers(mir: &Mir, ir: &mut impl Generator) {
        if ir.count_shadows() < mir.config.count_scratch() {
            let used = mir.used_registers();
            ir.save_used_registers(&used);
        }
    }

    fn restore_registers(mir: &Mir, ir: &mut impl Generator) {
        if ir.count_shadows() < mir.config.count_scratch() {
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
        count_params: usize,
    ) -> Result<()> {
        let cap = self.stack_size();
        ir.prologue_indirect(cap, count_states, count_obs, count_params);

        Self::save_registers(mir, ir);
        mir.rerun(ir)?;
        Self::restore_registers(mir, ir);

        ir.epilogue_indirect(cap, count_states, count_obs, count_params);
        ir.align();
        self.append_const_section(ir);
        self.append_vt_section(mir, ir);
        ir.seal();

        Ok(())
    }

    pub fn compile_fast_from_mir(
        &mut self,
        mir: &Mir,
        ir: &mut impl Generator,
        count_states: usize,
        count_obs: usize,
        idx_ret: i32,
    ) -> Result<()> {
        self.block().eliminate();
        let cap = self.stack_size();
        ir.prologue_fast(cap, count_states, count_obs);

        Self::save_registers(mir, ir);
        mir.rerun(ir)?;
        Self::restore_registers(mir, ir);

        ir.epilogue_fast(cap, count_states, count_obs, idx_ret);
        ir.align();
        self.append_const_section(ir);
        self.append_vt_section(mir, ir);
        ir.seal();

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

        /*
        if !self.config.is_complex() {
            let p = mir.find_op("sin_cos").expect("func not found");
            ir.add_func("sin_cos", p);
        }
        */
    }
}

impl Storage for Builder {
    fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&Self::MAGIC.to_le_bytes())?;
        stream.write_all(&self.count_loops.to_le_bytes())?;

        let stack_size = match self.count_stack {
            Some(size) => size,
            None => self.block_shared().sym_table.num_stack,
        };
        stream.write_all(&stack_size.to_le_bytes())?;

        stream.write_all(&self.consts.len().to_le_bytes())?;

        for x in self.consts.iter() {
            stream.write_all(&x.to_le_bytes())?;
        }

        stream.write_all(&self.ft.len().to_le_bytes())?;

        for s in self.ft.iter() {
            let bytes = s.as_bytes();
            let len = bytes.len();
            assert!(len < 256);

            stream.write_all(&[len as u8])?;
            stream.write_all(bytes)?;
        }

        Ok(())
    }

    fn load(stream: &mut impl Read, config: &Config) -> Result<Self> {
        let mut bytes: [u8; 8] = [0; 8];

        stream.read_exact(&mut bytes)?;

        if usize::from_le_bytes(bytes) != Self::MAGIC {
            return Err(anyhow!("invalid magic number (Program)"));
        }

        let mut builder = Builder::new(config.clone());

        stream.read_exact(&mut bytes)?;
        builder.count_loops = usize::from_le_bytes(bytes);

        stream.read_exact(&mut bytes)?;
        builder.count_stack = Some(usize::from_le_bytes(bytes));

        stream.read_exact(&mut bytes)?;
        let num_consts = usize::from_le_bytes(bytes);

        for _ in 0..num_consts {
            stream.read_exact(&mut bytes)?;
            builder.consts.push(f64::from_le_bytes(bytes));
        }

        stream.read_exact(&mut bytes)?;
        let num_ft = usize::from_le_bytes(bytes);

        for _ in 0..num_ft {
            stream.read_exact(&mut bytes[0..1])?;
            let n = bytes[0] as usize;
            let mut buf: Vec<u8> = vec![0; n];
            stream.read_exact(&mut buf)?;
            builder.ft.insert(String::from_utf8(buf)?);
        }

        Ok(builder)
    }
}
