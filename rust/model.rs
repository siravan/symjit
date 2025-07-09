use anyhow::{anyhow, Result};

use serde::Deserialize;

use crate::builder::Builder;
use crate::node::Node;

pub trait Transformer {
    fn transform(&self, builder: &mut Builder) -> Result<Node>;
}

/// Collects the intermediate code (builder) and interface variables
#[derive(Debug)]
pub struct Program {
    pub builder: Builder,
    pub count_states: usize,
    pub count_params: usize,
    pub count_obs: usize,
    pub count_diffs: usize,
}

impl Program {
    pub fn new(ml: &CellModel) -> Result<Program> {
        /*
            this section lays the memory format
            the order of different sections is important!

            the layout is:

            +------------------------+
            | state variables        |
            +------------------------+
            | independent variable   | *
            +------------------------+
            | parameters             |
            +------------------------+
            | observables (output)   | **
            +------------------------+
            | differentials (output) |
            +------------------------+

            * => the independent variable slot is always allocated, even if not an ODE
            ** => => the first observable is the return value for fast functions
        */

        let mut builder = Builder::new();

        for v in &ml.states {
            builder.sym_table.add_mem(&v.name);
        }

        builder.sym_table.add_mem(&ml.iv.name);

        for v in &ml.params {
            builder.sym_table.add_mem(&v.name);
        }

        for eq in &ml.obs {
            if let Some(name) = eq.lhs.var() {
                builder.sym_table.add_mem(&name);
            } else {
                return Err(anyhow!("lhs var not found"));
            }
        }

        for eq in &ml.odes {
            if let Some(name) = eq.lhs.diff_var() {
                let name = format!("δ{}", name);
                builder.sym_table.add_mem(&name);
            } else {
                return Err(anyhow!("lhs diff var not found"));
            }
        }

        ml.transform(&mut builder)?;

        let prog = Program {
            builder,
            count_states: ml.states.len(),
            count_params: ml.params.len(),
            count_obs: ml.obs.len(),
            count_diffs: ml.odes.len(),
        };

        Ok(prog)
    }
}

/// A defined (state or param) variable
#[derive(Debug, Clone, Deserialize)]
pub struct Variable {
    pub name: String,
}

/// Transforms the input tree to the intermediate representation (tree-like)
impl Transformer for Variable {
    fn transform(&self, builder: &mut Builder) -> Result<Node> {
        builder.create_var(&self.name)
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

    //**************** Transformations *****************//

    fn transform_unary(&self, builder: &mut Builder, op: &str, args: &[Expr]) -> Result<Node> {
        let x = args[0].transform(builder)?;

        if builder.intrinsic_unary.contains(&op) {
            builder.create_unary(op, x)
        } else {
            builder.add_call_unary(op, x)
        }
    }

    fn transform_binary(&self, builder: &mut Builder, op: &str, args: &[Expr]) -> Result<Node> {
        let l = args[0].transform(builder)?;
        let r = args[1].transform(builder)?;

        if builder.intrinsic_binary.contains(&op) {
            builder.create_binary(op, l, r)
        } else {
            builder.add_call_binary(op, l, r)
        }
    }

    /// Ternary operator is the conditional select operator
    fn transform_ternary(&self, builder: &mut Builder, op: &str, args: &[Expr]) -> Result<Node> {
        if op != "ifelse" {
            return self.transform_poly(builder, op, args);
        }

        let cond = args[0].transform(builder)?;
        let true_val = args[1].transform(builder)?;
        let false_val = args[2].transform(builder)?;

        builder.add_ifelse(cond, true_val, false_val)
    }

    /// Addition and Multiplication can haev multiple arguments
    /// The intermediate tree has only unary and binary nodes
    fn transform_poly(&self, builder: &mut Builder, op: &str, args: &[Expr]) -> Result<Node> {
        if !(op == "plus" || op == "times") {
            return Err(anyhow!("missing poly op: {}", op));
        }

        let mut x = args[0].transform(builder)?;

        for arg in args.iter().skip(1) {
            let y = arg.transform(builder)?;
            x = builder.create_binary(op, x, y)?;
        }

        Ok(x)
    }
}

impl Transformer for Expr {
    fn transform(&self, builder: &mut Builder) -> Result<Node> {
        let dst = match self {
            Expr::Const { val } => builder.create_const(*val)?,
            Expr::Var { name } => builder.create_var(name)?,
            Expr::Tree { op, args } => match args.len() {
                1 => self.transform_unary(builder, op.as_str(), args)?,
                2 => self.transform_binary(builder, op.as_str(), args)?,
                3 => self.transform_ternary(builder, op.as_str(), args)?,
                _ => self.transform_poly(builder, op.as_str(), args)?,
            },
        };
        Ok(dst)
    }
}

/// Represents lhs ~ rhs
#[derive(Debug, Clone, Deserialize)]
pub struct Equation {
    pub lhs: Expr,
    pub rhs: Expr,
}

impl Transformer for Equation {
    fn transform(&self, builder: &mut Builder) -> Result<Node> {
        let var = if let Some(var) = self.lhs.diff_var() {
            format!("δ{}", var)
        } else if let Some(var) = self.lhs.var() {
            var
        } else {
            return Err(anyhow!("lhs should be a variable"));
        };

        let rhs = self.rhs.transform(builder)?;
        let lhs = builder.create_var(var.as_str())?;
        builder.add_assign(lhs, rhs)?;
        builder.create_void()
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

impl Transformer for CellModel {
    fn transform(&self, builder: &mut Builder) -> Result<Node> {
        for eq in &self.obs {
            eq.transform(builder)?;
        }

        for eq in &self.odes {
            eq.transform(builder)?;
        }

        builder.create_void()
    }
}
