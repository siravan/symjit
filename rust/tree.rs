use crate::model::Expr;

#[derive(Debug, Clone)]
pub enum Node {
    Void,
    Const {
        val: f64,
    },
    Var {
        name: String,
    },
    Unary {
        op: String,
        arg: Box<Node>,
    },
    Binary {
        op: String,
        left: Box<Node>,
        right: Box<Node>,
    },
}

#[derive(Debug, Clone)]
pub enum Statement {
    Assign {
        lhs: String,
        rhs: Node,
    },
    Call {
        op: String,
        lhs: String,
        args: Vec<Node>,
    },
}

#[derive(Debug, Clone)]
pub struct Builder {
    stmts: Vec<Statement>,
    num_temps: usize,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            stmts: Vec::new(),
            num_temps: 0,
        }
    }

    pub fn add_stmt(&mut self, st: Statement) {
        self.stmts.push(st);
    }

    pub fn create_temp(&mut self) -> String {
        let name = format!("ψ{}", self.num_temps);
        self.num_temps += 1;
        name
    }
}

pub trait Transformer {
    fn transform(&self, builder: &mut Builder) -> Node;
}
