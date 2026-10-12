#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operation {
    Op(String),
    Plus,
    Minus,
    Times,
    Divide,
}

impl Operation {
    pub fn new(op: &str) -> Operation {
        match op {
            "plus" | "minus" | "times" | "divide" => panic!("invalid op {}", op),
            s => Operation::Op(s.into()),
        }
    }

    /// The operation `op` of a `Composer::append_fun` call. With the `composer` feature, the
    /// arithmetic operations ("minus", "divide", ...) are accepted, for the `-` and `/` of the
    /// kernel builder (`rust/kernel.rs`); without it, they panic, as `new` does.
    pub fn composer_op(op: &str) -> Operation {
        #[cfg(feature = "composer")]
        {
            Operation::new_checked(op)
        }
        #[cfg(not(feature = "composer"))]
        {
            Operation::new(op)
        }
    }

    pub fn new_checked(op: &str) -> Operation {
        match op {
            "plus" => Operation::Plus,
            "minus" => Operation::Minus,
            "times" => Operation::Times,
            "divide" => Operation::Divide,
            s => Operation::Op(s.into()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Operation::Plus => "plus",
            Operation::Minus => "minus",
            Operation::Times => "times",
            Operation::Divide => "divide",
            Operation::Op(s) => s.as_str(),
        }
    }

    pub fn is_plus(&self) -> bool {
        matches!(self, Operation::Plus)
    }

    pub fn is_minus(&self) -> bool {
        matches!(self, Operation::Minus)
    }

    pub fn is_times(&self) -> bool {
        matches!(self, Operation::Times)
    }

    pub fn is_divide(&self) -> bool {
        matches!(self, Operation::Divide)
    }

    pub fn to_string(&self) -> String {
        self.as_str().into()
    }
}
