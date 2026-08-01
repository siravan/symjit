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
