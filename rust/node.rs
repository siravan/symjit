use crate::generator::Generator;
use crate::symbol::Loc;
use crate::utils::Eval;


#[derive(Debug, Clone)]
pub enum Node {
    Void,
    Const {
        val: f64,
        idx: u32,
    },
    Var {
        name: String,
        loc: Loc,
    },
    Unary {
        op: String,
        arg: Box<Node>,
        ershov: usize,
    },
    Binary {
        op: String,
        left: Box<Node>,
        right: Box<Node>,
        ershov: usize,
    },
}

impl Node {
    pub fn ershov_number(&self) -> usize {
        match self {
            Node::Void => 0,
            Node::Const { .. } => 1,
            Node::Var { .. } => 1,
            Node::Unary { ershov, .. } => *ershov,
            Node::Binary { ershov, .. } => *ershov,
        }
    }

    pub fn compile(&self, ir: &mut dyn Generator, base: u8) -> u8 {
        match self {
            Node::Void => 0,
            Node::Const { .. } => self.compile_const(ir, base),
            Node::Var { .. } => self.compile_var(ir, base),
            Node::Unary { .. } => self.compile_unary(ir, base),
            Node::Binary { .. } => self.compile_binary(ir, base),
        }
    }

    fn compile_const(&self, ir: &mut dyn Generator, base: u8) -> u8 {
        if let Node::Const { idx, .. } = &self {
            let r = ir.first_shadow() + base;
            let label = format!("_const_{}_", idx);
            ir.load_const(r, &label);
            r
        } else {
            panic!("should not get here!");
        }
    }

    fn compile_var(&self, ir: &mut dyn Generator, base: u8) -> u8 {
        if let Node::Var { loc, .. } = &self {
            let r = ir.first_shadow() + base;
            match loc {
                Loc::Stack(idx) => ir.load_stack(r, *idx),
                Loc::Mem(idx) => ir.load_mem(r, *idx),
            };
            r
        } else {
            panic!("should not get here!");
        }
    }

    fn compile_unary(&self, ir: &mut dyn Generator, base: u8) -> u8 {
        if let Node::Unary { op, arg, ershov } = self {
            let r = arg.compile(ir, base);

            match op.as_str() {
                "neg" => ir.neg(r),
                "not" => ir.not(r),
                "abs" => ir.abs(r),
                "root" => ir.root(r),
                "square" => ir.square(r),
                "cube" => ir.cube(r),
                "recip" => ir.recip(r),
                "_call_" => {
                    if r != 0 {
                        ir.fmov(0, r);
                    }
                }
                _ => panic!("unary operation is not recognized"),
            };

            r
        } else {
            panic!("should not get here!");
        }
    }

    fn compile_binary(&self, ir: &mut dyn Generator, base: u8) -> u8 {
        if let Node::Binary {
            op,
            left,
            right,
            ershov,
        } = self
        {
            let (dst, l, r) = self.alloc(ir, base, left, right, *ershov);

            match op.as_str() {
                "plus" => ir.plus(dst, l, r),
                "minus" => ir.minus(dst, l, r),
                "times" => ir.times(dst, l, r),
                "divide" => ir.divide(dst, l, r),
                "gt" => ir.gt(dst, l, r),
                "geq" => ir.geq(dst, l, r),
                "lt" => ir.lt(dst, l, r),
                "leq" => ir.leq(dst, l, r),
                "eq" => ir.eq(dst, l, r),
                "neq" => ir.neq(dst, l, r),
                "and" => ir.and(dst, l, r),
                "or" => ir.or(dst, l, r),
                "xor" => ir.xor(dst, l, r),
                "select_if" => ir.select_if(dst, l, r),
                "select_else" => ir.select_else(dst, l, r),
                "_call_" => Self::call(ir, l, r),
                _ => panic!("binary operation is not recognized"),
            };

            dst
        } else {
            panic!("should not get here!");
        }
    }

    fn alloc(
        &self,
        ir: &mut dyn Generator,
        base: u8,
        left: &Node,
        right: &Node,
        ershov: usize,
    ) -> (u8, u8, u8) {
        let mut dst = ir.first_shadow() + base + (ershov as u8) - 1;

        let el = left.ershov_number();
        let er = right.ershov_number();

        let mut l = 0;
        let mut r = 0;

        let last = ir.first_shadow() + ir.count_shadows();

        if dst < last {
            if el == er {
                l = left.compile(ir, base + 1);
                r = right.compile(ir, base);
            } else if el > er {
                l = left.compile(ir, base);
                r = right.compile(ir, base);
            } else {
                r = right.compile(ir, base);
                l = left.compile(ir, base);
            }
        } else {
            let spill: u32 = (dst - last) as u32;

            if er <= el {
                l = left.compile(ir, 0);
                ir.save_stack(l, spill);
                r = right.compile(ir, 0);
                l = 1;
                ir.load_stack(l, spill);
            } else {
                r = right.compile(ir, 0);
                ir.save_stack(r, spill);
                l = left.compile(ir, 0);
                r = 1;
                ir.load_stack(r, spill);
            }

            dst = 0;
        };

        (dst, l, r)
    }

    fn call(ir: &mut dyn Generator, l: u8, r: u8) {
        if l == 1 && r == 0 {
            ir.fxchg(1, 0);
        } else if r == 0 {
            ir.fmov(1, 0);
            if l != 0 {
                ir.fmov(0, l);
            }
        } else {
            if l != 0 {
                ir.fmov(0, l);
            }
            if r != 0 {
                ir.fmov(1, r);
            }
        }
    }
}


impl Eval for Node {
    fn eval(&self, mem: &mut [f64], stack: &mut [f64]) -> f64 {
        const T: f64 = 1.0;
        const F: f64 = 0.0;

        match self {
            Node::Void => 0.0,
            Node::Const { val, .. } => *val,
            Node::Var { loc, .. } => match loc {
                Loc::Stack(idx) => stack[*idx as usize],
                Loc::Mem(idx) => mem[*idx as usize],
            },
            Node::Unary { op, arg, .. } => {
                let x = arg.eval(mem, stack);

                match op.as_str() {
                    "neg" => -x,
                    "not" => T - x,
                    "abs" => x.abs(),
                    "root" => x.sqrt(),
                    "square" => x * x,
                    "cube" => x * x * x,
                    "recip" => 1.0 / x,
                    "_call_" => {
                        stack[0] = x;
                        x
                    }
                    _ => f64::NAN,
                }
            }
            Node::Binary {
                op, left, right, ..
            } => {
                let x = left.eval(mem, stack);
                let y = right.eval(mem, stack);

                match op.as_str() {
                    "plus" => x + y,
                    "minus" => x - y,
                    "times" => x * y,
                    "divide" => x / y,
                    "gt" => {
                        if x > y {
                            T
                        } else {
                            F
                        }
                    }
                    "geq" => {
                        if x >= y {
                            T
                        } else {
                            F
                        }
                    }
                    "lt" => {
                        if x < y {
                            T
                        } else {
                            F
                        }
                    }
                    "leq" => {
                        if x <= y {
                            T
                        } else {
                            F
                        }
                    }
                    "eq" => {
                        if x == y {
                            T
                        } else {
                            F
                        }
                    }
                    "neq" => {
                        if x != y {
                            T
                        } else {
                            F
                        }
                    }
                    "and" => x * y,
                    "or" => x + y,
                    "xor" => {
                        if x != y {
                            T
                        } else {
                            F
                        }
                    }
                    "select_if" => {
                        if x != 0.0 {
                            y
                        } else {
                            0.0
                        }
                    }
                    "select_else" => {
                        if x == 0.0 {
                            y
                        } else {
                            0.0
                        }
                    }
                    "_call_" => {
                        stack[0] = x;
                        stack[1] = y;
                        x
                    }
                    _ => f64::NAN,
                }
            }
        }
    }
}

