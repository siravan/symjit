// Lowers MIR control flow (labels, jumps, conditional jumps and the internal
// subroutines entered with `call_funclet`) to WebAssembly's structured control flow.
//
// The code is split into basic blocks at labels and after every jump. Block k is
// placed after the `end` of the k-th of N nested `block`s inside a `loop`:
//
//     loop $dispatch
//       block $b[N-1] ... block $b1 block $b0
//         local.get $pc  br_table $b0 $b1 ... $b[N-1]
//       end  <code of block 0>
//       end  <code of block 1>
//       ...
//       end  <code of block N-1>
//     end
//
// so a block falls through into the next one. A forward jump from block k to block
// j > k is `br` to $b[j] (depth j-k-1), which lands at the start of block j. A
// backward jump stores j in $pc and branches to $dispatch (depth N-1-k), whose
// `br_table` continues at block j. A subroutine call stores the number of the block
// after the call in $ret_pc and jumps to the subroutine; its return dispatches on
// $ret_pc. Subroutines must not call subroutines (the only one, complexify's
// `@complex_root`, is a leaf); `lower` rejects nested calls.

use std::collections::HashMap;

use super::encoder::{BlockType, Op};

pub enum Flow {
    Op(Op),
    Label(String),
    Branch(String),
    /// jump if the f64 in local `cond` is not all zeros (`is_else`, as in the
    /// native backends and the bytecode interpreter) or if it is all zeros
    BranchIf {
        cond: u32,
        label: String,
        is_else: bool,
    },
    CallFunclet(String),
    Ret,
}

impl Flow {
    fn is_control(&self) -> bool {
        !matches!(self, Flow::Op(_))
    }
}

enum Term {
    Fall,
    Branch(usize),
    BranchIf { cond: u32, target: usize, is_else: bool },
    CallFunclet(usize),
    Ret,
}

struct Block {
    ops: Vec<Op>,
    term: Term,
}

/// Lowers `items` (the body of a function returning one i32, the last value left on
/// the stack by the final block) to structured code; `l_pc` and `l_ret_pc` are i32
/// locals. Fails with the name of an undefined label.
pub fn lower(items: Vec<Flow>, l_pc: u32, l_ret_pc: u32) -> Result<Vec<Op>, String> {
    if !items.iter().any(Flow::is_control) {
        return Ok(items
            .into_iter()
            .map(|f| match f {
                Flow::Op(op) => op,
                _ => unreachable!(),
            })
            .collect());
    }

    // pass 1: block boundaries and label -> block
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut count = 1;
    let mut open = false; // the current block has instructions or a terminator

    for f in items.iter() {
        match f {
            Flow::Op(_) => open = true,
            Flow::Label(l) => {
                if open {
                    count += 1;
                    open = false;
                }
                labels.entry(l.clone()).or_insert(count - 1);
            }
            _ => {
                count += 1;
                open = false;
            }
        }
    }

    let find = |l: &str| -> Result<usize, String> {
        labels
            .get(l)
            .copied()
            .ok_or_else(|| format!("branch to undefined label `{}`", l))
    };

    // pass 2: the blocks
    let mut blocks: Vec<Block> = Vec::with_capacity(count);
    let mut ops: Vec<Op> = Vec::new();
    let mut open = false;

    for f in items.into_iter() {
        let term = match f {
            Flow::Op(op) => {
                ops.push(op);
                open = true;
                continue;
            }
            Flow::Label(_) => {
                if open {
                    blocks.push(Block {
                        ops: std::mem::take(&mut ops),
                        term: Term::Fall,
                    });
                    open = false;
                }
                continue;
            }
            Flow::Branch(l) => Term::Branch(find(&l)?),
            Flow::BranchIf {
                cond,
                label,
                is_else,
            } => Term::BranchIf {
                cond,
                target: find(&label)?,
                is_else,
            },
            Flow::CallFunclet(l) => Term::CallFunclet(find(&l)?),
            Flow::Ret => Term::Ret,
        };
        blocks.push(Block {
            ops: std::mem::take(&mut ops),
            term,
        });
        open = false;
    }
    blocks.push(Block {
        ops,
        term: Term::Fall,
    });
    assert_eq!(blocks.len(), count);

    // a subroutine body (from its entry block to its first return) must not call a
    // subroutine: there is a single return slot
    for b in blocks.iter() {
        if let Term::CallFunclet(entry) = b.term {
            for body in blocks[entry..].iter() {
                match body.term {
                    Term::Ret => break,
                    Term::CallFunclet(_) => return Err("nested subroutine calls".to_string()),
                    _ => {}
                }
            }
        }
    }

    // pass 3: emit
    let n = blocks.len();
    let mut out: Vec<Op> = Vec::new();

    out.push(Op::Loop(BlockType::Empty));
    for _ in 0..n {
        out.push(Op::Block(BlockType::Empty));
    }
    out.push(Op::LocalGet(l_pc));
    out.push(Op::BrTable((0..n as u32).collect(), n as u32 - 1));

    // `br` from block k to the start of block j (forward) or through the dispatcher
    let jump = |out: &mut Vec<Op>, k: usize, j: usize, depth_extra: u32| {
        if j > k {
            out.push(Op::Br((j - k - 1) as u32 + depth_extra));
        } else {
            out.push(Op::I32Const(j as i32));
            out.push(Op::LocalSet(l_pc));
            out.push(Op::Br((n - 1 - k) as u32 + depth_extra));
        }
    };

    for (k, b) in blocks.into_iter().enumerate() {
        out.push(Op::End);
        out.extend(b.ops);

        match b.term {
            Term::Fall => {
                if k == n - 1 {
                    // the epilogue's status code is on the stack
                    out.push(Op::Return);
                }
            }
            Term::Branch(j) => jump(&mut out, k, j, 0),
            Term::BranchIf {
                cond,
                target,
                is_else,
            } => {
                out.push(Op::LocalGet(cond));
                out.push(Op::I64ReinterpretF64);
                out.push(Op::I64Eqz);
                if is_else {
                    out.push(Op::I32Eqz);
                }
                if target > k {
                    out.push(Op::BrIf((target - k - 1) as u32));
                } else {
                    out.push(Op::If(BlockType::Empty));
                    jump(&mut out, k, target, 1);
                    out.push(Op::End);
                }
            }
            Term::CallFunclet(j) => {
                if k + 1 >= n {
                    return Err("subroutine call without a continuation".to_string());
                }
                out.push(Op::I32Const(k as i32 + 1));
                out.push(Op::LocalSet(l_ret_pc));
                jump(&mut out, k, j, 0);
            }
            Term::Ret => {
                out.push(Op::LocalGet(l_ret_pc));
                out.push(Op::LocalSet(l_pc));
                out.push(Op::Br((n - 1 - k) as u32));
            }
        }
    }

    out.push(Op::End); // loop
    out.push(Op::Unreachable); // every path returns from the last block
    Ok(out)
}
