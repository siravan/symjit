"""Generates rust/wasm/tests.rs: WebAssembly encodings pinned against wabt's wat2wasm.

Each case is a function body in WAT and the same instructions as `encoder::Op`s.
The script assembles the body with wat2wasm, extracts the instruction bytes of the
function from the code section, and writes a test asserting that the encoder
produces identical bytes.

    python3 rust/wasm/gen_tests.py > rust/wasm/tests.rs
"""

import os
import subprocess
import sys
import tempfile

# (name, locals, params, results, wat body, rust ops)
# locals: WAT local types; local indices are params first, then locals.
CASES = [
    ("unreachable", "", "", "", "unreachable", "Op::Unreachable"),
    ("nop", "", "", "", "nop", "Op::Nop"),
    ("block_empty", "", "", "", "block end", "Op::Block(BlockType::Empty), Op::End"),
    ("block_f64", "", "", "f64", "block (result f64) f64.const 1 end",
     "Op::Block(BlockType::Value(ValType::F64)), Op::F64Const(1.0), Op::End"),
    ("loop_br_if", "i32", "", "", "loop local.get 0 br_if 0 end",
     "Op::Loop(BlockType::Empty), Op::LocalGet(0), Op::BrIf(0), Op::End"),
    ("if_else", "i32", "", "", "local.get 0 if nop else nop end",
     "Op::LocalGet(0), Op::If(BlockType::Empty), Op::Nop, Op::Else, Op::Nop, Op::End"),
    ("if_i32", "i32", "", "i32", "local.get 0 if (result i32) i32.const 1 else i32.const 2 end",
     "Op::LocalGet(0), Op::If(BlockType::Value(ValType::I32)), Op::I32Const(1), Op::Else, Op::I32Const(2), Op::End"),
    ("br", "", "", "", "block br 0 end", "Op::Block(BlockType::Empty), Op::Br(0), Op::End"),
    ("br_table", "i32", "", "",
     "block block block local.get 0 br_table 0 1 2 end end end",
     "Op::Block(BlockType::Empty), Op::Block(BlockType::Empty), Op::Block(BlockType::Empty), "
     "Op::LocalGet(0), Op::BrTable(vec![0, 1], 2), Op::End, Op::End, Op::End"),
    ("return_", "", "", "i32", "i32.const 0 return", "Op::I32Const(0), Op::Return"),
    ("drop_select", "i32", "", "f64",
     "f64.const 1 drop f64.const 1 f64.const 2 local.get 0 select",
     "Op::F64Const(1.0), Op::Drop, Op::F64Const(1.0), Op::F64Const(2.0), Op::LocalGet(0), Op::Select"),
    ("locals", "f64 f64", "", "",
     "local.get 1 local.set 0 local.get 0 local.tee 1 drop",
     "Op::LocalGet(1), Op::LocalSet(0), Op::LocalGet(0), Op::LocalTee(1), Op::Drop"),
    ("local_index_128", "f64 " * 129, "", "", "local.get 128 local.set 127",
     "Op::LocalGet(128), Op::LocalSet(127)"),
    ("globals", "", "", "", "global.get 0 global.set 0", "Op::GlobalGet(0), Op::GlobalSet(0)"),
    ("i32_load_store", "", "", "",
     "i32.const 0 i32.const 0 i32.load i32.store offset=4",
     "Op::I32Const(0), Op::I32Const(0), Op::I32Load(MemArg::i32(0)), Op::I32Store(MemArg::i32(4))"),
    ("i64_load_store", "", "", "",
     "i32.const 0 i32.const 8 i64.load i64.store offset=16",
     "Op::I32Const(0), Op::I32Const(8), Op::I64Load(MemArg::f64(0)), Op::I64Store(MemArg::f64(16))"),
    ("f64_load_store", "", "", "",
     "i32.const 0 i32.const 0 f64.load offset=8 f64.store offset=128",
     "Op::I32Const(0), Op::I32Const(0), Op::F64Load(MemArg::f64(8)), Op::F64Store(MemArg::f64(128))"),
    ("f64_load_offset_65536", "", "", "f64", "i32.const 0 f64.load offset=65536",
     "Op::I32Const(0), Op::F64Load(MemArg::f64(65536))"),
    ("i32_const_edges", "", "", "",
     "i32.const 0 drop i32.const 63 drop i32.const 64 drop i32.const -64 drop i32.const -65 drop "
     "i32.const 2147483647 drop i32.const -2147483648 drop",
     "Op::I32Const(0), Op::Drop, Op::I32Const(63), Op::Drop, Op::I32Const(64), Op::Drop, "
     "Op::I32Const(-64), Op::Drop, Op::I32Const(-65), Op::Drop, Op::I32Const(i32::MAX), Op::Drop, "
     "Op::I32Const(i32::MIN), Op::Drop"),
    ("i64_const_edges", "", "", "",
     "i64.const -1 drop i64.const 9223372036854775807 drop i64.const -9223372036854775808 drop",
     "Op::I64Const(-1), Op::Drop, Op::I64Const(i64::MAX), Op::Drop, Op::I64Const(i64::MIN), Op::Drop"),
    ("f64_const", "", "", "",
     "f64.const 0 drop f64.const -0 drop f64.const 3.141592653589793 drop f64.const inf drop",
     "Op::F64Const(0.0), Op::Drop, Op::F64Const(-0.0), Op::Drop, "
     "Op::F64Const(std::f64::consts::PI), Op::Drop, Op::F64Const(f64::INFINITY), Op::Drop"),
    ("i32_compare", "i32 i32", "", "",
     " ".join(f"local.get 0 local.get 1 i32.{c} drop" for c in
              ["eq", "ne", "lt_s", "lt_u", "gt_s", "gt_u", "le_s", "le_u", "ge_s", "ge_u"])
     + " local.get 0 i32.eqz drop",
     ", ".join(f"Op::LocalGet(0), Op::LocalGet(1), Op::{c}, Op::Drop" for c in
               ["I32Eq", "I32Ne", "I32LtS", "I32LtU", "I32GtS", "I32GtU", "I32LeS", "I32LeU", "I32GeS", "I32GeU"])
     + ", Op::LocalGet(0), Op::I32Eqz, Op::Drop"),
    ("i32_arith", "i32 i32", "", "",
     " ".join(f"local.get 0 local.get 1 i32.{c} drop" for c in ["add", "sub", "mul", "and", "shl"]),
     ", ".join(f"Op::LocalGet(0), Op::LocalGet(1), Op::{c}, Op::Drop" for c in
               ["I32Add", "I32Sub", "I32Mul", "I32And", "I32Shl"])),
    ("i64_ops", "i64 i64", "", "",
     " ".join(f"local.get 0 local.get 1 i64.{c} drop" for c in ["sub", "and", "or", "xor"])
     + " local.get 0 i64.eqz drop",
     ", ".join(f"Op::LocalGet(0), Op::LocalGet(1), Op::{c}, Op::Drop" for c in
               ["I64Sub", "I64And", "I64Or", "I64Xor"])
     + ", Op::LocalGet(0), Op::I64Eqz, Op::Drop"),
    ("f64_compare", "f64 f64", "", "",
     " ".join(f"local.get 0 local.get 1 f64.{c} drop" for c in ["eq", "ne", "lt", "gt", "le", "ge"]),
     ", ".join(f"Op::LocalGet(0), Op::LocalGet(1), Op::{c}, Op::Drop" for c in
               ["F64Eq", "F64Ne", "F64Lt", "F64Gt", "F64Le", "F64Ge"])),
    ("f64_unary", "f64", "", "",
     " ".join(f"local.get 0 f64.{c} drop" for c in ["abs", "neg", "ceil", "floor", "trunc", "nearest", "sqrt"]),
     ", ".join(f"Op::LocalGet(0), Op::{c}, Op::Drop" for c in
               ["F64Abs", "F64Neg", "F64Ceil", "F64Floor", "F64Trunc", "F64Nearest", "F64Sqrt"])),
    ("f64_binary", "f64 f64", "", "",
     " ".join(f"local.get 0 local.get 1 f64.{c} drop" for c in
              ["add", "sub", "mul", "div", "min", "max", "copysign"]),
     ", ".join(f"Op::LocalGet(0), Op::LocalGet(1), Op::{c}, Op::Drop" for c in
               ["F64Add", "F64Sub", "F64Mul", "F64Div", "F64Min", "F64Max", "F64Copysign"])),
    ("conversions", "i32 f64", "", "",
     "local.get 0 i64.extend_i32_u drop local.get 1 i64.reinterpret_f64 f64.reinterpret_i64 drop",
     "Op::LocalGet(0), Op::I64ExtendI32U, Op::Drop, Op::LocalGet(1), Op::I64ReinterpretF64, "
     "Op::F64ReinterpretI64, Op::Drop"),
    ("call", "", "", "", "f64.const 1 call 0 drop", "Op::F64Const(1.0), Op::Call(0), Op::Drop"),
]


def leb_u32(b, i):
    result, shift = 0, 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte & 0x80 == 0:
            return result, i


def body_bytes(wasm):
    """Instruction bytes of the last function (locals header skipped, final `end` kept)."""
    i = 8
    while i < len(wasm):
        sid = wasm[i]
        size, j = leb_u32(wasm, i + 1)
        if sid == 10:
            n, k = leb_u32(wasm, j)
            body = None
            for _ in range(n):
                fsize, k = leb_u32(wasm, k)
                body = (k, k + fsize)
                k += fsize
            start, end = body
            nloc, k = leb_u32(wasm, start)
            for _ in range(nloc):
                _, k = leb_u32(wasm, k)
                k += 1
            return wasm[k:end]
        i = j + size
    raise ValueError("no code section")


def assemble(locals_, results, body):
    loc = f"(local {locals_})" if locals_.strip() else ""
    res = f"(result {results})" if results else ""
    wat = f"""(module
  (import "env" "f" (func (param f64) (result f64)))
  (memory 1)
  (global (mut i32) (i32.const 0))
  (func {res} {loc} {body}))
"""
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "t.wat"), os.path.join(d, "t.wasm")
        with open(src, "w") as fd:
            fd.write(wat)
        subprocess.run(["wat2wasm", src, "-o", out], check=True)
        with open(out, "rb") as fd:
            return fd.read()


def main():
    version = subprocess.run(["wat2wasm", "--version"], capture_output=True, text=True).stdout.strip()
    out = sys.stdout
    out.write(f"""// Generated by rust/wasm/gen_tests.py from wabt wat2wasm {version}; do not edit by hand.
//
// Each test encodes a sequence of `Op`s and compares it with the function body
// that wat2wasm assembles from the equivalent WAT (shown in the comment).

use super::encoder::{{BlockType, MemArg, Op, ValType}};

fn encode(ops: &[Op]) -> Vec<u8> {{
    let mut buf = Vec::new();
    for op in ops {{
        op.encode(&mut buf);
    }}
    Op::End.encode(&mut buf);
    buf
}}
""")
    for name, locals_, _params, results, body, ops in CASES:
        expected = body_bytes(assemble(locals_, results, body))
        hexes = ", ".join(f"0x{b:02x}" for b in expected)
        out.write(f"""
#[test]
fn {name}() {{
    // {body}
    let ops = vec![{ops}];
    assert_eq!(encode(&ops), vec![{hexes}]);
}}
""")


if __name__ == "__main__":
    main()
