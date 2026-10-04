// Tests of the WebAssembly generator (module structure, imports, constants and
// error reporting). Execution is tested from Python (python/tests/test_wasm.py),
// which validates the modules with wabt and runs them in Node.

use super::WasmGenerator;
use crate::code::VirtualTable;
use crate::config::Config;
use crate::generator::Generator;
use crate::utils::Reg;

fn generator() -> WasmGenerator {
    WasmGenerator::new(Config::from_name("wasm", 0).unwrap())
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

// y = sin(x) * c, with x = mem[0], y = mem[1], c = consts[0]
fn build(g: &mut WasmGenerator, consts: &[f64]) {
    g.prologue_indirect(4, 1, 1, 0);
    g.load_mem(Reg::Ret, 0);
    g.call("sin", 1).unwrap();
    g.load_const(Reg::Gen(0), 0);
    g.save_mem_result(1);
    g.epilogue_indirect(4, 1, 1, 0);
    g.add_consts(consts);
    g.add_func("sin", VirtualTable::from_str("sin").unwrap());
    g.seal();
}

#[test]
fn module_header_and_exports() {
    let mut g = generator();
    build(&mut g, &[2.5]);
    g.check().unwrap();

    let m = g.bytes();
    assert_eq!(&m[..8], &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
    assert!(contains(&m, b"\x06symjit\x03sin"));
    assert!(contains(&m, b"\x03run\x00"));
    assert!(contains(&m, b"\x06memory\x02"));
    assert!(contains(&m, b"\x0b__heap_base\x03"));
}

#[test]
fn constants_are_resolved_at_seal() {
    let mut g = generator();
    build(&mut g, &[2.5]);

    let mut f64_const = vec![0x44];
    f64_const.extend_from_slice(&2.5f64.to_le_bytes());
    assert!(contains(&g.bytes(), &f64_const));
}

#[test]
fn missing_constant_is_an_error() {
    let mut g = generator();
    build(&mut g, &[]);
    assert!(g.check().is_err());
}

#[test]
fn import_arity_mismatch_is_an_error() {
    let mut g = generator();
    g.call("power", 2).unwrap();
    assert!(g.call("power", 1).is_err());
}

#[test]
fn imports_are_shared_between_calls() {
    let mut g = generator();
    g.prologue_indirect(4, 1, 1, 0);
    g.call("sin", 1).unwrap();
    g.call("sin", 1).unwrap();
    g.call("power", 2).unwrap();
    g.epilogue_indirect(4, 1, 1, 0);
    g.seal();
    g.check().unwrap();

    let m = g.bytes();
    // one import entry for both calls (the name section spells it `symjit.sin`)
    assert_eq!(m.windows(4).filter(|w| *w == b"\x03sin").count(), 1);
    assert!(contains(&m, b"\x05power"));
}

#[test]
fn rust_only_functions_are_unsupported() {
    // `Func::Slice`/`Func::App` (registered from Rust with `add_sliced_func`/`add_applet`)
    // are called through native pointers and cannot be imported
    let mut g = generator();
    g.add_func(
        "ext",
        crate::code::Func::Slice {
            f_scalar: std::ptr::null(),
            f_simd: std::ptr::null(),
            env: std::ptr::null(),
        },
    );
    let err = g.check().unwrap_err().to_string();
    assert!(err.contains("user-defined function `ext`"), "{}", err);
}

#[test]
fn jumps_are_structured() {
    // a backward jump (loop) and a forward jump produce valid nesting: one `loop`,
    // `block`s and a `br_table`, with the dispatcher locals named in the name section
    let mut g = generator();
    g.prologue_indirect(4, 1, 1, 0);
    g.set_label("top");
    g.load_mem(Reg::Gen(0), 0);
    g.branch_if(Reg::Gen(0), "skip", true);
    g.save_mem(Reg::Gen(0), 1);
    g.set_label("skip");
    g.branch_if(Reg::Gen(0), "top", false);
    g.epilogue_indirect(4, 1, 1, 0);
    g.seal();
    g.check().unwrap();

    let m = g.bytes();
    assert!(contains(&m, &[0x0e])); // br_table
    assert!(contains(&m, b"\x02pc"));
    assert!(contains(&m, b"\x06ret_pc"));
}

#[test]
fn undefined_label_is_an_error() {
    let mut g = generator();
    g.prologue_indirect(4, 1, 1, 0);
    g.branch("nowhere");
    g.epilogue_indirect(4, 1, 1, 0);
    g.seal();
    let err = g.check().unwrap_err().to_string();
    assert!(err.contains("undefined label `nowhere`"), "{}", err);
}

#[test]
fn nested_subroutine_calls_are_rejected() {
    // a subroutine body that calls a subroutine would overwrite the single return slot
    let mut g = generator();
    g.prologue_indirect(4, 1, 1, 0);
    g.call_funclet("@a");
    g.branch("@end");
    g.set_label("@a");
    g.call_funclet("@b");
    g.ret();
    g.set_label("@b");
    g.ret();
    g.set_label("@end");
    g.epilogue_indirect(4, 1, 1, 0);
    g.seal();
    let err = g.check().unwrap_err().to_string();
    assert!(err.contains("nested subroutine calls"), "{}", err);
}

#[test]
fn fast_is_exported_up_to_1000_states() {
    // the JavaScript API rejects functions with more than 1000 parameters
    for (n, exported) in [(1000, true), (1001, false)] {
        let mut g = generator();
        g.set_mem_size(n + 2);
        g.export_fast(n + 2);
        g.prologue_indirect(4, n, 1, 0);
        g.load_mem(Reg::Ret, 0);
        g.save_mem_result(n as u32);
        g.epilogue_indirect(4, n, 1, 0);
        g.seal();
        g.check().unwrap();
        assert_eq!(contains(&g.bytes(), b"\x04fast\x00"), exported, "{} states", n);
    }
}
