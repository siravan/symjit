//! Compile-fail tests of the `#[kernel]` macro (cargo feature `composer`): each file of
//! `tests/ui` must fail to compile with the error in its `.stderr` file. After a change of
//! the messages, `TRYBUILD=overwrite cargo test --features composer --test kernel_ui`
//! rewrites them.
#![cfg(feature = "composer")]

#[test]
fn compile_errors() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
