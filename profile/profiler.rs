use anyhow::Result;
use symjit::Compiler;

/*
 * For profiling using Samply,
 * 1. Build as cargo build --profile profiling --bin profiler
 * 2. Profile as samply record ./target/profiling/profiler
 */
pub fn main() -> Result<()> {
    let model = std::fs::read_to_string("3loop_instructions_2.txt")?;
    for _ in 0..10 {
        let mut compiler = Compiler::new();
        compiler.translate(model.clone(), 0)?;
    }
    println!("compiled ok!");
    Ok(())
}
