//! `compile`: applications from what the front ends of symjit-core make (`Composer::finish`).

use anyhow::Result;

use crate::{Application, Composer};

/// Compiles a front end's kernel into an application: `Composer::finish`, then the machine code
/// and the SIMD kernel (`Application::from_finished`). Implemented for every `Composer`
/// (`Translator`, ...); `use symjit::CompileExt` (or `use symjit::*`) to call it.
pub trait CompileExt {
    fn compile(&mut self) -> Result<Application>;
}

impl<T: Composer + ?Sized> CompileExt for T {
    fn compile(&mut self) -> Result<Application> {
        let salt = self.salt().map(|s| s.to_string());
        let compilation = self.finish()?;
        Application::from_finished(compilation, salt.as_deref())
    }
}
