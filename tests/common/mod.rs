use std::sync::Once;
use symjit::Config;

static TEST_TOML: Once = Once::new();

/// `Config::default()` reads `$SYMJIT_TOML`, else `symjit.toml` in the working directory.
/// Point `SYMJIT_TOML` at `tests/symjit.toml` so the tests run with known options. Every
/// test calls this before it compiles anything, so after the `Once` no thread writes the
/// environment again; the library's own `Config::default()` calls see the same file.
pub fn base_config() -> Config {
    TEST_TOML.call_once(|| {
        // SAFETY: no other thread reads or writes the environment meanwhile (see above).
        unsafe {
            std::env::set_var(
                "SYMJIT_TOML",
                concat!(env!("CARGO_MANIFEST_DIR"), "/tests/symjit.toml"),
            )
        };
    });
    Config::default()
}
