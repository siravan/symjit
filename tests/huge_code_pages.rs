// Machine code compiled with the `huge` option lives on huge pages (Linux, `hugepage_rs`); when
// the last owner of the code drops, those pages must go back through `hugepage_rs`, not the global
// allocator. Each round compiles a kernel on huge pages, evaluates it and drops it. Needs
// preallocated huge pages (`/proc/sys/vm/nr_hugepages`); without them the test does nothing.

mod common;
use common::base_config;

use num_complex::Complex;
use symjit::{Composer, Slot, Translator};

fn free_huge_pages() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|line| line.starts_with("HugePages_Free:"))
                .and_then(|line| line.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

#[test]
fn kernels_on_huge_pages_are_released_through_their_allocator() {
    if !cfg!(target_os = "linux") || free_huge_pages() < 2 {
        return;
    }
    for i in 0..64 {
        let mut config = base_config();
        config.set_complex(false);
        config.set_huge(true);
        let mut t = Translator::new(config);
        t.set_num_params(1);
        let c = i as f64 + 0.5;
        let k = Slot::Const(t.append_constant(Complex::new(c, 0.0)).unwrap());
        t.append_mul(&Slot::Out(0), &[Slot::Param(0), k], 0).unwrap();
        let app = t.compile().unwrap();
        assert_eq!(app.evaluate_single(&[2.0]), 2.0 * c);
        drop(app);
    }
}
