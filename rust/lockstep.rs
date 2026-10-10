//! Lockstep evaluation of a batch of points (cargo feature `async`, a prototype).
//!
//! The compiled kernels of large evaluators do not fit in L2, so evaluating them point by
//! point fetches every instruction again for each point. Here, the compiler inserts calls
//! to `_yield_` every `Config::yield_every` expression nodes (`Block::insert_yields`), and
//! `evaluate_matrix` runs `Config::lockstep` points at a time with `run_lockstep`, as
//! stackful coroutines (corosensei), round robin: each segment of code between two yield points runs for all points of the block
//! before the next segment starts, so it is fetched once per block.
//!
//! A yield is an ordinary call (`Func::Unary`/`Func::UnaryCplx`), so the register allocator
//! has spilled all live values to the kernel's stack frame, which lives on the coroutine's
//! stack. Outside `run_lockstep` the yield functions return at once.

use std::cell::{Cell, RefCell};

use corosensei::stack::DefaultStack;
use corosensei::{Coroutine, CoroutineResult, Yielder};
use num_complex::Complex;

type Yield = Yielder<(), ()>;

thread_local! {
    /// the yielder of the coroutine running on this thread (null outside `run_lockstep`)
    static CURRENT: Cell<*const Yield> = const { Cell::new(std::ptr::null()) };
    /// stacks kept for the next call of `run_lockstep` (mmap is expensive)
    static STACKS: RefCell<Vec<(usize, DefaultStack)>> = const { RefCell::new(Vec::new()) };
}

#[inline(always)]
fn suspend() {
    let y = CURRENT.with(|c| c.get());
    if !y.is_null() {
        // Safety: `y` is the yielder of the running coroutine; it is valid while the
        // coroutine runs, and the coroutine is running (we are on its stack).
        unsafe { (*y).suspend(()) };
        // other coroutines have set CURRENT in the meantime
        CURRENT.with(|c| c.set(y));
    }
}

pub extern "C" fn yield_point(x: f64) -> f64 {
    suspend();
    x
}

pub extern "C" fn cplx_yield_point(re: f64, im: f64, out: &mut Complex<f64>) {
    suspend();
    *out = Complex::new(re, im);
}

fn take_stack(size: usize) -> DefaultStack {
    STACKS
        .with(|s| {
            let mut s = s.borrow_mut();
            let k = s.iter().position(|(n, _)| *n >= size)?;
            Some(s.swap_remove(k).1)
        })
        .unwrap_or_else(|| DefaultStack::new(size).expect("cannot allocate a coroutine stack"))
}

fn give_stack(size: usize, stack: DefaultStack) {
    STACKS.with(|s| s.borrow_mut().push((size, stack)));
}

/// Calls `f(i)` for `i` in `0..n`, `block` points at a time in lockstep: each point runs
/// as a coroutine with a stack of `stack_size` bytes until its next yield point, then the
/// next point of the block runs, and so on until all of them return.
pub fn run_lockstep<F>(n: usize, block: usize, stack_size: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    let block = block.max(1);
    let f = &f;

    for start in (0..n).step_by(block) {
        let end = (start + block).min(n);
        let mut coroutines: Vec<Coroutine<(), (), (), DefaultStack>> = (start..end)
            .map(|i| {
                let stack = take_stack(stack_size);
                // Safety: `f` outlives the coroutines, which all run to completion below.
                unsafe {
                    Coroutine::with_stack_unchecked(stack, move |y: &Yield, ()| {
                        CURRENT.with(|c| c.set(y as *const Yield));
                        f(i);
                        CURRENT.with(|c| c.set(std::ptr::null()));
                    })
                }
            })
            .collect();

        let mut running = coroutines.len();
        while running > 0 {
            for c in coroutines.iter_mut() {
                if !c.done() {
                    if let CoroutineResult::Return(()) = c.resume(()) {
                        running -= 1;
                    }
                }
            }
        }

        for c in coroutines {
            give_stack(stack_size, c.into_stack());
        }
    }
}
