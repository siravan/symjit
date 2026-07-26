# Direct applications

`DirectApplication` lowers a portable complex SymJIT `Application` into a
callable that reads and writes point-contiguous planes through stable pointer
descriptors. It avoids packing evaluator inputs and scattering output buffers
when an owning runtime already stores values in split planes.

The API separates three independent choices:

- `DirectDestinationOperation::{Overwrite, Accumulate}` controls how a result
  is stored.
- `DirectInputSnapshot::{Live, BeforeWrite}` controls whether input planes are
  read in place or copied to the generated function's stack before any output
  is written.
- `DirectOutputScale::{Identity, ComplexScalar}` controls whether each complex
  output is stored directly or multiplied by the complex scalar in descriptor
  slots 0 and 1.

Each source state and parameter is mapped to either a point-dependent plane or
a point-independent scalar descriptor. An output may name an input plane that
it must alias, or use `DIRECT_NO_ALIAS` to select its distinct output
descriptor. These mechanisms are generic; scheduling, recurrence roles,
factor catalogs, and artifact policy belong to the caller.

`BeforeWrite` is intended for multi-output transforms whose output planes
alias inputs. It snapshots all input planes in the generated stack frame, so a
store for one output cannot affect a later output expression. It does not
allocate during a call.

Complex-scalar lowering currently accepts portable complex Symbolica O2
sources. Identity lowering accepts O0 through O3 sources and emits an O3
direct callable. Scalar and SIMD execution handle unaligned heads and odd
tails on x86-64 and AArch64.

The storage ABI retains portable source MIR and the direct metadata, never
host machine code. Loading recompiles for the current host. The format is an
execution format for trusted inputs, not a sandbox or a hostile-bytecode
validation boundary. Shape, descriptor, range, and alias checks remain part of
the API contract because they prevent accidental undefined behavior.

