# Direct table applications

`DirectTableApplication` is an optional table-driven companion to
`DirectApplication`. It emits the source MIR body once inside native row,
point, and attachment loops, allowing one prepared expression to consume many
plane bindings and fan out into many destinations without a packed evaluator
buffer.

The caller defines two fixed-width row layouts:

- an invocation row selects input planes and a contiguous attachment range;
- an attachment row selects destination planes, a complex scale, and either
  overwrite or accumulate semantics.

Field positions are byte offsets to little-endian `u32` values. The portable
descriptor records those offsets plus the source shape and parameter
bindings. It contains no pointers or executable machine code.

At execution time, `DirectTableCallViewV1` supplies immutable invocation and
attachment tables, plane and scalar catalogs, split real/imaginary complex
scale catalogs, and a point range. The generated callable evaluates rows
outermost, points next, and attachments in their declared order. Scalar
head/tail handling and a SIMD middle preserve odd point ranges on x86-64 and
AArch64.

The table mechanism has no knowledge of a scheduler, model, recurrence role,
or artifact format. Those policies—including how row tables and scale
catalogs are constructed—belong to the owning runtime.

Descriptor and call-view validation protects the raw-pointer execution
contract from accidental shape, range, and alias mistakes. As with ordinary
SymJIT storage, source bytecode is assumed to be trusted; DirectTable is not a
sandbox or a hostile-code inspection layer.

