> [!NOTE]
> *SymJit* is now a backend for *Symbolica*. Therefore, using `jit_compile` method of `ExpressionEvaluator`s is the preferable
> way to generate jit code for Symbolica. 

# Symbolica

In addition to Sympy, *SymJit* can generate code from [Symbolica](https://symbolica.io/)'s expressions. Symbolica is a modern computer algebra written in Rust with a Python binding. Considering that both *SymJit* and Symbolica are written in Rust, they can interact at multiple levels. However, here we limit the discussion to high-level interaction using Python. 

## Using Symbolica `Evaluator`

Using `compile_evaluator`, Symjit can benefit from Symbolica's optimization passes. A Symbolica expression (or a collection of multiple expressions) can be converted to an `Evaluator` object. Using Symjit `compile_evaluator`, it is possible to compile the `Evaluator`:

```python
import numpy as np
from symbolica import E, S
from symjit import compile_evaluator

x, y = S("x"), S("y")
ev = E("x + y^2").evaluator({}, {}, [x, y])
f = compile_evaluator(ev)
assert f.evaluate(np.array([[2, 3]])) == [[11]]
```

It is possible to compile for complex numbers:

```python
f = compile_evaluator(ev, dtype='complex128')
assert f.evaluate_complex(np.array([[2+1j, 3-1j]])) == [[10-5j]]
```

`compile_evaluator` accepts nmultiple optional named arguments. Some of the optimization switches, similar to other `compile` function. However, there are a few Symbolica-specific switches. The main one is `order`, which takes either `c` or `fortran`. The former is the default and instructs `compile_evaluator` to return a `SymbolicaFunc` object. However, if `order = 'fortran'`, a `Func` object (similar to Sympy interface) is returned.
