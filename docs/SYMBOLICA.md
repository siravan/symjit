# Symbolica

In addition to Sympy, *SymJit* can generate code from [Symbolica](https://symbolica.io/)'s expressions. Symbolica is a moderate computer algebra written in Rust with a Python binding. Considering that both *SymJit* and Symbolica are written in Rust, they can interact at multiple levels. However, here we limit the discussion to high-level interaction using Python. 

# Examples

*SymJit*'s main interface is composed of different `compile` functions: `compile_func`, `compile_ode`, and `compile_jac`. As of version 2.9.2, all three functions are aware of Symbolica's expressions (of type `symbolica.Expression`) and can accept them as inputs.

```Python
import numpy as np
from symjit import compile_func
from symbolica import S

x, y = S('x', 'y')
f = compile_func([x, y], [x+y, x*y])
assert(np.all(f(3, 5) == [8., 15.]))
```

It is usually easier to use Symbolica's parse function (usually abbreviated to `E`):

```python
import numpy as np
from scipy.integrate import nquad
from symbolica import S, E
from symjit import compile_func

N = 5
t, x = S('t', 'x')
f = compile_func([t, x], E(f'exp(-t*x)/t^{N}')  # note that f is a fast function

sol = nquad(f, [[1, np.inf], [0, np.inf]])
np.testing.assert_approx_equal(sol[0], 1/N)
```

## Conditions 

Conditional expressions are mainly implemented in Symbolica using pattern restriction and user-defined functions. 
Instead, *SymJit* uses the `if` function to implement conditions. In Symbolica, `if(cond, true_val, false_val)` returns
`true_val` if `cond` if non-zero, and `false_val` if it is zero. *SymJit* expands `cond` by adding comparison and boolean
functions. The comparison functions are `lt`, `leq`, `gt`, `geq`, `eq`, and `neq`. Equivalently, the comparison functions, 
with the exception of `neq`, can be written as `>`, `>=`, `<`, `<=`, and `==`. The boolean functions are `and`, `or`, `xor`, 
and `not`. For example:

```Python
import numpy as np
from symjit import compile_func
from symbolica import S

x, y = S('x', 'y')
f = compile_func([x, y], E('if(and(>(x, y), not(==(x, 5))), x^2, y^2)'))
assert(f(4, 2) == 16.0)
assert(f(2, 3) == 9.0)
assert(f(5, 2) == 4.0)
```

Another way to implement conditionals is to use `min` and `max` functions. The following table shows the equivalent expressions:

|   using comparisons   |     using `min`/`max`        |
|-----------------------|------------------------------|
|`if(gt(x, y), t, f)`   | `if(max(x-y, 0), t, f)`      |
|`if(geq(x, y), t, f)`  | `if(max(y-x, 0), f, t)`      |
|`if(lt(x, y), t, f)`   | `if(max(y-x, 0), t, f)`      |
|`if(leq(x, y), t, f)`  | `if(max(x-y, 0), f, t)`      |
|`if(eq(x, y), t, f)`   | `if(x-y, t, f)`              |
|`if(neq(x, y), t, f)`  | `if(x-y, f, t)`              |
|`if(and(p, q), t, f)`  | `if(p * q, t, f)`            |
|`if(or(p, q), t, f)`   | `if(p^2 + q^2, t, f)`        |
|`if(xor(p, q), t, f)`  | `if(max(p,q)-min(p,q), t, f)`|
|`if(not(p), t, f)`     | `if(p, f, t)`                |


## Loops

SymPy has reduction operators `Sum` and `Product`, which are supported by *SymJit* and can be used by Symbolica's expressions:

```Python
import numpy as np
from symjit import compile_func
from symbolica import S

n = S('n')
# k ranges from 1 to n (inclusing)
fact = compile_func([n], E('Product(k, (k, 1, n))'))   
assert(fact(6) == 720.0)   # fact is the factorial function!
```

## User-Defined functions

In *SymJit*, `compile_*` functions accepts user-defined functions, either Python's functions or compiled *SymJit*'s function, 
by defining them using `defuns` keyword. For example, we can re-use the `fact` function defined above to make an exponential 
function:

```Python
import math
...
x, F = S('x', 'F')
f = compile_func([x], E('Sum(x^k/F(k), (k, 0, 100))'), defuns={F: fact})   
assert(f(2.0) == Math.exp(2.0))   
```

## Auto-Vectorization

If NumPy arrays are passed aas inputs to *SymJit*-generated functions, it does auto-vectorization. Please refer to [README](../README.md)
for details.
