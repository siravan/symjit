# Composer

In addition to compiling SymPy or Symbolica expressions, Symjit exposes a low-level interface to give the user fine control over the generated code. The design of the composer interface is inspired by and similar to the [llvmlite](https://github.com/numba/llvmlite) interface.

The central object of this interface is called `Composer`. Generating a compiled function is done is three steps:

1. Creating a composer by calling its constructor `cp = Composer(num_params, num_outs)`, where `num_params` is the number of input arguments/parameters, and `num_outs` is the number of output variables.
2. Adding instructions by calling cp’s various methods (see below).
3. Finally, compiling the composer by passing it to `compiler_composer(cp, ...)`.

# Tutorial

## Basic Usage

Let’s look at a simple example:

```python
from symjit import Composer, compile_composer

cp = Composer(2, 1)
s1 = cp.fmul(cp.arg(0), cp.arg(1))
cp.assign(cp.out(0), s1)

f = compile_composer(cp)
assert(f(3, 4)[0] == 12)
```

First, we create a `Composer` object as `cp = Composer(2, 1)` with 2 input arguments and 1 output. Next, we add a multiplication instruction. The arguments are `cp.arg(0)` and `cp.arg(1)`, corresponding to the first two input arguments. The result of `fmul` is a temporary variable of type `Slot`. To return the final result to the user, we have to assign this slot to the output variable (`cp.out(0)` is the first output variable). Finally, we compile the composer.

Symjit supports `float64`, `complex128`, and various SIMD versions of these two main types. `Composer` is mostly type-agnostic. The same composer can be compiled for `float64` (default) and `complex128`. For example,

```python
cp = Composer(2, 1)
s1 = cp.fmul(cp.arg(0), cp.arg(1))
cp.assign(cp.out(0), s1)

f = compile_composer(cp, dtype = 'complex128')
assert(f(3 + 2j, 4 - 1j)[0] == 14 + 5j)
```

In the examples above, the compiler function `f` was called using scalar values (e.g., `f(3, 4)`). It is possible to call these functions with a `(num_samples x num_input)` matrix by calling `evaluate` or `evaluate_complex`. For example,

```python
from symjit import Composer, compile_composer
import numpy as np

cp = Composer(2, 1)
s1 = cp.fadd(cp.arg(0), cp.arg(1))
cp.assign(cp.out(0), s1)

f = compile_composer(cp, dtype = 'complex128')

X = np.random.rand((100, 2)) + np.random.rand((100, 2)) * 1j
np.testing.assert_array_almost_equal(f.evaluate_complex(X).ravel(), (X[:,0] + X[:,1]).ravel())
```

## Slots

A slot is a memory location or a constant. Most `Composer` methods expect one or more slots are arguments and return a slot. For example, the type annotation for `fadd is def fadd(self, x: Slot, y: Slot) -> Slot`.

There are four main types of slot:

1. `cp.arg(i)`: the `i`th input argument.
2. `cp.out(i)`: the `i`th output variable.
3. `cp.temp(i)`: the `i`th temoprary/stack variable. These slots are generally returned from Composer methods and created by calling cp.new_temp().
4. `cp.conatant(val)`: a numerical contact, either `float64` or `complex128`.

## Labels

In addition to slots, some Composer methods expect a label as an argument. The user can create a new label using `cp.new_label()`. Labels are used to create loops and jumps.

## Assignment

The majority of `Composer` operations return the result in a new temporary variable in accordance with the Static Single Assignment (SSA) form. On the other hand, `assign(lhs, rhs)` assigns the right-hand-side (rhs) to a previosely defined left-hand-side (lhs) variable. This is akin to the φ-block in compiler theory. 

## Operations

This is a list of operations defined for Composer:

* Arithmetic binary operations: `fadd`, `fsub`, `fmul`, `fdiv`, `idiv` (mimics integer division), and `mod`.
* Arithmetic unary operations: `neg`, `abs`, `sqrt`, `real_sqrt`, `square`, `cube`, `recip`, `round`, `floor`, `ceiling`, `trunc`.
* Power operations: `powi` (to an integer power) and `powf`.
* Comparison operations: `lt` (less than), `leq` (less than or equal), `gt` (greater than), `geq` (greater than or equal), `eq` (equal), `neq` (not equal). For complex values, comparison is based on their real part.
* Boolean operations: `and_`, `or_`, `xor`, `not_`, and `iszero`. Note the presence of `_` to distinguish these operations from the Python keywords of the same name.
* Join/select operation: `join(cond, true_val, false_val)` selects `true_val` or `false_val` based on the value of the condition `cond`.
* Branching operations: `branch(label)` (unconditional), `branch_if(cond, label)` (branch if cond is true), and `branch_else` (branch if condition is false).
* Trigonometrical functions: `sin`, `cos`, `tan`, `csc`, `sec`, and `cot`.
* Inverse trigonometrical functions: `asin`, `acos`, and `atan`.
* Hyperbolic functions: `sinh`, `cosh`, `tanh`, `csch`, `sech`, and `coth`.
* Inverse Hyperbolic functions: `asinh`, `acosh`, and `atanh`.
* Miscellaneous functions: `min`, `max`, `heaviside`, `cbrt`, `exp`, `exp2`, `log`, `log10`, `log2`, `expm1`, and `log1p`,
* Advanced functions: `erf`, `erfc`, `gamma`, and `loggamma`.

## External Functions

Similar to Symjit, you can pass external Python functions to `Composer`. Currently, only functions that accept one or two `float64` arguments and return a single `float64` are supported. Specifically, the function should be converted to one of the two following forms:

* `ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)`
* `ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)`

Let `g` be such a function or lambda expression. You can call it by using `Composer.call(g, arg0)` or `Composer.call(g, arg0, arg1)`. For example,

```python
    cp = Composer(1, 1)
    s = cp.call(lambda x: x**2 + 1, cp.arg(0))
    cp.assign(cp.out(0), s)
    f = compile_composer(cp)
    print(f(4)[0][0] == 17)
```

## Branching and SIMD

By default, Symjit generates SIMD instructions if possible, meaning `num_samples >= num_lanes`, where `num_lanes` is 4 for f64x4 (AVX/AVX2) or 2 (aarch64). Future versions of Symjit will generate code for f64x8 (e.g., on AVX512). The presence of SIMD requires caution when using branching instructions. To explain the problem and its solution, let's start with a simple example. Say we want to make a `min` function. `Composer` already has this function, which is defined as 

```python
def min(self, x: Slot, y: Slot) -> Slot:
        return self.join(self.lt(x, y), x, y)
```

Here, `join` handles SIMD instructions correctly and this method would work whether it is called with scalar or vectorized (SIMD) instructions. However, let's say we want to have a `min` operation which does short circuiting. We use branchin instructions to get

```python
from symjit import Composer, compile_composer

cp = Composer(2, 1)
label_else = cp.new_label()
label_done = cp.new_label()

cp.branch_else(cp.lt(cp.arg(0), cp.arg(1)), label_else)
cp.assign(cp.out(0), cp.arg(0))
cp.branch(label_done)
cp.set_label(label_else)
cp.assign(cp.out(0), cp.arg(1))
cp.set_label(label_done)
f = compile_composer(cp)

assert(f(3, 5)[0] == 3)
```

This works! However, if we apply the same function to a matrix input using `evaluate`, the result could be wrong. This is because different SIMD lanes are not necessarily convergent (all true or all false). An easy solution is to turn off automatic SIMD generation by passing `use_simd = False` to `compile_composer`. This works but would deny us a major optimization. A better option is to modify the code to make it work in case it is vectorized. The key is that both branches of `if` can be taken. In fact, for SIMD instructions, `branch_if` jumps only if all the lanes are true. Conversely, `branch_else` jumps only if all the lanes are false. For a mix of true and false lanes, both branch instructions are inactive. In this situation, we cannot overwrite `out(0)` in the else-branch, because it will erase whatever was written in the then-branch. The solution is to use temporary variables and then cap if-then-else with a `join` instructions to merge the results of the two branches. It is easier to see the code that to describe it! Here is the correct code:

```python
from symjit import Composer, compile_composer

cp = Composer(2, 1)
label_else = cp.new_label()
label_done = cp.new_label()

t1 = cp.new_temp()
t2 = cp.new_temp()

cond = cp.lt(cp.arg(0), cp.arg(1))

cp.branch_else(cond, label_else)

# the then-branch
cp.assign(t1, cp.arg(0))
cp.branch_if(cond, label_done)

# the else-branch
cp.set_label(label_else)
cp.assign(t2, cp.arg(1))

cp.set_label(label_done)
t = cp.join(cond, t1, t2)
cp.assign(cp.out(0), t)

f = compile_composer(cp)

assert(f(3, 5)[0] == 3)
```

Because the example above has lots of boilerplate code, `Composer` has helper functions to make it easier to write such code. For if-else code, the helper function is `append_if_else(cond, block_if, block_else)`, where `cond` is the condtion and `block_if` and `block_else` are code blocks. The utility function `Composer.new_block()` creates new empty blocks, which are then populated with the desired code before being passed to `append_if`. The above example becomes:

```python
from symjit import Composer, compile_composer

cp = Composer(2, 1)

b1 = cp.new_block()
t1 = cp.new_temp()
b1.assign(t1, cp.arg(0))

b2 = cp.new_block()
t2 = cp.new_temp()
b2.assign(t2, cp.arg(1))

cond = cp.lt(cp.arg(0), cp.arg(1))
cp.append_if_else(code, b1, b2)
t = cp.join(cond, t1, t2)
cp.assign(cp.out(0), t)

f = compile_composer(cp)

assert(f(5, 1)[0] == 1)
```

You can also directly insert a block into the current code stream using `Composer.append_block(block)`. Another helper function is `Composer.append_for(for_var, start, end, block)`, where `for_var` is the loop variable defined using `new_temp()`, `start` and `end` are constants marking the range of the for-loop, and `block` is a block defining the body of the loop. Finally, the following code shows the general pattern of a while-loop:

```python
label_while = cp.new_label()
label_done = cp.new_label()

cp.set_label(label_while)
cond = ...  # calculating the while condition
cp.branch_else(cond, label_done)
... # body of the while loop
t = cp.join(cond, ...)    # join instruction to make sure different lanes are updated correctly
cp.assign(..., t)
cp.branch(label_while)
set_label(label_done)
```
