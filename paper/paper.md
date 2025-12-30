---
title: 'Symjit: a lightweight just-in-time (JIT) optimizer compiler for Sympy symbolic expressions'
tags:
  - Python
  - SymPy
  - Computer Algebra
  - Compiler Technology  
authors:
  - name: Shahriar Iravanian
    orcid: 0000-0003-2132-1543
    equal-contrib: true
    affiliation: "1, 2" 
affiliations:
 - name: Emory University, Atlanta, GA
   index: 1 
 - name: Independent Researcher, Atlanta, GA
   index: 2
date: 17 November 2025
bibliography: paper.bib
---

# Summary

`Symjit` is a lightweight JIT compiler that directly translates Sympy Computer 
Algebra System [@10.7717/peerj-cs.103] expressions into machine code without 
using a separate general-purpose compiler like LLVM [@10.5555/977395.977673]. 
Its main utility is to generate fast numerical functions to feed into various 
solvers provided by the scientific Python ecosystem, including numerical 
integration routines, ordinary differential equation (ODE) solvers, and 
image filtering functions.

# Statement of need

Computer Algebra Systems create and manipulate symbolic expressions. A common 
utility of such systems is to generate input for numerical routines. Sympy is 
Python's main symbolic algebra package. Considering the importance of Python in
scientific computing and machine learning, Sympy has a critical role in providing
input to various numerical packages. However, Sympy cannot directly generate 
performant numerical functions. The link between Sympy and the numerical packages is
usually indirect and goes through intermediate steps by translating symbolic 
expressions into C or Python code first. 

`Symjit` tries to provide a fast but lightweight interface between symbolic 
expressions and machine code. The key to the success of `Symjit` is that it
understands and directly works on symbolic expressions. Therefore, it can utilize 
the information encoded in the symbolic expressions to generate high-quality, optimized 
code. Moreover, `Symjit` integrates seamlessly with the Python ecosystem and can 
be used easily in REPL environments. 

# Software Description

The core of `Symjit` is a library written in the Rust programming language [@rust_book] 
with minimum external dependencies. The Rust backend translates Sympy symbolic 
expressions into x86-64, aarch64 (arm64), and RISC-V machine code on Windows and 
Unix-like platforms (Linux and macOS). 

`Symjit`'s main interface is composed of three compile functions: `compile_func`, 
`compile_ode`, and `compile_jac`. The primary function, `compile_func`, mimics Sympy's 
`lambdify`, accepts a list of variables and symbolic expressions, and returns a function 
linked directly to the compiled machine code. The returned function accepts a list of 
scalars or vectors as input. The other two functions work similarly to generate functions 
suitable for numerical ODE solvers. 

`Symjit` performs different optimizations. When available, it can use SIMD 
instructions or multi-threading to improve the performance of vectorized code. 
It also utilizes standard compiler optimization techniques, such as graph coloring 
for register allocation, common sub-expression elimination, and generating fused 
multiply-add instructions. 

# Results

The test of `Symjit` on a library of test problems shows a performance gain of 2-50x 
for scalar and 10-250x for vectorized operations compared to plain Python 
implementations. 

# Acknowledgements

We acknowledge Dr. Jason Moore for help with automatic package management and 
distribution, and Peter Stahlecker for extensive testing and providing valuable 
feedback.

# References
