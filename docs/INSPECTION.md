## Code Inspection

To inspect the generated code, you can use either `dump` function of various `Func` callables to write the binary into a file or use `dumps` to return a hex string. The output of `dump` is a flat binary file with no header or other extras that can be disassembled. For example,

```python
from symjit import compile_func
from sympy import symbols

x, y = symbols('x y')
f = compile_func([x, y], [x+y, x*y])
f.dump('test.bin', what='scalar')
```

Passing `what='simd'` dumps the vectorized version of the function and `what='fast'` to dump the fast function. 

On a Linux system, we can invoke `objdump` to disassemble the output as below:

```
objdump -b binary -m i386:x86-64 -M intel -D test.bin
```

On a MacOS `aarch64` (Apple Silicon), the `-b` option is not available. You need to first use `objcopy` to make
an object file:

```
objcopy -I binary -O elf64-littleaarch64 test.bin test.elf
objdump -D test.bin
```


The output (assuming a Linux x86-64 machine) is

```
0000000000000000 <.data>:
   0:	55                   	push   rbp
   1:	53                   	push   rbx
   2:	48 8b ef             	mov    rbp,rdi
   5:	48 81 ec 88 00 00 00 	sub    rsp,0x88
   c:	c5 fb 10 5d 00       	vmovsd xmm3,QWORD PTR [rbp+0x0]
  11:	c5 fb 10 55 08       	vmovsd xmm2,QWORD PTR [rbp+0x8]
  16:	c5 e3 58 da          	vaddsd xmm3,xmm3,xmm2
  1a:	c5 fb 11 5d 18       	vmovsd QWORD PTR [rbp+0x18],xmm3
  1f:	c5 fb 10 5d 00       	vmovsd xmm3,QWORD PTR [rbp+0x0]
  24:	c5 fb 10 55 08       	vmovsd xmm2,QWORD PTR [rbp+0x8]
  29:	c5 e3 59 da          	vmulsd xmm3,xmm3,xmm2
  2d:	c5 fb 11 5d 20       	vmovsd QWORD PTR [rbp+0x20],xmm3
  32:	c5 f8 77             	vzeroupper
  35:	48 81 c4 88 00 00 00 	add    rsp,0x88
  3c:	5b                   	pop    rbx
  3d:	5d                   	pop    rbp
  3e:	c3                   	ret
```

Note that this is the output from an older version, and the more recent versions have a more complex prologue and epilogue.

To see the intermediate representation in a pseudo-LLVM textual format (useful for debugging), pass `what='bytecode'`. For example:

```python
x, y = symbols('x y')
f = compile_func([x, y], [x+y, x*y])
print(f.dumps('bytecode'))
```

The output is

```
#!
00000	%0 := Mem[0]
00001	%1 := Mem[1]
00002	%2 := %0 Plus %1
00003	Mem[2] := %2
00004	%3 := %0 Times %1
00005	Mem[3] := %3
```
