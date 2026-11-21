"""Unit tests for `symjit.disassemble`.

Run from the repository root with

    python -m unittest discover -s python/tests -v

Tests that need an optional tool (GNU objdump for x86-64, the `capstone` package,
Symbolica, binutils to assemble `assembly`'s text) are skipped when it is missing; symjit's
own decoder (`tool="symjit"`) needs none.
"""

import contextlib
import io
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))

from sympy import cos, sin, symbols

import symjit
from symjit import assembly, compile_func, disassemble, disassembly
from symjit import disasm

try:
    import capstone  # noqa: F401

    HAS_CAPSTONE = True
except ImportError:
    HAS_CAPSTONE = False

try:
    from symbolica import S, E

    HAS_SYMBOLICA = True
except ImportError:
    HAS_SYMBOLICA = False

HOST_X86 = platform.machine().lower() in ("x86_64", "amd64")
HAS_OBJDUMP_X86 = disasm._find_objdump("x86_64") is not None

x, y = symbols("x y")

LINE = re.compile(r"^\s*([0-9a-f]+):  ((?:[0-9a-f]{2} ?)+)")
LABEL = re.compile(r"^[@._A-Za-z][^\s]*:$")


def compile_sample(**kwargs):
    return compile_func([x, y], [x * y + 3.5, sin(x)], **kwargs)


def raw_kernel(f, kernel):
    """The bytes of a kernel, obtained independently from `disassemble`."""
    with tempfile.TemporaryDirectory() as d:
        return disasm._compiler_of(f).dump(os.path.join(d, "k.bin"), kernel)


def parse_listing(text):
    """Returns ([(address, bytes, instruction text)], [comment lines]) from a listing."""
    rows, comments = [], []
    for line in text.splitlines():
        if line.startswith(";"):
            comments.append(line)
            continue
        if LABEL.match(line):  # symjit's listing names its labels
            continue
        m = LINE.match(line)
        assert m, f"unparsable line: {line!r}"
        rows.append((int(m.group(1), 16), bytes.fromhex(m.group(2).replace(" ", "")), line[m.end() :].strip()))
    return rows, comments


def mnemonics(text):
    rows, _ = parse_listing(text)
    return [text.split(" ")[0].lower() for _, _, text in rows]


class ListingChecks:
    """Structural checks shared by every architecture."""

    def check_listing(self, f, kernel, last_mnemonic="ret", tool="auto"):
        text = disassembly(f, kernel, tool=tool)
        rows, comments = parse_listing(text)
        code = raw_kernel(f, kernel)

        self.assertTrue(rows)
        self.assertTrue(comments[0].startswith(f"; symjit `{kernel}` kernel"))

        # addresses are contiguous and the listed bytes are exactly the start of the kernel
        expected = 0
        for address, raw, _ in rows:
            self.assertEqual(address, expected)
            expected += len(raw)
        self.assertEqual(b"".join(r for _, r, _ in rows), code[:expected])

        # the code ends with the final `ret`, and the rest is reported as the constant pool
        self.assertTrue(mnemonics(text)[-1].startswith(last_mnemonic))
        pool = [c for c in comments if c.startswith("; constant pool")]
        if expected < len(code):
            self.assertEqual(len(pool), 1)
            self.assertIn(f"{len(code) - expected} bytes at offset {expected:#x}", pool[0])
        return text, rows, code


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.f = compile_sample()

    def test_exported(self):
        self.assertIs(symjit.disassemble, disassemble)
        self.assertIs(symjit.disassembly, disassembly)
        self.assertIs(symjit.assembly, assembly)

    def test_rejects_other_objects(self):
        for bad in (42, "code", None, [1, 2], lambda z: z):
            with self.assertRaises(TypeError):
                disassemble(bad)

    def test_unknown_kernel(self):
        with self.assertRaisesRegex(ValueError, "kernel should be one of"):
            disassemble(self.f, "bogus")

    def test_unknown_tool(self):
        with self.assertRaisesRegex(ValueError, "tool should be"):
            disassemble(self.f, tool="gdb")

    def test_function_without_code(self):
        with self.assertRaisesRegex(ValueError, "no compiled code"):
            disassemble(symjit.SymbolicaFunc(None))

    def test_bytecode_type_has_no_machine_code(self):
        f = compile_func([x], [x + 1], ty="bytecode")
        with self.assertRaisesRegex(ValueError, "does not generate machine code"):
            disassemble(f)

    def test_bytecode_kernel_is_text(self):
        text = disassembly(self.f, "bytecode")
        self.assertTrue(text.startswith("#! bytecode"))
        self.assertIn("call sin", text)

    def test_unavailable_kernel(self):
        # the RISC-V backend has no SIMD kernel
        f = compile_sample(ty="riscv")
        with self.assertRaisesRegex(ValueError, "`simd` kernel is not available"):
            disassemble(f, "simd")


class LoadedEvaluatorTests(ListingChecks, unittest.TestCase):
    """A saved Symbolica-bridge function loaded back (load_func): its compiler is the dtype's,
    for the complex one too (it was `f.compiler`, None: "the function has no compiled code")."""

    # out0 = p0 * p1 + p0, as compile_evaluator's instruction text (no symbolica package needed)
    EVALUATOR = str(([("mul", ("temp", 0), [("param", 0), ("param", 1)], 0),
                      ("add", ("out", 0), [("temp", 0), ("param", 0)], 0)], 1, []))

    def loaded(self, dtype):
        f = symjit.compile_evaluator(self.EVALUATOR, dtype=dtype, num_params=2, use_threads=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.sjb")
            f.save(path, dtype=dtype)
            return f, symjit.load_func(path)

    def test_compiler_is_the_dtypes(self):
        for dtype in ("float64", "complex128"):
            f, g = self.loaded(dtype)
            self.assertIsInstance(g, symjit.SymbolicaFunc)
            self.assertEqual(g.is_complex, dtype == "complex128")
            c = disasm._compiler_of(g)
            self.assertIs(c, g.complex_compiler if dtype == "complex128" else g.compiler)
            self.assertEqual(c.measure("ker-scalar-size"), disasm._compiler_of(f).measure("ker-scalar-size"))

    def test_loaded_complex_is_called_as_complex(self):
        _, g = self.loaded("complex128")
        a, b = 0.5 + 1j, 2.0 - 0.5j
        self.assertEqual(complex(g(a, b)[0, 0]), a * b + a)

    @unittest.skipUnless(HOST_X86 and (HAS_OBJDUMP_X86 or HAS_CAPSTONE), "needs an x86-64 host and a disassembler")
    def test_listing(self):
        for dtype in ("float64", "complex128"):
            f, g = self.loaded(dtype)
            text, _, code = self.check_listing(g, "scalar")
            self.assertEqual(code, raw_kernel(f, "scalar"))


class PrintingTests(unittest.TestCase):
    def setUp(self):
        self.f = compile_sample()

    def test_print_to_stdout(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            disassemble(self.f)
        self.assertEqual(out.getvalue(), disassembly(self.f) + "\n")

    def test_print_to_file(self):
        out = io.StringIO()
        disassemble(self.f, file=out)
        self.assertEqual(out.getvalue(), disassembly(self.f) + "\n")

    def test_returns_none(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(disassemble(self.f))


@unittest.skipUnless(HOST_X86, "needs an x86-64 host")
class X86Tests(ListingChecks, unittest.TestCase):
    def setUp(self):
        self.f = compile_sample()

    @unittest.skipUnless(HAS_OBJDUMP_X86, "needs objdump")
    def test_scalar_objdump(self):
        text, rows, _ = self.check_listing(self.f, "scalar", tool="objdump")
        self.assertIn("disassembled with objdump", text.splitlines()[0])
        ms = mnemonics(text)
        self.assertEqual(ms[0], "push")  # prologue
        self.assertIn("vmulsd", ms)  # x * y
        self.assertIn("vaddsd", ms)  # + 3.5
        self.assertEqual(sum(m == "call" for m in ms), 1)  # sin(x)

    @unittest.skipUnless(HAS_OBJDUMP_X86, "needs objdump")
    def test_simd_objdump(self):
        text, rows, _ = self.check_listing(self.f, "simd", tool="objdump")
        self.assertTrue(any(m in ("vmulpd", "vmulsd") for m in mnemonics(text)))

    @unittest.skipUnless(HAS_CAPSTONE, "needs capstone")
    def test_capstone(self):
        for kernel in ("scalar", "simd"):
            text, _, _ = self.check_listing(self.f, kernel, tool="capstone")
            self.assertIn("disassembled with capstone", text.splitlines()[0])

    @unittest.skipUnless(HAS_CAPSTONE and HAS_OBJDUMP_X86, "needs objdump and capstone")
    def test_objdump_and_capstone_agree(self):
        for kernel in ("scalar", "simd"):
            a, _ = parse_listing(disassembly(self.f, kernel, tool="objdump"))
            b, _ = parse_listing(disassembly(self.f, kernel, tool="capstone"))
            self.assertEqual([(p, r) for p, r, _ in a], [(p, r) for p, r, _ in b])
            self.assertEqual([t.split(" ")[0] for _, _, t in a], [t.split(" ")[0] for _, _, t in b])

    def test_constant_pool_is_not_disassembled(self):
        text, rows, code = self.check_listing(self.f, "scalar")
        self.assertLess(len(b"".join(r for _, r, _ in rows)), len(code))
        self.assertIn("use show_data=True", text.splitlines()[-1])

    def test_show_data(self):
        text = disassembly(self.f, "scalar", show_data=True)
        pool = [line for line in text.splitlines() if ".quad" in line]
        self.assertTrue(pool)
        self.assertNotIn("use show_data=True", text)

        # the pool holds 3.5 and the address of `sin`
        self.assertTrue(any("; 3.5" in line for line in pool))

        # the listed pool words are exactly the bytes of the kernel after the code
        rows, _ = parse_listing("\n".join(l for l in text.splitlines() if not l.startswith(";")))
        code = raw_kernel(self.f, "scalar")
        listed = b"".join(r for a, r, _ in rows)
        self.assertEqual(listed, code)

    def test_complex_function(self):
        g = compile_func([x, y], [x * y + x, sin(y) * x], dtype="complex128")
        self.assertEqual(type(g).__name__, "FuncComplex")
        for kernel in ("scalar", "simd"):
            self.check_listing(g, kernel)

    def test_intermediate_options_change_code(self):
        small = compile_sample(opt_level=0)
        text0 = disassembly(small, "scalar")
        text2 = disassembly(self.f, "scalar")
        self.assertNotEqual(text0, text2)

    def test_more_functions(self):
        f = compile_func([x], [sin(x) * cos(x) + x**3])
        text, _, _ = self.check_listing(f, "scalar")
        self.assertEqual(sum(m == "call" for m in mnemonics(text)), 2)  # sin and cos


@unittest.skipUnless(HAS_SYMBOLICA, "needs symbolica")
class SymbolicaTests(ListingChecks, unittest.TestCase):
    def setUp(self):
        a, b = S("a"), S("b")
        self.evaluator = E("a*b + a^2").evaluator([a, b])

    @unittest.skipUnless(HOST_X86, "needs an x86-64 host")
    def test_real_and_complex(self):
        for dtype in ("float64", "complex128"):
            f = symjit.compile_evaluator(self.evaluator, dtype=dtype)
            self.assertIsInstance(f, symjit.SymbolicaFunc)
            self.check_listing(f, "scalar")
            self.check_listing(f, "simd")


@unittest.skipUnless(HAS_CAPSTONE, "needs capstone")
class CrossArchitectureTests(ListingChecks, unittest.TestCase):
    """ARM and RISC-V code can be generated (not run) on any host."""

    def test_aarch64(self):
        f = compile_sample(ty="arm")
        for kernel in ("scalar", "simd"):
            text, rows, _ = self.check_listing(f, kernel)
            self.assertIn("aarch64", text.splitlines()[0])
            self.assertTrue(all(len(r) == 4 for _, r, _ in rows))  # fixed-width instructions
            ms = mnemonics(text)
            self.assertIn("stp", ms)  # prologue
            self.assertTrue(any(m in ("fmadd", "fmul", "fmla") for m in ms))  # x * y + 3.5
            self.assertIn("blr", ms)  # call to sin

    def test_riscv64(self):
        f = compile_sample(ty="riscv")
        text, rows, _ = self.check_listing(f, "scalar")
        self.assertIn("riscv64", text.splitlines()[0])
        self.assertTrue(all(len(r) == 4 for _, r, _ in rows))
        ms = mnemonics(text)
        self.assertIn("sd", ms)
        self.assertTrue(any(m.startswith("fmadd") or m.startswith("fmul") for m in ms))

    @unittest.skipIf(disasm._find_objdump("aarch64") is not None, "an aarch64 objdump is installed")
    def test_objdump_without_target_support(self):
        f = compile_sample(ty="arm")
        with self.assertRaisesRegex(RuntimeError, "no objdump supporting aarch64"):
            disassemble(f, tool="objdump")


class MissingToolsTests(unittest.TestCase):
    def test_no_disassembler_available(self):
        # without objdump and capstone, `auto` uses symjit's own decoder
        f = compile_sample(ty="riscv")
        with mock.patch.object(disasm, "_find_objdump", return_value=None), mock.patch.object(
            disasm, "_run_capstone", return_value=None
        ):
            text = disassembly(f)
        self.assertIn("disassembled with symjit", text.splitlines()[0])

    def test_capstone_missing(self):
        f = compile_sample(ty="riscv")
        with mock.patch.object(disasm, "_run_capstone", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "capstone"):
                disassemble(f, tool="capstone")


class ArchitectureTests(unittest.TestCase):
    def test_explicit_types(self):
        self.assertEqual(disasm._architecture("amd"), "x86_64")
        self.assertEqual(disasm._architecture("amd-avx"), "x86_64")
        self.assertEqual(disasm._architecture("amd-sse"), "x86_64")
        self.assertEqual(disasm._architecture("arm"), "aarch64")
        self.assertEqual(disasm._architecture("riscv"), "riscv64")

    def test_native_follows_the_host(self):
        for machine, arch in [("x86_64", "x86_64"), ("AMD64", "x86_64"), ("aarch64", "aarch64"),
                              ("arm64", "aarch64"), ("riscv64", "riscv64")]:
            with mock.patch("platform.machine", return_value=machine):
                self.assertEqual(disasm._architecture("native"), arch)
                self.assertEqual(disasm._architecture("debug"), arch)

    def test_unsupported(self):
        with mock.patch("platform.machine", return_value="sparc64"):
            with self.assertRaises(RuntimeError):
                disasm._architecture("native")
        with self.assertRaises(ValueError):
            disasm._architecture("bytecode")


class HelperTests(unittest.TestCase):
    """Pure-function tests of the parsing heuristics with synthetic input."""

    def test_x86_data_target(self):
        t = disasm._x86_data_target
        # objdump: the target is given in a comment
        self.assertEqual(t(0x84, 8, "vaddsd xmm3,xmm3,QWORD PTR [rip+0x74] # 0x100"), 0x100)
        # capstone: compute next-ip + displacement
        self.assertEqual(t(0x84, 8, "vaddsd xmm3, xmm3, qword ptr [rip + 0x74]"), 0x84 + 8 + 0x74)
        self.assertEqual(t(0x100, 7, "mov rax, qword ptr [rip - 0x10]"), 0x100 + 7 - 0x10)
        # `lea` may form a code address and is ignored; other instructions do not use rip
        self.assertIsNone(t(0, 7, "lea rax,[rip+0x10] # 0x17"))
        self.assertIsNone(t(0, 3, "mov rax,rbx"))

    @staticmethod
    def records(items):
        out, address = [], 0
        for raw, text in items:
            out.append((address, bytes(raw), text))
            address += len(raw)
        return out

    def test_split_uses_lowest_rip_target_on_x86(self):
        recs = self.records(
            [
                ([0x55], "push rbp"),
                ([0xC5] * 4 + [0], "vmovsd xmm0,QWORD PTR [rip+0x6] # 0xc"),
                ([0xC5] * 4 + [0], "vaddsd xmm0,xmm0,QWORD PTR [rip+0x1e] # 0x20"),  # a later pool entry
                ([0xC3], "ret"),
                ([0x90] * 3, "nop"),
                ([0xC3], "ret"),  # a fake `ret` decoded from the constant pool
                ([0, 0], "add BYTE PTR [rax],al"),
            ]
        )
        # the pool starts at 0xc (just after the real `ret`); the later `ret` is data
        code, data = disasm._split_code_data(recs, "x86_64")
        self.assertEqual(code[-1][2], "ret")
        self.assertEqual(code[-1][0], recs[3][0])
        self.assertEqual([r[2] for r in data], ["nop", "ret", "add BYTE PTR [rax],al"])

    def test_split_ignores_lea_targets(self):
        recs = self.records(
            [
                ([0x48] * 7, "lea rax,[rip+0x0] # 0x2"),  # points into the code
                ([0xC3], "ret"),
                ([0x00] * 8, "add BYTE PTR [rax],al"),
            ]
        )
        code, data = disasm._split_code_data(recs, "x86_64")
        self.assertEqual(len(code), 2)  # the `lea` target did not truncate the code

    def test_split_without_targets_uses_last_ret(self):
        recs = self.records(
            [([0x55], "push rbp"), ([0xC3], "ret"), ([0x55], "push rbp"), ([0xC3], "ret"), ([0x0], "add")]
        )
        code, data = disasm._split_code_data(recs, "x86_64")
        self.assertEqual(len(code), 4)
        self.assertEqual(len(data), 1)

    def test_split_on_risc_architectures(self):
        word = bytes(4)
        recs = self.records(
            [(word, "addi sp, sp, -64"), (word, "ret"), (word, "ret"), (word, ".word"), (word, ".word")]
        )
        for arch in ("aarch64", "riscv64"):
            code, data = disasm._split_code_data(recs, arch)
            self.assertEqual(len(code), 3)
            self.assertEqual(len(data), 2)

    def test_split_without_ret(self):
        recs = self.records([([0x90], "nop"), ([0x90], "nop")])
        code, data = disasm._split_code_data(recs, "x86_64")
        self.assertEqual((len(code), len(data)), (2, 0))

    def test_format_data(self):
        pool = struct.pack("<d", 3.5) + struct.pack("<Q", 0xFFFFFFFFFFFFFFFF) + struct.pack("<d", -0.0)
        # the pool starts at an unaligned address: 5 padding bytes, then aligned words
        data = [(0xFB + i, bytes([b]), "x") for i, b in enumerate(bytes(5) + pool)]
        lines = disasm._format_data(data, 0xFB, 0xFB + len(bytes(5) + pool))
        self.assertIn("(padding)", lines[1])
        self.assertIn("3.5", lines[2])
        self.assertIn("0xffffffffffffffff", lines[3])
        self.assertNotIn(";", lines[3])  # NaN pattern: no float annotation
        self.assertIn("-0.0", lines[4])
        self.assertTrue(lines[2].lstrip().startswith("100:"))  # 0xfb + 5 is 8-byte aligned

    def test_objdump_parser(self):
        out = "\n".join(
            [
                "code.bin:     file format binary",
                "Disassembly of section .data:",
                "0000000000000000 <.data>:",
                "   0:\t55                   \tpush   rbp",
                "   1:\t48 81 ec 30 00 00 00 \tsub    rsp,0x30",
                "   8:\tc3                   \tret",
            ]
        )
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(stdout=out)
            recs = disasm._run_objdump("/usr/bin/objdump", "x86_64", b"\x55")
        self.assertEqual(recs[0], (0, b"\x55", "push rbp"))
        self.assertEqual(recs[1][1], bytes.fromhex("4881ec30000000"))
        self.assertEqual(recs[2], (8, b"\xc3", "ret"))


class SymjitToolTests(unittest.TestCase):
    """`tool="symjit"`: symjit's own decoder, on every architecture, with no external tool."""

    TYPES = (("riscv", "riscv64"), ("arm", "aarch64"), ("amd", "x86_64"))

    def check(self, f, kernel, arch):
        text = disassembly(f, kernel, tool="symjit")
        rows, comments = parse_listing(text)
        code = raw_kernel(f, kernel)
        head = comments[0]
        self.assertTrue(head.startswith(f"; symjit `{kernel}` kernel, {arch}"), head)
        self.assertIn("disassembled with symjit", head)

        # contiguous rows, exactly the kernel's code; the rest is the constant pool
        expected = 0
        for address, raw, _ in rows:
            self.assertEqual(address, expected)
            expected += len(raw)
        self.assertEqual(b"".join(r for _, r, _ in rows), code[:expected])
        self.assertIn("ret", mnemonics(text))
        pool = [c for c in comments if c.startswith("; constant pool")]
        self.assertEqual(len(pool), 1)
        self.assertIn(f"{len(code) - expected} bytes at offset {expected:#x}", pool[0])

        # every instruction decoded, and the labels named
        for _, _, ins in rows:
            self.assertFalse(ins.startswith((".byte", ".word", ".inst ")), ins)
        self.assertIn("@self:", text.splitlines())
        return text

    def test_kernels(self):
        for ty, arch in self.TYPES:
            for dtype in ("float64", "complex128"):
                f = compile_sample(ty=ty, dtype=dtype, use_threads=False)
                for kernel in ("scalar", "simd", "fast"):
                    with self.subTest(ty=ty, dtype=dtype, kernel=kernel):
                        try:
                            raw_kernel(f, kernel)
                        except ValueError:
                            continue  # e.g. no SIMD kernel on RISC-V
                        self.check(f, kernel, arch)

    def test_names_targets_and_constants(self):
        f = compile_func([x, y], [x * y + 3.5], ty="riscv")
        text = disassembly(f, tool="symjit", show_data=True)
        self.assertIn("<@main>", text)  # a branch names its label
        self.assertIn("<_const_0_> = 3.5", text)  # a constant load names the constant
        self.assertIn("_const_0_:", text.splitlines())
        self.assertIn(".quad 0x400c000000000000", text)

    def test_hides_data(self):
        f = compile_func([x, y], [x * y + 3.5], ty="arm")
        text = disassembly(f, tool="symjit")
        self.assertNotIn(".quad", text)
        self.assertIn("use show_data=True", text)

    @unittest.skipUnless(HOST_X86 and HAS_OBJDUMP_X86, "needs an x86-64 host and objdump")
    def test_agrees_with_objdump(self):
        # the same instruction boundaries as objdump, up to the last `ret`
        f = compile_sample(use_threads=False)
        for kernel in ("scalar", "simd"):
            ours, _ = parse_listing(disassembly(f, kernel, tool="symjit"))
            theirs, _ = parse_listing(disassembly(f, kernel, tool="objdump"))
            self.assertEqual([a for a, _, _ in theirs], [a for a, _, _ in ours][: len(theirs)])


def _binutils(arch):
    """as, ld, objcopy for `arch`, or None."""
    prefix = {"x86_64": "x86_64-linux-gnu-", "aarch64": "aarch64-linux-gnu-", "riscv64": "riscv64-linux-gnu-"}[arch]
    native = sys.platform.startswith("linux") and disasm._architecture("native") == arch
    for p in (prefix, "") if native else (prefix,):
        tools = [shutil.which(p + t) for t in ("as", "ld", "objcopy")]
        if all(tools):
            return tools
    return None


class AssemblyTests(unittest.TestCase):
    """`assembly`: the kernel as GNU `as` input, which assembles back to the same bytes."""

    def test_form(self):
        f = compile_sample(ty="riscv")
        text = assembly(f)
        lines = text.splitlines()
        self.assertIn("\t.globl symjit_scalar", lines)
        self.assertIn("symjit_scalar:", lines)
        self.assertIn(".L_self:", lines)
        self.assertIn(".L_const_0_:", lines)
        # no offsets or bytes
        self.assertFalse(any(re.match(r"^\s*[0-9a-f]+:", l) for l in lines))

    def test_arguments(self):
        f = compile_sample()
        with self.assertRaisesRegex(ValueError, "kernel should be one of"):
            assembly(f, "bytecode")
        with self.assertRaisesRegex(ValueError, "does not generate machine code"):
            assembly(compile_func([x], [x + 1], ty="bytecode"))
        with self.assertRaisesRegex(ValueError, "`simd` kernel is not available"):
            assembly(compile_sample(ty="riscv"), "simd")

    def test_assembles_to_the_kernel(self):
        tested = 0
        for ty, arch in SymjitToolTests.TYPES:
            tools = _binutils(arch)
            if tools is None:
                continue
            asm, ld, objcopy = tools
            for dtype in ("float64", "complex128"):
                f = compile_sample(ty=ty, dtype=dtype, use_threads=False)
                for kernel in ("scalar", "simd", "fast"):
                    try:
                        code = raw_kernel(f, kernel)
                    except ValueError:
                        continue
                    with self.subTest(ty=ty, dtype=dtype, kernel=kernel), tempfile.TemporaryDirectory() as d:
                        with open(os.path.join(d, "k.s"), "w") as fd:
                            fd.write(assembly(f, kernel))
                        for cmd in (
                            [asm, "-o", "k.o", "k.s"],
                            [ld, "-Ttext=0", "-e", f"symjit_{kernel}", "-o", "k.elf", "k.o"],
                            [objcopy, "-O", "binary", "-j", ".text", "k.elf", "k.bin"],
                        ):
                            subprocess.run(cmd, cwd=d, check=True, capture_output=True)
                        with open(os.path.join(d, "k.bin"), "rb") as fd:
                            got = fd.read()
                        # the assembler may pad the section to its alignment
                        self.assertEqual(got[: len(code)], code)
                        self.assertLess(len(got) - len(code), 16)
                        tested += 1
        if tested == 0:
            self.skipTest("needs binutils for x86-64, aarch64 or riscv64")


if __name__ == "__main__":
    unittest.main()
