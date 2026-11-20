"""Unit tests for the MIR as JSON: `dump(path, what="json" | "json-complex")` and `symjit.mir_json`,
which reads a saved function (.sjb) without compiling it.

Run from the repository root with

    python -m unittest discover -s python/tests -v
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(os.path.dirname(__file__), "..")))

from sympy import exp, sin, symbols

import symjit
from symjit import compile_func


class TestMirJson(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        x, y = symbols("x y")
        self.funcs = {dtype: compile_func([x, y], [x * sin(y) + exp(x)], dtype=dtype)
                      for dtype in ("float64", "complex128")}

    def tearDown(self):
        self.dir.cleanup()

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def dumped(self, f, what):
        f.dump(self.path("dump.json"), what=what)
        with open(self.path("dump.json")) as fd:
            return fd.read()

    def test_mir_json_is_the_dump(self):
        for dtype, f in self.funcs.items():
            sjb = self.path(f"{dtype}.sjb")
            f.save(sjb)
            with open(sjb, "rb") as fd:
                data = fd.read()
            loaded = symjit.load_func(sjb)
            for what in ("json", "json-complex"):
                text = symjit.mir_json(data, what)
                self.assertEqual(text, self.dumped(loaded, what), f"{dtype} {what}")
                self.assertEqual(symjit.mir_json(sjb, what), text)  # a file name reads the same
                self.assertEqual(symjit.mir_json(bytearray(data), what), text)

    def test_format(self):
        for dtype, f in self.funcs.items():
            sjb = self.path(f"{dtype}.sjb")
            f.save(sjb)
            for what in ("json", "json-complex"):
                d = json.loads(symjit.mir_json(sjb, what))
                self.assertEqual((d["format"], d["format_version"]), ("symjit-mir", 1))
                self.assertEqual(d["complexified"], dtype == "complex128" and what == "json-complex")
                self.assertTrue(all(c.startswith("0x") and len(c) == 18 for c in d["consts"]))
        self.assertEqual(symjit.mir_json(self.path("float64.sjb")), symjit.mir_json(self.path("float64.sjb"), "json"))

    def test_refused(self):
        f = self.funcs["float64"]
        f.save(self.path("f.sjb"))
        with self.assertRaises(ValueError):
            symjit.mir_json(b"not a saved function")
        with self.assertRaises(ValueError):
            symjit.mir_json(self.path("f.sjb"), "scalar")


if __name__ == "__main__":
    unittest.main()
