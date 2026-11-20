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
                self.assertEqual((d["format"], d["format_version"]), ("symjit-application", 1))
                self.assertEqual(d["mir"]["complexified"], dtype == "complex128" and what == "json-complex")
                self.assertTrue(all(c.startswith("0x") and len(c) == 18 for c in d["mir"]["consts"]))
                self.assertEqual(d["program"]["config"]["options"]["complex"], dtype == "complex128")
        self.assertEqual(symjit.mir_json(self.path("float64.sjb")), symjit.mir_json(self.path("float64.sjb"), "json"))

    def test_refused(self):
        f = self.funcs["float64"]
        f.save(self.path("f.sjb"))
        with self.assertRaises(ValueError):
            symjit.mir_json(b"not a saved function")
        with self.assertRaises(ValueError):
            symjit.mir_json(self.path("f.sjb"), "scalar")



class TestLoadJson(unittest.TestCase):
    """symjit.load_json: the JSON of a function ("symjit-application") loaded back, from its text, its
    bytes or its file, gives the function: the same values and the same saved bytes."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        x, y = symbols("x y")
        self.funcs = {dtype: compile_func([x, y], [x * sin(y) + exp(x)], dtype=dtype)
                      for dtype in ("float64", "complex128")}
        self.points = {"float64": [(0.5, 2.0), (-1.25, 3.5)], "complex128": [(0.5 + 1j, 2.0 - 0.5j), (-1.25, 3.5j)]}

    def tearDown(self):
        self.dir.cleanup()

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def saved(self, f, name):
        f.save(self.path(name))
        with open(self.path(name), "rb") as fd:
            return fd.read()

    def test_load_json_is_the_function(self):
        for dtype, f in self.funcs.items():
            f.dump(self.path("f.json"), what="json")
            with open(self.path("f.json")) as fd:
                text = fd.read()
            for src in (text, text.encode("utf-8"), self.path("f.json")):
                g = symjit.load_json(src)
                self.assertIsInstance(g, type(f))
                for p in self.points[dtype]:
                    self.assertEqual(complex(g(*p)[0]), complex(f(*p)[0]), f"{dtype} {p}")
                self.assertEqual(self.saved(g, "g.sjb"), self.saved(f, "f.sjb"), dtype)

    def test_mir_json_text_loads(self):
        f = self.funcs["float64"]
        f.save(self.path("f.sjb"))
        g = symjit.load_json(symjit.mir_json(self.path("f.sjb"), "json"))
        self.assertEqual(g(0.5, 2.0)[0], f(0.5, 2.0)[0])

    def test_refused(self):
        # a complex function's complexified MIR, a file that is not the format, and a bad text
        self.funcs["complex128"].dump(self.path("c.json"), what="json-complex")
        for src in (self.path("c.json"), '{"format": "symjit-mir"}', "{not json"):
            with self.assertRaises(ValueError):
                symjit.load_json(src)
        with self.assertRaises(FileNotFoundError):
            symjit.load_json(self.path("missing.json"))


if __name__ == "__main__":
    unittest.main()
