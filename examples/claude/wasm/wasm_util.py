"""Helpers for the WebAssembly examples: export a module and run it in Node."""

import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def have_node():
    return shutil.which("node") is not None


def export(f, name, dtype="float64"):
    """Writes out/<name>.wasm and out/<name>.json (the memory layout host.mjs needs).

    `f` is a function compiled with ty="wasm" (compile_func, compile_ode,
    compile_composer or compile_evaluator)."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.wasm")
    if hasattr(f, "complex_compiler"):  # the Symbolica bridge
        c = f.complex_compiler if dtype == "complex128" else f.compiler
        f.dump(path, "wasm", dtype=dtype)
    else:
        c = f.compiler
        f.dump(path, "wasm")

    layout = {
        "count_states": c.count_states,
        "count_obs": c.count_obs,
        "count_params": c.count_params,
        "mem_size": c.count_states + c.count_obs + c.count_diffs + 1,
    }
    with open(os.path.join(OUT, f"{name}.json"), "w") as fd:
        json.dump(layout, fd)

    print(f"wrote out/{name}.wasm ({os.path.getsize(path)} bytes), layout {layout}")
    return path, layout


def node(path, layout, mode, args, params=(), host=None):
    """Evaluates the module with host.mjs (see run.mjs for the modes)."""
    req = {"module": path, "layout": layout, "mode": mode, "args": args, "params": list(params), "host": host or {}}
    res = subprocess.run(
        ["node", os.path.join(HERE, "run.mjs")], input=json.dumps(req), capture_output=True, text=True, timeout=600
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    return json.loads(res.stdout)
