// Command-line driver for the examples: evaluates a module with host.mjs.
//
//     node run.mjs < request.json
//
// request: {"module": "f.wasm", "layout": {...}, "mode": "evaluate" | "evaluateMany" |
//           "evaluateInputs" | "fast", "args": [...], "params": [...],
//           "host": {"name": "JavaScript function source", ...}}
// prints the result as JSON.

import { readFileSync } from "node:fs";
import { load } from "./host.mjs";

const req = JSON.parse(readFileSync(0, "utf8"));
const extra = Object.fromEntries(Object.entries(req.host ?? {}).map(([k, src]) => [k, new Function(`return ${src}`)()]));
const model = await load(readFileSync(req.module), req.layout, extra);

let result;
switch (req.mode) {
    case "evaluate":
        result = req.args.map((pt) => model.evaluate(pt, req.params ?? []));
        break;
    case "evaluateMany":
        result = model.evaluateMany(req.args.map((col) => Float64Array.from(col)), req.params ?? []).map((a) => Array.from(a));
        break;
    case "evaluateInputs":
        result = req.args.map((pt) => {
            try {
                return model.evaluateInputs(pt);
            } catch (e) {
                return String(e.message);
            }
        });
        break;
    case "fast":
        result = req.args.map((pt) => model.fast(...pt));
        break;
    default:
        throw new Error(`unknown mode ${req.mode}`);
}
console.log(JSON.stringify(result));
