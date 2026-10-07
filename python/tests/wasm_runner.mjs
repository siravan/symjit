// Runs a symjit WebAssembly module in Node for python/tests/test_wasm.py.
//
// Reads a JSON request from stdin:
//   {"module": path, "mode": "direct" | "indirect" | "fast" | "params",
//    "count_states": n, "count_obs": m, "mem_size": k,
//    "points": [[x0, x1, ...], ...], "params": [p0, ...]}
// ("params" is the Symbolica/Composer convention: each point holds the parameters,
// and run(outs, 0, 0, params) writes `count_obs` outputs to `outs`)
// and prints {"obs": [[y0, y1, ...], ...], "status": [...], "imports": [...]},
// or, for a JSON array of requests, an array of results in which a failing
// request gives {"error": message}.
//
// An optional "host": {name: "JavaScript function source"} supplies further
// imports (user-defined functions).
//
// Numbers are exchanged exactly: NaN, Infinity, -Infinity and -0 (which JSON
// cannot represent) travel as the strings "NaN", "Infinity", "-Infinity", "-0".
//
// It is also the reference host for the module interface in docs/WASM.md.

import { readFileSync } from "node:fs";

const recip = (f) => (x) => 1 / f(x);

// symjit function names -> JavaScript implementations
const MATH = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    csc: recip(Math.sin), sec: recip(Math.cos), cot: recip(Math.tan),
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    csch: recip(Math.sinh), sech: recip(Math.cosh), coth: recip(Math.tanh),
    arcsin: Math.asin, arccos: Math.acos, arctan: Math.atan,
    arcsinh: Math.asinh, arccosh: Math.acosh, arctanh: Math.atanh,
    sinc: (x) => (x === 0 ? 1 : Math.sin(x) / x),
    exp: Math.exp, expm1: Math.expm1, exp2: (x) => 2 ** x,
    ln: Math.log, log: Math.log, log1p: Math.log1p, log2: Math.log2,
    // C99 pow: pow(1, y) = 1 and pow(-1, +-inf) = 1 (JavaScript's Math.pow gives NaN)
    power: (x, y) => (x === 1 || (x === -1 && Math.abs(y) === Infinity) ? 1 : Math.pow(x, y)),
    cbrt: Math.cbrt, atan2: Math.atan2,
    arg: (x) => (x < 0 ? Math.PI : 0),
    random: Math.random,
};

// complex functions take (re, im[, re, im]) and return [re, im] (principal branches,
// as in Rust's num_complex)
const c = {
    add: ([a, b], [x, y]) => [a + x, b + y],
    sub: ([a, b], [x, y]) => [a - x, b - y],
    mul: ([a, b], [x, y]) => [a * x - b * y, a * y + b * x],
    div: ([a, b], [x, y]) => { const d = x * x + y * y; return [(a * x + b * y) / d, (b * x - a * y) / d]; },
    exp: ([a, b]) => { const e = Math.exp(a); return [e * Math.cos(b), e * Math.sin(b)]; },
    ln: ([a, b]) => [Math.log(Math.hypot(a, b)), Math.atan2(b, a)],
    sqrt: ([a, b]) => {
        if (a === 0 && b === 0) return [0, b];
        const t = Math.sqrt((Math.hypot(a, b) + Math.abs(a)) / 2);
        return a >= 0 ? [t, b / (2 * t)] : [Math.abs(b) / (2 * t), Math.sign(b) * t || t];
    },
    sin: ([a, b]) => [Math.sin(a) * Math.cosh(b), Math.cos(a) * Math.sinh(b)],
    cos: ([a, b]) => [Math.cos(a) * Math.cosh(b), -Math.sin(a) * Math.sinh(b)],
    sinh: ([a, b]) => [Math.sinh(a) * Math.cos(b), Math.cosh(a) * Math.sin(b)],
    cosh: ([a, b]) => [Math.cosh(a) * Math.cos(b), Math.sinh(a) * Math.sin(b)],
};
c.recip = (z) => c.div([1, 0], z);
c.tan = (z) => c.div(c.sin(z), c.cos(z));
c.tanh = (z) => c.div(c.sinh(z), c.cosh(z));
c.pow = (z, w) => (z[0] === 0 && z[1] === 0 ? [0, 0] : c.exp(c.mul(w, c.ln(z))));
const I = [0, 1], ONE = [1, 0];
c.asin = (z) => c.mul([0, -1], c.ln(c.add(c.mul(I, z), c.sqrt(c.sub(ONE, c.mul(z, z))))));
c.atan = (z) => c.mul([0, 0.5], c.ln(c.div(c.add(I, z), c.sub(I, z))));
c.asinh = (z) => c.ln(c.add(z, c.sqrt(c.add(c.mul(z, z), ONE))));
c.acosh = (z) => c.ln(c.add(z, c.mul(c.sqrt(c.add(z, ONE)), c.sqrt(c.sub(z, ONE)))));
c.atanh = (z) => c.mul([0.5, 0], c.ln(c.div(c.add(ONE, z), c.sub(ONE, z))));

const unary = (f) => (re, im) => f([re, im]);
const COMPLEX = {
    cplx_sin: unary(c.sin), cplx_cos: unary(c.cos), cplx_tan: unary(c.tan),
    cplx_csc: unary((z) => c.recip(c.sin(z))), cplx_sec: unary((z) => c.recip(c.cos(z))),
    cplx_cot: unary((z) => c.recip(c.tan(z))),
    cplx_sinh: unary(c.sinh), cplx_cosh: unary(c.cosh), cplx_tanh: unary(c.tanh),
    cplx_csch: unary((z) => c.recip(c.sinh(z))), cplx_sech: unary((z) => c.recip(c.cosh(z))),
    cplx_coth: unary((z) => c.recip(c.tanh(z))),
    cplx_sinc: unary((z) => (z[0] === 0 && z[1] === 0 ? [1, 0] : c.div(c.sin(z), z))),
    cplx_arcsin: unary(c.asin), cplx_arccos: unary((z) => c.sub([Math.PI / 2, 0], c.asin(z))),
    cplx_arctan: unary(c.atan), cplx_arcsinh: unary(c.asinh), cplx_arccosh: unary(c.acosh),
    cplx_arctanh: unary(c.atanh),
    cplx_root: unary(c.sqrt), cplx_cbrt: unary((z) => c.pow(z, [1 / 3, 0])),
    cplx_exp: unary(c.exp), cplx_ln: unary(c.ln), cplx_log: unary(c.ln),
    cplx_arg: unary(([a, b]) => [Math.atan2(b, a), 0]),
    cplx_power: (a, b, x, y) => c.pow([a, b], [x, y]),
};
Object.assign(MATH, COMPLEX);

const decode = (v) => (typeof v === "string" ? (v === "-0" ? -0 : Number(v)) : v);
const encode = (v) => (Number.isFinite(v) && !Object.is(v, -0) ? v : Object.is(v, -0) ? "-0" : String(v));

function evaluate(req) {
    req = { ...req, points: req.points.map((p) => p.map(decode)), params: req.params.map(decode) };
    const module = new WebAssembly.Module(readFileSync(req.module));
    const requested = WebAssembly.Module.imports(module).map((i) => i.name);

    const host = { ...MATH };
    for (const [name, src] of Object.entries(req.host ?? {})) host[name] = new Function(`return ${src}`)();

    const missing = requested.filter((name) => !(name in host));
    if (missing.length > 0) {
        throw new Error(`no host implementation for: ${missing.join(", ")}`);
    }

    const instance = new WebAssembly.Instance(module, { symjit: host });
    const { run, memory, __heap_base } = instance.exports;

    const cs = req.count_states, co = req.count_obs;
    const points = req.points, n = points.length;

    // heap layout (bytes from __heap_base): params | mem | state/obs arrays | table
    let top = __heap_base.value;
    const alloc = (bytes) => { const p = top; top += Math.ceil(bytes / 16) * 16; return p; };
    const pParams = alloc(8 * Math.max(1, req.params.length, ...points.map((p) => p.length)));
    const pMem = alloc(8 * req.mem_size);
    const pArrays = alloc(8 * n * (cs + co));
    const pTable = alloc(8 * (cs + co));

    const need = Math.ceil(top / 65536) - memory.buffer.byteLength / 65536;
    if (need > 0) memory.grow(need);

    const f64 = new Float64Array(memory.buffer);
    const i32 = new Int32Array(memory.buffer);
    f64.set(req.params, pParams / 8);

    const obs = [], status = [];

    if (req.mode === "direct") {
        for (const p of points) {
            f64.fill(0, pMem / 8, pMem / 8 + req.mem_size);
            f64.set(p, pMem / 8);
            status.push(run(pMem, 0, 0, pParams));
            obs.push(Array.from(f64.subarray(pMem / 8 + cs, pMem / 8 + cs + co)));
        }
    } else if (req.mode === "params") {
        for (const p of points) {
            f64.fill(0, pMem / 8, pMem / 8 + req.mem_size);
            f64.set(p, pParams / 8);
            status.push(run(pMem, 0, 0, pParams));
            obs.push(Array.from(f64.subarray(pMem / 8, pMem / 8 + co)));
        }
    } else if (req.mode === "fast") {
        if (!instance.exports.fast) throw new Error("the module has no `fast` export");
        for (const p of points) {
            obs.push([instance.exports.fast(...p)]);
            status.push(0);
        }
    } else {
        // one array of length n per state, then per observable; the table holds (ptr, len)
        for (let k = 0; k < cs + co; k++) {
            const ptr = pArrays + 8 * n * k;
            i32[pTable / 4 + 2 * k] = ptr;
            i32[pTable / 4 + 2 * k + 1] = n;
            for (let j = 0; j < n; j++) f64[ptr / 8 + j] = k < cs ? points[j][k] : NaN;
        }
        for (let j = 0; j < n; j++) status.push(run(0, pTable, j, pParams));
        for (let j = 0; j < n; j++) {
            const row = [];
            for (let k = cs; k < cs + co; k++) row.push(f64[(pArrays + 8 * n * k) / 8 + j]);
            obs.push(row);
        }
    }

    return { obs: obs.map((row) => row.map(encode)), status, imports: requested };
}

const req = JSON.parse(readFileSync(0, "utf8"));

if (Array.isArray(req)) {
    const results = req.map((r) => {
        try {
            return evaluate(r);
        } catch (e) {
            return { error: String(e) };
        }
    });
    console.log(JSON.stringify(results));
} else {
    console.log(JSON.stringify(evaluate(req)));
}
