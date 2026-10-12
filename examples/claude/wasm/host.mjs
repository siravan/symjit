// A small WebAssembly host for symjit modules (`ty="wasm"`), usable in Node and in
// browsers. See docs/WASM.md for the module interface.
//
//     import { load } from "./host.mjs";
//     const model = await load(bytes, layout);       // layout: the .json written by wasm_util.py
//     model.evaluate([x, y]);                         // one point -> outputs
//     model.evaluateMany([xs, ys]);                   // arrays (one per state) -> arrays
//     model.evaluateInputs([a, b, c]);                // Composer/Symbolica models
//     model.fast?.(x, y);                             // single-output models
//
// `layout` is {count_states, count_obs, count_params, mem_size}; complex models count
// real and imaginary parts separately (values are interleaved re, im).

const recip = (f) => (x) => 1 / f(x);

// symjit's real functions, by name. C semantics where JavaScript differs:
// pow(1, y) = 1 and pow(-1, +-inf) = 1 (Math.pow gives NaN).
export const MATH = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    csc: recip(Math.sin), sec: recip(Math.cos), cot: recip(Math.tan),
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    csch: recip(Math.sinh), sech: recip(Math.cosh), coth: recip(Math.tanh),
    arcsin: Math.asin, arccos: Math.acos, arctan: Math.atan,
    arcsinh: Math.asinh, arccosh: Math.acosh, arctanh: Math.atanh,
    sinc: (x) => (x === 0 ? 1 : Math.sin(x) / x),
    exp: Math.exp, expm1: Math.expm1, exp2: (x) => 2 ** x,
    ln: Math.log, log: Math.log, log1p: Math.log1p, log2: Math.log2,
    cbrt: Math.cbrt, atan2: Math.atan2,
    power: (x, y) => (x === 1 || (x === -1 && Math.abs(y) === Infinity) ? 1 : Math.pow(x, y)),
};

// complex helpers on [re, im] pairs (principal branches)
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
        return a >= 0 ? [t, b / (2 * t)] : [Math.abs(b) / (2 * t), b < 0 ? -t : t];
    },
    sin: ([a, b]) => [Math.sin(a) * Math.cosh(b), Math.cos(a) * Math.sinh(b)],
    cos: ([a, b]) => [Math.cos(a) * Math.cosh(b), -Math.sin(a) * Math.sinh(b)],
    sinh: ([a, b]) => [Math.sinh(a) * Math.cos(b), Math.cosh(a) * Math.sin(b)],
    cosh: ([a, b]) => [Math.cosh(a) * Math.cos(b), Math.sinh(a) * Math.sin(b)],
};
c.tan = (z) => c.div(c.sin(z), c.cos(z));
c.tanh = (z) => c.div(c.sinh(z), c.cosh(z));
c.pow = (z, w) => (z[0] === 0 && z[1] === 0 ? [0, 0] : c.exp(c.mul(w, c.ln(z))));

// symjit's complex functions: (re, im[, re, im]) -> [re, im]
const unary = (f) => (re, im) => f([re, im]);
export const COMPLEX = {
    cplx_sin: unary(c.sin), cplx_cos: unary(c.cos), cplx_tan: unary(c.tan),
    cplx_sinh: unary(c.sinh), cplx_cosh: unary(c.cosh), cplx_tanh: unary(c.tanh),
    cplx_exp: unary(c.exp), cplx_ln: unary(c.ln), cplx_log: unary(c.ln),
    cplx_root: unary(c.sqrt),
    cplx_power: (a, b, x, y) => c.pow([a, b], [x, y]),
};

/** Compiles and instantiates a symjit module; `extra` adds or overrides imports. */
export async function load(bytes, layout, extra = {}) {
    const module = await WebAssembly.compile(bytes);
    const host = { ...MATH, ...COMPLEX, ...extra };
    const missing = WebAssembly.Module.imports(module).map((i) => i.name).filter((n) => !(n in host));
    if (missing.length > 0) throw new Error(`the host lacks: ${missing.join(", ")}`);
    const instance = await WebAssembly.instantiate(module, { symjit: host });
    return new Model(instance.exports, layout);
}

export class Model {
    constructor(exports, layout) {
        this.exports = exports;
        this.layout = layout;
        this.fast = exports.fast; // undefined unless exported
    }

    // a fresh bump allocator above __heap_base for each call (8-byte aligned)
    #alloc(sizes) {
        const { memory, __heap_base } = this.exports;
        let top = __heap_base.value;
        const ptrs = sizes.map((n) => { const p = top; top += 8 * Math.max(n, 1); return p; });
        const pages = Math.ceil(top / 65536) - memory.buffer.byteLength / 65536;
        if (pages > 0) memory.grow(pages);
        return ptrs;
    }

    #call(...args) {
        const status = this.exports.run(...args);
        if (status !== 0) throw new Error(`run failed with status ${status} (stack exhausted)`);
    }

    /** One point, direct mode: states -> outputs. */
    evaluate(states, params = []) {
        const { count_states: cs, count_obs: co, mem_size } = this.layout;
        const [mem, par] = this.#alloc([mem_size, params.length]);
        const f64 = new Float64Array(this.exports.memory.buffer);
        f64.fill(0, mem / 8, mem / 8 + mem_size);
        f64.set(states, mem / 8);
        f64.set(params, par / 8);
        this.#call(mem, 0, 0, par);
        return Array.from(f64.subarray(mem / 8 + cs, mem / 8 + cs + co));
    }

    /** Many points, indirect mode: one array per state -> one Float64Array per output. */
    evaluateMany(columns, params = []) {
        const { count_states: cs, count_obs: co } = this.layout;
        const n = columns[0].length;
        const [table, par, ...arrays] = this.#alloc([cs + co, params.length, ...Array(cs + co).fill(n)]);
        const f64 = new Float64Array(this.exports.memory.buffer);
        const i32 = new Int32Array(this.exports.memory.buffer);
        arrays.forEach((p, k) => {
            i32[table / 4 + 2 * k] = p; // (ptr, len) pairs: states, then outputs
            i32[table / 4 + 2 * k + 1] = n;
            if (k < cs) f64.set(columns[k], p / 8);
        });
        f64.set(params, par / 8);
        for (let i = 0; i < n; i++) this.#call(0, table, i, par);
        return arrays.slice(cs).map((p) => f64.slice(p / 8, p / 8 + n));
    }

    /** Composer/Symbolica models: inputs (the parameters) -> outputs. */
    evaluateInputs(inputs) {
        const { count_obs: co } = this.layout;
        const [outs, par] = this.#alloc([co, inputs.length]);
        const f64 = new Float64Array(this.exports.memory.buffer);
        f64.set(inputs, par / 8);
        this.#call(outs, 0, 0, par);
        return Array.from(f64.subarray(outs / 8, outs / 8 + co));
    }
}
