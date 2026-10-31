//! Common-subexpression elimination with a conditional. `e = (p1 p2 + p3) p1` is computed in
//! the taken-if-nonzero branch of an `IfElse` on `p0` and twice after its `Join`. The first pass
//! of `Block::elimination_pass` finds `e` shared by the two statements after the join (it only
//! looks at depth 0); the second pass rewrote every occurrence, the one inside the branch too,
//! and assigned the shared temporary at its first occurrence, inside the branch. At `p0 = 0` the
//! branch does not run, and the two later statements read an unset temporary (it reads `0` here,
//! at every opt level, so `out1 = p2` and `out2 = p3`).
mod common;
use common::base_config;

use anyhow::Result;
use num_complex::Complex;
use symjit::{Applet, Application, Composer, Config, Slot, Translator};

/// out0 = if p0 != 0 { e } else { p2 }, out1 = e + p2, out2 = e + p3, with e = (p1 p2 + p3) p1.
fn compile(config: Config) -> Result<Application> {
    let mut t = Translator::new(config);
    t.set_num_params(4);
    let (p0, p1, p2, p3) = (
        Slot::Param(0),
        Slot::Param(1),
        Slot::Param(2),
        Slot::Param(3),
    );
    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(0), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(1), &[Slot::Temp(0), p3], 0)?;
    t.append_mul(&Slot::Temp(2), &[Slot::Temp(1), p1], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_assign(&Slot::Temp(3), &p2)?;
    t.append_label(1)?;
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(2), &Slot::Temp(3))?;
    t.append_mul(&Slot::Temp(4), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(5), &[Slot::Temp(4), p3], 0)?;
    t.append_mul(&Slot::Temp(6), &[Slot::Temp(5), p1], 0)?;
    t.append_add(&Slot::Out(1), &[Slot::Temp(6), p2], 0)?;
    t.append_mul(&Slot::Temp(7), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(8), &[Slot::Temp(7), p3], 0)?;
    t.append_mul(&Slot::Temp(9), &[Slot::Temp(8), p1], 0)?;
    t.append_add(&Slot::Out(2), &[Slot::Temp(9), p3], 0)?;
    let mut app = t.compile()?;
    app.prepare_simd();
    Ok(app)
}

#[test]
fn shared_subexpression_after_an_untaken_branch() -> Result<()> {
    for opt_level in 0..=3 {
        for complex in [false, true] {
            let mut config = base_config();
            config.set_opt_level(opt_level);
            config.set_complex(complex);
            let app = compile(config)?.seal()?;
            // p0 = 0 first (nothing computed e yet), then p0 = 1, then p0 = 0 with other values
            // (a value of e left from the p0 = 1 row is wrong there)
            for [p0, a, b, d] in [
                [0.0, 1.5, -2.0, 0.25],
                [1.0, 2.0, 3.0, 5.0],
                [0.0, -1.0, 4.0, 2.0],
            ] {
                let e = (a * b + d) * a;
                let want = [if p0 != 0.0 { e } else { b }, e + b, e + d];
                let what =
                    format!("opt level {opt_level}, complex = {complex}, p = {p0} {a} {b} {d}");
                let got: Vec<f64> = if complex {
                    let x = [p0, a, b, d].map(|v| Complex::new(v, 0.0));
                    let mut outs = [Complex::new(f64::NAN, f64::NAN); 3];
                    app.evaluate_matrix(&x, &mut outs, 1);
                    for z in outs {
                        assert_eq!(z.im, 0.0, "{what}: {outs:?}");
                    }
                    outs.iter().map(|z| z.re).collect()
                } else {
                    let mut outs = [f64::NAN; 3];
                    app.evaluate_matrix(&[p0, a, b, d], &mut outs, 1);
                    outs.to_vec()
                };
                assert_eq!(got, want, "{what}");
            }
        }
    }
    Ok(())
}

/// out0 = if p0 != 0 { p1 p2 } else { p3 p2 },
/// out1 = if p0 != 0 { e } else { if p1 != 0 { e } else { p3 } }, with e as above.
///
/// With `simd_branch` the jump over an else part is not emitted (`translate_goto`), so an
/// if/else is a conditional jump and two labels. Counting labels against jumps then left the
/// statements after the first if/else at depth 1, the then part of the second at depth 0, and
/// the then part of the if/else in its else part at depth 0 too: `e` was shared between the two
/// exclusive branches and assigned in the first, which does not run when `p0 = 0`.
fn compile_two_ifs(config: Config) -> Result<Application> {
    let mut t = Translator::new(config);
    t.set_num_params(4);
    let (p0, p1, p2, p3) = (
        Slot::Param(0),
        Slot::Param(1),
        Slot::Param(2),
        Slot::Param(3),
    );
    t.append_if_else(&p0, 0)?;
    t.append_mul(&Slot::Temp(0), &[p1, p2], 0)?;
    t.append_goto(1)?;
    t.append_label(0)?;
    t.append_mul(&Slot::Temp(1), &[p3, p2], 0)?;
    t.append_label(1)?;
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(0), &Slot::Temp(1))?;
    t.append_if_else(&p0, 2)?;
    t.append_mul(&Slot::Temp(2), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(3), &[Slot::Temp(2), p3], 0)?;
    t.append_mul(&Slot::Temp(4), &[Slot::Temp(3), p1], 0)?;
    t.append_goto(3)?;
    t.append_label(2)?;
    t.append_if_else(&p1, 4)?;
    t.append_mul(&Slot::Temp(5), &[p1, p2], 0)?;
    t.append_add(&Slot::Temp(6), &[Slot::Temp(5), p3], 0)?;
    t.append_mul(&Slot::Temp(7), &[Slot::Temp(6), p1], 0)?;
    t.append_goto(5)?;
    t.append_label(4)?;
    t.append_assign(&Slot::Temp(8), &p3)?;
    t.append_label(5)?;
    t.append_join(&Slot::Temp(9), &p1, &Slot::Temp(7), &Slot::Temp(8))?;
    t.append_label(3)?;
    t.append_join(&Slot::Out(1), &p0, &Slot::Temp(4), &Slot::Temp(9))?;
    let mut app = t.compile()?;
    app.prepare_simd();
    Ok(app)
}

#[test]
fn shared_subexpression_in_exclusive_branches() -> Result<()> {
    for opt_level in 0..=3 {
        for simd_branch in [false, true] {
            for complex in [false, true] {
                let mut config = base_config();
                config.set_opt_level(opt_level);
                config.set_simd_branch(simd_branch);
                config.set_complex(complex);
                let app = compile_two_ifs(config)?.seal()?;
                for [p0, a, b, d] in [
                    [0.0, 1.5, -2.0, 0.25],
                    [1.0, 2.0, 3.0, 5.0],
                    [0.0, -1.0, 4.0, 2.0],
                ] {
                    let e = (a * b + d) * a;
                    let want = [
                        if p0 != 0.0 { a * b } else { d * b },
                        e, // p1 != 0 in every row
                    ];
                    let what = format!(
                        "opt level {opt_level}, simd_branch = {simd_branch}, complex = {complex}, p = {p0} {a} {b} {d}"
                    );
                    assert_eq!(
                        evaluate(&app, complex, &[p0, a, b, d], 1, 2, &what),
                        want,
                        "{what}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Evaluates `rows` rows of four parameters (`x`, row-major) to `outs` outputs per row; a
/// complex application gets real inputs and must return real outputs.
fn evaluate(
    app: &Applet,
    complex: bool,
    x: &[f64],
    rows: usize,
    outs: usize,
    what: &str,
) -> Vec<f64> {
    if complex {
        let xc: Vec<Complex<f64>> = x.iter().map(|v| Complex::new(*v, 0.0)).collect();
        let mut o = vec![Complex::new(f64::NAN, f64::NAN); rows * outs];
        app.evaluate_matrix(&xc, &mut o, rows);
        for z in &o {
            assert_eq!(z.im, 0.0, "{what}: {o:?}");
        }
        o.iter().map(|z| z.re).collect()
    } else {
        let mut o = vec![f64::NAN; rows * outs];
        app.evaluate_matrix(x, &mut o, rows);
        o
    }
}

// The random programs of `random_nested_branches`.
#[derive(Debug)]
enum Item {
    Expr(usize),
    If(usize, Vec<Item>, Vec<Item>),
}

// expression k is (p_a p_b + p_c) p_d with [a, b, c, d] = EXPRS[k]
const EXPRS: [[usize; 4]; 4] = [[1, 2, 3, 1], [2, 3, 0, 1], [3, 1, 2, 0], [0, 2, 1, 3]];

struct Rng(u64);

impl Rng {
    // xorshift64
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }

    fn items(&mut self, depth: usize) -> Vec<Item> {
        (0..1 + self.below(3))
            .map(|_| {
                if depth < 3 && self.below(3) == 0 {
                    Item::If(
                        self.below(4) as usize,
                        self.items(depth + 1),
                        self.items(depth + 1),
                    )
                } else {
                    Item::Expr(self.below(EXPRS.len() as u64) as usize)
                }
            })
            .collect()
    }
}

fn sum(items: &[Item], p: &[f64]) -> f64 {
    let value = |item: &Item| match item {
        Item::Expr(k) => {
            let [a, b, c, d] = EXPRS[*k];
            (p[a] * p[b] + p[c]) * p[d]
        }
        Item::If(c, then, other) => sum(if p[*c] != 0.0 { then } else { other }, p),
    };
    items[1..]
        .iter()
        .fold(value(&items[0]), |acc, item| acc + value(item))
}

struct Emitter {
    t: Translator,
    temps: usize,
    labels: usize,
}

impl Emitter {
    fn temp(&mut self) -> Slot {
        self.temps += 1;
        Slot::Temp(self.temps - 1)
    }

    fn sum(&mut self, items: &[Item]) -> Result<Slot> {
        let mut acc: Option<Slot> = None;
        for item in items {
            let v = match item {
                Item::Expr(k) => {
                    let [a, b, c, d] = EXPRS[*k].map(Slot::Param);
                    let (x, y, z) = (self.temp(), self.temp(), self.temp());
                    self.t.append_mul(&x, &[a, b], 0)?;
                    self.t.append_add(&y, &[x, c], 0)?;
                    self.t.append_mul(&z, &[y, d], 0)?;
                    z
                }
                Item::If(c, then, other) => {
                    let (other_label, done) = (self.labels, self.labels + 1);
                    self.labels += 2;
                    let cond = Slot::Param(*c);
                    self.t.append_if_else(&cond, other_label)?;
                    let then = self.sum(then)?;
                    self.t.append_goto(done)?;
                    self.t.append_label(other_label)?;
                    let other = self.sum(other)?;
                    self.t.append_label(done)?;
                    let v = self.temp();
                    self.t.append_join(&v, &cond, &then, &other)?;
                    v
                }
            };
            acc = Some(match acc {
                None => v,
                Some(a) => {
                    let s = self.temp();
                    self.t.append_add(&s, &[a, v], 0)?;
                    s
                }
            });
        }
        Ok(acc.unwrap())
    }
}

/// Random programs: each output is a sum of items, an item is one of four expressions of the
/// shape of `e` or an if/else on a parameter whose two parts are such sums, nested up to three
/// deep. The expressions repeat, so CSE finds shared subexpressions in and around branches.
/// Compiled with CSE at every opt level, with and without `simd_branch`, real and complex,
/// against the same sums in Rust, over rows whose conditions differ.
#[test]
fn random_nested_branches() -> Result<()> {
    const ROWS: usize = 19; // SIMD blocks and a remainder
    const OUTS: usize = 3;
    for seed in 1..=300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let program: Vec<Vec<Item>> = (0..OUTS).map(|_| rng.items(0)).collect();
        // a third of the parameters are 0, so the conditions differ between rows
        let x: Vec<f64> = (0..ROWS * 4)
            .map(|_| match rng.below(3) {
                0 => 0.0,
                _ => (rng.below(4001) as f64 - 2000.0) / 1000.0 + 0.0005,
            })
            .collect();
        let want: Vec<f64> = (0..ROWS)
            .flat_map(|i| program.iter().map(move |items| (items, i)))
            .map(|(items, i)| sum(items, &x[4 * i..4 * i + 4]))
            .collect();
        for opt_level in 0..=3 {
            for simd_branch in [false, true] {
                for complex in [false, true] {
                    let mut config = base_config();
                    config.set_opt_level(opt_level);
                    config.set_simd_branch(simd_branch);
                    config.set_complex(complex);
                    let mut e = Emitter {
                        t: Translator::new(config),
                        temps: 0,
                        labels: 0,
                    };
                    e.t.set_num_params(4);
                    for (k, items) in program.iter().enumerate() {
                        let v = e.sum(items)?;
                        e.t.append_assign(&Slot::Out(k), &v)?;
                    }
                    let mut app = e.t.compile()?;
                    app.prepare_simd();
                    let app = app.seal()?;
                    let what = format!(
                        "seed {seed}, opt level {opt_level}, simd_branch = {simd_branch}, complex = {complex}: {program:?}"
                    );
                    let got = evaluate(&app, complex, &x, ROWS, OUTS, &what);
                    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
                            "{what}: value {k}: {g} for {w}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
