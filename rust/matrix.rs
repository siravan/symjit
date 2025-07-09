use crate::utils::f64x4;

pub struct Matrix {
    p: Vec<*mut f64>,
    pub ncols: usize,
}

impl Matrix {
    pub fn new() -> Matrix {
        Matrix {
            p: Vec::new(),
            ncols: 0,
        }
    }

    pub fn from_buf(buf: &mut [f64], nrows: usize, ncols: usize) -> Matrix {
        assert!(buf.len() >= nrows * ncols);
        let mut p: Vec<*mut f64> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let q = &mut buf[row * ncols] as *mut f64;
            p.push(q);
        }

        Matrix { p, ncols }
    }

    pub fn add_row(&mut self, v: *mut f64, n: usize) {
        self.ncols = if self.p.is_empty() {
            n
        } else {
            self.ncols.min(n)
        };
        self.p.push(v);
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        let u: &[f64] = unsafe { std::slice::from_raw_parts(self.p[row], self.ncols) };
        u[col]
    }

    pub fn get_simd(&self, row: usize, col: usize) -> f64x4 {
        let u: &[f64] = unsafe { std::slice::from_raw_parts(self.p[row], self.ncols) };
        f64x4::from_slice(&u[col..col + 4])
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        let u: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(self.p[row], self.ncols) };
        u[col] = val;
    }

    pub fn set_simd(&mut self, row: usize, col: usize, val: f64x4) {
        let u: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(self.p[row], self.ncols) };
        val.copy_to_slice(&mut u[col..col + 4]);
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::new()
    }
}
