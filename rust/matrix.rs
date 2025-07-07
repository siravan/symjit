use crate::utils::f64x4;

pub struct Matrix {
    p: Vec<*mut f64>,
    ncols: usize,
}

impl Matrix {
    pub fn from_buf(buf: &mut [f64], nrows: usize, ncols: usize) -> Matrix {
        assert!(buf.len() >= nrows * ncols);
        let mut p: Vec<*mut f64> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let q = &mut buf[row * ncols] as *mut f64;
            p.push(q);
        }

        Matrix { p, ncols }
    }

    pub fn get(&self, row: usize, idx: usize) -> f64 {
        let u: &[f64] = unsafe { std::slice::from_raw_parts(self.p[row], self.ncols) };
        u[idx]
    }

    pub fn get_simd(&self, row: usize, idx: usize) -> f64x4 {
        let u: &[f64] = unsafe { std::slice::from_raw_parts(self.p[row], self.ncols) };
        f64x4::from_slice(&u[idx..idx + 4])
    }

    pub fn set(&mut self, row: usize, idx: usize, val: f64) {
        let u: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(self.p[row], self.ncols) };
        u[idx] = val;
    }

    pub fn set_simd(&mut self, row: usize, idx: usize, val: f64x4) {
        let u: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(self.p[row], self.ncols) };
        val.copy_to_slice(&mut u[idx..idx + 4]);
    }
}
