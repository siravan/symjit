pub struct Matrix<'a> {
    pub p: Vec<&'a mut [f64]>,
    pub ncols: usize,
}

impl<'a> Matrix<'a> {
    pub fn new() -> Matrix<'a> {
        Matrix {
            p: Vec::new(),
            ncols: 0,
        }
    }

    pub fn from_buf(buf: &'a mut [f64], nrows: usize, ncols: usize) -> Matrix<'a> {
        assert!(buf.len() >= nrows * ncols);
        let mut p: Vec<&mut [f64]> = Vec::with_capacity(nrows);
        for row in buf.chunks_mut(ncols) {
            p.push(row);
        }

        Matrix { p, ncols }
    }

    pub fn add_row(&mut self, v: *mut f64, n: usize) {
        self.ncols = if self.p.is_empty() {
            n
        } else {
            self.ncols.min(n)
        };
        let q = unsafe { std::slice::from_raw_parts_mut(v, self.ncols) };
        self.p.push(q);
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.p[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.p[row][col] = val;
    }
}

pub fn combine_matrixes<'a>(a: &'a mut Matrix, b: &'a mut Matrix) -> Matrix<'a> {
    assert!(a.ncols == b.ncols);
    let mut c = Matrix::new();
    c.p.extend(std::mem::take(&mut a.p));
    c.p.extend(std::mem::take(&mut b.p));
    c
}

impl<'a> Default for Matrix<'a> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<'a> Sync for Matrix<'a> {}
