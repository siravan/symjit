pub struct Matrix<'a> {
    pub p: Vec<&'a mut [f64]>,
    pub vecsize: usize,
}

impl<'a> Matrix<'a> {
    pub fn new() -> Matrix<'a> {
        Matrix {
            p: Vec::new(),
            vecsize: 0,
        }
    }

    pub fn from_buf(buf: &'a mut [f64], nvecs: usize, vecsize: usize) -> Matrix<'a> {
        assert!(buf.len() >= nvecs * vecsize);
        let mut p: Vec<&mut [f64]> = Vec::with_capacity(nvecs);
        for row in buf.chunks_mut(vecsize) {
            p.push(row);
        }

        Matrix { p, vecsize }
    }

    /// # Safety
    /// v should point to a valid memory area of at least size n, which should
    /// stay alive for the duration of the Matrix life
    pub unsafe fn add_row(&mut self, v: *mut f64, n: usize) {
        self.vecsize = if self.p.is_empty() {
            n
        } else {
            self.vecsize.min(n)
        };
        let q = unsafe { std::slice::from_raw_parts_mut(v, self.vecsize) };
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
    assert!(a.vecsize == b.vecsize);
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
