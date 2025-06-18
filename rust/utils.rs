pub trait Compiled<T> {
    fn exec(&mut self);
    fn mem(&self) -> &[T];
    fn mem_mut(&mut self) -> &mut [T];
    fn dump(&self, name: &str);
    fn func(&self) -> fn(&[T]);
}

pub trait Eval {
    fn eval(&self, mem: &mut [f64], stack: &mut [f64]) -> f64;
}

pub fn bool_to_f64(b: bool) -> f64 {
    const T: f64 = f64::from_bits(!0);
    const F: f64 = f64::from_bits(0);
    if b {
        T
    } else {
        F
    }
}

/// aligns at a multiple of 32 (to cover different ABIs)
pub fn align_stack(n: u32) -> u32 {
    n + 16 - (n & 15)
}

/*****************************************/

#[cfg(target_arch = "x86_64")]
mod simd {
    use std::ops::{Add, Div, Mul, Sub};

    use std::arch::x86_64::__m256d;
    use std::arch::x86_64::{_mm256_add_pd, _mm256_div_pd, _mm256_mul_pd, _mm256_sub_pd};
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, Debug)]
    pub struct f64x4(__m256d);

    impl f64x4 {
        pub fn splat(x: f64) -> f64x4 {
            unsafe { f64x4(_mm256_set1_pd(x)) }
        }

        pub fn from_slice(slice: &[f64]) -> f64x4 {
            let x = unsafe { _mm256_loadu_pd(slice.as_ptr()) };
            f64x4(x)
        }

        pub fn copy_to_slice(self, slice: &mut [f64]) {
            unsafe {
                _mm256_storeu_pd(slice.as_mut_ptr(), self.0);
            }
        }
    }

    impl Add for f64x4 {
        type Output = f64x4;

        fn add(self, rhs: f64x4) -> f64x4 {
            let x = unsafe { _mm256_add_pd(self.0, rhs.0) };
            f64x4(x)
        }
    }

    impl Sub for f64x4 {
        type Output = f64x4;

        fn sub(self, rhs: f64x4) -> f64x4 {
            let x = unsafe { _mm256_sub_pd(self.0, rhs.0) };
            f64x4(x)
        }
    }

    impl Mul for f64x4 {
        type Output = f64x4;

        fn mul(self, rhs: f64x4) -> f64x4 {
            let x = unsafe { _mm256_mul_pd(self.0, rhs.0) };
            f64x4(x)
        }
    }

    impl Div for f64x4 {
        type Output = f64x4;

        fn div(self, rhs: f64x4) -> f64x4 {
            let x = unsafe { _mm256_div_pd(self.0, rhs.0) };
            f64x4(x)
        }
    }

    #[test]
    fn test_simd() {
        let x = f64x4::splat(2.0);
        let y = f64x4::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let z = f64x4::from_slice(&[5.0, 6.0, 7.0, 8.0]);
        let u = (x - y) * z;
        let mut v: Vec<f64> = vec![0.0; 4];
        u.copy_to_slice(&mut v);
        let s: f64 = v.iter().sum();
        assert_eq!(s, -18.0);
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod simd {
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, Debug)]
    pub struct f64x4([f64; 4]);

    impl f64x4 {
        pub fn splat(x: f64) -> f64x4 {
            f64x4([x; 4])
        }

        pub fn from_slice(slice: &[f64]) -> f64x4 {
            let mut x = [0.0; 4];
            for i in 0..4 {
                x[i] = slice[i];
            }
            f64x4(x)
        }

        pub fn copy_to_slice(self, slice: &mut [f64]) {
            for i in 0..4 {
                slice[i] = self.0[i];
            }
        }
    }
}

#[allow(non_camel_case_types)]
pub type f64x4 = simd::f64x4;
