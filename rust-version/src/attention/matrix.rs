//! The few matrix operations attention needs.
//!
//! Tip: there is no tensor library here, so `Matrix` is a row-major `Vec<f32>`.
//! Everything attention does is these operations: matmul, transpose, softmax.

use crate::tokenization::rng::Rng;

/// A row-major rows x cols matrix of f32.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Build from rows of equal length.
    pub fn from_rows(rows: &[&[f32]]) -> Self {
        let mut m = Self::new(rows.len(), rows[0].len());
        for (i, r) in rows.iter().enumerate() {
            m.data[i * m.cols..(i + 1) * m.cols].copy_from_slice(r);
        }
        m
    }

    /// A rows x cols matrix, uniform in [0, 1), like `torch.rand`.
    pub fn rand(rows: usize, cols: usize, rng: &mut Rng) -> Self {
        let data = (0..rows * cols).map(|_| rng.next_f64() as f32).collect();
        Self { rows, cols, data }
    }

    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// a x b: [n,k] x [k,m] -> [n,m].
    pub fn matmul(&self, b: &Matrix) -> Matrix {
        let mut out = Matrix::new(self.rows, b.cols);
        for i in 0..self.rows {
            for j in 0..b.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.at(i, k) * b.at(k, j);
                }
                out.data[i * b.cols + j] = sum;
            }
        }
        out
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[j * self.rows + i] = self.at(i, j);
            }
        }
        out
    }

    /// Turn every row into weights that are positive and sum to 1.
    ///
    /// Tip: subtracting the row maximum first does not change the result but keeps
    /// `exp` from overflowing. A `-inf` entry becomes exactly 0.
    pub fn softmax_rows(&self) -> Matrix {
        let mut out = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            let row = self.row(i);
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for (j, &v) in row.iter().enumerate() {
                let e = ((v - max) as f64).exp() as f32;
                out.data[i * self.cols + j] = e;
                sum += e;
            }
            for j in 0..self.cols {
                out.data[i * self.cols + j] /= sum;
            }
        }
        out
    }

    /// Multiply every element by `s`.
    pub fn scale(&self, s: f32) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| v * s).collect(),
        }
    }

    /// Set every element above the diagonal to `-inf` (the "future" tokens).
    pub fn mask_future(&self) -> Matrix {
        let mut out = self.clone();
        for i in 0..self.rows {
            for j in i + 1..self.cols {
                out.data[i * self.cols + j] = f32::NEG_INFINITY;
            }
        }
        out
    }

    /// Columns `[start, end)` as a new matrix.
    pub fn slice_cols(&self, start: usize, end: usize) -> Matrix {
        let mut out = Matrix::new(self.rows, end - start);
        for i in 0..self.rows {
            out.data[i * out.cols..(i + 1) * out.cols].copy_from_slice(&self.row(i)[start..end]);
        }
        out
    }

    /// Zero each element with probability `p` and scale the rest by 1/(1-p).
    /// The scaling keeps the average unchanged, so nothing needs rescaling at
    /// inference time (when dropout is simply switched off).
    pub fn dropout(&self, p: f32, rng: &mut Rng) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|&v| {
                if rng.next_f64() >= p as f64 {
                    v / (1.0 - p)
                } else {
                    0.0
                }
            })
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
