//! Steps 10-12 (book 4.2 and 4.3) - layer normalization, GELU and the feed forward block.
//!
//! Layer normalization rescales each token's vector to mean 0 and variance 1,
//! which keeps activations in a range where training is stable:
//!
//! ```text
//! x [T, D]  ->  (x - mean) / sqrt(var + eps)  ->  scale * that + shift
//! ```
//!
//! The mean and variance are taken over the D numbers of ONE token (one row),
//! never across tokens. `scale` and `shift` are learned, so the model can undo
//! the normalization where that helps.
//!
//! GELU is a smooth cousin of ReLU, and the feed forward block expands each
//! token's vector 4x, applies GELU, and shrinks it back:
//!
//! ```text
//! [T, D] -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> [T, D]
//! ```

use crate::attention::Matrix;

pub struct LayerNorm {
    pub scale: Vec<f32>,
    pub shift: Vec<f32>,
}

const EPS: f32 = 1e-5;

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            scale: vec![1.0; dim],
            shift: vec![0.0; dim],
        }
    }

    /// Normalize every row of `x`.
    pub fn forward(&self, x: &Matrix) -> Matrix {
        let mut out = Matrix::new(x.rows, x.cols);
        let n = x.cols as f32;
        for i in 0..x.rows {
            let row = x.row(i);
            let mean = row.iter().sum::<f32>() / n;
            // divide by n (biased), like GPT-2
            let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv = (1.0 / ((variance + EPS) as f64).sqrt()) as f32;
            for (j, &v) in row.iter().enumerate() {
                out.data[i * x.cols + j] = self.scale[j] * ((v - mean) * inv) + self.shift[j];
            }
        }
        out
    }
}

/// The tanh approximation GPT-2 was trained with.
pub fn gelu(x: f32) -> f32 {
    let v = x as f64;
    (0.5 * v * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh()))
        as f32
}

/// `x @ W + b` with W stored [in, out] (PyTorch's `nn.Linear` stores it transposed).
pub struct Linear {
    pub w: Matrix,
    pub b: Vec<f32>,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            w: Matrix::new(input, output),
            b: vec![0.0; output],
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let mut out = x.matmul(&self.w);
        for i in 0..out.rows {
            for j in 0..out.cols {
                out.data[i * out.cols + j] += self.b[j];
            }
        }
        out
    }
}

pub struct FeedForward {
    pub expand: Linear,
    pub shrink: Linear,
}

impl FeedForward {
    pub fn new(dim: usize) -> Self {
        Self {
            expand: Linear::new(dim, 4 * dim),
            shrink: Linear::new(4 * dim, dim),
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        let mut h = self.expand.forward(x);
        for v in h.data.iter_mut() {
            *v = gelu(*v);
        }
        self.shrink.forward(&h)
    }
}
