//! Step 7 (book 3.4) - self-attention with trainable weights (scaled dot-product).
//!
//! Three weight matrices turn each input vector into three roles:
//!
//! ```text
//! query = x x W_query    "what am I looking for?"
//! key   = x x W_key      "what do I offer to a search?"
//! value = x x W_value    "what do I hand over if I am chosen?"
//!
//! scores  = query x key^T                 how well does each query match each key?
//! weights = softmax(scores / sqrt(d_k))   scale, then normalise each row
//! context = weights x value               mix the values
//! ```
//!
//! Why divide by sqrt(d_k)? Big dot products push softmax towards a hard 0/1
//! choice, which makes gradients tiny. Scaling keeps training healthy.
//!
//! Tip: do not confuse *weight parameters* (W_*, learned) with *attention weights*
//! (softmax output, computed fresh for every input).

use super::Matrix;
use crate::tokenization::rng::Rng;

pub struct SelfAttention {
    pub w_query: Matrix,
    pub w_key: Matrix,
    pub w_value: Matrix,
}

impl SelfAttention {
    /// Weights drawn uniform in [0, 1), in query, key, value order.
    pub fn random(d_in: usize, d_out: usize, rng: &mut Rng) -> Self {
        let w_query = Matrix::rand(d_in, d_out, rng);
        let w_key = Matrix::rand(d_in, d_out, rng);
        let w_value = Matrix::rand(d_in, d_out, rng);
        Self {
            w_query,
            w_key,
            w_value,
        }
    }

    /// `x` is [T, d_in]; returns [T, T] weights.
    pub fn attention_weights(&self, x: &Matrix) -> Matrix {
        let queries = x.matmul(&self.w_query);
        let keys = x.matmul(&self.w_key);
        let scores = queries.matmul(&keys.transpose());
        scale_by_sqrt(&scores, keys.cols).softmax_rows()
    }

    /// Context vectors [T, d_out].
    pub fn forward(&self, x: &Matrix) -> Matrix {
        self.attention_weights(x).matmul(&x.matmul(&self.w_value))
    }
}

/// Divide scores by sqrt(d_k).
pub(super) fn scale_by_sqrt(scores: &Matrix, d_k: usize) -> Matrix {
    scores.scale(1.0 / (d_k as f64).sqrt() as f32)
}
