//! Step 8 (book 3.5) - causal attention: hide the future.
//!
//! A GPT writes text left to right, so token i may only look at tokens 0..i.
//! We hide the later tokens with a mask. Set their scores to -inf *before*
//! softmax: exp(-inf) = 0, and the remaining weights in each row still sum to 1.
//!
//! ```text
//! scores (T x T)                       after the mask
//! [ s00 s01 s02 ]                      [ s00 -inf -inf ]
//! [ s10 s11 s12 ]      -------->       [ s10  s11 -inf ]
//! [ s20 s21 s22 ]                      [ s20  s21  s22 ]
//! ```
//!
//! Dropout (training only) randomly zeroes some attention weights so the model
//! does not lean on any single connection.

use super::self_attention::scale_by_sqrt;
use super::{Matrix, SelfAttention};
use crate::tokenization::rng::Rng;

pub struct CausalAttention {
    pub base: SelfAttention,
    pub dropout_p: f32,
}

impl CausalAttention {
    pub fn new(w_query: Matrix, w_key: Matrix, w_value: Matrix, dropout_p: f32) -> Self {
        Self {
            base: SelfAttention {
                w_query,
                w_key,
                w_value,
            },
            dropout_p,
        }
    }

    /// `x` is [T, d_in]. Pass an `rng` and `train = true` to apply dropout.
    pub fn attention_weights(&self, x: &Matrix, rng: Option<&mut Rng>) -> Matrix {
        let queries = x.matmul(&self.base.w_query);
        let keys = x.matmul(&self.base.w_key);
        let scores = queries.matmul(&keys.transpose()).mask_future();
        let weights = scale_by_sqrt(&scores, keys.cols).softmax_rows();
        match rng {
            Some(rng) if self.dropout_p > 0.0 => weights.dropout(self.dropout_p, rng),
            _ => weights,
        }
    }

    pub fn forward(&self, x: &Matrix, rng: Option<&mut Rng>) -> Matrix {
        self.attention_weights(x, rng)
            .matmul(&x.matmul(&self.base.w_value))
    }
}
