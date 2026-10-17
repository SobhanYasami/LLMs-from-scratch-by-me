//! Step 14 (book 4.5) - the transformer block.
//!
//! One block is two sub-layers, each wrapped in the same recipe
//! (normalize first, run the layer, add the input back):
//!
//! ```text
//! x = x + attention(norm1(x))      <- tokens look at each other
//! x = x + feed_forward(norm2(x))   <- each token is processed alone
//! ```
//!
//! The output has the same shape as the input, so blocks can be stacked as many
//! times as you like. Normalizing BEFORE each layer is called pre-LayerNorm; it
//! trains better than the original transformer's normalize-after. Dropout, which
//! the book puts after each sub-layer, only runs in training, so it is left out.
//!
//! Tip: the attention here is the chapter 3 `MultiHeadAttention` unchanged.

use super::layers::{FeedForward, LayerNorm};
use super::model::Config;
use crate::attention::{Matrix, MultiHeadAttention};

pub struct TransformerBlock {
    pub att: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(cfg: &Config) -> Self {
        let d = cfg.emb_dim;
        let att = MultiHeadAttention {
            w_query: Matrix::new(d, d),
            w_key: Matrix::new(d, d),
            w_value: Matrix::new(d, d),
            out_weight: Matrix::new(d, d),
            out_bias: vec![0.0; d],
            num_heads: cfg.n_heads,
            head_dim: d / cfg.n_heads,
        };
        Self {
            att,
            ff: FeedForward::new(d),
            norm1: LayerNorm::new(d),
            norm2: LayerNorm::new(d),
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        // add the original input back after each sub-layer
        let x = add(self.att.forward(&self.norm1.forward(x)), x);
        add(self.ff.forward(&self.norm2.forward(&x)), &x)
    }
}

fn add(mut a: Matrix, b: &Matrix) -> Matrix {
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x += y;
    }
    a
}
