//! Step 9 (book 3.6) - multi-head attention.
//!
//! One attention head learns one way of relating tokens. Several heads, each
//! with its own smaller query/key/value, can look for different patterns in
//! parallel (one for grammar, one for topic, ...). Their outputs are joined
//! and mixed by a final linear layer.
//!
//! ```text
//! d_out = 4, num_heads = 2  ->  each head works on head_dim = 2 columns
//!
//! W_query columns: [ head 0 | head 1 ]     (same for W_key and W_value)
//! per head: causal attention on its own slice -> [T, head_dim]
//! concatenate heads -> [T, d_out] -> out_proj
//! ```
//!
//! Tip: one big weight matrix cut into column slices is the same as separate
//! small matrices per head, just faster (one matmul instead of many).

use super::Matrix;
use super::self_attention::scale_by_sqrt;
use crate::tokenization::rng::Rng;

pub struct MultiHeadAttention {
    pub w_query: Matrix,
    pub w_key: Matrix,
    pub w_value: Matrix,
    pub out_weight: Matrix,
    pub out_bias: Vec<f32>,
    /// Biases of the query, key and value projections: `None` in chapters 3 and 4,
    /// `Some` for OpenAI's GPT-2 weights (5.5).
    pub qkv_bias: Option<[Vec<f32>; 3]>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    /// Uniform [0, 1) weights (query, key, value, out_proj) and a zero bias.
    /// `d_out` must divide evenly by `num_heads`.
    pub fn random(d_in: usize, d_out: usize, num_heads: usize, rng: &mut Rng) -> Self {
        let w_query = Matrix::rand(d_in, d_out, rng);
        let w_key = Matrix::rand(d_in, d_out, rng);
        let w_value = Matrix::rand(d_in, d_out, rng);
        let out_weight = Matrix::rand(d_out, d_out, rng);
        Self {
            w_query,
            w_key,
            w_value,
            out_weight,
            out_bias: vec![0.0; d_out],
            qkv_bias: None,
            num_heads,
            head_dim: d_out / num_heads,
        }
    }

    /// `x` is [T, d_in]; returns [T, d_out].
    pub fn forward(&self, x: &Matrix) -> Matrix {
        let (mut q, mut k, mut v) = (
            x.matmul(&self.w_query),
            x.matmul(&self.w_key),
            x.matmul(&self.w_value),
        );
        if let Some([bq, bk, bv]) = &self.qkv_bias {
            for (m, b) in [(&mut q, bq), (&mut k, bk), (&mut v, bv)] {
                for row in m.data.chunks_mut(b.len()) {
                    for (v, bias) in row.iter_mut().zip(b) {
                        *v += bias;
                    }
                }
            }
        }
        let mut context = Matrix::new(x.rows, self.num_heads * self.head_dim);
        for h in 0..self.num_heads {
            let (lo, hi) = (h * self.head_dim, (h + 1) * self.head_dim); // this head's column slice
            let (qh, kh, vh) = (
                q.slice_cols(lo, hi),
                k.slice_cols(lo, hi),
                v.slice_cols(lo, hi),
            );
            let scores = qh.matmul(&kh.transpose()).mask_future();
            let head = scale_by_sqrt(&scores, self.head_dim)
                .softmax_rows()
                .matmul(&vh);
            for i in 0..head.rows {
                // glue the heads back together
                context.data[i * context.cols + lo..i * context.cols + hi]
                    .copy_from_slice(head.row(i));
            }
        }
        let mut out = context.matmul(&self.out_weight);
        for i in 0..out.rows {
            for j in 0..out.cols {
                out.data[i * out.cols + j] += self.out_bias[j];
            }
        }
        out
    }
}
