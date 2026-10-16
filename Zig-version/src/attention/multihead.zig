//! Step 9 (book 3.6) - multi-head attention.
//!
//! One attention head learns one way of relating tokens. Several heads, each
//! with its own smaller query/key/value, can look for different patterns in
//! parallel (one for grammar, one for topic, ...). Their outputs are joined
//! and mixed by a final linear layer.
//!
//! ```
//! d_out = 4, num_heads = 2  ->  each head works on head_dim = 2 columns
//!
//! W_query columns: [ head 0 | head 1 ]     (same for W_key and W_value)
//! per head: causal attention on its own slice -> [T, head_dim]
//! concatenate heads -> [T, d_out] -> out_proj
//! ```
//!
//! Tip: one big weight matrix cut into column slices is the same as separate
//! small matrices per head, just faster (one matmul instead of many).

const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const sa = @import("self_attention.zig");
const Rng = @import("../tokenization/rng.zig").Rng;

pub const MultiHeadAttention = struct {
    w_query: Matrix,
    w_key: Matrix,
    w_value: Matrix,
    out_weight: Matrix,
    out_bias: Matrix, // 1 x d_out
    num_heads: usize,
    head_dim: usize,

    /// Uniform [0, 1) weights (query, key, value, out_proj) and a zero bias.
    /// `d_out` must divide evenly by `num_heads`.
    pub fn random(gpa: std.mem.Allocator, d_in: usize, d_out: usize, num_heads: usize, rng: *Rng) !MultiHeadAttention {
        const w_query = try Matrix.rand(gpa, d_in, d_out, rng);
        const w_key = try Matrix.rand(gpa, d_in, d_out, rng);
        const w_value = try Matrix.rand(gpa, d_in, d_out, rng);
        const out_weight = try Matrix.rand(gpa, d_out, d_out, rng);
        return .{
            .w_query = w_query,
            .w_key = w_key,
            .w_value = w_value,
            .out_weight = out_weight,
            .out_bias = try Matrix.init(gpa, 1, d_out),
            .num_heads = num_heads,
            .head_dim = d_out / num_heads,
        };
    }

    /// `x` is [T, d_in]; returns [T, d_out].
    pub fn forward(self: MultiHeadAttention, gpa: std.mem.Allocator, x: Matrix) !Matrix {
        const q = try x.matmul(gpa, self.w_query);
        const k = try x.matmul(gpa, self.w_key);
        const v = try x.matmul(gpa, self.w_value);
        const context = try Matrix.init(gpa, x.rows, self.num_heads * self.head_dim);
        for (0..self.num_heads) |h| {
            const lo = h * self.head_dim; // this head's column slice
            const hi = lo + self.head_dim;
            const qh = try q.sliceCols(gpa, lo, hi);
            const kh = try k.sliceCols(gpa, lo, hi);
            const vh = try v.sliceCols(gpa, lo, hi);
            const scores = try (try qh.matmul(gpa, try kh.transpose(gpa))).maskFuture(gpa);
            const weights = try (try sa.scaleBySqrt(gpa, scores, self.head_dim)).softmaxRows(gpa);
            const head = try weights.matmul(gpa, vh);
            for (0..head.rows) |i| @memcpy(context.row(i)[lo..hi], head.row(i)); // glue the heads back together
        }
        const out = try context.matmul(gpa, self.out_weight);
        for (0..out.rows) |i| {
            for (0..out.cols) |j| out.data[i * out.cols + j] += self.out_bias.data[j];
        }
        return out;
    }
};
