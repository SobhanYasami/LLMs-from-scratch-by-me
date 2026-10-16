//! Step 7 (book 3.4) - self-attention with trainable weights (scaled dot-product).
//!
//! Three weight matrices turn each input vector into three roles:
//!
//! ```
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

const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const Rng = @import("../tokenization/rng.zig").Rng;

pub const SelfAttention = struct {
    w_query: Matrix,
    w_key: Matrix,
    w_value: Matrix,

    /// Weights drawn uniform in [0, 1), in query, key, value order.
    pub fn random(gpa: std.mem.Allocator, d_in: usize, d_out: usize, rng: *Rng) !SelfAttention {
        const w_query = try Matrix.rand(gpa, d_in, d_out, rng);
        const w_key = try Matrix.rand(gpa, d_in, d_out, rng);
        const w_value = try Matrix.rand(gpa, d_in, d_out, rng);
        return .{ .w_query = w_query, .w_key = w_key, .w_value = w_value };
    }

    /// `x` is [T, d_in]; returns [T, T] weights.
    pub fn attentionWeights(self: SelfAttention, gpa: std.mem.Allocator, x: Matrix) !Matrix {
        const queries = try x.matmul(gpa, self.w_query);
        const keys = try x.matmul(gpa, self.w_key);
        const scores = try queries.matmul(gpa, try keys.transpose(gpa));
        return (try scaleBySqrt(gpa, scores, keys.cols)).softmaxRows(gpa);
    }

    /// Context vectors [T, d_out].
    pub fn forward(self: SelfAttention, gpa: std.mem.Allocator, x: Matrix) !Matrix {
        const weights = try self.attentionWeights(gpa, x);
        return weights.matmul(gpa, try x.matmul(gpa, self.w_value));
    }
};

/// scores / sqrt(d_k), shared by the attention variants.
pub fn scaleBySqrt(gpa: std.mem.Allocator, scores: Matrix, d_k: usize) !Matrix {
    const root: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(d_k))));
    return scores.scale(gpa, 1.0 / root);
}
