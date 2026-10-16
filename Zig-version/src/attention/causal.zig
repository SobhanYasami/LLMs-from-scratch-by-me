//! Step 8 (book 3.5) - causal attention: hide the future.
//!
//! A GPT writes text left to right, so token i may only look at tokens 0..i.
//! We hide the later tokens with a mask. Set their scores to -inf *before*
//! softmax: exp(-inf) = 0, and the remaining weights in each row still sum to 1.
//!
//! ```
//! scores (T x T)                       after the mask
//! [ s00 s01 s02 ]                      [ s00 -inf -inf ]
//! [ s10 s11 s12 ]      -------->       [ s10  s11 -inf ]
//! [ s20 s21 s22 ]                      [ s20  s21  s22 ]
//! ```
//!
//! Dropout (training only) randomly zeroes some attention weights so the model
//! does not lean on any single connection.

const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const sa = @import("self_attention.zig");
const Rng = @import("../tokenization/rng.zig").Rng;

pub const CausalAttention = struct {
    base: sa.SelfAttention,
    dropout_p: f32 = 0,

    /// `x` is [T, d_in]. Pass an `rng` to apply dropout (training), or null (inference).
    pub fn attentionWeights(self: CausalAttention, gpa: std.mem.Allocator, x: Matrix, rng: ?*Rng) !Matrix {
        const queries = try x.matmul(gpa, self.base.w_query);
        const keys = try x.matmul(gpa, self.base.w_key);
        const scores = try (try queries.matmul(gpa, try keys.transpose(gpa))).maskFuture(gpa);
        const weights = try (try sa.scaleBySqrt(gpa, scores, keys.cols)).softmaxRows(gpa);
        if (rng) |r| {
            if (self.dropout_p > 0) return weights.dropout(gpa, self.dropout_p, r);
        }
        return weights;
    }

    pub fn forward(self: CausalAttention, gpa: std.mem.Allocator, x: Matrix, rng: ?*Rng) !Matrix {
        const weights = try self.attentionWeights(gpa, x, rng);
        return weights.matmul(gpa, try x.matmul(gpa, self.base.w_value));
    }
};
