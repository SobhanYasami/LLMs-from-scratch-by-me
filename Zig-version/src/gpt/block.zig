//! Step 14 (book 4.5) - the transformer block.
//!
//! One block is two sub-layers, each wrapped in the same recipe
//! (normalize first, run the layer, add the input back):
//!
//! ```
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

const std = @import("std");
const att = @import("../attention/root.zig");
const layers = @import("layers.zig");

const Matrix = att.Matrix;

pub const TransformerBlock = struct {
    att: att.MultiHeadAttention,
    ff: layers.FeedForward,
    norm1: layers.LayerNorm,
    norm2: layers.LayerNorm,

    pub fn init(gpa: std.mem.Allocator, emb_dim: usize, n_heads: usize, qkv_bias: bool) !TransformerBlock {
        return .{
            .att = .{
                .w_query = try Matrix.init(gpa, emb_dim, emb_dim),
                .w_key = try Matrix.init(gpa, emb_dim, emb_dim),
                .w_value = try Matrix.init(gpa, emb_dim, emb_dim),
                .out_weight = try Matrix.init(gpa, emb_dim, emb_dim),
                .out_bias = try Matrix.init(gpa, 1, emb_dim),
                .num_heads = n_heads,
                .head_dim = emb_dim / n_heads,
                .qkv_bias = if (qkv_bias) .{ try Matrix.init(gpa, 1, emb_dim), try Matrix.init(gpa, 1, emb_dim), try Matrix.init(gpa, 1, emb_dim) } else null,
            },
            .ff = try layers.FeedForward.init(gpa, emb_dim),
            .norm1 = try layers.LayerNorm.init(gpa, emb_dim),
            .norm2 = try layers.LayerNorm.init(gpa, emb_dim),
        };
    }

    pub fn forward(self: TransformerBlock, gpa: std.mem.Allocator, x: Matrix) !Matrix {
        // add the original input back after each sub-layer
        const a = add(try self.att.forward(gpa, try self.norm1.forward(gpa, x)), x);
        return add(try self.ff.forward(gpa, try self.norm2.forward(gpa, a)), a);
    }
};

fn add(a: Matrix, b: Matrix) Matrix {
    for (a.data, b.data) |*x, y| x.* += y;
    return a;
}
