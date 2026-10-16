//! Attention building blocks (book chapter 3).
//!
//! Files, in the order you learn them:
//!
//! - `matrix`: the few matrix operations attention needs.
//! - `simple`: simplified self-attention, no trainable weights (3.3).
//! - `self_attention`: scaled dot-product attention with W_query/key/value (3.4).
//! - `causal`: hide future tokens with a mask, optional dropout (3.5).
//! - `multihead`: several heads in parallel plus an output projection (3.6).
//!
//! The seeded random generator is shared with chapter 2: `tokenization/rng.zig`.

pub const matrix = @import("matrix.zig");
pub const simple = @import("simple.zig");
pub const self_attention = @import("self_attention.zig");
pub const causal = @import("causal.zig");
pub const multihead = @import("multihead.zig");

pub const Matrix = matrix.Matrix;
pub const SelfAttention = self_attention.SelfAttention;
pub const CausalAttention = causal.CausalAttention;
pub const MultiHeadAttention = multihead.MultiHeadAttention;
pub const simpleAttention = simple.simpleAttention;

test {
    _ = matrix;
}
