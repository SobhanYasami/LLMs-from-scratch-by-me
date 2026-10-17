//! Pretraining (book chapter 5).
//!
//! Files, in the order you learn them:
//!
//! - `loss`: cross entropy loss and perplexity (5.1).
//! - `grad`: the backward pass, written by hand, and `adamw`, the optimizer (5.2).
//! - `sample`: temperature and top-k decoding (5.3).
//! - `checkpoint`: saving and loading weights and optimizer state (5.4).
//! - `openai`: loading OpenAI's pretrained GPT-2 weights (5.5).
//!
//! The Python version lets torch's autograd do the backward pass. Here there is no autograd, so
//! `grad` applies the chain rule layer by layer.

pub const loss = @import("loss.zig");
pub const grad = @import("grad.zig");
pub const adamw = @import("adamw.zig");
pub const sample = @import("sample.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const openai = @import("openai.zig");

pub const Batch = loss.Batch;
pub const AdamW = adamw.AdamW;

test {
    _ = loss;
    _ = grad;
    _ = sample;
}
