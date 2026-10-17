//! The GPT model (book chapter 4).
//!
//! Files, in the order you learn them:
//!
//! - `layers`: layer normalization, GELU, Linear and the feed forward block (4.2, 4.3).
//! - `shortcut`: a tiny deep network that shows what shortcut connections do (4.4).
//! - `block`: the transformer block (4.5).
//! - `model`: the GPT model, its configuration and text generation (4.6, 4.7).
//!
//! Tip: like chapter 3 there is no tensor library. A sequence of T tokens is a
//! `Matrix` [T, D]; the Zig code handles one sequence at a time (the demo loops
//! over the batch), and the math per sequence is the same as the batched Python.

pub const layers = @import("layers.zig");
pub const shortcut = @import("shortcut.zig");
pub const block = @import("block.zig");
pub const model = @import("model.zig");

pub const LayerNorm = layers.LayerNorm;
pub const Linear = layers.Linear;
pub const FeedForward = layers.FeedForward;
pub const gelu = layers.gelu;
pub const ExampleLayer = shortcut.ExampleLayer;
pub const ExampleDeepNeuralNetwork = shortcut.ExampleDeepNeuralNetwork;
pub const TransformerBlock = block.TransformerBlock;
pub const Config = model.Config;
pub const GptModel = model.GptModel;
pub const config_124m = model.config_124m;
pub const config_tiny = model.config_tiny;
pub const sizes = model.sizes;
pub const parameterCount = model.parameterCount;
pub const generateTextSimple = model.generateTextSimple;

test {
    _ = layers;
    _ = shortcut;
    _ = model;
}
