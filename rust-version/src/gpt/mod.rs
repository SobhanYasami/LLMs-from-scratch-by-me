//! The GPT model (book chapter 4).
//!
//! Modules, in the order you learn them:
//!
//! - `layers`: layer normalization, GELU, Linear and the feed forward block (4.2, 4.3).
//! - `shortcut`: a tiny deep network that shows what shortcut connections do (4.4).
//! - `block`: the transformer block (4.5).
//! - `model`: the GPT model, its configuration and text generation (4.6, 4.7).
//!
//! Tip: like chapter 3 there is no tensor library. A sequence of T tokens is a
//! `Matrix` [T, D]; the Rust code handles one sequence at a time (the demo loops
//! over the batch), and the math per sequence is the same as the batched Python.

pub mod block;
pub mod layers;
pub mod model;
pub mod shortcut;

pub use block::TransformerBlock;
pub use layers::{FeedForward, LayerNorm, Linear, gelu};
pub use model::{
    CONFIG_124M, CONFIG_TINY, Config, GptModel, SIZES, generate_text_simple, parameter_count,
};
pub use shortcut::{ExampleDeepNeuralNetwork, ExampleLayer};
