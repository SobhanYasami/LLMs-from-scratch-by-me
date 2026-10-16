//! Attention building blocks (book chapter 3).
//!
//! Modules, in the order you learn them:
//!
//! - `matrix`: the few matrix operations attention needs.
//! - `simple`: simplified self-attention, no trainable weights (3.3).
//! - `self_attention`: scaled dot-product attention with W_query/key/value (3.4).
//! - `causal`: hide future tokens with a mask, optional dropout (3.5).
//! - `multihead`: several heads in parallel plus an output projection (3.6).
//!
//! The seeded random generator is shared with chapter 2: `tokenization::rng`.

pub mod causal;
pub mod matrix;
pub mod multihead;
pub mod self_attention;
pub mod simple;

pub use causal::CausalAttention;
pub use matrix::Matrix;
pub use multihead::MultiHeadAttention;
pub use self_attention::SelfAttention;
pub use simple::simple_attention;
