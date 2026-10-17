//! Pretraining (book chapter 5).
//!
//! Modules, in the order you learn them:
//!
//! - `loss`: cross entropy loss and perplexity (5.1).
//! - `grad`: the backward pass, written by hand, and `adamw`, the optimizer (5.2).
//! - `sample`: temperature and top-k decoding (5.3).
//! - `checkpoint`: saving and loading weights and optimizer state (5.4).
//! - `openai`: loading OpenAI's pretrained GPT-2 weights (5.5).
//!
//! The Python version lets torch's autograd do the backward pass. Here there is no
//! autograd, so `grad` applies the chain rule layer by layer.

pub mod adamw;
pub mod checkpoint;
pub mod grad;
pub mod loss;
pub mod openai;
pub mod sample;
