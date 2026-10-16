//! Educational tokenizer building blocks.
//!
//! Modules, in the order you learn them:
//!
//! - `preprocess`: split raw text into tokens and join them back.
//! - `vocab`: map each unique token to an integer ID.
//! - `simple`: `SimpleTokenizerV1` and `SimpleTokenizerV2` built on the above.
//! - `dataset`: sliding-window (input, target) pairs for next-token prediction.
//! - `dataloader`: batching and shuffling of those pairs.
//! - `embedding`: token + positional embeddings (`rng` supplies seeded randomness).

pub mod dataloader;
pub mod dataset;
pub mod embedding;
pub mod preprocess;
pub mod rng;
pub mod simple;
pub mod vocab;

pub use dataloader::{Batch, DataLoader};
pub use dataset::GptDataset;
pub use embedding::{Embedding, InputEmbedding};
pub use preprocess::{END_OF_TEXT, UNK_TOKEN, detokenize, split_text};
pub use simple::{SimpleTokenizerV1, SimpleTokenizerV2};
pub use vocab::build_vocab;
