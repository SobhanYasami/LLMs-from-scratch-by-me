//! Educational tokenizer building blocks.
//!
//! Files, in the order you learn them:
//!
//! - `preprocess`: split raw text into tokens and join them back.
//! - `vocab`: map each unique token to an integer ID.
//! - `simple`: `SimpleTokenizer` (V1 and V2) built on the above.
//! - `dataset`: sliding-window (input, target) pairs for next-token prediction.
//! - `dataloader`: batching and shuffling of those pairs.
//! - `embedding`: token + positional embeddings (`rng` supplies seeded randomness).
//! - `bpe`: byte pair encoding, the GPT-2 tokenizer (bonus).

pub const preprocess = @import("preprocess.zig");
pub const vocab = @import("vocab.zig");
pub const simple = @import("simple.zig");
pub const dataset = @import("dataset.zig");
pub const dataloader = @import("dataloader.zig");
pub const embedding = @import("embedding.zig");
pub const rng = @import("rng.zig");
pub const bpe = @import("bpe.zig");

pub const END_OF_TEXT = preprocess.END_OF_TEXT;
pub const UNK_TOKEN = preprocess.UNK_TOKEN;
pub const splitText = preprocess.splitText;
pub const detokenize = preprocess.detokenize;
pub const Vocab = vocab.Vocab;
pub const SimpleTokenizer = simple.SimpleTokenizer;
pub const GptDataset = dataset.GptDataset;
pub const DataLoader = dataloader.DataLoader;
pub const InputEmbedding = embedding.InputEmbedding;
pub const Bpe = bpe.Bpe;

test {
    // `zig build test` runs the tests in every file listed here.
    _ = preprocess;
    _ = vocab;
    _ = simple;
    _ = dataset;
    _ = bpe;
}
