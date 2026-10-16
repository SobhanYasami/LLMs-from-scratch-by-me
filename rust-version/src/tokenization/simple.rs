//! Step 3 - simple tokenizers: split text, then look each token up in the vocab.
//!
//! ```text
//! encode: "Hello, world" -> ["Hello", ",", "world"] -> [1131, 5, 987]
//! decode: [1131, 5, 987] -> ["Hello", ",", "world"] -> "Hello, world"
//! ```
//!
//! Tip: compare the two types. V1 fails on a word it never saw; V2 swaps it for
//! `<|unk|>` and keeps going. That one difference is why real models use BPE.

use std::collections::HashMap;

use super::preprocess::{UNK_TOKEN, detokenize, split_text};

/// Encode and decode with a vocabulary; unknown tokens are an error.
pub struct SimpleTokenizerV1 {
    str_to_int: HashMap<String, usize>, // token -> ID, used by encode
    int_to_str: HashMap<usize, String>, // ID -> token, used by decode (the vocab, inverted)
}

impl SimpleTokenizerV1 {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        let int_to_str = vocab.iter().map(|(s, &i)| (i, s.clone())).collect();
        Self {
            str_to_int: vocab,
            int_to_str,
        }
    }

    /// Shared by V1 and V2. The only difference is what happens to a token that
    /// is not in the vocab: error (V1) or `<|unk|>` (V2).
    fn encode_with(&self, text: &str, unknown_to_unk: bool) -> Result<Vec<usize>, String> {
        split_text(text)
            .iter()
            .map(|token| {
                let known = self.str_to_int.get(token);
                let id = match (known, unknown_to_unk) {
                    (None, true) => self.str_to_int.get(UNK_TOKEN),
                    _ => known,
                };
                id.copied()
                    .ok_or_else(|| format!("token not in vocab: {token:?}"))
            })
            .collect()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        self.encode_with(text, false)
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, String> {
        let tokens = ids
            .iter()
            .map(|id| {
                self.int_to_str
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("id not in vocab: {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(detokenize(&tokens))
    }
}

/// V1, except unknown tokens become `<|unk|>`. Wraps V1, so decode is reused.
pub struct SimpleTokenizerV2(SimpleTokenizerV1);

impl SimpleTokenizerV2 {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        Self(SimpleTokenizerV1::new(vocab))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        self.0.encode_with(text, true)
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, String> {
        self.0.decode(ids)
    }
}
