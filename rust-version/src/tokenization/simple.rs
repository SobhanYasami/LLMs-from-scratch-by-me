//! Step 3 - simple tokenizers that combine splitting with the vocabulary.
//!
//! Both types expose `encode` (text -> IDs) and `decode` (IDs -> text).
//! V1 fails on tokens outside its vocabulary; V2 replaces them with `<|unk|>`.

use std::collections::HashMap;

use super::preprocess::{UNK_TOKEN, detokenize, split_text};

/// Encode and decode with a vocabulary; unknown tokens are an error.
pub struct SimpleTokenizerV1 {
    str_to_int: HashMap<String, usize>,
    int_to_str: HashMap<usize, String>,
}

impl SimpleTokenizerV1 {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        let int_to_str = vocab.iter().map(|(s, &i)| (i, s.clone())).collect();
        Self { str_to_int: vocab, int_to_str }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        split_text(text)
            .iter()
            .map(|t| {
                self.str_to_int
                    .get(t)
                    .copied()
                    .ok_or_else(|| format!("token not in vocab: {t:?}"))
            })
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, String> {
        let tokens = ids
            .iter()
            .map(|i| {
                self.int_to_str
                    .get(i)
                    .cloned()
                    .ok_or_else(|| format!("id not in vocab: {i}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(detokenize(&tokens))
    }
}

/// Same as V1, but unknown tokens become `<|unk|>` instead of failing.
pub struct SimpleTokenizerV2(SimpleTokenizerV1);

impl SimpleTokenizerV2 {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        Self(SimpleTokenizerV1::new(vocab))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        let v1 = &self.0;
        split_text(text)
            .iter()
            .map(|t| {
                let t = if v1.str_to_int.contains_key(t) { t.as_str() } else { UNK_TOKEN };
                v1.str_to_int
                    .get(t)
                    .copied()
                    .ok_or_else(|| format!("token not in vocab: {t:?}"))
            })
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, String> {
        self.0.decode(ids)
    }
}
