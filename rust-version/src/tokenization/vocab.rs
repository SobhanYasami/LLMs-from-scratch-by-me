//! Step 2 - build a vocabulary: a mapping from each token to a unique integer ID.

use std::collections::{BTreeSet, HashMap};

use super::preprocess::{END_OF_TEXT, UNK_TOKEN};

pub const SPECIAL_TOKENS: [&str; 2] = [END_OF_TEXT, UNK_TOKEN];

/// Assign IDs to the unique tokens in sorted order, then append special tokens.
///
/// Sorting makes IDs deterministic for a given corpus. Special tokens go last so
/// regular word IDs stay stable.
pub fn build_vocab(tokens: &[String]) -> HashMap<String, usize> {
    let unique: BTreeSet<&str> = tokens.iter().map(String::as_str).collect();
    unique
        .into_iter()
        .chain(SPECIAL_TOKENS)
        .enumerate()
        .map(|(i, t)| (t.to_string(), i))
        .collect()
}
