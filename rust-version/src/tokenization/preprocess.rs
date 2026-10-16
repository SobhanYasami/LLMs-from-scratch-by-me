//! Step 1 - split raw text into tokens (and join them back).
//!
//! A token is a word or a punctuation mark. The pattern keeps its delimiters,
//! so punctuation becomes its own token. Rust's `regex` has no capturing split,
//! so `split_text` walks the matches by hand (Go does the same).

use std::sync::LazyLock;

use regex::Regex;

/// Special tokens. They are not produced by splitting text; they are added on purpose.
pub const END_OF_TEXT: &str = "<|endoftext|>";
pub const UNK_TOKEN: &str = "<|unk|>";

/// Matches: , . ? _ ! " ( ) ' , the double dash "--", and any whitespace.
static SPLIT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"[,.?_!"()']|--|\s"#).unwrap());

/// Used when decoding: removes the space that join(" ") put before punctuation.
static DETOKENIZE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\s+([,.?!"()'])"#).unwrap());

/// Return the tokens of `text`, with whitespace-only pieces removed.
///
/// ```
/// use llm_from_scratch::tokenization::split_text;
/// assert_eq!(split_text("Hello, world!"), ["Hello", ",", "world", "!"]);
/// ```
pub fn split_text(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut last = 0;
    for m in SPLIT_RE.find_iter(text) {
        pieces.push(&text[last..m.start()]);
        pieces.push(m.as_str());
        last = m.end();
    }
    pieces.push(&text[last..]);
    pieces
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(String::from)
        .collect()
}

/// Join tokens back into text, undoing the spaces added before punctuation.
///
/// Not a perfect inverse: the original spacing is lost.
pub fn detokenize(tokens: &[String]) -> String {
    DETOKENIZE_RE.replace_all(&tokens.join(" "), "$1").into_owned()
}
