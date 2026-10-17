//! Step 19 (book 5.3) - decoding strategies: temperature and top-k sampling.
//!
//! Greedy decoding always takes the most likely token, so the same prompt gives the same text,
//! often copied straight from the training data. Two knobs add variety:
//!
//! - temperature: divide the logits by T before softmax. T < 1 sharpens the distribution (more
//!   confident), T > 1 flattens it (more surprising).
//! - top-k: keep only the k best logits, set the others to -inf (probability 0).
//!
//! Then draw the next token IN PROPORTION to its probability. The draw uses the repo's shared
//! generator (not torch.multinomial), so every language agrees.

use crate::gpt::GptModel;
use crate::tokenization::rng::Rng;

/// Softmax of one row of scores, like torch.softmax: subtract the max, exp, divide by the sum.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|&v| ((v - max) as f64).exp() as f32)
        .collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

pub fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|v| v / temperature).collect();
    softmax(&scaled)
}

/// Keep the k largest logits (ties at the threshold are kept), -inf for the rest.
pub fn top_k_logits(logits: &[f32], k: usize) -> Vec<f32> {
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[k - 1];
    logits
        .iter()
        .map(|&v| if v < threshold { f32::NEG_INFINITY } else { v })
        .collect()
}

/// Draw one index in proportion to `probas`: inverse-CDF sampling. One uniform number u in [0, 1);
/// walk the cumulative sum and stop at the first entry where it passes u. Tokens with probability 0
/// can never be picked.
pub fn sample_index(probas: &[f32], rng: &mut Rng) -> usize {
    let u = rng.next_f64();
    let (mut cumulative, mut last) = (0.0f64, 0);
    for (i, &p) in probas.iter().enumerate() {
        if p > 0.0 {
            last = i;
        }
        cumulative += p as f64;
        if u < cumulative {
            return i;
        }
    }
    last // rounding left the total a hair below u
}

/// Like `generate_text_simple` but with temperature and top-k. `temperature == 0` means greedy.
pub fn generate(
    model: &GptModel,
    ids: &[u32],
    max_new_tokens: usize,
    context_size: usize,
    rng: &mut Rng,
    temperature: f32,
    top_k: Option<usize>,
) -> Vec<u32> {
    let mut ids = ids.to_vec();
    for _ in 0..max_new_tokens {
        let start = ids.len().saturating_sub(context_size);
        let logits = model.forward(&ids[start..]);
        let mut last = logits.row(logits.rows - 1).to_vec(); // scores for the next token
        if let Some(k) = top_k {
            last = top_k_logits(&last, k);
        }
        let next = if temperature > 0.0 {
            sample_index(&softmax_with_temperature(&last, temperature), rng)
        } else {
            last.iter()
                .enumerate()
                .fold(0, |best, (i, &v)| if v > last[best] { i } else { best })
        };
        ids.push(next as u32);
    }
    ids
}
