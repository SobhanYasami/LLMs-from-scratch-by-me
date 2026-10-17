//! Steps 15-16 (book 4.6 and 4.7) - the GPT model and text generation.
//!
//! ```text
//! token IDs [T]
//!   -> token embedding + position embedding        [T, D]
//!   -> N x transformer block                       [T, D]
//!   -> final layer norm                            [T, D]
//!   -> output head (Linear D -> vocab, no bias)    [T, vocab]   = "logits"
//! ```
//!
//! The logits of the LAST position are the model's scores for the next token.
//! `generate_text_simple` picks the best one, appends it, and repeats.
//!
//! Weights start as random numbers from the repo's shared generator
//! (Normal(0, 0.02), biases 0, layer norm scale 1 / shift 0), in a fixed order,
//! so every language builds exactly the same untrained model. Untrained means
//! the generated text is gibberish; training is chapter 5.

use super::block::TransformerBlock;
use super::layers::LayerNorm;
use crate::attention::Matrix;
use crate::tokenization::rng::Rng;

#[derive(Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,     // BPE vocabulary of GPT-2
    pub context_length: usize, // most tokens the model can look at
    pub emb_dim: usize,        // size of every token vector
    pub n_heads: usize,        // attention heads per block
    pub n_layers: usize,       // transformer blocks
    pub drop_rate: f32,        // dropout (training only)
    pub qkv_bias: bool,        // bias in the query/key/value projections
}

pub const CONFIG_124M: Config = Config {
    vocab_size: 50257,
    context_length: 1024,
    emb_dim: 768,
    n_heads: 12,
    n_layers: 12,
    drop_rate: 0.1,
    qkv_bias: false,
};

/// The same architecture, small enough to run in a blink in any language.
pub const CONFIG_TINY: Config = Config {
    context_length: 16,
    emb_dim: 32,
    n_heads: 4,
    n_layers: 2,
    ..CONFIG_124M
};

/// Book exercise 4.2: bigger GPT-2 models differ only in the configuration.
/// (name, emb_dim, n_layers, n_heads)
pub const SIZES: [(&str, usize, usize, usize); 4] = [
    ("gpt2-small", 768, 12, 12),
    ("gpt2-medium", 1024, 24, 16),
    ("gpt2-large", 1280, 36, 20),
    ("gpt2-xl", 1600, 48, 25),
];

pub struct GptModel {
    pub cfg: Config,
    pub tok_emb: Matrix,
    pub pos_emb: Matrix,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
    pub out_head: Matrix, // [D, vocab]
}

impl GptModel {
    pub fn new(cfg: Config) -> Self {
        assert!(
            !cfg.qkv_bias,
            "this implementation has no query/key/value bias (chapter 6 adds it)"
        );
        Self {
            cfg,
            tok_emb: Matrix::new(cfg.vocab_size, cfg.emb_dim),
            pos_emb: Matrix::new(cfg.context_length, cfg.emb_dim),
            blocks: (0..cfg.n_layers)
                .map(|_| TransformerBlock::new(&cfg))
                .collect(),
            final_norm: LayerNorm::new(cfg.emb_dim),
            out_head: Matrix::new(cfg.emb_dim, cfg.vocab_size),
        }
    }

    /// Build a model and fill it from the shared generator. The ORDER is part of
    /// the recipe: token emb, position emb, then per block wq, wk, wv, wo, ff1,
    /// ff2, then the output head.
    pub fn new_initialized(cfg: Config, seed: u64) -> Self {
        let mut m = Self::new(cfg);
        let mut rng = Rng::new(seed);
        let mut fill = |w: &mut Matrix| {
            for v in w.data.iter_mut() {
                *v = rng.normal() * 0.02;
            }
        };
        fill(&mut m.tok_emb);
        fill(&mut m.pos_emb);
        for b in m.blocks.iter_mut() {
            fill(&mut b.att.w_query);
            fill(&mut b.att.w_key);
            fill(&mut b.att.w_value);
            fill(&mut b.att.out_weight);
            fill(&mut b.ff.expand.w);
            fill(&mut b.ff.shrink.w);
        }
        fill(&mut m.out_head);
        m
    }

    /// Token embedding + position embedding: what the model feeds its first block.
    pub fn embed(&self, ids: &[u32]) -> Matrix {
        let d = self.cfg.emb_dim;
        let mut x = Matrix::new(ids.len(), d);
        for (t, &id) in ids.iter().enumerate() {
            let (tok, pos) = (self.tok_emb.row(id as usize), self.pos_emb.row(t));
            for j in 0..d {
                x.data[t * d + j] = tok[j] + pos[j];
            }
        }
        x
    }

    /// Token IDs [T] in, logits [T, vocab] out.
    pub fn forward(&self, ids: &[u32]) -> Matrix {
        let mut x = self.embed(ids);
        for b in &self.blocks {
            x = b.forward(&x);
        }
        self.final_norm.forward(&x).matmul(&self.out_head)
    }

    /// Add up every weight the model holds.
    pub fn count_parameters(&self) -> usize {
        let mut n = self.tok_emb.data.len()
            + self.pos_emb.data.len()
            + self.out_head.data.len()
            + 2 * self.final_norm.scale.len();
        for b in &self.blocks {
            let a = &b.att;
            n += a.w_query.data.len()
                + a.w_key.data.len()
                + a.w_value.data.len()
                + a.out_weight.data.len()
                + a.out_bias.len();
            n += b.ff.expand.w.data.len()
                + b.ff.expand.b.len()
                + b.ff.shrink.w.data.len()
                + b.ff.shrink.b.len();
            n += 2 * (b.norm1.scale.len() + b.norm2.scale.len());
        }
        n
    }
}

/// The same count by arithmetic only (no model is built).
pub fn parameter_count(cfg: &Config) -> usize {
    let (v, c, d, n) = (
        cfg.vocab_size,
        cfg.context_length,
        cfg.emb_dim,
        cfg.n_layers,
    );
    let attention = 4 * d * d + d; // wq, wk, wv, wo and the output bias
    let feed_forward = d * 4 * d + 4 * d + 4 * d * d + d;
    let norms = 2 * 2 * d; // two layer norms per block, scale + shift each
    v * d + c * d + n * (attention + feed_forward + norms) + 2 * d + d * v
}

/// Greedy decoding: `ids` are the tokens so far; the result has new IDs appended.
pub fn generate_text_simple(
    model: &GptModel,
    ids: &[u32],
    max_new_tokens: usize,
    context_size: usize,
) -> Vec<u32> {
    let mut ids = ids.to_vec();
    for _ in 0..max_new_tokens {
        let start = ids.len().saturating_sub(context_size); // keep only what fits in the context
        let logits = model.forward(&ids[start..]);
        let last = logits.row(logits.rows - 1); // scores for the NEXT token only
        // Softmax would turn these into probabilities, but it keeps the order,
        // so the biggest score is also the most likely token.
        let mut best = 0;
        for (i, &v) in last.iter().enumerate() {
            if v > last[best] {
                best = i;
            }
        }
        ids.push(best as u32);
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count_matches_the_book() {
        assert_eq!(parameter_count(&CONFIG_124M), 163_009_536);
    }

    #[test]
    fn building_and_arithmetic_agree() {
        let m = GptModel::new(CONFIG_TINY);
        assert_eq!(m.count_parameters(), parameter_count(&CONFIG_TINY));
    }

    #[test]
    fn untrained_generation_is_repeatable() {
        let m = GptModel::new_initialized(CONFIG_TINY, 123);
        let a = generate_text_simple(&m, &[15496, 11], 3, CONFIG_TINY.context_length);
        let b = generate_text_simple(&m, &[15496, 11], 3, CONFIG_TINY.context_length);
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
    }
}
