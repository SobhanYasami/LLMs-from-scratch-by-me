//! Step 5 - token and positional embeddings.
//!
//! A token ID is just an integer. An embedding layer is a lookup table with one
//! vector per ID, so every token becomes a vector the model can learn from:
//!
//! ```text
//! ids [B][T]  --token embedding-->  [B][T][D]
//! positions 0..T-1 --pos embedding-> [T][D]  (added to every batch row)
//! input embeddings = token + position  -> [B][T][D]
//! ```
//!
//! Why add positions? Attention (later chapters) treats its input as a bag of
//! vectors, so word order must be put into the vectors themselves.
//!
//! No tensor library: tensors are nested `Vec`s, and weights are random
//! (normal, mean 0, std 1) and untrained, like torch's `nn.Embedding` at init.

use super::rng::Rng;

/// A lookup table of `num_rows` vectors of size `dim`.
pub struct Embedding {
    pub weight: Vec<Vec<f32>>,
}

impl Embedding {
    pub fn new(num_rows: usize, dim: usize, rng: &mut Rng) -> Self {
        let weight = (0..num_rows)
            .map(|_| (0..dim).map(|_| rng.normal()).collect())
            .collect();
        Self { weight }
    }

    /// Look up one vector per ID.
    pub fn forward(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        ids.iter()
            .map(|&id| self.weight[id as usize].clone())
            .collect()
    }
}

/// Token embedding plus absolute positional embedding.
pub struct InputEmbedding {
    pub token: Embedding,
    pub pos: Embedding,
}

impl InputEmbedding {
    pub fn new(vocab_size: usize, context_length: usize, dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let token = Embedding::new(vocab_size, dim, &mut rng);
        let pos = Embedding::new(context_length, dim, &mut rng);
        Self { token, pos }
    }

    /// `ids` has shape [batch][seq_len]; returns [batch][seq_len][dim].
    pub fn forward(&self, ids: &[Vec<u32>]) -> Vec<Vec<Vec<f32>>> {
        ids.iter()
            .map(|row| {
                self.token
                    .forward(row)
                    .into_iter()
                    .zip(&self.pos.weight)
                    .map(|(tok, pos)| tok.iter().zip(pos).map(|(a, b)| a + b).collect())
                    .collect()
            })
            .collect()
    }
}
