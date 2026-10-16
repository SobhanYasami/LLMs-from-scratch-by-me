//! Step 4 - sliding-window data sampling for next-token prediction.
//!
//! ```text
//! ids      = [10 11 12 13 14 15]
//! window 0: input [10 11 12 13]  target [11 12 13 14]
//! ```
//!
//! The target is the input shifted by one position. `stride` is how far the
//! window moves each step; `stride == max_length` means no overlap.

/// Holds every (input, target) window of one tokenized text.
pub struct GptDataset {
    input_ids: Vec<Vec<u32>>,
    target_ids: Vec<Vec<u32>>,
}

impl GptDataset {
    /// Cut `token_ids` into windows, like Python's `GPTDatasetV1`.
    pub fn new(token_ids: &[u32], max_length: usize, stride: usize) -> Self {
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();
        for i in (0..token_ids.len().saturating_sub(max_length)).step_by(stride) {
            input_ids.push(token_ids[i..i + max_length].to_vec());
            target_ids.push(token_ids[i + 1..i + max_length + 1].to_vec());
        }
        Self { input_ids, target_ids }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    /// The input and target of window `idx`.
    pub fn get(&self, idx: usize) -> (&[u32], &[u32]) {
        (&self.input_ids[idx], &self.target_ids[idx])
    }

    /// Group consecutive windows, dropping a final short batch.
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = (&[Vec<u32>], &[Vec<u32>])> {
        self.input_ids
            .chunks_exact(batch_size)
            .zip(self.target_ids.chunks_exact(batch_size))
    }
}
