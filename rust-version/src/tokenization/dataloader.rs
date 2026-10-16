//! Step 4b - batching. Mirrors torch's `DataLoader`.
//!
//! Why batch? The GPU processes many windows at once, which is much faster.
//! Why shuffle? Windows in book order make training swing; random order smooths it.
//! Why drop the last short batch? A different size can upset training.

use super::dataset::GptDataset;
use super::rng::Rng;

/// One group of windows: `inputs[b]` and `targets[b]` are one (input, target) pair.
pub struct Batch {
    pub inputs: Vec<Vec<u32>>,
    pub targets: Vec<Vec<u32>>,
}

/// Serves a `GptDataset` in batches.
pub struct DataLoader<'a> {
    dataset: &'a GptDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    rng: Rng,
}

impl<'a> DataLoader<'a> {
    /// `seed` makes the shuffle repeatable.
    pub fn new(
        dataset: &'a GptDataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        seed: u64,
    ) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            rng: Rng::new(seed),
        }
    }

    /// Every batch once. With shuffle, each call uses a new order.
    pub fn epoch(&mut self) -> Vec<Batch> {
        let mut order: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            self.rng.shuffle(&mut order);
        }
        order
            .chunks(self.batch_size)
            .filter(|c| !(self.drop_last && c.len() < self.batch_size))
            .map(|chunk| {
                let (inputs, targets) = chunk
                    .iter()
                    .map(|&i| {
                        let (x, y) = self.dataset.get(i);
                        (x.to_vec(), y.to_vec())
                    })
                    .unzip();
                Batch { inputs, targets }
            })
            .collect()
    }
}
