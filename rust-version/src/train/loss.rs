//! Step 17 (book 5.1) - how wrong is the model? Cross entropy loss and perplexity.
//!
//! For every position the model gives a score (logit) for each of the 50,257 tokens.
//! Softmax turns the scores into probabilities. The loss looks only at the probability
//! given to the token that REALLY came next:
//!
//! ```text
//! loss = - mean( log p(target) )        (cross entropy)
//! perplexity = exp(loss)                "among how many tokens is the model torn?"
//! ```
//!
//! A model that guesses uniformly has loss ln(50257) = 10.82. A perfect model has loss 0.

use crate::attention::Matrix;
use crate::gpt::GptModel;
use crate::tokenization::dataloader::Batch;

/// log of the sum of exp(x - max), plus the max: the log of softmax's denominator.
pub fn log_sum_exp(row: &[f32]) -> f32 {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &v in row {
        sum += ((v - max) as f64).exp() as f32;
    }
    (sum as f64).ln() as f32 + max
}

/// Cross entropy of `logits` [T, vocab] against the `targets` (one token ID per row), summed over rows.
pub fn cross_entropy_sum(logits: &Matrix, targets: &[u32]) -> f32 {
    let mut total = 0.0f32;
    for (i, &t) in targets.iter().enumerate() {
        let row = logits.row(i);
        total += log_sum_exp(row) - row[t as usize]; // -log p(target)
    }
    total
}

pub fn perplexity(loss: f32) -> f32 {
    (loss as f64).exp() as f32
}

/// Mean cross entropy of one batch (all its tokens count equally).
pub fn calc_loss_batch(batch: &Batch, model: &GptModel) -> f32 {
    let mut total = 0.0f32;
    let mut n = 0;
    for (x, y) in batch.inputs.iter().zip(&batch.targets) {
        total += cross_entropy_sum(&model.forward(x), y);
        n += y.len();
    }
    total / n as f32
}

/// Average loss over (at most `num_batches`) batches.
pub fn calc_loss_loader(batches: &[Batch], model: &GptModel, num_batches: Option<usize>) -> f32 {
    if batches.is_empty() {
        return f32::NAN;
    }
    let n = num_batches.map_or(batches.len(), |k| k.min(batches.len()));
    batches[..n]
        .iter()
        .map(|b| calc_loss_batch(b, model))
        .sum::<f32>()
        / n as f32
}
