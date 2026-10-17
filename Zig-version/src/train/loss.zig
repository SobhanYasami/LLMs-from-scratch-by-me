//! Step 17 (book 5.1) - how wrong is the model? Cross entropy loss and perplexity.
//!
//! For every position the model gives a score (logit) for each of the 50,257 tokens. Softmax turns
//! the scores into probabilities. The loss looks only at the probability given to the token that
//! REALLY came next:
//!
//! ```
//! loss = - mean( log p(target) )        (cross entropy)
//! perplexity = exp(loss)                "among how many tokens is the model torn?"
//! ```
//!
//! A model that guesses uniformly has loss ln(50257) = 10.82. A perfect model has loss 0.

const std = @import("std");
const Matrix = @import("../attention/root.zig").Matrix;
const GptModel = @import("../gpt/root.zig").GptModel;

/// One batch: `inputs[r]` and `targets[r]` are one (input, target) pair; targets are the inputs shifted by one.
pub const Batch = struct {
    inputs: []const []const u32,
    targets: []const []const u32,
};

/// The log of softmax's denominator: log(sum(exp(x - max))) + max.
pub fn logSumExp(row: []const f32) f32 {
    var max: f32 = -std.math.inf(f32);
    for (row) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (row) |v| sum += @floatCast(@exp(@as(f64, v - max)));
    return @as(f32, @floatCast(@log(@as(f64, sum)))) + max;
}

/// Cross entropy of `logits` [T, vocab] against `targets` (one token ID per row), summed over rows.
pub fn crossEntropySum(logits: Matrix, targets: []const u32) f32 {
    var total: f32 = 0;
    for (targets, 0..) |t, i| {
        const r = logits.row(i);
        total += logSumExp(r) - r[t]; // -log p(target)
    }
    return total;
}

pub fn perplexity(l: f32) f32 {
    return @floatCast(@exp(@as(f64, l)));
}

/// Mean cross entropy of one batch (all its tokens count equally).
pub fn calcLossBatch(model: GptModel, batch: Batch) !f32 {
    var total: f32 = 0;
    var n: usize = 0;
    for (batch.inputs, batch.targets) |x, y| {
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        total += crossEntropySum(try model.forward(scratch.allocator(), x), y);
        n += y.len;
    }
    return total / @as(f32, @floatFromInt(n));
}

/// Average loss over (at most `num_batches`; 0 means all) batches.
pub fn calcLossLoader(model: GptModel, batches: []const Batch, num_batches: usize) !f32 {
    if (batches.len == 0) return std.math.nan(f32);
    const n = if (num_batches > 0 and num_batches < batches.len) num_batches else batches.len;
    var total: f32 = 0;
    for (batches[0..n]) |b| total += try calcLossBatch(model, b);
    return total / @as(f32, @floatFromInt(n));
}
