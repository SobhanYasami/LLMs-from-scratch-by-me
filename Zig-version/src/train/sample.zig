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

const std = @import("std");
const GptModel = @import("../gpt/root.zig").GptModel;
const Rng = @import("../tokenization/rng.zig").Rng;

/// Softmax of one row of scores, like torch.softmax: subtract the max, exp, divide by the sum.
pub fn softmax(gpa: std.mem.Allocator, logits: []const f32) ![]f32 {
    var max: f32 = -std.math.inf(f32);
    for (logits) |v| max = @max(max, v);
    const out = try gpa.alloc(f32, logits.len);
    var sum: f32 = 0;
    for (logits, out) |v, *o| {
        o.* = @floatCast(@exp(@as(f64, v - max)));
        sum += o.*;
    }
    for (out) |*o| o.* /= sum;
    return out;
}

pub fn softmaxWithTemperature(gpa: std.mem.Allocator, logits: []const f32, temperature: f32) ![]f32 {
    const scaled = try gpa.alloc(f32, logits.len);
    for (logits, scaled) |v, *s| s.* = v / temperature;
    return softmax(gpa, scaled);
}

fn descending(_: void, a: f32, b: f32) bool {
    return a > b;
}

/// Keep the k largest logits (ties at the threshold are kept), -inf for the rest.
pub fn topKLogits(gpa: std.mem.Allocator, logits: []const f32, k: usize) ![]f32 {
    const sorted = try gpa.dupe(f32, logits);
    std.mem.sort(f32, sorted, {}, descending);
    const threshold = sorted[k - 1];
    const out = try gpa.alloc(f32, logits.len);
    for (logits, out) |v, *o| o.* = if (v < threshold) -std.math.inf(f32) else v;
    return out;
}

/// Draw one index in proportion to `probas`: inverse-CDF sampling. One uniform number u in [0, 1);
/// walk the cumulative sum and stop at the first entry where it passes u. Tokens with probability 0
/// can never be picked.
pub fn sampleIndex(probas: []const f32, rng: *Rng) usize {
    const u = rng.nextF64();
    var cumulative: f64 = 0;
    var last: usize = 0;
    for (probas, 0..) |p, i| {
        if (p > 0) last = i;
        cumulative += p;
        if (u < cumulative) return i;
    }
    return last; // rounding left the total a hair below u
}

/// Like `generateTextSimple` but with temperature and top-k (`top_k` 0 = off). `temperature` 0 means
/// greedy. Every forward pass gets its own scratch arena that is freed right after.
pub fn generate(gpa: std.mem.Allocator, model: GptModel, ids: []const u32, max_new_tokens: usize, context_size: usize, rng: *Rng, temperature: f32, top_k: usize) ![]u32 {
    var out = try std.ArrayList(u32).initCapacity(gpa, ids.len + max_new_tokens);
    out.appendSliceAssumeCapacity(ids);
    for (0..max_new_tokens) |_| {
        const start = if (out.items.len > context_size) out.items.len - context_size else 0;
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const s = scratch.allocator();
        const logits = try model.forward(s, out.items[start..]);
        var last = logits.row(logits.rows - 1); // scores for the next token
        if (top_k > 0) last = try topKLogits(s, last, top_k);
        var next: usize = 0;
        if (temperature > 0) {
            next = sampleIndex(try softmaxWithTemperature(s, last, temperature), rng);
        } else {
            for (last, 0..) |v, i| {
                if (v > last[next]) next = i;
            }
        }
        out.appendAssumeCapacity(@intCast(next));
    }
    return out.items;
}

test "top-k keeps exactly the k largest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const out = try topKLogits(arena.allocator(), &.{ 4.51, 0.89, -1.9, 6.75, 1.63, 6.28 }, 3);
    try std.testing.expectEqual(@as(f32, 4.51), out[0]);
    try std.testing.expect(std.math.isNegativeInf(out[1]));
    try std.testing.expectEqual(@as(f32, 6.75), out[3]);
}

test "temperature sharpens and flattens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const logits = [_]f32{ 1, 2, 3 };
    const cold = try softmaxWithTemperature(arena.allocator(), &logits, 0.1);
    const hot = try softmaxWithTemperature(arena.allocator(), &logits, 5);
    try std.testing.expect(cold[2] > 0.99);
    try std.testing.expect(hot[2] < 0.5);
}
