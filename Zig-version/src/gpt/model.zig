//! Steps 15-16 (book 4.6 and 4.7) - the GPT model and text generation.
//!
//! ```
//! token IDs [T]
//!   -> token embedding + position embedding        [T, D]
//!   -> N x transformer block                       [T, D]
//!   -> final layer norm                            [T, D]
//!   -> output head (Linear D -> vocab, no bias)    [T, vocab]   = "logits"
//! ```
//!
//! The logits of the LAST position are the model's scores for the next token.
//! `generateTextSimple` picks the best one, appends it, and repeats.
//!
//! Weights start as random numbers from the repo's shared generator
//! (Normal(0, 0.02), biases 0, layer norm scale 1 / shift 0), in a fixed order,
//! so every language builds exactly the same untrained model. Untrained means
//! the generated text is gibberish; training is chapter 5.

const std = @import("std");
const att = @import("../attention/root.zig");
const Rng = @import("../tokenization/rng.zig").Rng;
const block_mod = @import("block.zig");
const layers = @import("layers.zig");

const Matrix = att.Matrix;

pub const Config = struct {
    vocab_size: usize, // BPE vocabulary of GPT-2
    context_length: usize, // most tokens the model can look at
    emb_dim: usize, // size of every token vector
    n_heads: usize, // attention heads per block
    n_layers: usize, // transformer blocks
    drop_rate: f32, // dropout (training only)
    qkv_bias: bool, // bias in the query/key/value projections
};

pub const config_124m = Config{
    .vocab_size = 50257,
    .context_length = 1024,
    .emb_dim = 768,
    .n_heads = 12,
    .n_layers = 12,
    .drop_rate = 0.1,
    .qkv_bias = false,
};

/// The same architecture, small enough to run in a blink in any language.
pub const config_tiny = Config{
    .vocab_size = 50257,
    .context_length = 16,
    .emb_dim = 32,
    .n_heads = 4,
    .n_layers = 2,
    .drop_rate = 0.1,
    .qkv_bias = false,
};

/// Book exercise 4.2: bigger GPT-2 models differ only in the configuration.
pub const Size = struct { name: []const u8, emb_dim: usize, n_layers: usize, n_heads: usize };
pub const sizes = [_]Size{
    .{ .name = "gpt2-small", .emb_dim = 768, .n_layers = 12, .n_heads = 12 },
    .{ .name = "gpt2-medium", .emb_dim = 1024, .n_layers = 24, .n_heads = 16 },
    .{ .name = "gpt2-large", .emb_dim = 1280, .n_layers = 36, .n_heads = 20 },
    .{ .name = "gpt2-xl", .emb_dim = 1600, .n_layers = 48, .n_heads = 25 },
};

pub const GptModel = struct {
    cfg: Config,
    tok_emb: Matrix,
    pos_emb: Matrix,
    blocks: []block_mod.TransformerBlock,
    final_norm: layers.LayerNorm,
    out_head: Matrix, // [D, vocab]

    pub fn init(gpa: std.mem.Allocator, cfg: Config) !GptModel {
        const blocks = try gpa.alloc(block_mod.TransformerBlock, cfg.n_layers);
        for (blocks) |*b| b.* = try block_mod.TransformerBlock.init(gpa, cfg.emb_dim, cfg.n_heads, cfg.qkv_bias);
        return .{
            .cfg = cfg,
            .tok_emb = try Matrix.init(gpa, cfg.vocab_size, cfg.emb_dim),
            .pos_emb = try Matrix.init(gpa, cfg.context_length, cfg.emb_dim),
            .blocks = blocks,
            .final_norm = try layers.LayerNorm.init(gpa, cfg.emb_dim),
            .out_head = try Matrix.init(gpa, cfg.emb_dim, cfg.vocab_size),
        };
    }

    /// Build a model and fill it from the shared generator. The ORDER is part of the
    /// recipe: token emb, position emb, then per block wq, wk, wv, wo, ff1, ff2, then the output head.
    pub fn initRandom(gpa: std.mem.Allocator, cfg: Config, seed: u64) !GptModel {
        const m = try init(gpa, cfg);
        var rng = Rng.init(seed);
        fill(m.tok_emb, &rng);
        fill(m.pos_emb, &rng);
        for (m.blocks) |b| {
            fill(b.att.w_query, &rng);
            fill(b.att.w_key, &rng);
            fill(b.att.w_value, &rng);
            fill(b.att.out_weight, &rng);
            fill(b.ff.expand.w, &rng);
            fill(b.ff.shrink.w, &rng);
        }
        fill(m.out_head, &rng);
        return m;
    }

    fn fill(w: Matrix, rng: *Rng) void {
        for (w.data) |*v| v.* = rng.normal() * 0.02;
    }

    /// Token embedding + position embedding: what the model feeds its first block.
    pub fn embed(self: GptModel, gpa: std.mem.Allocator, ids: []const u32) !Matrix {
        const d = self.cfg.emb_dim;
        const x = try Matrix.init(gpa, ids.len, d);
        for (ids, 0..) |id, t| {
            const tok = self.tok_emb.row(id);
            const pos = self.pos_emb.row(t);
            for (0..d) |j| x.data[t * d + j] = tok[j] + pos[j];
        }
        return x;
    }

    /// Token IDs [T] in, logits [T, vocab] out.
    pub fn forward(self: GptModel, gpa: std.mem.Allocator, ids: []const u32) !Matrix {
        var x = try self.embed(gpa, ids);
        for (self.blocks) |b| x = try b.forward(gpa, x);
        return (try self.final_norm.forward(gpa, x)).matmul(gpa, self.out_head);
    }

    /// Every weight as a flat slice (they share memory with the model), in the canonical order
    /// shared by all languages (see train/checkpoint.zig): token emb, position emb, then per block
    /// wq, wk, wv, [bq, bk, bv], wo, bo, ln1 scale, ln1 shift, ff1 w, ff1 b, ff2 w, ff2 b, ln2 scale,
    /// ln2 shift; then the final norm scale and shift and the output head.
    pub fn tensors(self: GptModel, gpa: std.mem.Allocator) ![][]f32 {
        var list = std.ArrayList([]f32).empty;
        try list.append(gpa, self.tok_emb.data);
        try list.append(gpa, self.pos_emb.data);
        for (self.blocks) |b| {
            const a = b.att;
            try list.appendSlice(gpa, &.{ a.w_query.data, a.w_key.data, a.w_value.data });
            if (a.qkv_bias) |qb| try list.appendSlice(gpa, &.{ qb[0].data, qb[1].data, qb[2].data });
            try list.appendSlice(gpa, &.{ a.out_weight.data, a.out_bias.data, b.norm1.scale, b.norm1.shift });
            try list.appendSlice(gpa, &.{ b.ff.expand.w.data, b.ff.expand.b, b.ff.shrink.w.data, b.ff.shrink.b, b.norm2.scale, b.norm2.shift });
        }
        try list.appendSlice(gpa, &.{ self.final_norm.scale, self.final_norm.shift, self.out_head.data });
        return list.items;
    }

    /// A model of the same shape with every number 0 (gradients and optimizer state live in one).
    pub fn zerosLike(self: GptModel, gpa: std.mem.Allocator) !GptModel {
        const z = try init(gpa, self.cfg);
        for (try z.tensors(gpa)) |t| @memset(t, 0);
        return z;
    }

    /// Add up every weight the model holds.
    pub fn countParameters(self: GptModel) usize {
        var n = self.tok_emb.data.len + self.pos_emb.data.len + self.out_head.data.len + 2 * self.final_norm.scale.len;
        for (self.blocks) |b| {
            const a = b.att;
            n += a.w_query.data.len + a.w_key.data.len + a.w_value.data.len + a.out_weight.data.len + a.out_bias.data.len;
            if (a.qkv_bias) |qb| n += qb[0].data.len + qb[1].data.len + qb[2].data.len;
            n += b.ff.expand.w.data.len + b.ff.expand.b.len + b.ff.shrink.w.data.len + b.ff.shrink.b.len;
            n += 2 * (b.norm1.scale.len + b.norm2.scale.len);
        }
        return n;
    }
};

/// The same count by arithmetic only (no model is built).
pub fn parameterCount(cfg: Config) usize {
    const v = cfg.vocab_size;
    const c = cfg.context_length;
    const d = cfg.emb_dim;
    const n = cfg.n_layers;
    const attention = 4 * d * d + d + (if (cfg.qkv_bias) 3 * d else 0); // wq, wk, wv, wo, the output bias (+ qkv biases)
    const feed_forward = d * 4 * d + 4 * d + 4 * d * d + d;
    const norms = 2 * 2 * d; // two layer norms per block, scale + shift each
    return v * d + c * d + n * (attention + feed_forward + norms) + 2 * d + d * v;
}

/// Greedy decoding: `ids` are the tokens so far; the result has new IDs appended.
/// Every forward pass gets its own scratch arena that is freed right after.
pub fn generateTextSimple(gpa: std.mem.Allocator, model: GptModel, ids: []const u32, max_new_tokens: usize, context_size: usize) ![]u32 {
    var out = try std.ArrayList(u32).initCapacity(gpa, ids.len + max_new_tokens);
    out.appendSliceAssumeCapacity(ids);
    for (0..max_new_tokens) |_| {
        const start = if (out.items.len > context_size) out.items.len - context_size else 0; // keep only what fits
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const logits = try model.forward(scratch.allocator(), out.items[start..]);
        const last = logits.row(logits.rows - 1); // scores for the NEXT token only
        // Softmax would turn these into probabilities, but it keeps the order,
        // so the biggest score is also the most likely token.
        var best: usize = 0;
        for (last, 0..) |v, i| {
            if (v > last[best]) best = i;
        }
        out.appendAssumeCapacity(@intCast(best));
    }
    return out.items;
}

test "parameter count matches the book: 163,009,536 for GPT-2 small" {
    try std.testing.expectEqual(@as(usize, 163_009_536), parameterCount(config_124m));
}

test "a tiny model counts the same by building and by arithmetic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const m = try GptModel.init(arena.allocator(), config_tiny);
    try std.testing.expectEqual(parameterCount(config_tiny), m.countParameters());
}
