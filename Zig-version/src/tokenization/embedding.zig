//! Step 5 (book 2.7-2.8) - token and positional embeddings.
//!
//! A token ID is just an integer. An embedding layer is a lookup table with one
//! vector per ID, so every token becomes a vector the model can learn from:
//!
//! ```
//! ids [B][T]  --token embedding-->  [B][T][D]
//! positions 0..T-1 --pos embedding-> [T][D]  (added to every batch row)
//! input embeddings = token + position  -> [B][T][D]
//! ```
//!
//! Why add positions? Attention (later chapters) treats its input as a bag of
//! vectors, so word order must be put into the vectors themselves.
//!
//! No tensor library: a tensor is one flat `[]f32`, indexed [b][t][d] as
//! `(b * T + t) * D + d`. Weights are random (normal, mean 0, std 1) and untrained,
//! like torch's `nn.Embedding` at init.

const std = @import("std");
const Rng = @import("rng.zig").Rng;

/// A lookup table of `rows` vectors of size `dim`.
pub const Embedding = struct {
    rows: usize,
    dim: usize,
    weight: []f32, // rows x dim

    pub fn init(gpa: std.mem.Allocator, rows: usize, dim: usize, rng: *Rng) !Embedding {
        const weight = try gpa.alloc(f32, rows * dim);
        for (weight) |*w| w.* = rng.normal();
        return .{ .rows = rows, .dim = dim, .weight = weight };
    }

    pub fn deinit(self: *Embedding, gpa: std.mem.Allocator) void {
        gpa.free(self.weight);
    }

    /// Look up one vector per ID; `out` holds `ids.len * dim` floats.
    pub fn forward(self: Embedding, ids: []const u32, out: []f32) void {
        for (ids, 0..) |id, i| { // a lookup: copy row `id`
            @memcpy(out[i * self.dim ..][0..self.dim], self.weight[@as(usize, id) * self.dim ..][0..self.dim]);
        }
    }
};

/// Token embedding plus absolute positional embedding.
pub const InputEmbedding = struct {
    token: Embedding,
    pos: Embedding,

    pub fn init(gpa: std.mem.Allocator, vocab_size: usize, context_length: usize, dim: usize, seed: u64) !InputEmbedding {
        var rng = Rng.init(seed);
        var token = try Embedding.init(gpa, vocab_size, dim, &rng);
        errdefer token.deinit(gpa);
        const pos = try Embedding.init(gpa, context_length, dim, &rng);
        return .{ .token = token, .pos = pos };
    }

    pub fn deinit(self: *InputEmbedding, gpa: std.mem.Allocator) void {
        self.token.deinit(gpa);
        self.pos.deinit(gpa);
    }

    /// `ids` is [batch][seq_len] flat; `out` is [batch][seq_len][dim] flat.
    pub fn forward(self: InputEmbedding, ids: []const u32, batch: usize, seq_len: usize, out: []f32) void {
        const dim = self.token.dim;
        for (0..batch) |b| {
            const row = out[b * seq_len * dim ..][0 .. seq_len * dim];
            self.token.forward(ids[b * seq_len ..][0..seq_len], row);
            for (0..seq_len) |t| { // token + position
                for (0..dim) |d| row[t * dim + d] += self.pos.weight[t * dim + d];
            }
        }
    }
};

test "embedding forward is a row lookup" {
    const gpa = std.testing.allocator;
    var rng = Rng.init(1);
    var e = try Embedding.init(gpa, 6, 3, &rng);
    defer e.deinit(gpa);
    var out: [3]f32 = undefined;
    e.forward(&.{3}, &out);
    try std.testing.expectEqualSlices(f32, e.weight[9..12], &out);
}
