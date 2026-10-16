//! Chapter 3 demos: attention, one idea at a time (book 3.3 to 3.6).
//!
//! Run from Zig-version/:  `zig build run -- ch03`
//!
//! Tip: read the demos top to bottom; each builds on the one before. The numbers
//! in 3.3 and 3.4 match the book exactly (3.4 uses the book's seed-123 weights,
//! hard-coded so every language shares them).

const std = @import("std");
const att = @import("attention/root.zig");
const Rng = @import("tokenization/rng.zig").Rng;

const Matrix = att.Matrix;
const Out = *std.Io.Writer;

/// "Your journey starts with one step." as 3-dimensional embeddings (book 3.3.1).
const inputs_data = [_]f32{
    0.43, 0.15, 0.89, // Your
    0.55, 0.87, 0.66, // journey
    0.57, 0.85, 0.64, // starts
    0.22, 0.58, 0.33, // with
    0.77, 0.25, 0.10, // one
    0.05, 0.80, 0.55, // step
};

// The book's torch.manual_seed(123) weights (d_in=3, d_out=2), written out so no language needs torch.
const w_query_data = [_]f32{ 0.29611194, 0.51656228, 0.25167072, 0.68855679, 0.07397246, 0.86652195 };
const w_key_data = [_]f32{ 0.13657987, 0.10247904, 0.18405646, 0.72644675, 0.31525391, 0.68710667 };
const w_value_data = [_]f32{ 0.07563531, 0.19663817, 0.31641197, 0.40174013, 0.11856830, 0.82739538 };

fn printRow(out: Out, values: []const f32) !void {
    try out.writeAll("[");
    for (values, 0..) |v, i| {
        if (i > 0) try out.writeAll(", ");
        try out.print("{d:.4}", .{v});
    }
    try out.writeAll("]");
}

fn showVec(out: Out, label: []const u8, values: []const f32) !void {
    try out.print("{s}: ", .{label});
    try printRow(out, values);
    try out.writeAll("\n");
}

fn showMatrix(out: Out, label: []const u8, m: Matrix) !void {
    try out.print("{s}:\n", .{label});
    for (0..m.rows) |i| {
        try out.writeAll("  ");
        try printRow(out, m.row(i));
        try out.writeAll("\n");
    }
}

/// 3.3: one query by hand, then all tokens at once.
fn demoSimpleAttention(gpa: std.mem.Allocator, out: Out) !void {
    const inputs = try Matrix.fromSlice(gpa, 6, 3, &inputs_data);
    try out.writeAll("== 3.3.1 simple attention for one query (x2, 'journey') ==\n");
    const query = try Matrix.fromSlice(gpa, 1, 3, inputs.row(1));
    const scores_2 = try query.matmul(gpa, try inputs.transpose(gpa)); // dot product of every token with the query
    const weights_2 = try scores_2.softmaxRows(gpa);
    const context_2 = try weights_2.matmul(gpa, inputs);
    try showVec(out, "scores    ", scores_2.row(0));
    try showVec(out, "weights   ", weights_2.row(0));
    try showVec(out, "context z2", context_2.row(0));

    try out.writeAll("== 3.3.2 all tokens at once ==\n");
    const r = try att.simpleAttention(gpa, inputs);
    try showMatrix(out, "scores (T x T)", r.scores);
    try showMatrix(out, "weights (rows sum to 1)", r.weights);
    var sums: [6]f32 = undefined;
    for (0..6) |i| {
        var total: f32 = 0;
        for (r.weights.row(i)) |w| total += w;
        sums[i] = total;
    }
    try showVec(out, "row sums", &sums);
    try showMatrix(out, "context vectors (T x d)", r.context);
}

/// 3.4: query/key/value with trainable (here: fixed) weights.
fn demoSelfAttention(gpa: std.mem.Allocator, out: Out) !void {
    const inputs = try Matrix.fromSlice(gpa, 6, 3, &inputs_data);
    try out.writeAll("== 3.4 self-attention with trainable weights (d_in=3, d_out=2) ==\n");
    const sa = att.SelfAttention{
        .w_query = try Matrix.fromSlice(gpa, 3, 2, &w_query_data),
        .w_key = try Matrix.fromSlice(gpa, 3, 2, &w_key_data),
        .w_value = try Matrix.fromSlice(gpa, 3, 2, &w_value_data),
    };
    const x2 = try Matrix.fromSlice(gpa, 1, 3, inputs.row(1));
    const query_2 = try x2.matmul(gpa, sa.w_query);
    const keys = try inputs.matmul(gpa, sa.w_key);
    const values = try inputs.matmul(gpa, sa.w_value);
    try showVec(out, "query_2", query_2.row(0));
    try out.print("keys shape: ({d}, {d}), values shape: ({d}, {d})\n", .{ keys.rows, keys.cols, values.rows, values.cols });
    const scores_2 = try query_2.matmul(gpa, try keys.transpose(gpa));
    try showVec(out, "scores_2 (q2 . keys)", scores_2.row(0));
    const weights_2 = try (try att.self_attention.scaleBySqrt(gpa, scores_2, keys.cols)).softmaxRows(gpa);
    try showVec(out, "weights_2 (scaled)", weights_2.row(0));
    try showVec(out, "context_2", (try weights_2.matmul(gpa, values)).row(0));
    try showMatrix(out, "all context vectors (T x d_out)", try sa.forward(gpa, inputs));
}

/// 3.5: mask the future, then dropout.
fn demoCausalAttention(gpa: std.mem.Allocator, out: Out) !void {
    const inputs = try Matrix.fromSlice(gpa, 6, 3, &inputs_data);
    try out.writeAll("== 3.5 causal attention ==\n");
    const ca = att.CausalAttention{ .base = .{
        .w_query = try Matrix.fromSlice(gpa, 3, 2, &w_query_data),
        .w_key = try Matrix.fromSlice(gpa, 3, 2, &w_key_data),
        .w_value = try Matrix.fromSlice(gpa, 3, 2, &w_value_data),
    } };
    try showMatrix(out, "causal weights (upper triangle is 0)", try ca.attentionWeights(gpa, inputs, null));
    try showMatrix(out, "causal context vectors", try ca.forward(gpa, inputs, null));
    const ones = try Matrix.init(gpa, 6, 6);
    @memset(ones.data, 1);
    var rng = Rng.init(123);
    try showMatrix(out, "dropout 0.5 on ones(6, 6), seed 123 (kept values are doubled)", try ones.dropout(gpa, 0.5, &rng));
}

/// 3.6: split the work across heads.
fn demoMultihead(gpa: std.mem.Allocator, out: Out) !void {
    const inputs = try Matrix.fromSlice(gpa, 6, 3, &inputs_data);
    try out.writeAll("== 3.6 multi-head attention (d_in=3, d_out=2, heads=2) ==\n");
    var rng = Rng.init(123);
    const mha = try att.MultiHeadAttention.random(gpa, 3, 2, 2, &rng);
    // A batch is just the same computation repeated per example; two copies = a batch of 2.
    const result = try mha.forward(gpa, inputs);
    try out.print("input shape: (2, {d}, {d})  output shape: (2, {d}, {d})\n", .{ inputs.rows, inputs.cols, result.rows, result.cols });
    try showMatrix(out, "output[0]", result);
}

pub fn run(gpa: std.mem.Allocator, out: Out) !void {
    try demoSimpleAttention(gpa, out);
    try demoSelfAttention(gpa, out);
    try demoCausalAttention(gpa, out);
    try demoMultihead(gpa, out);
}
