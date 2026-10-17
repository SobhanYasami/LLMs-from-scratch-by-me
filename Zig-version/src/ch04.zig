//! Chapter 4 demos: the GPT model, one building block at a time (book 4.1 to 4.7).
//!
//! Run from Zig-version/:  `zig build run -- ch04`
//!                         `zig build run -Doptimize=ReleaseFast -- ch04-124m`   (the real 124M model)
//!
//! Tip: read the demos top to bottom. Numbers in 4.2, 4.3 and 4.4 match the book
//! (the 4.4 weights are the book's torch.manual_seed(123) ones, written out so
//! every language shares them). The model demos use random weights from the repo's
//! shared generator, so all five languages print the same numbers; they differ from
//! the book's torch output, and since the model is untrained its text is gibberish.

const std = @import("std");
const att = @import("attention/root.zig");
const gpt = @import("gpt/root.zig");
const tk = @import("tokenization/root.zig");
const Rng = @import("tokenization/rng.zig").Rng;

const Matrix = att.Matrix;
const Out = *std.Io.Writer;

const ranks_path = "../assets/gpt2-ranks.txt";

/// Book 4.1: two short texts, already turned into token IDs by the GPT-2 tokenizer.
const batch = [2][4]u32{ .{ 6109, 3626, 6100, 345 }, .{ 6109, 1110, 6622, 257 } }; // "Every effort moves you" / "Every day holds a"

/// Book 4.2: the outputs of a Linear(5, 6) + ReLU layer for two inputs.
const layer_out_data = [_]f32{
    0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000,
    0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000,
};

/// Book 4.4: weights and biases of five Linear layers (3->3 four times, then 3->1), torch.manual_seed(123).
const shortcut_layers = [_]gpt.ExampleLayer{
    .{ .out = 3, .in = 3, .w = &.{ -0.23542964, 0.01912448, -0.28674594, 0.21772662, -0.49193421, 0.42322308, -0.41964141, -0.45901766, -0.36482018 }, .b = &.{ 0.26147819, -0.21332639, 0.21605217 } },
    .{ .out = 3, .in = 3, .w = &.{ -0.49001414, -0.35029206, -0.21198919, -0.11346072, -0.44043937, 0.37804362, -0.13615717, 0.18532233, 0.40826949 }, .b = &.{ 0.10756382, 0.15787685, 0.55729234 } },
    .{ .out = 3, .in = 3, .w = &.{ -0.26039040, 0.18287641, -0.25687245, 0.41260317, 0.46110451, -0.53230095, 0.49285263, 0.27569306, 0.25159022 }, .b = &.{ 0.23768058, 0.47995073, -0.07623307 } },
    .{ .out = 3, .in = 3, .w = &.{ -0.48826423, -0.16567004, -0.40661314, 0.03816671, -0.10779562, -0.30968198, -0.05249349, 0.54698384, -0.04553875 }, .b = &.{ 0.01833092, -0.09004387, 0.09076351 } },
    .{ .out = 1, .in = 3, .w = &.{ 0.51442641, 0.35304558, 0.20493712 }, .b = &.{0.12546431} },
};

/// 4 decimals; a tiny negative number is just zero.
fn print4(out: Out, v: f32) !void {
    var buf: [32]u8 = undefined;
    const s = try std.fmt.bufPrint(&buf, "{d:.4}", .{v});
    try out.writeAll(if (std.mem.eql(u8, s, "-0.0000")) "0.0000" else s);
}

fn printRow(out: Out, values: []const f32) !void {
    try out.writeAll("[");
    for (values, 0..) |v, i| {
        if (i > 0) try out.writeAll(", ");
        try print4(out, v);
    }
    try out.writeAll("]");
}

fn show4(out: Out, label: []const u8, values: []const f32) !void {
    try out.print("{s}: ", .{label});
    try printRow(out, values);
    try out.writeAll("\n");
}

/// 1234567 -> "1,234,567".
fn commas(buf: []u8, n: usize) ![]const u8 {
    var digits: [32]u8 = undefined;
    const s = try std.fmt.bufPrint(&digits, "{d}", .{n});
    var o: usize = 0;
    for (s, 0..) |c, i| {
        if (i > 0 and (s.len - i) % 3 == 0) {
            buf[o] = ',';
            o += 1;
        }
        buf[o] = c;
        o += 1;
    }
    return buf[0..o];
}

/// A float with two decimals and thousands separators.
fn commasF(buf: []u8, v: f64) ![]const u8 {
    var s: [48]u8 = undefined;
    const text = try std.fmt.bufPrint(&s, "{d:.2}", .{v});
    const dot = std.mem.indexOfScalar(u8, text, '.').?;
    var head: [32]u8 = undefined;
    const int_part = try commas(&head, try std.fmt.parseInt(usize, text[0..dot], 10));
    return std.fmt.bufPrint(buf, "{s}{s}", .{ int_part, text[dot..] });
}

fn printIds(out: Out, ids: []const u32) !void {
    try out.writeAll("[");
    for (ids, 0..) |id, i| {
        if (i > 0) try out.writeAll(", ");
        try out.print("{d}", .{id});
    }
    try out.writeAll("]");
}

fn build(gpa: std.mem.Allocator, cfg: gpt.Config) !gpt.GptModel {
    return gpt.GptModel.initRandom(gpa, cfg, 123);
}

/// 4.1: token IDs in, one score per vocabulary word out, for every position.
fn demoBigPicture(gpa: std.mem.Allocator, out: Out) !void {
    try out.writeAll("== 4.1 the big picture (tiny GPT: emb 32, 4 heads, 2 blocks) ==\n");
    try out.writeAll("batch of token IDs, shape (2, 4): [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]\n");
    const model = try build(gpa, gpt.config_tiny);
    const logits = try model.forward(gpa, &batch[0]); // the batch is processed one sequence at a time
    try out.print("logits shape: (2, {d}, {d})  <- [batch, tokens, vocab_size]\n", .{ logits.rows, logits.cols });
    try show4(out, "logits[0][0], first 4 of 50257", logits.row(0)[0..4]);
}

/// 4.2: make each token's numbers mean 0, variance 1.
fn demoLayerNorm(gpa: std.mem.Allocator, out: Out) !void {
    try out.writeAll("== 4.2 layer normalization ==\n");
    const x = try Matrix.fromSlice(gpa, 2, 6, &layer_out_data);
    var means: [2]f32 = undefined;
    var vars: [2]f32 = undefined;
    for (0..2) |i| {
        const r = x.row(i);
        var sum: f32 = 0;
        for (r) |v| sum += v;
        means[i] = sum / 6;
        var sq: f32 = 0;
        for (r) |v| sq += (v - means[i]) * (v - means[i]);
        vars[i] = sq / 5; // the book first uses the unbiased variance
    }
    var norm0: [6]f32 = undefined;
    for (x.row(0), 0..) |v, j| norm0[j] = (v - means[0]) / @as(f32, @floatCast(@sqrt(@as(f64, vars[0]))));
    try show4(out, "mean per row", &means);
    try show4(out, "variance per row", &vars);
    try show4(out, "normalized, row 0", &norm0);
    const ln = try gpt.LayerNorm.init(gpa, 6);
    const y = try ln.forward(gpa, x);
    var out_mean: [2]f32 = undefined;
    var out_var: [2]f32 = undefined;
    for (0..2) |i| {
        var sum: f32 = 0;
        for (y.row(i)) |v| sum += v;
        out_mean[i] = sum / 6;
        var sq: f32 = 0;
        for (y.row(i)) |v| sq += (v - out_mean[i]) * (v - out_mean[i]);
        out_var[i] = sq / 6;
    }
    try show4(out, "LayerNorm output mean", &out_mean);
    try show4(out, "LayerNorm output variance (biased)", &out_var);
}

/// 4.3: a smooth ReLU, and the expand-then-shrink feed forward block.
fn demoGeluAndFeedForward(gpa: std.mem.Allocator, out: Out) !void {
    try out.writeAll("== 4.3 GELU and the feed forward network ==\n");
    var x: [7]f32 = undefined;
    var g: [7]f32 = undefined;
    var r: [7]f32 = undefined;
    for (0..7) |i| {
        x[i] = @as(f32, @floatFromInt(i)) - 3;
        g[i] = gpt.gelu(x[i]);
        r[i] = @max(0, x[i]);
    }
    try show4(out, "x   ", &x);
    try show4(out, "GELU", &g);
    try show4(out, "ReLU", &r);
    var rng = Rng.init(123);
    var ffn = try gpt.FeedForward.init(gpa, 32);
    ffn.expand.w = try Matrix.randn(gpa, 32, 128, 0.02, &rng);
    ffn.shrink.w = try Matrix.randn(gpa, 128, 32, 0.02, &rng);
    const y = try ffn.forward(gpa, try Matrix.rand(gpa, 3, 32, &rng));
    try out.print("feed forward: input (2, 3, 32) -> output (2, {d}, {d}) (hidden layer is 4 x 32 = 128 wide)\n", .{ y.rows, y.cols });
}

/// 4.4: shortcut connections keep the gradient alive.
fn demoShortcut(out: Out) !void {
    try out.writeAll("== 4.4 shortcut connections (mean |gradient| of each layer's weights) ==\n");
    for ([_]bool{ false, true }) |use_shortcut| {
        const net = gpt.ExampleDeepNeuralNetwork{ .layers = &shortcut_layers, .use_shortcut = use_shortcut };
        var grads: [5]f64 = undefined;
        net.gradientMeans(&.{ 1, 0, -1 }, 0, &grads);
        try out.print("{s} shortcut: ", .{if (use_shortcut) "with   " else "without"});
        for (grads, 0..) |gr, i| {
            if (i > 0) try out.writeAll("  ");
            try out.print("{d:.6}", .{gr});
        }
        try out.writeAll("   (layer 0 ... layer 4)\n");
    }
}

/// 4.5: attention + feed forward, each wrapped in norm and shortcut. Shape in = shape out.
fn demoTransformerBlock(gpa: std.mem.Allocator, out: Out) !void {
    try out.writeAll("== 4.5 the transformer block ==\n");
    const model = try build(gpa, gpt.config_tiny);
    const x = try model.embed(gpa, &batch[0]);
    const y = try model.blocks[0].forward(gpa, x);
    try out.print("input shape: (2, {d}, {d})  output shape: (2, {d}, {d})\n", .{ x.rows, x.cols, y.rows, y.cols });
    try show4(out, "input[0][0], first 4 numbers", x.row(0)[0..4]);
    try show4(out, "output[0][0], first 4 numbers", y.row(0)[0..4]);
}

fn mb(n: usize) f64 {
    return @as(f64, @floatFromInt(n)) * 4 / (1024.0 * 1024.0);
}

/// 4.6: how big is the model?
fn demoParameters(gpa: std.mem.Allocator, out: Out) !void {
    try out.writeAll("== 4.6 model size ==\n");
    var b1: [48]u8 = undefined;
    var b2: [48]u8 = undefined;
    var b3: [48]u8 = undefined;
    const tiny = try build(gpa, gpt.config_tiny);
    try out.print("tiny model: {s} parameters (arithmetic says {s})\n", .{ try commas(&b1, tiny.countParameters()), try commas(&b2, gpt.parameterCount(gpt.config_tiny)) });
    const cfg = gpt.config_124m;
    const total = gpt.parameterCount(cfg);
    const head = cfg.vocab_size * cfg.emb_dim;
    try out.print("GPT-2 small, no model built: {s} parameters\n", .{try commas(&b1, total)});
    try out.print("  minus the output head (weight tying in the original GPT-2): {s}\n", .{try commas(&b1, total - head)});
    try out.print("  float32 size: {d:.2} MB\n", .{mb(total)});
    const d = cfg.emb_dim;
    const ff = d * 4 * d + 4 * d + 4 * d * d + d;
    const att_params = 4 * d * d + d;
    try out.print("exercise 4.1: feed forward {s} vs attention {s} parameters per block\n", .{ try commas(&b1, ff), try commas(&b2, att_params) });
    try out.writeAll("exercise 4.2: bigger models, same code, different configuration\n");
    try out.print("  {s:<20} {s:>14} {s:>14} {s:>11}\n", .{ "", "parameters", "weight-tied", "float32 MB" });
    for (gpt.sizes) |s| {
        var c = cfg;
        c.emb_dim = s.emb_dim;
        c.n_layers = s.n_layers;
        c.n_heads = s.n_heads;
        const n = gpt.parameterCount(c);
        try out.print("  {s:<20} {s:>14} {s:>14} {s:>11}\n", .{ s.name, try commas(&b1, n), try commas(&b2, n - c.vocab_size * c.emb_dim), try commasF(&b3, mb(n)) });
    }
}

/// 4.7: predict one token, append it, repeat.
fn demoGenerate(gpa: std.mem.Allocator, out: Out, cfg: gpt.Config, label: []const u8) !void {
    try out.print("== 4.7 generating text ({s}) ==\n", .{label});
    const model = try build(gpa, cfg);
    const ranks = try std.fs.cwd().readFileAlloc(ranks_path, gpa, .limited(8 << 20));
    const bpe = try tk.Bpe.init(gpa, ranks);
    const encoded = try bpe.encode(gpa, "Hello, I am");
    try out.writeAll("encoded: ");
    try printIds(out, encoded);
    try out.writeAll("\n");
    const ids = try gpt.generateTextSimple(gpa, model, encoded, 6, cfg.context_length);
    try out.writeAll("output IDs: ");
    try printIds(out, ids);
    try out.print("\noutput length: {d}\n", .{ids.len});
    try out.print("decoded: {s}\n", .{try bpe.decode(gpa, ids)});
    try out.writeAll("(gibberish: the weights are random, nothing has been trained yet)\n");
}

pub fn run(gpa: std.mem.Allocator, out: Out) !void {
    try demoBigPicture(gpa, out);
    try demoLayerNorm(gpa, out);
    try demoGeluAndFeedForward(gpa, out);
    try demoShortcut(out);
    try demoTransformerBlock(gpa, out);
    try demoParameters(gpa, out);
    try demoGenerate(gpa, out, gpt.config_tiny, "tiny model");
}

/// The real 124M-parameter model: forward pass, then generation. Needs ~1 GB of memory.
pub fn run124m(gpa: std.mem.Allocator, out: Out) !void {
    const model = try build(gpa, gpt.config_124m);
    var buf: [48]u8 = undefined;
    try out.writeAll("== GPT-2 small (124M) ==\n");
    try out.print("parameters built: {s}\n", .{try commas(&buf, model.countParameters())});
    const logits = try model.forward(gpa, &batch[0]);
    try out.print("logits shape: (2, {d}, {d})\n", .{ logits.rows, logits.cols });
    try show4(out, "logits[0][0], first 4 of 50257", logits.row(0)[0..4]);
    try demoGenerate(gpa, out, gpt.config_124m, "124M model");
}
