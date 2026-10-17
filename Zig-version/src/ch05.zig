//! Chapter 5 demos: pretraining, one step at a time (book 5.1 to 5.5).
//!
//! Run from Zig-version/:  `zig build run -Doptimize=ReleaseFast -- ch05`
//!                         `zig build run -Doptimize=ReleaseFast -- ch05-gpt2`   (OpenAI's weights, 5.5)
//!
//! Tip: read the demos top to bottom. The model is the tiny GPT of chapter 4 (emb 32, 2 blocks,
//! context 16) so that every language can TRAIN it in seconds; the book trains the 124M model on a
//! GPU. All five languages use the same weights, data order (no shuffling) and sampling generator,
//! so they print the same numbers. They differ from the book's torch output. Build in release mode:
//! training in Debug mode is much slower.

const std = @import("std");
const gpt = @import("gpt/root.zig");
const tk = @import("tokenization/root.zig");
const train = @import("train/root.zig");
const Rng = @import("tokenization/rng.zig").Rng;

const Out = *std.Io.Writer;
const Allocator = std.mem.Allocator;

const data_path = "../assets/the-verdict.txt";
const ranks_path = "../assets/gpt2-ranks.txt";
const gpt2_path = "../assets/gpt2/model.safetensors";
const ckpt_path = "../assets/checkpoints/zig.bin";
const batch_size = 4;
const epochs = 2;
const eval_freq = 24;
const eval_iter = 2;
const lr: f64 = 3e-3;
const weight_decay: f64 = 0.1;

/// The tiny GPT with dropout off: our loop has no random masks.
fn cfg5() gpt.Config {
    var c = gpt.config_tiny;
    c.drop_rate = 0;
    return c;
}

/// `digits` decimals; a tiny negative number is just zero.
fn printNum(out: Out, v: f32, digits: usize) !void {
    var buf: [64]u8 = undefined;
    const s = try std.fmt.bufPrint(&buf, "{d:.[1]}", .{ v, digits });
    var zero: [64]u8 = undefined;
    const z = try std.fmt.bufPrint(&zero, "-{d:.[1]}", .{ @as(f32, 0), digits });
    try out.writeAll(if (std.mem.eql(u8, s, z)) s[1..] else s);
}

fn printRow(out: Out, values: []const f32, digits: usize) !void {
    try out.writeAll("[");
    for (values, 0..) |v, i| {
        if (i > 0) try out.writeAll(", ");
        try printNum(out, v, digits);
    }
    try out.writeAll("]");
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

/// Text in single quotes with newlines flattened (same in every language).
fn printQuoted(out: Out, text: []const u8, quotes: bool) !void {
    if (quotes) try out.writeAll("'");
    for (text) |c| try out.writeByte(if (c == '\n') ' ' else c);
    if (quotes) try out.writeAll("'");
}

fn build(gpa: Allocator, cfg: gpt.Config) !gpt.GptModel {
    return gpt.GptModel.initRandom(gpa, cfg, 123);
}

const Data = struct { bpe: tk.Bpe, text: []const u8, split: usize };

fn loadData(gpa: Allocator) !Data {
    const ranks = try std.fs.cwd().readFileAlloc(ranks_path, gpa, .limited(8 << 20));
    const text = try std.fs.cwd().readFileAlloc(data_path, gpa, .limited(1 << 20));
    return .{ .bpe = try tk.Bpe.init(gpa, ranks), .text = text, .split = @intFromFloat(0.9 * @as(f64, @floatFromInt(text.len))) };
}

/// The batches of one text, in book order: windows cut and grouped like chapter 2's data loader (no shuffling).
fn makeBatches(gpa: Allocator, bpe: tk.Bpe, text: []const u8, ctx: usize, rows_per_batch: usize, drop_last: bool) ![]train.Batch {
    const ids = try bpe.encode(gpa, text);
    const ds = tk.GptDataset.init(ids, ctx, ctx); // stride = window size: no overlap
    var dl = try tk.DataLoader.init(gpa, &ds, rows_per_batch, false, drop_last, 123);
    const n = dl.startEpoch();
    const batches = try gpa.alloc(train.Batch, n);
    const flat_in = try gpa.alloc(u32, rows_per_batch * ctx);
    const flat_out = try gpa.alloc(u32, rows_per_batch * ctx);
    for (batches, 0..) |*b, bi| {
        const rows = dl.batch(bi, flat_in, flat_out);
        const inputs = try gpa.alloc([]const u32, rows);
        const targets = try gpa.alloc([]const u32, rows);
        for (0..rows) |r| {
            inputs[r] = try gpa.dupe(u32, flat_in[r * ctx .. (r + 1) * ctx]);
            targets[r] = try gpa.dupe(u32, flat_out[r * ctx .. (r + 1) * ctx]);
        }
        b.* = .{ .inputs = inputs, .targets = targets };
    }
    return batches;
}

/// 5.1: from probabilities to a single number.
fn demoLoss(gpa: Allocator, out: Out, d: Data) !void {
    try out.writeAll("== 5.1 evaluating generative text models ==\n");
    const cfg = cfg5();
    const model = try build(gpa, cfg);
    const inputs = [2][3]u32{ .{ 16833, 3626, 6100 }, .{ 40, 1107, 588 } }; // ["every effort moves", "I really like"]
    const targets = [2][3]u32{ .{ 3626, 6100, 345 }, .{ 588, 428, 11311 } }; // [" effort moves you", " really like chocolate"]
    try out.writeAll("inputs : [[16833, 3626, 6100], [40, 1107, 588]]\n");
    try out.writeAll("targets: [[3626, 6100, 345], [588, 428, 11311]]\n");
    try out.print("probas shape: (2, 3, {d})  <- [batch, tokens, vocab_size]\n", .{cfg.vocab_size});
    var log_probas: [6]f32 = undefined;
    var ce_total: f32 = 0;
    for (0..2) |i| {
        const logits = try model.forward(gpa, &inputs[i]);
        ce_total += train.loss.crossEntropySum(logits, &targets[i]);
        var picked: [3]f32 = undefined;
        for (0..3) |t| {
            picked[t] = (try train.sample.softmax(gpa, logits.row(t)))[targets[i][t]];
            log_probas[i * 3 + t] = @floatCast(@log(@as(f64, picked[t])));
        }
        try out.print("probability of the target, text {d}: ", .{i + 1});
        try printRow(out, &picked, 6);
        try out.writeAll("\n");
    }
    try out.writeAll("log probabilities: ");
    try printRow(out, &log_probas, 4);
    var sum: f32 = 0;
    for (log_probas) |v| sum += v;
    const avg = sum / 6;
    try out.writeAll("\naverage log probability: ");
    try printNum(out, avg, 4);
    try out.writeAll("\nnegative average (the loss): ");
    try printNum(out, -avg, 4);
    const loss = ce_total / 6;
    try out.writeAll("\ncross_entropy of the whole batch at once: ");
    try printNum(out, loss, 4);
    try out.writeAll("\nperplexity: ");
    try printNum(out, train.loss.perplexity(loss), 0);
    try out.writeAll("  (a model that guesses uniformly is torn between 50,257 tokens)\n");
    const tr = try makeBatches(gpa, d.bpe, d.text[0..d.split], cfg.context_length, batch_size, true);
    const va = try makeBatches(gpa, d.bpe, d.text[d.split..], cfg.context_length, batch_size, false);
    try out.print("training batches: {d} of {d} x {d} tokens, validation batches: {d}\n", .{ tr.len, batch_size, cfg.context_length, va.len });
    try out.writeAll("untrained: training loss ");
    try printNum(out, try train.loss.calcLossLoader(model, tr, 0), 3);
    try out.writeAll(", validation loss ");
    try printNum(out, try train.loss.calcLossLoader(model, va, 0), 3);
    try out.writeAll("  (a uniform guess gives ln(50257) = ");
    try printNum(out, @floatCast(@log(50257.0)), 3);
    try out.writeAll(")\n");
}

fn generateAndPrintSample(gpa: Allocator, out: Out, model: gpt.GptModel, bpe: tk.Bpe, start: []const u8) !void {
    const ids = try bpe.encode(gpa, start);
    const generated = try gpt.generateTextSimple(gpa, model, ids, 20, model.cfg.context_length);
    try printQuoted(out, try bpe.decode(gpa, generated), false);
    try out.writeAll("\n");
}

const Trainer = struct {
    model: gpt.GptModel,
    grads: gpt.GptModel,
    opt: train.AdamW,

    /// One training step: zero the old gradients, forward and backward, update the weights.
    fn step(self: *Trainer, gpa: Allocator, batch: train.Batch) !f32 {
        try train.grad.zeroGrads(gpa, self.grads);
        const l = try train.grad.lossAndGrads(self.model, self.grads, batch);
        try self.opt.step(gpa, self.model, self.grads);
        return l;
    }
};

fn newTrainer(gpa: Allocator, model: gpt.GptModel) !Trainer {
    return .{ .model = model, .grads = try model.zerosLike(gpa), .opt = try train.AdamW.init(gpa, model, lr, weight_decay) };
}

/// 5.2: train for a few epochs.
fn trainDemo(gpa: Allocator, out: Out, d: Data) !Trainer {
    try out.writeAll("== 5.2 training an LLM ==\n");
    var t = try newTrainer(gpa, try build(gpa, cfg5()));
    const tr = try makeBatches(gpa, d.bpe, d.text[0..d.split], t.model.cfg.context_length, batch_size, true);
    const va = try makeBatches(gpa, d.bpe, d.text[d.split..], t.model.cfg.context_length, batch_size, false);
    var global_step: usize = 0;
    for (0..epochs) |epoch| {
        for (tr) |b| {
            _ = try t.step(gpa, b);
            if (global_step % eval_freq == 0) {
                try out.print("Ep {d} (Step {d:0>6}): Train loss ", .{ epoch + 1, global_step });
                try printNum(out, try train.loss.calcLossLoader(t.model, tr, eval_iter), 3);
                try out.writeAll(", Val loss ");
                try printNum(out, try train.loss.calcLossLoader(t.model, va, eval_iter), 3);
                try out.writeAll("\n");
            }
            global_step += 1;
        }
        try generateAndPrintSample(gpa, out, t.model, d.bpe, "Every effort moves you");
    }
    return t;
}

const vocab = [9][]const u8{ "closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you" };
const next_token_logits = [9]f32{ 4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79 };

/// 5.3: temperature and top-k.
fn demoDecoding(gpa: Allocator, out: Out, t: Trainer, d: Data) !void {
    try out.writeAll("== 5.3 decoding strategies ==\n");
    const probas = try train.sample.softmax(gpa, &next_token_logits);
    try out.writeAll("next-token probabilities: ");
    try printRow(out, probas, 4);
    var greedy_pick: usize = 0;
    for (probas, 0..) |p, i| {
        if (p > probas[greedy_pick]) greedy_pick = i;
    }
    try out.print("\ngreedy pick: {s}\n", .{vocab[greedy_pick]});
    const labels = [3][]const u8{ "1", "0.1", "5" };
    const temps = [3]f32{ 1.0, 0.1, 5.0 };
    for (labels, temps) |label, temp| {
        const p = try train.sample.softmaxWithTemperature(gpa, &next_token_logits, temp);
        var rng = Rng.init(123);
        var counts = [_]usize{0} ** 9;
        for (0..1000) |_| counts[train.sample.sampleIndex(p, &rng)] += 1;
        try out.print("temperature {s}: P(pizza) = ", .{label});
        try printNum(out, p[6], 4);
        try out.writeAll("  1000 draws: ");
        var first = true;
        for (counts, vocab) |c, w| {
            if (c == 0) continue;
            try out.print("{s}{d} x {s}", .{ if (first) "" else ", ", c, w });
            first = false;
        }
        try out.writeAll("\n");
    }
    var order = [9]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    for (0..9) |i| { // a tiny selection sort, biggest logit first
        for (i + 1..9) |j| {
            if (next_token_logits[order[j]] > next_token_logits[order[i]]) std.mem.swap(usize, &order[i], &order[j]);
        }
    }
    const top = [3]f32{ next_token_logits[order[0]], next_token_logits[order[1]], next_token_logits[order[2]] };
    try out.writeAll("top-3 logits: ");
    try printRow(out, &top, 2);
    try out.print(" at positions [{d}, {d}, {d}]\n", .{ order[0], order[1], order[2] });
    const masked = try train.sample.topKLogits(gpa, &next_token_logits, 3);
    try out.writeAll("after masking with -inf: [");
    for (masked, 0..) |v, i| {
        if (i > 0) try out.writeAll(", ");
        try out.writeAll("'");
        if (std.math.isInf(v)) try out.writeAll("-inf") else try printNum(out, v, 2);
        try out.writeAll("'");
    }
    try out.writeAll("]\ntop-3 probabilities: ");
    try printRow(out, try train.sample.softmax(gpa, masked), 4);
    const start = try d.bpe.encode(gpa, "Every effort moves you");
    const ctx = t.model.cfg.context_length;
    const greedy = try gpt.generateTextSimple(gpa, t.model, start, 15, ctx);
    try out.writeAll("\ngreedy   : ");
    try printQuoted(out, try d.bpe.decode(gpa, greedy), true);
    var rng = Rng.init(123);
    const sampled = try train.sample.generate(gpa, t.model, start, 15, ctx, &rng, 1.4, 25);
    try out.writeAll("\ntop-k 25, temperature 1.4: ");
    try printQuoted(out, try d.bpe.decode(gpa, sampled), true);
    var rng0 = Rng.init(123);
    const same = try train.sample.generate(gpa, t.model, start, 15, ctx, &rng0, 0.0, 0);
    try out.print("\ntemperature 0 equals greedy: {}\n", .{std.mem.eql(u32, same, greedy)});
}

/// 5.4: save, load, and carry on exactly where we stopped.
fn demoCheckpoint(gpa: Allocator, out: Out, t: *Trainer, d: Data) !void {
    try out.writeAll("== 5.4 saving and loading ==\n");
    try train.checkpoint.save(gpa, ckpt_path, t.model, &t.opt);
    const stat = try std.fs.cwd().statFile(ckpt_path);
    var buf: [32]u8 = undefined;
    try out.print("saved model and optimizer state: {s} bytes\n", .{try commas(&buf, stat.size)});
    const loaded = try train.checkpoint.load(gpa, ckpt_path, lr, weight_decay);
    var r = Trainer{ .model = loaded.model, .grads = try loaded.model.zerosLike(gpa), .opt = loaded.opt.? };
    const x = [4]u32{ 6109, 3626, 6100, 345 };
    const a = try t.model.forward(gpa, &x);
    const b = try r.model.forward(gpa, &x);
    var diff: f32 = 0;
    for (a.data, b.data) |p, q| diff = @max(diff, @abs(p - q));
    try out.writeAll("largest logit difference between the two models: ");
    try printNum(out, diff, 6);
    const tr = try makeBatches(gpa, d.bpe, d.text[0..d.split], t.model.cfg.context_length, batch_size, true);
    var continued: [3]f32 = undefined;
    var resumed: [3]f32 = undefined;
    for (0..3) |i| continued[i] = try t.step(gpa, tr[i]);
    for (0..3) |i| resumed[i] = try r.step(gpa, tr[i]);
    try out.writeAll("\n3 more steps, continued: losses ");
    try printRow(out, &continued, 3);
    try out.writeAll("\n3 more steps, restored : losses ");
    try printRow(out, &resumed, 3);
    try out.print("\nsame losses: {}\n", .{std.mem.eql(f32, &continued, &resumed)});
}

pub fn run(gpa: Allocator, out: Out) !void {
    const d = try loadData(gpa);
    try demoLoss(gpa, out, d);
    var t = try trainDemo(gpa, out, d);
    try demoDecoding(gpa, out, t, d);
    try demoCheckpoint(gpa, out, &t, d);
}

/// 5.5: OpenAI's GPT-2 small, loaded into our model. Needs assets/gpt2/model.safetensors.
pub fn runGpt2(gpa: Allocator, out: Out) !void {
    const d = try loadData(gpa);
    try out.writeAll("== 5.5 loading pretrained weights from OpenAI ==\n");
    const model = train.openai.loadGpt2Weights(gpa, gpt2_path) catch |err| {
        std.debug.print("could not load {s}: {s} (download the weights first: command in src/train/openai.zig)\n", .{ gpt2_path, @errorName(err) });
        return;
    };
    var buf: [32]u8 = undefined;
    try out.print("loaded GPT-2 small: {s} parameters (with the attention biases)\n", .{try commas(&buf, model.countParameters())});
    const start = try d.bpe.encode(gpa, "Every effort moves you");
    const ctx = train.openai.gpt2_small.context_length;
    var rng = Rng.init(123);
    const generated = try train.sample.generate(gpa, model, start, 25, ctx, &rng, 1.5, 50);
    try out.writeAll("generated: ");
    try printQuoted(out, try d.bpe.decode(gpa, generated), true);
    const greedy = try gpt.generateTextSimple(gpa, model, start, 25, ctx);
    try out.writeAll("\ngreedy   : ");
    try printQuoted(out, try d.bpe.decode(gpa, greedy), true);
    const val = try makeBatches(gpa, d.bpe, d.text[d.split..], 64, 2, true);
    try out.writeAll("\nexercise 5.5: loss of the pretrained model on The Verdict (validation part, 2 batches of 2 x 64 tokens): ");
    try printNum(out, try train.loss.calcLossLoader(model, val, 2), 3);
    try out.writeAll("\n");
}
