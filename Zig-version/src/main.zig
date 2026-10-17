//! Demo runner for the educational tokenizers and data sampling.
//!
//! Run from Zig-version/:
//!
//!     zig build run              # all chapters
//!     zig build run -- ch04      # one chapter
//!
//! Tip: read `main()` first, then jump into `tokenization/` following each demo.
//!
//! Pipeline mirrors python-version/main.py:
//!     raw text -> splitText -> Vocab.build -> SimpleTokenizer V1/V2 -> BPE (GPT-2)
//!     -> sliding-window dataset -> data loader -> embeddings

const std = @import("std");
const tk = @import("tokenization/root.zig");
const ch03 = @import("ch03.zig");
const ch04 = @import("ch04.zig");
const ch05 = @import("ch05.zig");

/// Run from Zig-version/, like go-version resolves ../assets.
const data_path = "../assets/the-verdict.txt";
const ranks_path = "../assets/gpt2-ranks.txt";

const v1_text = "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride.";
const v2_text = "Hello, do you like tea? " ++ tk.END_OF_TEXT ++ " In the sunlit terraces of the palace.";

const Out = *std.Io.Writer;

fn printIdsInline(out: Out, ids: []const u32) !void {
    try out.writeAll("[");
    for (ids, 0..) |id, i| {
        if (i > 0) try out.writeAll(", ");
        try out.print("{d}", .{id});
    }
    try out.writeAll("]");
}

fn printIds(out: Out, ids: []const u32) !void {
    try printIdsInline(out, ids);
    try out.writeAll("\n");
}

/// Print the last `count` vocabulary entries (the highest IDs).
fn showVocabTail(out: Out, vocab: tk.Vocab, count: usize) !void {
    for (vocab.size() - count..vocab.size()) |id| try out.print("('{s}', {d})\n", .{ vocab.tokens[id], id });
}

/// encode, print IDs, decode, print text.
fn roundtrip(gpa: std.mem.Allocator, out: Out, t: tk.SimpleTokenizer, text: []const u8) !void {
    const ids = try t.encode(gpa, text);
    try printIds(out, ids);
    const decoded = try t.decode(gpa, ids);
    try out.print("{s}\n", .{decoded});
}

/// V1 works only on words it has seen in the corpus.
fn demoV1(gpa: std.mem.Allocator, out: Out, vocab: *const tk.Vocab) !void {
    try out.writeAll("version 1 tokenizer :\n");
    try roundtrip(gpa, out, tk.SimpleTokenizer.v1(vocab), v1_text);
    try out.writeAll("-------------------------\n");
}

/// V2 maps unknown words to <|unk|> and handles the <|endoftext|> separator.
fn demoV2(gpa: std.mem.Allocator, out: Out, vocab: *const tk.Vocab) !void {
    try out.writeAll("version 2 tokenizer :\n");
    try roundtrip(gpa, out, tk.SimpleTokenizer.v2(vocab), v2_text);
    try out.writeAll("---------------------\n");
}

/// Byte pair encoding: sub-word pieces, so no <|unk|> is ever needed.
fn demoBpe(gpa: std.mem.Allocator, out: Out, bpe: tk.Bpe) !void {
    try out.writeAll("bpe tokenizer :\n");
    const ids = try bpe.encode(gpa, v2_text);
    try printIds(out, ids);
    try out.print("{s}\n", .{try bpe.decode(gpa, ids)});
    try out.writeAll("---------------------\n");
}

/// Next-word prediction pairs: the target is the input shifted by one.
fn demoSampling(gpa: std.mem.Allocator, out: Out, bpe: tk.Bpe, text: []const u8) !void {
    const all = try bpe.encode(gpa, text);
    // Drop the first 50 tokens: gives a slightly more interesting passage.
    const sample = all[50..];
    const context_size = 4;

    try out.writeAll("data sampling :\n");
    try out.writeAll("x: ");
    try printIds(out, sample[0..context_size]);
    try out.writeAll("y:      ");
    try printIds(out, sample[1 .. context_size + 1]);
    for (1..context_size + 1) |i| {
        const context = try bpe.decode(gpa, sample[0..i]);
        const desired = try bpe.decode(gpa, sample[i .. i + 1]);
        try out.print("{s} ----> {s}\n", .{ context, desired });
    }

    // The same idea, packaged: every window of the whole text.
    const ds = tk.GptDataset.init(all, 4, 4);
    const first = ds.get(0);
    try out.print("windows: {d} first pair: ", .{ds.count});
    try printIdsInline(out, first.input);
    try out.writeAll(" ");
    try printIds(out, first.target);

    // the same windows, batched
    var dl = try tk.DataLoader.init(gpa, &ds, 8, false, true, 123);
    _ = dl.startEpoch();
    var inputs: [8 * 4]u32 = undefined;
    var targets: [8 * 4]u32 = undefined;
    _ = dl.batch(0, &inputs, &targets);
    try out.writeAll("first batch inputs:\n");
    for (0..8) |r| {
        try out.writeAll("  ");
        try printIds(out, inputs[r * 4 ..][0..4]);
    }
    try out.writeAll("first batch targets:\n");
    for (0..8) |r| {
        try out.writeAll("  ");
        try printIds(out, targets[r * 4 ..][0..4]);
    }
    try out.writeAll("---------------------\n");
}

/// Batches of windows. stride < max_length overlaps windows.
fn demoDataloader(gpa: std.mem.Allocator, out: Out, bpe: tk.Bpe, text: []const u8) !void {
    const ids = try bpe.encode(gpa, text);
    var inputs: [8 * 4]u32 = undefined;
    var targets: [8 * 4]u32 = undefined;
    try out.writeAll("data loader :\n");

    const overlap = tk.GptDataset.init(ids, 4, 1);
    var dl = try tk.DataLoader.init(gpa, &overlap, 1, false, true, 123);
    _ = dl.startEpoch();
    for (0..2) |b| { // first window, then second: shifted by one
        _ = dl.batch(b, &inputs, &targets);
        try printIdsInline(out, inputs[0..4]);
        try out.writeAll(" ");
        try printIds(out, targets[0..4]);
    }

    const ds = tk.GptDataset.init(ids, 4, 4);
    var dl8 = try tk.DataLoader.init(gpa, &ds, 8, false, true, 123);
    _ = dl8.startEpoch();
    const rows = dl8.batch(0, &inputs, &targets);
    try out.print("inputs shape: ({d}, 4)\n", .{rows});
    try out.writeAll("---------------------\n");
}

/// Token IDs -> vectors, then add position vectors.
fn demoEmbeddings(gpa: std.mem.Allocator, out: Out, bpe: tk.Bpe, text: []const u8) !void {
    const max_length = 4;
    const output_dim = 256;
    const batch = 8;
    const ids = try bpe.encode(gpa, text);
    const ds = tk.GptDataset.init(ids, max_length, max_length);
    var dl = try tk.DataLoader.init(gpa, &ds, batch, false, true, 123);
    _ = dl.startEpoch();
    var inputs: [batch * max_length]u32 = undefined;
    var targets: [batch * max_length]u32 = undefined;
    _ = dl.batch(0, &inputs, &targets);

    const emb = try tk.InputEmbedding.init(gpa, tk.bpe.ENDOFTEXT_ID + 1, max_length, output_dim, 123);
    const token_emb = try gpa.alloc(f32, batch * max_length * output_dim);
    const input_emb = try gpa.alloc(f32, batch * max_length * output_dim);
    emb.token.forward(&inputs, token_emb); // token vectors only
    emb.forward(&inputs, batch, max_length, input_emb); // token + position

    // Shapes are the thing to notice: [batch, tokens] grows a new axis of size D.
    try out.writeAll("embeddings :\n");
    try out.print("token ids shape:      ({d}, {d})\n", .{ batch, max_length });
    try out.print("token embeddings:     ({d}, {d}, {d})\n", .{ batch, max_length, output_dim });
    try out.print("position embeddings:  ({d}, {d})  <- one vector per position, shared by all rows\n", .{ emb.pos.rows, emb.pos.dim });
    try out.print("input embeddings:     ({d}, {d}, {d})  <- token + position\n", .{ batch, max_length, output_dim });
    try out.writeAll("---------------------\n");
}

fn runChapter2(gpa: std.mem.Allocator, out: Out) !void {
    const text = try std.fs.cwd().readFileAlloc(data_path, gpa, .limited(1 << 20));
    const tokens = try tk.splitText(gpa, text);
    const vocab = try tk.Vocab.build(gpa, tokens);
    const ranks = try std.fs.cwd().readFileAlloc(ranks_path, gpa, .limited(8 << 20));
    const bpe = try tk.Bpe.init(gpa, ranks);

    try showVocabTail(out, vocab, 5);
    try demoV1(gpa, out, &vocab);
    try demoV2(gpa, out, &vocab);
    try demoBpe(gpa, out, bpe);
    try demoSampling(gpa, out, bpe, text);
    try demoDataloader(gpa, out, bpe, text);
    try demoEmbeddings(gpa, out, bpe, text);
}

/// `zig build run -- ch03` runs one chapter's demos; with no argument, all of them.
pub fn main() !void {
    // Tip: an arena frees everything at once when the program ends, so the
    // demos do not need a `defer free` on every allocation.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var buf: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&buf);
    const out = &stdout.interface;
    defer out.flush() catch {};

    const args = try std.process.argsAlloc(gpa);
    if (args.len == 1) {
        try runChapter2(gpa, out);
        try ch03.run(gpa, out);
        try ch04.run(gpa, out);
        try ch05.run(gpa, out);
        return;
    }
    for (args[1..]) |chapter| {
        if (std.mem.eql(u8, chapter, "ch02")) {
            try runChapter2(gpa, out);
        } else if (std.mem.eql(u8, chapter, "ch03")) {
            try ch03.run(gpa, out);
        } else if (std.mem.eql(u8, chapter, "ch04")) {
            try ch04.run(gpa, out);
        } else if (std.mem.eql(u8, chapter, "ch04-124m")) {
            try ch04.run124m(gpa, out);
        } else if (std.mem.eql(u8, chapter, "ch05")) {
            try ch05.run(gpa, out);
        } else if (std.mem.eql(u8, chapter, "ch05-gpt2")) {
            try ch05.runGpt2(gpa, out);
        } else {
            std.debug.print("unknown chapter '{s}'; use ch02, ch03, ch04, ch04-124m, ch05 or ch05-gpt2\n", .{chapter});
            std.process.exit(1);
        }
    }
}
