//! Step 3 (book 2.3-2.4) - simple tokenizers: split text, look tokens up.
//!
//! ```
//! encode: "Hello, world" -> ["Hello", ",", "world"] -> [1131, 5, 987]
//! decode: [1131, 5, 987] -> ["Hello", ",", "world"] -> "Hello, world"
//! ```
//!
//! Tip: compare V1 and V2. V1 fails on a word it never saw; V2 swaps it for
//! `<|unk|>` and keeps going. That one difference is why real models use BPE.

const std = @import("std");
const preprocess = @import("preprocess.zig");
const Vocab = @import("vocab.zig").Vocab;

pub const SimpleTokenizer = struct {
    vocab: *const Vocab,
    /// false = V1 (error), true = V2 (`<|unk|>`).
    unknown_to_unk: bool,

    pub fn v1(vocab: *const Vocab) SimpleTokenizer {
        return .{ .vocab = vocab, .unknown_to_unk = false };
    }

    pub fn v2(vocab: *const Vocab) SimpleTokenizer {
        return .{ .vocab = vocab, .unknown_to_unk = true };
    }

    /// text -> IDs. V1 fails with `error.UnknownToken` on a word outside the
    /// vocabulary; V2 uses the `<|unk|>` ID instead. Caller frees.
    pub fn encode(self: SimpleTokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        const tokens = try preprocess.splitText(gpa, text);
        defer gpa.free(tokens);
        const ids = try gpa.alloc(u32, tokens.len);
        errdefer gpa.free(ids);
        for (tokens, ids) |token, *id| {
            id.* = self.vocab.idOf(token) orelse
                (if (self.unknown_to_unk) self.vocab.idOf(preprocess.UNK_TOKEN) else null) orelse
                return error.UnknownToken;
        }
        return ids;
    }

    /// IDs -> text. Fails with `error.UnknownId` for an ID outside the vocabulary. Caller frees.
    pub fn decode(self: SimpleTokenizer, gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
        const tokens = try gpa.alloc([]const u8, ids.len);
        defer gpa.free(tokens);
        for (ids, tokens) |id, *token| token.* = self.vocab.tokenOf(id) orelse return error.UnknownId;
        return preprocess.detokenize(gpa, tokens);
    }
};

fn testVocab(gpa: std.mem.Allocator) !Vocab {
    return Vocab.build(gpa, &.{ "Hello", ",", "world" });
}

test "V1 fails on unknown words, V2 uses <|unk|>" {
    const gpa = std.testing.allocator;
    var vocab = try testVocab(gpa);
    defer vocab.deinit(gpa);

    try std.testing.expectError(error.UnknownToken, SimpleTokenizer.v1(&vocab).encode(gpa, "Hello there"));

    const ids = try SimpleTokenizer.v2(&vocab).encode(gpa, "Hello there");
    defer gpa.free(ids);
    try std.testing.expectEqual(vocab.idOf(preprocess.UNK_TOKEN).?, ids[1]);
}

test "encode then decode round-trips simple text" {
    const gpa = std.testing.allocator;
    var vocab = try testVocab(gpa);
    defer vocab.deinit(gpa);
    const t = SimpleTokenizer.v1(&vocab);
    const ids = try t.encode(gpa, "Hello, world");
    defer gpa.free(ids);
    const text = try t.decode(gpa, ids);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("Hello, world", text);
}
