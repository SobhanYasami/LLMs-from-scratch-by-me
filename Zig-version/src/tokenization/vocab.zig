//! Step 2 (book 2.3) - build a vocabulary: token -> unique integer ID.
//!
//! ```
//! {"!": 0, "\"": 1, ..., "<|endoftext|>": 1159, "<|unk|>": 1160}
//! ```
//!
//! Models only understand numbers, so this table is the bridge between text and IDs.
//!
//! Tip: a hash map has no fixed order. Sorting before numbering is what makes
//! IDs the same on every run.

const std = @import("std");
const preprocess = @import("preprocess.zig");

pub const Vocab = struct {
    /// ID -> token, used by decode.
    tokens: []const []const u8,
    /// token -> ID, used by encode.
    ids: std.StringHashMapUnmanaged(u32),

    /// Unique tokens in sorted order, then `<|endoftext|>` and `<|unk|>`
    /// (last, so regular word IDs stay stable). The vocab borrows the token
    /// strings: keep the text they came from alive.
    pub fn build(gpa: std.mem.Allocator, tokens: []const []const u8) !Vocab {
        const sorted = try gpa.dupe([]const u8, tokens);
        defer gpa.free(sorted);
        std.mem.sort([]const u8, sorted, {}, lessThan);

        var list: std.ArrayList([]const u8) = .empty;
        errdefer list.deinit(gpa);
        for (sorted, 0..) |token, i| {
            if (i == 0 or !std.mem.eql(u8, token, sorted[i - 1])) try list.append(gpa, token);
        }
        try list.append(gpa, preprocess.END_OF_TEXT);
        try list.append(gpa, preprocess.UNK_TOKEN);
        const all = try list.toOwnedSlice(gpa);
        errdefer gpa.free(all);

        var ids: std.StringHashMapUnmanaged(u32) = .empty;
        errdefer ids.deinit(gpa);
        try ids.ensureTotalCapacity(gpa, @intCast(all.len));
        for (all, 0..) |token, id| ids.putAssumeCapacity(token, @intCast(id));
        return .{ .tokens = all, .ids = ids };
    }

    pub fn deinit(self: *Vocab, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        self.ids.deinit(gpa);
    }

    pub fn size(self: Vocab) usize {
        return self.tokens.len;
    }

    /// token -> ID, or null if the token is not in the vocabulary.
    pub fn idOf(self: Vocab, text: []const u8) ?u32 {
        return self.ids.get(text);
    }

    /// ID -> token, or null if out of range.
    pub fn tokenOf(self: Vocab, id: u32) ?[]const u8 {
        return if (id < self.tokens.len) self.tokens[id] else null;
    }
};

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

test "vocab is sorted with special tokens last" {
    const gpa = std.testing.allocator;
    var v = try Vocab.build(gpa, &.{ "b", "a", "b", "c" });
    defer v.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5), v.size());
    try std.testing.expectEqual(@as(?u32, 0), v.idOf("a"));
    try std.testing.expectEqual(@as(?u32, 3), v.idOf(preprocess.END_OF_TEXT));
    try std.testing.expectEqual(@as(?u32, 4), v.idOf(preprocess.UNK_TOKEN));
    try std.testing.expectEqual(@as(?u32, null), v.idOf("zzz"));
}
