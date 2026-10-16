//! Bonus (book 2.5) - byte pair encoding, the GPT-2 tokenizer.
//!
//! There is no tiktoken for Zig, so this is a small from-scratch encoder that
//! reads the GPT-2 vocabulary from `assets/gpt2-ranks.txt` (one line per token:
//! the token's bytes in hex, then its rank). It gives the same IDs as tiktoken's "gpt2".
//!
//! How it works, for one piece of text:
//!   1. start with one part per byte
//!   2. find the neighbouring pair whose joined bytes have the LOWEST rank
//!   3. merge it, repeat until no pair is a known token
//! Low rank = merged early in training = most frequent. Any byte string can be
//! encoded, so no `<|unk|>` is needed.

const std = @import("std");
const END_OF_TEXT = @import("preprocess.zig").END_OF_TEXT;

pub const ENDOFTEXT_ID: u32 = 50256;

pub const Bpe = struct {
    /// rank -> token bytes.
    tokens: [][]const u8,
    /// token bytes -> rank (keys are the slices in `tokens`).
    ranks: std.StringHashMapUnmanaged(u32),

    /// Parse the ranks file contents. Lines look like `48656c6c6f 1234`.
    pub fn init(gpa: std.mem.Allocator, file_contents: []const u8) !Bpe {
        var tokens: std.ArrayList([]const u8) = .empty;
        errdefer tokens.deinit(gpa);
        var lines = std.mem.splitScalar(u8, file_contents, '\n');
        while (lines.next()) |line| {
            const space = std.mem.indexOfScalar(u8, line, ' ') orelse continue;
            const bytes = try gpa.alloc(u8, space / 2);
            _ = try std.fmt.hexToBytes(bytes, line[0..space]);
            try tokens.append(gpa, bytes); // ranks are 0, 1, 2, ... in file order
        }
        var ranks: std.StringHashMapUnmanaged(u32) = .empty;
        errdefer ranks.deinit(gpa);
        try ranks.ensureTotalCapacity(gpa, @intCast(tokens.items.len));
        for (tokens.items, 0..) |t, rank| ranks.putAssumeCapacity(t, @intCast(rank));
        return .{ .tokens = try tokens.toOwnedSlice(gpa), .ranks = ranks };
    }

    pub fn deinit(self: *Bpe, gpa: std.mem.Allocator) void {
        for (self.tokens) |t| gpa.free(t);
        gpa.free(self.tokens);
        self.ranks.deinit(gpa);
    }

    /// Text -> token IDs. `<|endoftext|>` in the text becomes ID 50256. Caller frees.
    pub fn encode(self: Bpe, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        var i: usize = 0;
        while (i < text.len) {
            if (std.mem.startsWith(u8, text[i..], END_OF_TEXT)) { // special token
                try out.append(gpa, ENDOFTEXT_ID);
                i += END_OF_TEXT.len;
                continue;
            }
            var end = nextPiece(text, i);
            // never let a piece swallow the start of a special token
            for (i + 1..end) |j| {
                if (std.mem.startsWith(u8, text[j..], END_OF_TEXT)) {
                    end = j;
                    break;
                }
            }
            try self.encodePiece(gpa, text[i..end], &out);
            i = end;
        }
        return out.toOwnedSlice(gpa);
    }

    /// IDs -> text. Caller frees.
    pub fn decode(self: Bpe, gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (ids) |id| try out.appendSlice(gpa, if (id == ENDOFTEXT_ID) END_OF_TEXT else self.tokens[id]);
        return out.toOwnedSlice(gpa);
    }

    /// Encode one piece (a few bytes) by repeatedly merging the best neighbour pair.
    fn encodePiece(self: Bpe, gpa: std.mem.Allocator, piece: []const u8, out: *std.ArrayList(u32)) !void {
        // bounds[i]..bounds[i+1] is part i; begin with one part per byte
        var bounds: std.ArrayList(usize) = .empty;
        defer bounds.deinit(gpa);
        for (0..piece.len + 1) |i| try bounds.append(gpa, i);

        while (bounds.items.len > 2) {
            var best_rank: ?u32 = null;
            var best: usize = 0;
            for (0..bounds.items.len - 2) |i| { // rank of parts i and i+1 joined
                const joined = piece[bounds.items[i]..bounds.items[i + 2]];
                if (self.ranks.get(joined)) |r| {
                    if (best_rank == null or r < best_rank.?) {
                        best_rank = r;
                        best = i;
                    }
                }
            }
            if (best_rank == null) break;
            _ = bounds.orderedRemove(best + 1); // merge: drop the boundary between the two parts
        }
        for (0..bounds.items.len - 1) |i| {
            try out.append(gpa, self.ranks.get(piece[bounds.items[i]..bounds.items[i + 1]]).?);
        }
    }
};

// ---- splitting text into pieces (GPT-2's regex, written by hand) ----

// The GPT-2 pattern, in words:
//   'm 'd 's 't 're 've 'll | optional space + letters | optional space + digits
//   | optional space + other symbols | whitespace
// Bytes >= 0x80 (UTF-8) are treated as letters, which is right for most text.
fn isLetter(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c >= 0x80;
}
fn isDigit(c: u8) bool {
    return std.ascii.isDigit(c);
}
fn isSpace(c: u8) bool {
    return c < 0x80 and std.ascii.isWhitespace(c);
}
fn isOther(c: u8) bool {
    return !isSpace(c) and !isLetter(c) and !isDigit(c);
}

fn consume(s: []const u8, start: usize, comptime pred: fn (u8) bool) usize {
    var i = start;
    while (i < s.len and pred(s[i])) i += 1;
    return i;
}

/// End of the piece starting at `i`.
fn nextPiece(s: []const u8, i: usize) usize {
    if (s[i] == '\'' and i + 1 < s.len) {
        if (std.mem.indexOfScalar(u8, "sdmt", s[i + 1]) != null) return i + 2;
        if (i + 2 < s.len) {
            const two = s[i + 1 .. i + 3];
            if (std.mem.eql(u8, two, "ll") or std.mem.eql(u8, two, "ve") or std.mem.eql(u8, two, "re")) return i + 3;
        }
    }
    const k = if (s[i] == ' ') i + 1 else i; // optional single leading space
    if (k < s.len) {
        if (isLetter(s[k])) return consume(s, k, isLetter);
        if (isDigit(s[k])) return consume(s, k, isDigit);
        if (isOther(s[k])) return consume(s, k, isOther);
    }
    // whitespace run: leave its last char for the next piece, unless it is the end
    const e = consume(s, i, isSpace);
    return if (e < s.len and e - i > 1) e - 1 else e;
}

test "pieces follow the GPT-2 pattern" {
    try std.testing.expectEqual(@as(usize, 5), nextPiece("Hello world", 0)); // "Hello"
    try std.testing.expectEqual(@as(usize, 11), nextPiece("Hello world", 5)); // " world"
    try std.testing.expectEqual(@as(usize, 2), nextPiece("'s x", 0)); // "'s"
}

test "bpe merges by lowest rank" {
    const gpa = std.testing.allocator;
    // bytes: a=0 b=1, then "ab"=2
    var bpe = try Bpe.init(gpa, "61 0\n62 1\n6162 2\n");
    defer bpe.deinit(gpa);
    const ids = try bpe.encode(gpa, "abab");
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2 }, ids);
}
