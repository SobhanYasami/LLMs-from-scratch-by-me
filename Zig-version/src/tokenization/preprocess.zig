//! Step 1 (book 2.2) - split raw text into tokens, and join them back.
//!
//! ```
//! "Hello, world!"  ->  ["Hello", ",", "world", "!"]
//! ```
//!
//! Tip: start reading here, then vocab -> simple -> dataset -> dataloader ->
//! embedding. Each file uses the one before it. bpe is the bonus (book 2.5).

const std = @import("std");

/// Special tokens. They are not produced by splitting text; they are added on purpose.
pub const END_OF_TEXT = "<|endoftext|>";
pub const UNK_TOKEN = "<|unk|>";

/// Delimiters that become tokens of their own. "--" and whitespace are handled
/// in `splitText`. Same set as the Python regex: , . ? _ ! " ( ) '
const split_punct = ",.?_!\"()'";
/// Punctuation that `detokenize` glues to the previous word.
const detokenize_punct = ",.?!\"()'";

fn flush(gpa: std.mem.Allocator, out: *std.ArrayList([]const u8), text: []const u8, start: usize, end: usize) !void {
    if (end > start) try out.append(gpa, text[start..end]);
}

/// Return the tokens of `text`; whitespace-only pieces are dropped.
///
/// Tip: tokens are *slices into `text`*, nothing is copied. They are only
/// valid while `text` is alive. The returned list is owned by the caller.
///
/// There is no regex library in the standard library, so we scan by hand: walk the
/// bytes and cut a token whenever we meet whitespace, "--" or a punctuation mark.
pub fn splitText(gpa: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(gpa);

    var word_start: usize = 0;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (std.ascii.isWhitespace(c)) {
            try flush(gpa, &out, text, word_start, i);
            word_start = i + 1;
        } else if (c == '-' and i + 1 < text.len and text[i + 1] == '-') {
            try flush(gpa, &out, text, word_start, i);
            try out.append(gpa, text[i .. i + 2]);
            i += 1;
            word_start = i + 1;
        } else if (std.mem.indexOfScalar(u8, split_punct, c) != null) {
            try flush(gpa, &out, text, word_start, i);
            try out.append(gpa, text[i .. i + 1]);
            word_start = i + 1;
        }
    }
    try flush(gpa, &out, text, word_start, text.len);
    return out.toOwnedSlice(gpa);
}

/// Join tokens with spaces, then remove the space before punctuation.
/// Not a perfect inverse: the original spacing is lost. Caller frees.
pub fn detokenize(gpa: std.mem.Allocator, tokens: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (tokens, 0..) |token, i| {
        const glue = token.len == 1 and std.mem.indexOfScalar(u8, detokenize_punct, token[0]) != null;
        if (i > 0 and !glue) try out.append(gpa, ' ');
        try out.appendSlice(gpa, token);
    }
    return out.toOwnedSlice(gpa);
}

test "splitText keeps punctuation as tokens" {
    const gpa = std.testing.allocator;
    const tokens = try splitText(gpa, "Hello, world!");
    defer gpa.free(tokens);
    try std.testing.expectEqual(@as(usize, 4), tokens.len);
    try std.testing.expectEqualStrings("Hello", tokens[0]);
    try std.testing.expectEqualStrings(",", tokens[1]);
    try std.testing.expectEqualStrings("!", tokens[3]);
}

test "double dash is one token" {
    const gpa = std.testing.allocator;
    const tokens = try splitText(gpa, "Is this-- a test?");
    defer gpa.free(tokens);
    try std.testing.expectEqualStrings("--", tokens[2]);
}

test "detokenize removes space before punctuation" {
    const gpa = std.testing.allocator;
    const text = try detokenize(gpa, &.{ "Hello", ",", "world", "!" });
    defer gpa.free(text);
    try std.testing.expectEqualStrings("Hello, world!", text);
}
