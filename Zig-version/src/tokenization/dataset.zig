//! Step 4 (book 2.6) - sliding-window data sampling for next-token prediction.
//!
//! ```
//! ids      = [10 11 12 13 14 15]
//! window 0: input [10 11 12 13]  target [11 12 13 14]
//! ```
//!
//! Why shifted by one? The model's job is: given the tokens so far, guess the
//! next one. Every position in a window is a free training example.
//! `stride` is how far the window moves; `stride == max_length` means no overlap.

const std = @import("std");

pub const GptDataset = struct {
    /// Borrowed: the caller keeps the token IDs alive.
    ids: []const u32,
    max_length: usize,
    stride: usize,
    /// Number of windows.
    count: usize,

    pub fn init(ids: []const u32, max_length: usize, stride: usize) GptDataset {
        // Windows start at 0, stride, 2*stride, ... and stop before len - max_length:
        // the target reaches one token past the input.
        const count = if (ids.len > max_length) (ids.len - max_length + stride - 1) / stride else 0;
        return .{ .ids = ids, .max_length = max_length, .stride = stride, .count = count };
    }

    pub const Window = struct { input: []const u32, target: []const u32 };

    /// Window `idx`: two slices into the ID array, the target shifted by one.
    pub fn get(self: GptDataset, idx: usize) Window {
        const start = idx * self.stride;
        return .{
            .input = self.ids[start .. start + self.max_length],
            .target = self.ids[start + 1 .. start + self.max_length + 1],
        };
    }
};

test "windows are shifted by one" {
    const ids = [_]u32{ 10, 11, 12, 13, 14, 15 };
    const ds = GptDataset.init(&ids, 4, 1);
    try std.testing.expectEqual(@as(usize, 2), ds.count);
    const w = ds.get(0);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 12, 13 }, w.input);
    try std.testing.expectEqualSlices(u32, &.{ 11, 12, 13, 14 }, w.target);
}

test "stride equal to max_length gives no overlap" {
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const ds = GptDataset.init(&ids, 4, 4);
    try std.testing.expectEqual(@as(usize, 2), ds.count);
}
