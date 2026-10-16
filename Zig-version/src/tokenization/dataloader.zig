//! Step 4b (book 2.6) - batching. Mirrors torch's `DataLoader`.
//!
//! Why batch? The GPU processes many windows at once, which is much faster.
//! Why shuffle? Windows in book order make training swing; random order smooths it.
//! Why drop the last short batch? A different size can upset training.

const std = @import("std");
const GptDataset = @import("dataset.zig").GptDataset;
const Rng = @import("rng.zig").Rng;

pub const DataLoader = struct {
    dataset: *const GptDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    rng: Rng,
    /// Window indices in the order this epoch serves them.
    order: []usize,

    /// `seed` makes the shuffle repeatable.
    pub fn init(gpa: std.mem.Allocator, dataset: *const GptDataset, batch_size: usize, shuffle: bool, drop_last: bool, seed: u64) !DataLoader {
        const order = try gpa.alloc(usize, dataset.count);
        for (order, 0..) |*o, i| o.* = i;
        return .{ .dataset = dataset, .batch_size = batch_size, .shuffle = shuffle, .drop_last = drop_last, .rng = Rng.init(seed), .order = order };
    }

    pub fn deinit(self: *DataLoader, gpa: std.mem.Allocator) void {
        gpa.free(self.order);
    }

    /// Start an epoch (reshuffles if shuffle is on). Returns the number of batches.
    pub fn startEpoch(self: *DataLoader) usize {
        if (self.shuffle) self.rng.shuffle(usize, self.order);
        const n = self.dataset.count;
        return if (self.drop_last) n / self.batch_size else (n + self.batch_size - 1) / self.batch_size;
    }

    /// Copy batch `b` into flat arrays of `batch_size * max_length` IDs each
    /// (row r is `inputs[r*max_length .. (r+1)*max_length]`). Returns rows filled.
    pub fn batch(self: DataLoader, b: usize, inputs: []u32, targets: []u32) usize {
        const len = self.dataset.max_length;
        const first = b * self.batch_size;
        const end = @min(first + self.batch_size, self.dataset.count);
        var rows: usize = 0;
        for (self.order[first..end]) |idx| {
            const w = self.dataset.get(idx);
            @memcpy(inputs[rows * len ..][0..len], w.input);
            @memcpy(targets[rows * len ..][0..len], w.target);
            rows += 1;
        }
        return rows;
    }
};

test "drop_last drops the short batch" {
    const gpa = std.testing.allocator;
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const ds = GptDataset.init(&ids, 2, 2); // 5 windows
    var dl = try DataLoader.init(gpa, &ds, 2, false, true, 1);
    defer dl.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), dl.startEpoch());
}

test "same seed gives the same shuffle" {
    const gpa = std.testing.allocator;
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const ds = GptDataset.init(&ids, 2, 2);
    var a = try DataLoader.init(gpa, &ds, 2, true, true, 123);
    defer a.deinit(gpa);
    var b = try DataLoader.init(gpa, &ds, 2, true, true, 123);
    defer b.deinit(gpa);
    _ = a.startEpoch();
    _ = b.startEpoch();
    try std.testing.expectEqualSlices(usize, a.order, b.order);
}
