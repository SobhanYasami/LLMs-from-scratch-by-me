//! Tiny seeded random generator (splitmix64), no library needed.
//!
//! Used for shuffling batches and for random embedding weights. Same seed,
//! same numbers, on every run. (Same algorithm as the Rust and C versions.)

const std = @import("std");

pub const Rng = struct {
    state: u64,

    pub fn init(seed: u64) Rng {
        return .{ .state = seed };
    }

    fn nextU64(self: *Rng) u64 {
        self.state +%= 0x9E37_79B9_7F4A_7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58_476D_1CE4_E5B9;
        z = (z ^ (z >> 27)) *% 0x94D0_49BB_1331_11EB;
        return z ^ (z >> 31);
    }

    /// Uniform in [0, 1).
    pub fn nextF64(self: *Rng) f64 {
        return @as(f64, @floatFromInt(self.nextU64() >> 11)) / 9007199254740992.0; // 2^53
    }

    /// Uniform integer in [0, n).
    pub fn below(self: *Rng, n: usize) usize {
        return @intCast(self.nextU64() % n);
    }

    /// Normal(0, 1) via the Box-Muller transform.
    pub fn normal(self: *Rng) f32 {
        const a = 1.0 - self.nextF64(); // avoid log(0)
        const b = self.nextF64();
        return @floatCast(@sqrt(-2.0 * @log(a)) * @cos(2.0 * std.math.pi * b));
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle(self: *Rng, comptime T: type, items: []T) void {
        var i = items.len;
        while (i > 1) {
            i -= 1;
            std.mem.swap(T, &items[i], &items[self.below(i + 1)]);
        }
    }
};
