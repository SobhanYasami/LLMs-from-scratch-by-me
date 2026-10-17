//! Steps 10-12 (book 4.2 and 4.3) - layer normalization, GELU and the feed forward block.
//!
//! Layer normalization rescales each token's vector to mean 0 and variance 1,
//! which keeps activations in a range where training is stable:
//!
//! ```
//! x [T, D]  ->  (x - mean) / sqrt(var + eps)  ->  scale * that + shift
//! ```
//!
//! The mean and variance are taken over the D numbers of ONE token (one row),
//! never across tokens. `scale` and `shift` are learned, so the model can undo
//! the normalization where that helps.
//!
//! GELU is a smooth cousin of ReLU, and the feed forward block expands each
//! token's vector 4x, applies GELU, and shrinks it back:
//!
//! ```
//! [T, D] -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> [T, D]
//! ```

const std = @import("std");
const Matrix = @import("../attention/matrix.zig").Matrix;

pub const LayerNorm = struct {
    scale: []f32,
    shift: []f32,

    const eps: f32 = 1e-5;

    /// scale 1, shift 0.
    pub fn init(gpa: std.mem.Allocator, dim: usize) !LayerNorm {
        const scale = try gpa.alloc(f32, dim);
        @memset(scale, 1);
        const shift = try gpa.alloc(f32, dim);
        @memset(shift, 0);
        return .{ .scale = scale, .shift = shift };
    }

    /// Normalize every row of `x`.
    pub fn forward(self: LayerNorm, gpa: std.mem.Allocator, x: Matrix) !Matrix {
        const out = try Matrix.init(gpa, x.rows, x.cols);
        const n: f32 = @floatFromInt(x.cols);
        for (0..x.rows) |i| {
            const r = x.row(i);
            var mean: f32 = 0;
            for (r) |v| mean += v;
            mean /= n;
            var variance: f32 = 0; // divide by n (biased), like GPT-2
            for (r) |v| variance += (v - mean) * (v - mean);
            variance /= n;
            const inv: f32 = @floatCast(1.0 / @sqrt(@as(f64, variance + eps)));
            for (r, 0..) |v, j| out.data[i * x.cols + j] = self.scale[j] * ((v - mean) * inv) + self.shift[j];
        }
        return out;
    }
};

/// The tanh approximation GPT-2 was trained with.
pub fn gelu(x: f32) f32 {
    const v: f64 = x;
    return @floatCast(0.5 * v * (1.0 + std.math.tanh(@sqrt(2.0 / std.math.pi) * (v + 0.044715 * v * v * v))));
}

/// `x @ W + b` with W stored [in, out] (PyTorch's `nn.Linear` stores it transposed).
pub const Linear = struct {
    w: Matrix,
    b: []f32,

    pub fn init(gpa: std.mem.Allocator, input: usize, output: usize) !Linear {
        const b = try gpa.alloc(f32, output);
        @memset(b, 0);
        return .{ .w = try Matrix.init(gpa, input, output), .b = b };
    }

    pub fn forward(self: Linear, gpa: std.mem.Allocator, x: Matrix) !Matrix {
        const out = try x.matmul(gpa, self.w);
        for (0..out.rows) |i| {
            for (out.row(i), self.b) |*v, bias| v.* += bias;
        }
        return out;
    }
};

pub const FeedForward = struct {
    expand: Linear,
    shrink: Linear,

    pub fn init(gpa: std.mem.Allocator, dim: usize) !FeedForward {
        return .{ .expand = try Linear.init(gpa, dim, 4 * dim), .shrink = try Linear.init(gpa, 4 * dim, dim) };
    }

    pub fn forward(self: FeedForward, gpa: std.mem.Allocator, x: Matrix) !Matrix {
        const h = try self.expand.forward(gpa, x);
        for (h.data) |*v| v.* = gelu(v.*);
        return self.shrink.forward(gpa, h);
    }
};

test "layer norm gives mean 0 and variance 1" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const ln = try LayerNorm.init(a, 4);
    const x = try Matrix.fromSlice(a, 1, 4, &.{ 1, 2, 3, 10 });
    const y = try ln.forward(a, x);
    var mean: f32 = 0;
    for (y.data) |v| mean += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0), mean / 4, 1e-6);
}

test "gelu is about x for big x and about 0 for very negative x" {
    try std.testing.expectApproxEqAbs(@as(f32, 3), gelu(3), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0), gelu(-5), 0.001);
    try std.testing.expectEqual(@as(f32, 0), gelu(0));
}
