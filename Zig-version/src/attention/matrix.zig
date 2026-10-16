//! The few matrix operations attention needs.
//!
//! Tip: there is no tensor library, so a `Matrix` is a row-major `[]f32`.
//! Everything attention does is these operations: matmul, transpose, softmax.
//! Every operation allocates its result from the allocator you pass in; the
//! demos use an arena, so nothing needs freeing one by one.

const std = @import("std");
const Rng = @import("../tokenization/rng.zig").Rng;

pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f32,

    /// Zero-filled.
    pub fn init(gpa: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try gpa.alloc(f32, rows * cols);
        @memset(data, 0);
        return .{ .rows = rows, .cols = cols, .data = data };
    }

    /// A copy of `values`, which holds rows * cols numbers.
    pub fn fromSlice(gpa: std.mem.Allocator, rows: usize, cols: usize, values: []const f32) !Matrix {
        const m = try init(gpa, rows, cols);
        @memcpy(m.data, values);
        return m;
    }

    /// Uniform in [0, 1), like `torch.rand`.
    pub fn rand(gpa: std.mem.Allocator, rows: usize, cols: usize, rng: *Rng) !Matrix {
        const m = try init(gpa, rows, cols);
        for (m.data) |*v| v.* = @floatCast(rng.nextF64());
        return m;
    }

    pub fn at(self: Matrix, i: usize, j: usize) f32 {
        return self.data[i * self.cols + j];
    }

    pub fn row(self: Matrix, i: usize) []f32 {
        return self.data[i * self.cols ..][0..self.cols];
    }

    /// self x other: [n,k] x [k,m] -> [n,m].
    pub fn matmul(self: Matrix, gpa: std.mem.Allocator, other: Matrix) !Matrix {
        const out = try init(gpa, self.rows, other.cols);
        for (0..self.rows) |i| {
            for (0..other.cols) |j| {
                var sum: f32 = 0;
                for (0..self.cols) |k| sum += self.at(i, k) * other.at(k, j);
                out.data[i * out.cols + j] = sum;
            }
        }
        return out;
    }

    pub fn transpose(self: Matrix, gpa: std.mem.Allocator) !Matrix {
        const out = try init(gpa, self.cols, self.rows);
        for (0..self.rows) |i| {
            for (0..self.cols) |j| out.data[j * out.cols + i] = self.at(i, j);
        }
        return out;
    }

    /// Turn every row into weights that are positive and sum to 1.
    ///
    /// Tip: subtracting the row maximum first does not change the result but keeps
    /// `exp` from overflowing. A `-inf` entry becomes exactly 0.
    pub fn softmaxRows(self: Matrix, gpa: std.mem.Allocator) !Matrix {
        const out = try init(gpa, self.rows, self.cols);
        for (0..self.rows) |i| {
            const r = self.row(i);
            var max = r[0];
            for (r[1..]) |v| max = @max(max, v);
            var sum: f32 = 0;
            for (r, 0..) |v, j| {
                const e: f32 = @floatCast(@exp(@as(f64, v - max)));
                out.data[i * out.cols + j] = e;
                sum += e;
            }
            for (0..self.cols) |j| out.data[i * out.cols + j] /= sum;
        }
        return out;
    }

    /// Multiply every element by `s`.
    pub fn scale(self: Matrix, gpa: std.mem.Allocator, s: f32) !Matrix {
        const out = try init(gpa, self.rows, self.cols);
        for (self.data, out.data) |v, *o| o.* = v * s;
        return out;
    }

    /// Set every element above the diagonal to `-inf` (the "future" tokens).
    pub fn maskFuture(self: Matrix, gpa: std.mem.Allocator) !Matrix {
        const out = try fromSlice(gpa, self.rows, self.cols, self.data);
        for (0..self.rows) |i| {
            for (i + 1..self.cols) |j| out.data[i * out.cols + j] = -std.math.inf(f32);
        }
        return out;
    }

    /// Columns `[start, end)` as a new matrix.
    pub fn sliceCols(self: Matrix, gpa: std.mem.Allocator, start: usize, end: usize) !Matrix {
        const out = try init(gpa, self.rows, end - start);
        for (0..self.rows) |i| @memcpy(out.row(i), self.row(i)[start..end]);
        return out;
    }

    /// Zero each element with probability `p` and scale the rest by 1/(1-p).
    /// The scaling keeps the average unchanged, so nothing needs rescaling at
    /// inference time (when dropout is simply switched off).
    pub fn dropout(self: Matrix, gpa: std.mem.Allocator, p: f32, rng: *Rng) !Matrix {
        const out = try init(gpa, self.rows, self.cols);
        for (self.data, out.data) |v, *o| {
            if (rng.nextF64() >= p) o.* = v / (1.0 - p);
        }
        return out;
    }
};

test "softmax rows sum to 1 and -inf becomes 0" {
    const gpa = std.testing.allocator;
    const m = try Matrix.fromSlice(gpa, 1, 3, &.{ 1, 2, -std.math.inf(f32) });
    defer gpa.free(m.data);
    const s = try m.softmaxRows(gpa);
    defer gpa.free(s.data);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s.data[0] + s.data[1] + s.data[2], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), s.data[2]);
}

test "matmul and transpose" {
    const gpa = std.testing.allocator;
    const a = try Matrix.fromSlice(gpa, 2, 2, &.{ 1, 2, 3, 4 });
    defer gpa.free(a.data);
    const t = try a.transpose(gpa);
    defer gpa.free(t.data);
    const p = try a.matmul(gpa, t); // a x a^T
    defer gpa.free(p.data);
    try std.testing.expectEqualSlices(f32, &.{ 5, 11, 11, 25 }, p.data);
}
