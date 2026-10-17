//! Step 13 (book 4.4) - shortcut connections, and why they matter.
//!
//! A shortcut (residual) connection adds a layer's input to its output:
//!
//! ```
//! x = x + layer(x)        instead of        x = layer(x)
//! ```
//!
//! During training the gradient of the loss must travel backwards through every
//! layer. Without shortcuts it shrinks a little at each layer (the vanishing
//! gradient problem), so the first layers barely learn. With shortcuts the
//! gradient also has a short path straight back, so it stays healthy.
//!
//! Python gets the gradients from torch's autograd. Here there is no autograd,
//! so this file does the backward pass by hand, in f64: the chain rule, layer by
//! layer, from the loss back to the first layer. It reproduces the book's numbers.

const std = @import("std");

/// Linear + GELU. `w` is [out][in] like `nn.Linear`, flattened row by row.
pub const ExampleLayer = struct {
    out: usize,
    in: usize,
    w: []const f64,
    b: []const f64,
};

const max_width = 3;
const max_layers = 8;

fn gelu64(v: f64) f64 {
    return 0.5 * v * (1.0 + std.math.tanh(@sqrt(2.0 / std.math.pi) * (v + 0.044715 * v * v * v)));
}

/// The derivative of the tanh-approximated GELU.
fn geluGrad(v: f64) f64 {
    const c = @sqrt(2.0 / std.math.pi);
    const t = std.math.tanh(c * (v + 0.044715 * v * v * v));
    return 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * v * v);
}

pub const ExampleDeepNeuralNetwork = struct {
    layers: []const ExampleLayer,
    use_shortcut: bool,

    /// One forward and backward pass (loss = (output - target)^2). Fills `means` with the
    /// mean absolute gradient of each layer's weight matrix, first layer first.
    pub fn gradientMeans(self: ExampleDeepNeuralNetwork, x: []const f64, target: f64, means: []f64) void {
        var acts: [max_layers + 1][max_width]f64 = undefined; // input of every layer
        var pre: [max_layers][max_width]f64 = undefined; // Linear output (before GELU) of every layer
        var shortcut: [max_layers]bool = undefined;
        for (x, 0..) |v, j| acts[0][j] = v;

        for (self.layers, 0..) |l, li| {
            for (0..l.out) |i| {
                var z = l.b[i];
                for (0..l.in) |j| z += l.w[i * l.in + j] * acts[li][j];
                pre[li][i] = z;
                acts[li + 1][i] = gelu64(z);
            }
            // A shortcut needs matching shapes, so the last (3 -> 1) layer has none.
            shortcut[li] = self.use_shortcut and l.out == l.in;
            if (shortcut[li]) {
                for (0..l.out) |i| acts[li + 1][i] += acts[li][i];
            }
        }

        var grad = [_]f64{0} ** max_width;
        grad[0] = 2.0 * (acts[self.layers.len][0] - target); // d loss / d output
        var li = self.layers.len;
        while (li > 0) {
            li -= 1;
            const l = self.layers[li];
            var dz: [max_width]f64 = undefined; // gradient at the Linear output
            for (0..l.out) |i| dz[i] = grad[i] * geluGrad(pre[li][i]);
            var sum: f64 = 0;
            for (0..l.out) |i| {
                for (0..l.in) |j| sum += @abs(dz[i] * acts[li][j]); // d loss / d W[i][j]
            }
            means[li] = sum / @as(f64, @floatFromInt(l.out * l.in));
            var next = [_]f64{0} ** max_width; // gradient handed to the layer below
            for (0..l.in) |j| {
                for (0..l.out) |i| next[j] += l.w[i * l.in + j] * dz[i];
                if (shortcut[li]) next[j] += grad[j]; // the shortcut: gradient also flows straight through
            }
            grad = next;
        }
    }
};

test "shortcuts keep the first layer's gradient from vanishing" {
    // Two layers of 1 x 1 weights 0.5: without a shortcut the gradient at layer 0 shrinks.
    const layers = [_]ExampleLayer{
        .{ .out = 1, .in = 1, .w = &.{0.5}, .b = &.{0.0} },
        .{ .out = 1, .in = 1, .w = &.{0.5}, .b = &.{0.0} },
    };
    var plain: [2]f64 = undefined;
    var with: [2]f64 = undefined;
    (ExampleDeepNeuralNetwork{ .layers = &layers, .use_shortcut = false }).gradientMeans(&.{1.0}, 0, &plain);
    (ExampleDeepNeuralNetwork{ .layers = &layers, .use_shortcut = true }).gradientMeans(&.{1.0}, 0, &with);
    try std.testing.expect(with[0] > plain[0]);
}
