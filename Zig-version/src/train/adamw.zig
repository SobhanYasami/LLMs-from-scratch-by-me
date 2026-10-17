//! The AdamW optimizer (book 5.2), the same arithmetic as `torch.optim.AdamW`.
//!
//! Plain gradient descent moves every weight by `lr * gradient`. Adam keeps two running averages per
//! weight, so each weight gets its own step size:
//!
//! ```
//! m = b1*m + (1-b1)*g          average of the gradient         (the direction)
//! v = b2*v + (1-b2)*g*g        average of the squared gradient (how noisy it is)
//! w = w - lr * m_hat / (sqrt(v_hat) + eps)
//! ```
//!
//! "W" means the weight decay is applied straight to the weights (`w *= 1 - lr*wd`) instead of being
//! mixed into the gradient, which regularizes better.

const std = @import("std");
const GptModel = @import("../gpt/root.zig").GptModel;

pub const AdamW = struct {
    // The settings are f64 like Python's floats; the per-step scalars are rounded to f32 once.
    lr: f64,
    beta1: f64 = 0.9,
    beta2: f64 = 0.999,
    eps: f64 = 1e-8,
    weight_decay: f64,
    step_count: u32 = 0,
    /// Running averages, stored in models of the same shape as the one being trained.
    exp_avg: GptModel,
    exp_avg_sq: GptModel,

    pub fn init(gpa: std.mem.Allocator, model: GptModel, lr: f64, weight_decay: f64) !AdamW {
        return .{ .lr = lr, .weight_decay = weight_decay, .exp_avg = try model.zerosLike(gpa), .exp_avg_sq = try model.zerosLike(gpa) };
    }

    /// One update of every weight in `model` from `grads` (a model-shaped bag of gradients).
    pub fn step(self: *AdamW, gpa: std.mem.Allocator, model: GptModel, grads: GptModel) !void {
        self.step_count += 1;
        const t: f64 = @floatFromInt(self.step_count);
        const step_size: f32 = @floatCast(self.lr / (1 - std.math.pow(f64, self.beta1, t)));
        const bias2_sqrt: f32 = @floatCast(@sqrt(1 - std.math.pow(f64, self.beta2, t)));
        const decay: f32 = @floatCast(1 - self.lr * self.weight_decay);
        const b2: f32 = @floatCast(self.beta2);
        const eps: f32 = @floatCast(self.eps);
        const one_minus_b1: f32 = @floatCast(1 - self.beta1);
        const one_minus_b2: f32 = @floatCast(1 - self.beta2);
        const params = try model.tensors(gpa);
        const ms = try self.exp_avg.tensors(gpa);
        const vs = try self.exp_avg_sq.tensors(gpa);
        const gs = try grads.tensors(gpa);
        for (params, ms, vs, gs) |p, m, v, g| {
            for (p, m, v, g) |*pi, *mi, *vi, gi| {
                pi.* *= decay;
                mi.* += (gi - mi.*) * one_minus_b1; // lerp
                vi.* = vi.* * b2 + one_minus_b2 * gi * gi;
                const denom = @as(f32, @floatCast(@sqrt(@as(f64, vi.*)))) / bias2_sqrt + eps;
                pi.* += (-step_size * mi.*) / denom;
            }
        }
    }
};
