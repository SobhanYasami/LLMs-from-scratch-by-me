//! Step 20 (book 5.4) - saving and loading a model (and the optimizer).
//!
//! Training is expensive, so save the weights and reload them later. The book uses
//! `torch.save(model.state_dict())`. To keep the file readable by every language in this repo, we
//! write one plain format:
//!
//! ```
//! "LLMCKPT1" | 7 x u32: vocab, context, emb, heads, layers, qkv_bias, has_optimizer
//! then raw little-endian f32 numbers, tensor after tensor, in the order of `GptModel.tensors`
//! (the weights), then the AdamW state if present: step (as f32), then exp_avg and exp_avg_sq of
//! every tensor.
//! ```
//!
//! A checkpoint written by Zig can be loaded by Python, Go, Rust or C, and the other way round.

const std = @import("std");
const gpt = @import("../gpt/root.zig");
const AdamW = @import("adamw.zig").AdamW;

const magic = "LLMCKPT1";

fn writeTensors(gpa: std.mem.Allocator, w: *std.Io.Writer, m: gpt.GptModel) !void {
    for (try m.tensors(gpa)) |t| try w.writeAll(std.mem.sliceAsBytes(t));
}

fn readTensors(gpa: std.mem.Allocator, r: *std.Io.Reader, m: gpt.GptModel) !void {
    for (try m.tensors(gpa)) |t| try r.readSliceAll(std.mem.sliceAsBytes(t));
}

/// Writes the model and, if `opt` is not null, the optimizer state.
pub fn save(gpa: std.mem.Allocator, path: []const u8, m: gpt.GptModel, opt: ?*const AdamW) !void {
    if (std.fs.path.dirname(path)) |dir| try std.fs.cwd().makePath(dir);
    var file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(&buf);
    const w = &fw.interface;
    try w.writeAll(magic);
    const c = m.cfg;
    const header = [7]u32{ @intCast(c.vocab_size), @intCast(c.context_length), @intCast(c.emb_dim), @intCast(c.n_heads), @intCast(c.n_layers), @intFromBool(c.qkv_bias), @intFromBool(opt != null) };
    try w.writeAll(std.mem.sliceAsBytes(&header));
    try writeTensors(gpa, w, m);
    if (opt) |o| {
        const step = [1]f32{@floatFromInt(o.step_count)};
        try w.writeAll(std.mem.sliceAsBytes(&step));
        try writeTensors(gpa, w, o.exp_avg);
        try writeTensors(gpa, w, o.exp_avg_sq);
    }
    try w.flush();
}

pub const Loaded = struct { model: gpt.GptModel, opt: ?AdamW };

/// Read a checkpoint. Returns the model, and the optimizer state if the file has it
/// (`lr` and `weight_decay` are settings, not saved).
pub fn load(gpa: std.mem.Allocator, path: []const u8, lr: f64, weight_decay: f64) !Loaded {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var buf: [1 << 16]u8 = undefined;
    var fr = file.reader(&buf);
    const r = &fr.interface;
    var head: [8]u8 = undefined;
    try r.readSliceAll(&head);
    if (!std.mem.eql(u8, &head, magic)) return error.NotACheckpoint;
    var h: [7]u32 = undefined;
    try r.readSliceAll(std.mem.sliceAsBytes(&h));
    const cfg = gpt.Config{ .vocab_size = h[0], .context_length = h[1], .emb_dim = h[2], .n_heads = h[3], .n_layers = h[4], .drop_rate = 0, .qkv_bias = h[5] != 0 };
    const model = try gpt.GptModel.init(gpa, cfg);
    try readTensors(gpa, r, model);
    if (h[6] == 0) return .{ .model = model, .opt = null };
    var opt = try AdamW.init(gpa, model, lr, weight_decay);
    var step: [1]f32 = undefined;
    try r.readSliceAll(std.mem.sliceAsBytes(&step));
    opt.step_count = @intFromFloat(step[0]);
    try readTensors(gpa, r, opt.exp_avg);
    try readTensors(gpa, r, opt.exp_avg_sq);
    return .{ .model = model, .opt = opt };
}
