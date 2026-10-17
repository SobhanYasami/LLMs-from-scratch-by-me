//! Step 21 (book 5.5) - loading OpenAI's pretrained GPT-2 weights.
//!
//! Training a GPT-2 from scratch costs a fortune; OpenAI published the result. The weights come as
//! a `.safetensors` file, a very plain format: 8 bytes with the header length, a JSON header that
//! says where each tensor lives, then raw f32 numbers. No library needed.
//!
//! Download once (about 500 MB, the file is ignored by git):
//!
//! ```
//! curl -L -o assets/gpt2/model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
//! ```
//!
//! OpenAI's layout differs from ours in a few ways, which `loadGpt2Weights` fixes: one big `c_attn`
//! matrix holds query, key and value side by side (we keep three); the attention projections carry
//! biases (`qkv_bias = true`); and the output head re-uses the token embedding (weight tying), so we
//! copy `wte` into it. Matrices are stored [in, out], like ours.

const std = @import("std");
const gpt = @import("../gpt/root.zig");
const Matrix = @import("../attention/root.zig").Matrix;

pub const gpt2_small = gpt.Config{ .vocab_size = 50257, .context_length = 1024, .emb_dim = 768, .n_heads = 12, .n_layers = 12, .drop_rate = 0, .qkv_bias = true };

/// A `.safetensors` file opened for reading tensors by name.
pub const SafeTensors = struct {
    file: std.fs.File,
    header: []u8,
    base: u64,

    pub fn open(gpa: std.mem.Allocator, path: []const u8) !SafeTensors {
        const file = try std.fs.cwd().openFile(path, .{});
        var len_bytes: [8]u8 = undefined;
        if (try file.preadAll(&len_bytes, 0) != 8) return error.BadFile;
        const n = std.mem.readInt(u64, &len_bytes, .little);
        const header = try gpa.alloc(u8, n);
        if (try file.preadAll(header, 8) != n) return error.BadFile;
        return .{ .file = file, .header = header, .base = 8 + n };
    }

    pub fn close(self: SafeTensors) void {
        self.file.close();
    }

    /// Reads `"key":[1,2,3]` from an entry into `out`; returns how many numbers.
    fn readList(entry: []const u8, key: []const u8, out: []u64) usize {
        var pattern: [64]u8 = undefined;
        const p = std.fmt.bufPrint(&pattern, "\"{s}\":[", .{key}) catch return 0;
        const at = std.mem.indexOf(u8, entry, p) orelse return 0;
        const rest = entry[at + p.len ..];
        const inner = rest[0 .. std.mem.indexOfScalar(u8, rest, ']') orelse return 0];
        var n: usize = 0;
        var parts = std.mem.splitScalar(u8, inner, ',');
        while (parts.next()) |part| {
            if (part.len == 0) continue;
            out[n] = std.fmt.parseInt(u64, std.mem.trim(u8, part, " "), 10) catch return 0;
            n += 1;
        }
        return n;
    }

    /// Reads the tensor `name` into `dst`. The header looks like
    /// `"name":{"dtype":"F32","shape":[768,2304],"data_offsets":[0,7077888]}`; we find the entry by name.
    pub fn get(self: SafeTensors, name: []const u8, dst: []f32) !void {
        var key_buf: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "\"{s}\":", .{name});
        const at = std.mem.indexOf(u8, self.header, key) orelse return error.TensorNotInFile;
        const entry = self.header[at .. at + (std.mem.indexOfScalar(u8, self.header[at..], '}') orelse return error.BadHeader)];
        if (std.mem.indexOf(u8, entry, "F32") == null) return error.OnlyFloat32;
        var off: [2]u64 = undefined;
        if (readList(entry, "data_offsets", &off) != 2) return error.BadHeader;
        if (off[1] - off[0] != dst.len * 4) return error.ShapeMismatch;
        if (try self.file.preadAll(std.mem.sliceAsBytes(dst), self.base + off[0]) != dst.len * 4) return error.ShortRead;
    }
};

/// Build a GPT-2 small model from the weights file.
pub fn loadGpt2Weights(gpa: std.mem.Allocator, path: []const u8) !gpt.GptModel {
    var st = try SafeTensors.open(gpa, path);
    defer st.close();
    const cfg = gpt2_small;
    const m = try gpt.GptModel.init(gpa, cfg);
    const d = cfg.emb_dim;
    try st.get("wte.weight", m.tok_emb.data);
    try st.get("wpe.weight", m.pos_emb.data);
    const c_attn = try gpa.alloc(f32, d * 3 * d);
    defer gpa.free(c_attn);
    for (m.blocks, 0..) |b, i| {
        var name_buf: [96]u8 = undefined;
        // c_attn is [d, 3d]: query, key and value side by side
        try st.get(try std.fmt.bufPrint(&name_buf, "h.{d}.attn.c_attn.weight", .{i}), c_attn);
        const proj = [3]Matrix{ b.att.w_query, b.att.w_key, b.att.w_value };
        for (proj, 0..) |w, which| {
            for (0..d) |r| @memcpy(w.data[r * d .. (r + 1) * d], c_attn[r * 3 * d + which * d .. r * 3 * d + (which + 1) * d]);
        }
        try st.get(try std.fmt.bufPrint(&name_buf, "h.{d}.attn.c_attn.bias", .{i}), c_attn[0 .. 3 * d]);
        for (b.att.qkv_bias.?, 0..) |bias, which| @memcpy(bias.data, c_attn[which * d .. (which + 1) * d]);
        const items = [_]struct { []const u8, []f32 }{
            .{ "attn.c_proj.weight", b.att.out_weight.data }, .{ "attn.c_proj.bias", b.att.out_bias.data },
            .{ "mlp.c_fc.weight", b.ff.expand.w.data },       .{ "mlp.c_fc.bias", b.ff.expand.b },
            .{ "mlp.c_proj.weight", b.ff.shrink.w.data },     .{ "mlp.c_proj.bias", b.ff.shrink.b },
            .{ "ln_1.weight", b.norm1.scale },                .{ "ln_1.bias", b.norm1.shift },
            .{ "ln_2.weight", b.norm2.scale },                .{ "ln_2.bias", b.norm2.shift },
        };
        for (items) |it| try st.get(try std.fmt.bufPrint(&name_buf, "h.{d}.{s}", .{ i, it[0] }), it[1]);
    }
    try st.get("ln_f.weight", m.final_norm.scale);
    try st.get("ln_f.bias", m.final_norm.shift);
    // weight tying: the head is the embedding again, [vocab, d] -> [d, vocab]
    const wte = try gpa.alloc(f32, cfg.vocab_size * d);
    defer gpa.free(wte);
    try st.get("wte.weight", wte);
    for (0..cfg.vocab_size) |tok| {
        for (0..d) |r| m.out_head.data[r * cfg.vocab_size + tok] = wte[tok * d + r];
    }
    return m;
}
