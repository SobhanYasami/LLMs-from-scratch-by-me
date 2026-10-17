//! Step 18 (book 5.2) - the backward pass, by hand.
//!
//! Training needs to know, for EVERY weight, "if I make this weight a little bigger, does the loss
//! go up or down, and how fast?" That number is the weight's gradient. torch's autograd finds all
//! of them automatically; here we do the same with the chain rule:
//!
//! 1. run the forward pass and REMEMBER what each layer saw and produced (the "cache");
//! 2. start at the loss and walk the layers backwards, turning "how the loss depends on this
//!    layer's output" into "how it depends on its weights and on its input".
//!
//! Every layer has a small backward rule:
//!
//! ```
//! y = x W + b       dW = x^T dy     db = sum of dy over tokens     dx = dy W^T
//! y = gelu(x)       dx = dy * gelu'(x)
//! y = layer_norm(x) see lnBackward
//! y = x + f(x)      dx = dy + (what flows back through f)        <- the shortcut, chapter 4.4
//! attention         softmax backward: dS = P * (dP - sum(dP * P) per row)
//! ```
//!
//! The gradients are stored in a `GptModel` of the same shape (a handy trick: the optimizer and the
//! checkpoint code then treat weights, gradients and optimizer state alike). Every sequence gets
//! its own scratch arena, freed as soon as its gradients are added in.

const std = @import("std");
const att = @import("../attention/root.zig");
const gpt = @import("../gpt/root.zig");
const loss = @import("loss.zig");

const Matrix = att.Matrix;
const Allocator = std.mem.Allocator;
const ln_eps: f32 = 1e-5;

fn attnScale(head_dim: usize) f32 {
    const root: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(head_dim))));
    return 1.0 / root;
}

fn addInto(dst: Matrix, src: Matrix) void {
    for (dst.data, src.data) |*d, s| d.* += s;
}

fn addBias(m: Matrix, bias: []const f32) void {
    for (0..m.rows) |i| {
        for (m.row(i), bias) |*v, b| v.* += b;
    }
}

/// d gelu(x) / dx for the tanh approximation.
fn geluGrad(x: f32) f32 {
    const v: f64 = x;
    const c = @sqrt(2.0 / std.math.pi);
    const t = std.math.tanh(c * (v + 0.044715 * v * v * v));
    return @floatCast(0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * v * v));
}

/// What layer norm remembers for the backward pass.
const LnCache = struct {
    xhat: Matrix, // the normalized values, before scale and shift
    inv: []f32, // 1 / sqrt(variance + eps) per row
};

fn lnForward(gpa: Allocator, ln: gpt.LayerNorm, x: Matrix) !struct { Matrix, LnCache } {
    const out = try Matrix.init(gpa, x.rows, x.cols);
    const xhat = try Matrix.init(gpa, x.rows, x.cols);
    const inv = try gpa.alloc(f32, x.rows);
    const n: f32 = @floatFromInt(x.cols);
    for (0..x.rows) |i| {
        const r = x.row(i);
        var mean: f32 = 0;
        for (r) |v| mean += v;
        mean /= n;
        var variance: f32 = 0;
        for (r) |v| variance += (v - mean) * (v - mean);
        variance /= n;
        inv[i] = @floatCast(1.0 / @sqrt(@as(f64, variance + ln_eps)));
        for (r, 0..) |v, j| {
            const h = (v - mean) * inv[i];
            xhat.data[i * x.cols + j] = h;
            out.data[i * x.cols + j] = ln.scale[j] * h + ln.shift[j];
        }
    }
    return .{ out, .{ .xhat = xhat, .inv = inv } };
}

/// With dxhat = dy * scale: dx = inv * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat));
/// dscale += dy * xhat; dshift += dy.
fn lnBackward(gpa: Allocator, ln: gpt.LayerNorm, c: LnCache, dy: Matrix, g: gpt.LayerNorm) !Matrix {
    const dx = try Matrix.init(gpa, dy.rows, dy.cols);
    const n: f32 = @floatFromInt(dy.cols);
    for (0..dy.rows) |i| {
        var s1: f32 = 0;
        var s2: f32 = 0;
        for (0..dy.cols) |j| {
            const dxhat = dy.at(i, j) * ln.scale[j];
            s1 += dxhat;
            s2 += dxhat * c.xhat.at(i, j);
            g.scale[j] += dy.at(i, j) * c.xhat.at(i, j);
            g.shift[j] += dy.at(i, j);
        }
        for (0..dy.cols) |j| {
            const dxhat = dy.at(i, j) * ln.scale[j];
            dx.data[i * dy.cols + j] = c.inv[i] * (dxhat - s1 / n - c.xhat.at(i, j) * (s2 / n));
        }
    }
    return dx;
}

/// y = x W (+ b): returns dx and adds to dW and db (db may be null).
fn linearBackward(gpa: Allocator, x: Matrix, w: Matrix, dy: Matrix, dw: Matrix, db: ?[]f32) !Matrix {
    addInto(dw, try (try x.transpose(gpa)).matmul(gpa, dy));
    if (db) |d| {
        for (0..dy.rows) |i| {
            for (dy.row(i), d) |v, *dj| dj.* += v;
        }
    }
    return dy.matmul(gpa, try w.transpose(gpa));
}

/// Everything one transformer block saw in the forward pass.
const BlockCache = struct {
    ln1: LnCache,
    n1: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    probs: []Matrix, // attention weights of each head [T, T]
    ctx: Matrix,
    ln2: LnCache,
    n2: Matrix,
    h1: Matrix,
    g: Matrix,
};

/// Mean cross entropy of the batch, and the gradient of that mean for every weight, ADDED into
/// `grads` (zero it first with `zeroGrads`).
pub fn lossAndGrads(model: gpt.GptModel, grads: gpt.GptModel, batch: loss.Batch) !f32 {
    for (model.blocks) |b| std.debug.assert(b.att.qkv_bias == null); // training without the attention biases only
    var n_tokens: usize = 0;
    for (batch.targets) |t| n_tokens += t.len;
    const n_f: f32 = @floatFromInt(n_tokens);
    var total: f32 = 0;
    for (batch.inputs, batch.targets) |ids, targets| {
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const gpa = scratch.allocator();

        // forward pass, keeping the cache
        var x = try model.embed(gpa, ids);
        const caches = try gpa.alloc(BlockCache, model.blocks.len);
        for (model.blocks, caches) |b, *c| {
            const a = b.att;
            const x_in = x;
            const n1 = try lnForward(gpa, b.norm1, x_in);
            c.n1 = n1[0];
            c.ln1 = n1[1];
            c.q = try c.n1.matmul(gpa, a.w_query);
            c.k = try c.n1.matmul(gpa, a.w_key);
            c.v = try c.n1.matmul(gpa, a.w_value);
            c.ctx = try Matrix.init(gpa, x.rows, a.num_heads * a.head_dim);
            c.probs = try gpa.alloc(Matrix, a.num_heads);
            for (0..a.num_heads) |h| {
                const lo = h * a.head_dim;
                const hi = lo + a.head_dim;
                const qh = try c.q.sliceCols(gpa, lo, hi);
                const kh = try c.k.sliceCols(gpa, lo, hi);
                const vh = try c.v.sliceCols(gpa, lo, hi);
                const scores = try (try qh.matmul(gpa, try kh.transpose(gpa))).maskFuture(gpa);
                c.probs[h] = try (try scores.scale(gpa, attnScale(a.head_dim))).softmaxRows(gpa);
                const head = try c.probs[h].matmul(gpa, vh);
                for (0..head.rows) |i| @memcpy(c.ctx.row(i)[lo..hi], head.row(i));
            }
            const x_mid = try c.ctx.matmul(gpa, a.out_weight);
            addBias(x_mid, a.out_bias.data);
            addInto(x_mid, x_in);
            const n2 = try lnForward(gpa, b.norm2, x_mid);
            c.n2 = n2[0];
            c.ln2 = n2[1];
            c.h1 = try c.n2.matmul(gpa, b.ff.expand.w);
            addBias(c.h1, b.ff.expand.b);
            c.g = try Matrix.init(gpa, c.h1.rows, c.h1.cols);
            for (c.h1.data, c.g.data) |h, *o| o.* = gpt.gelu(h);
            const f = try c.g.matmul(gpa, b.ff.shrink.w);
            addBias(f, b.ff.shrink.b);
            addInto(f, x_mid);
            x = f;
        }
        const nfr = try lnForward(gpa, model.final_norm, x);
        const nf = nfr[0];
        const lnf = nfr[1];
        const logits = try nf.matmul(gpa, model.out_head);

        // loss and its gradient: d loss / d logits = (softmax - one_hot) / N
        const dlogits = try Matrix.init(gpa, logits.rows, logits.cols);
        for (targets, 0..) |t, i| {
            const r = logits.row(i);
            total += loss.logSumExp(r) - r[t];
            var max: f32 = -std.math.inf(f32);
            for (r) |v| max = @max(max, v);
            var sum: f32 = 0;
            for (r) |v| sum += @floatCast(@exp(@as(f64, v - max)));
            for (r, 0..) |v, j| {
                const p = @as(f32, @floatCast(@exp(@as(f64, v - max)))) / sum;
                const one_hot: f32 = if (j == t) 1.0 else 0.0;
                dlogits.data[i * logits.cols + j] = (p - one_hot) / n_f;
            }
        }

        // output head and final layer norm
        const dnf = try linearBackward(gpa, nf, model.out_head, dlogits, grads.out_head, null);
        var dx = try lnBackward(gpa, model.final_norm, lnf, dnf, grads.final_norm);

        // blocks, last to first
        var bi = model.blocks.len;
        while (bi > 0) {
            bi -= 1;
            const b = model.blocks[bi];
            const c = caches[bi];
            const gb = grads.blocks[bi];
            const a = b.att;
            // feed forward: x_out = x_mid + ff(ln2(x_mid))
            const dh1 = try linearBackward(gpa, c.g, b.ff.shrink.w, dx, gb.ff.shrink.w, gb.ff.shrink.b);
            for (dh1.data, c.h1.data) |*d, h| d.* *= geluGrad(h);
            const dn2 = try linearBackward(gpa, c.n2, b.ff.expand.w, dh1, gb.ff.expand.w, gb.ff.expand.b);
            const dx_mid = try lnBackward(gpa, b.norm2, c.ln2, dn2, gb.norm2);
            addInto(dx_mid, dx); // the shortcut
            // attention: x_mid = x_in + att(ln1(x_in))
            const dctx = try linearBackward(gpa, c.ctx, a.out_weight, dx_mid, gb.att.out_weight, gb.att.out_bias.data);
            const dq = try Matrix.init(gpa, c.q.rows, c.q.cols);
            const dk = try Matrix.init(gpa, c.q.rows, c.q.cols);
            const dv = try Matrix.init(gpa, c.q.rows, c.q.cols);
            for (0..a.num_heads) |h| {
                const lo = h * a.head_dim;
                const hi = lo + a.head_dim;
                const qh = try c.q.sliceCols(gpa, lo, hi);
                const kh = try c.k.sliceCols(gpa, lo, hi);
                const vh = try c.v.sliceCols(gpa, lo, hi);
                const p = c.probs[h];
                const dctx_h = try dctx.sliceCols(gpa, lo, hi);
                const dp = try dctx_h.matmul(gpa, try vh.transpose(gpa));
                const dvh = try (try p.transpose(gpa)).matmul(gpa, dctx_h);
                // softmax backward, then the 1/sqrt(d) scale (masked entries have p = 0, so no gradient)
                var ds = try Matrix.init(gpa, p.rows, p.cols);
                for (0..p.rows) |i| {
                    var dot: f32 = 0;
                    for (0..p.cols) |j| dot += dp.at(i, j) * p.at(i, j);
                    for (0..p.cols) |j| ds.data[i * p.cols + j] = p.at(i, j) * (dp.at(i, j) - dot);
                }
                ds = try ds.scale(gpa, attnScale(a.head_dim));
                const dqh = try ds.matmul(gpa, kh);
                const dkh = try (try ds.transpose(gpa)).matmul(gpa, qh);
                for (0..dqh.rows) |i| {
                    for (0..a.head_dim) |j| {
                        dq.data[i * dq.cols + lo + j] = dqh.at(i, j);
                        dk.data[i * dk.cols + lo + j] = dkh.at(i, j);
                        dv.data[i * dv.cols + lo + j] = dvh.at(i, j);
                    }
                }
            }
            const dn1 = try linearBackward(gpa, c.n1, a.w_query, dq, gb.att.w_query, null);
            addInto(dn1, try linearBackward(gpa, c.n1, a.w_key, dk, gb.att.w_key, null));
            addInto(dn1, try linearBackward(gpa, c.n1, a.w_value, dv, gb.att.w_value, null));
            dx = try lnBackward(gpa, b.norm1, c.ln1, dn1, gb.norm1);
            addInto(dx, dx_mid); // the shortcut
        }

        // embeddings: every token's row of dx goes back to its row of the token table
        for (ids, 0..) |id, t| {
            for (0..dx.cols) |j| {
                grads.tok_emb.data[id * dx.cols + j] += dx.at(t, j);
                grads.pos_emb.data[t * dx.cols + j] += dx.at(t, j);
            }
        }
    }
    return total / n_f;
}

/// Set every gradient to 0.
pub fn zeroGrads(gpa: Allocator, grads: gpt.GptModel) !void {
    for (try grads.tensors(gpa)) |t| @memset(t, 0);
}
