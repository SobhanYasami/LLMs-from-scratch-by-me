//! Step 18 (book 5.2) - the backward pass, by hand.
//!
//! Training needs to know, for EVERY weight, "if I make this weight a little bigger, does the
//! loss go up or down, and how fast?" That number is the weight's gradient. torch's autograd
//! finds all of them automatically; here we do the same with the chain rule:
//!
//! 1. run the forward pass and REMEMBER what each layer saw and produced (the "cache");
//! 2. start at the loss and walk the layers backwards, turning "how the loss depends on this
//!    layer's output" into "how it depends on its weights and on its input".
//!
//! Every layer has a small backward rule:
//!
//! ```text
//! y = x W + b       dW = x^T dy     db = sum of dy over tokens     dx = dy W^T
//! y = gelu(x)       dx = dy * gelu'(x)
//! y = layer_norm(x) see ln_backward
//! y = x + f(x)      dx = dy + (what flows back through f)        <- the shortcut, chapter 4.4
//! attention         softmax backward: dS = P * (dP - sum(dP * P) per row)
//! ```
//!
//! The gradients are stored in a `GptModel` of the same shape (a handy trick: the optimizer and
//! the checkpoint code then treat weights, gradients and optimizer state alike).

use super::loss::log_sum_exp;
use crate::attention::Matrix;
use crate::gpt::{GptModel, LayerNorm, gelu};
use crate::tokenization::dataloader::Batch;

const LN_EPS: f32 = 1e-5;

fn attn_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f64).sqrt() as f32
}

fn add_into(dst: &mut Matrix, src: &Matrix) {
    for (d, s) in dst.data.iter_mut().zip(&src.data) {
        *d += s;
    }
}

fn add_bias(m: &mut Matrix, bias: &[f32]) {
    for row in m.data.chunks_mut(bias.len()) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// d gelu(x) / dx for the tanh approximation.
fn gelu_grad(x: f32) -> f32 {
    let v = x as f64;
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let t = (c * (v + 0.044715 * v * v * v)).tanh();
    (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * v * v)) as f32
}

/// What layer norm remembers for the backward pass.
struct LnCache {
    xhat: Matrix,  // the normalized values, before scale and shift
    inv: Vec<f32>, // 1 / sqrt(variance + eps) per row
}

fn ln_forward(ln: &LayerNorm, x: &Matrix) -> (Matrix, LnCache) {
    let (mut out, mut xhat) = (Matrix::new(x.rows, x.cols), Matrix::new(x.rows, x.cols));
    let mut inv = vec![0.0; x.rows];
    let n = x.cols as f32;
    for (i, inv_i) in inv.iter_mut().enumerate() {
        let row = x.row(i);
        let mean = row.iter().sum::<f32>() / n;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        *inv_i = (1.0 / ((variance + LN_EPS) as f64).sqrt()) as f32;
        for (j, &v) in row.iter().enumerate() {
            let h = (v - mean) * *inv_i;
            xhat.data[i * x.cols + j] = h;
            out.data[i * x.cols + j] = ln.scale[j] * h + ln.shift[j];
        }
    }
    (out, LnCache { xhat, inv })
}

/// Layer norm backward. With dxhat = dy * scale:
/// dx = inv * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)); dscale += dy * xhat; dshift += dy.
fn ln_backward(
    ln: &LayerNorm,
    c: &LnCache,
    dy: &Matrix,
    dscale: &mut [f32],
    dshift: &mut [f32],
) -> Matrix {
    let mut dx = Matrix::new(dy.rows, dy.cols);
    let n = dy.cols as f32;
    for i in 0..dy.rows {
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for j in 0..dy.cols {
            let dxhat = dy.at(i, j) * ln.scale[j];
            s1 += dxhat;
            s2 += dxhat * c.xhat.at(i, j);
            dscale[j] += dy.at(i, j) * c.xhat.at(i, j);
            dshift[j] += dy.at(i, j);
        }
        for j in 0..dy.cols {
            let dxhat = dy.at(i, j) * ln.scale[j];
            dx.data[i * dy.cols + j] = c.inv[i] * (dxhat - s1 / n - c.xhat.at(i, j) * (s2 / n));
        }
    }
    dx
}

/// y = x W (+ b): returns dx and adds to dW and db.
fn linear_backward(
    x: &Matrix,
    w: &Matrix,
    dy: &Matrix,
    dw: &mut Matrix,
    db: Option<&mut [f32]>,
) -> Matrix {
    add_into(dw, &x.transpose().matmul(dy));
    if let Some(db) = db {
        for row in dy.data.chunks(dy.cols) {
            for (d, v) in db.iter_mut().zip(row) {
                *d += v;
            }
        }
    }
    dy.matmul(&w.transpose())
}

/// Everything one transformer block saw in the forward pass.
struct BlockCache {
    x_in: Matrix,
    ln1: LnCache,
    n1: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    probs: Vec<Matrix>, // attention weights of each head [T, T]
    ctx: Matrix,
    ln2: LnCache,
    n2: Matrix,
    h1: Matrix,
    g: Matrix,
}

/// Forward pass of the whole model for one sequence, keeping the cache. Returns the logits.
fn forward_cached(model: &GptModel, ids: &[u32]) -> (Matrix, Vec<BlockCache>, Matrix, LnCache) {
    let mut x = model.embed(ids);
    let mut caches = Vec::new();
    for b in &model.blocks {
        let a = &b.att;
        let x_in = x.clone();
        let (n1, ln1) = ln_forward(&b.norm1, &x_in);
        let (q, k, v) = (
            n1.matmul(&a.w_query),
            n1.matmul(&a.w_key),
            n1.matmul(&a.w_value),
        );
        let mut ctx = Matrix::new(x.rows, a.num_heads * a.head_dim);
        let mut probs = Vec::new();
        for h in 0..a.num_heads {
            let (lo, hi) = (h * a.head_dim, (h + 1) * a.head_dim);
            let (qh, kh, vh) = (
                q.slice_cols(lo, hi),
                k.slice_cols(lo, hi),
                v.slice_cols(lo, hi),
            );
            let scores = qh.matmul(&kh.transpose()).mask_future();
            let weights = scores.scale(attn_scale(a.head_dim)).softmax_rows();
            let head = weights.matmul(&vh);
            for i in 0..head.rows {
                ctx.data[i * ctx.cols + lo..i * ctx.cols + hi].copy_from_slice(head.row(i));
            }
            probs.push(weights);
        }
        let mut att_out = ctx.matmul(&a.out_weight);
        add_bias(&mut att_out, &a.out_bias);
        add_into(&mut att_out, &x_in);
        let x_mid = att_out;
        let (n2, ln2) = ln_forward(&b.norm2, &x_mid);
        let mut h1 = n2.matmul(&b.ff.expand.w);
        add_bias(&mut h1, &b.ff.expand.b);
        let mut g = h1.clone();
        g.data.iter_mut().for_each(|v| *v = gelu(*v));
        let mut f = g.matmul(&b.ff.shrink.w);
        add_bias(&mut f, &b.ff.shrink.b);
        add_into(&mut f, &x_mid);
        x = f;
        caches.push(BlockCache {
            x_in,
            ln1,
            n1,
            q,
            k,
            v,
            probs,
            ctx,
            ln2,
            n2,
            h1,
            g,
        });
    }
    let (nf, lnf) = ln_forward(&model.final_norm, &x);
    let logits = nf.matmul(&model.out_head);
    (logits, caches, nf, lnf)
}

/// Mean cross entropy of the batch, and the gradient of that mean for every weight, ADDED into `grads`
/// (zero it first with `zero_grads`).
pub fn loss_and_grads(model: &GptModel, grads: &mut GptModel, batch: &Batch) -> f32 {
    assert!(
        model.blocks.iter().all(|b| b.att.qkv_bias.is_none()),
        "training without the attention biases only"
    );
    let n_tokens: usize = batch.targets.iter().map(Vec::len).sum();
    let mut total = 0.0f32;
    for (ids, targets) in batch.inputs.iter().zip(&batch.targets) {
        let (logits, caches, nf, lnf) = forward_cached(model, ids);

        // loss and its gradient: d loss / d logits = (softmax - one_hot) / N
        let mut dlogits = Matrix::new(logits.rows, logits.cols);
        for (i, &t) in targets.iter().enumerate() {
            let row = logits.row(i);
            let lse = log_sum_exp(row);
            total += lse - row[t as usize];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &v in row {
                sum += ((v - max) as f64).exp() as f32;
            }
            for (j, &v) in row.iter().enumerate() {
                let p = ((v - max) as f64).exp() as f32 / sum;
                let one_hot = if j == t as usize { 1.0 } else { 0.0 };
                dlogits.data[i * logits.cols + j] = (p - one_hot) / n_tokens as f32;
            }
        }

        // output head and final layer norm
        let dnf = linear_backward(&nf, &model.out_head, &dlogits, &mut grads.out_head, None);
        let mut dx = ln_backward(
            &model.final_norm,
            &lnf,
            &dnf,
            &mut grads.final_norm.scale,
            &mut grads.final_norm.shift,
        );

        // blocks, last to first
        for bi in (0..model.blocks.len()).rev() {
            let (b, c) = (&model.blocks[bi], &caches[bi]);
            let gb = &mut grads.blocks[bi];
            let a = &b.att;
            // feed forward: x_out = x_mid + ff(ln2(x_mid))
            let dg = linear_backward(
                &c.g,
                &b.ff.shrink.w,
                &dx,
                &mut gb.ff.shrink.w,
                Some(&mut gb.ff.shrink.b),
            );
            let mut dh1 = dg;
            for (d, h) in dh1.data.iter_mut().zip(&c.h1.data) {
                *d *= gelu_grad(*h);
            }
            let dn2 = linear_backward(
                &c.n2,
                &b.ff.expand.w,
                &dh1,
                &mut gb.ff.expand.w,
                Some(&mut gb.ff.expand.b),
            );
            let mut dx_mid = ln_backward(
                &b.norm2,
                &c.ln2,
                &dn2,
                &mut gb.norm2.scale,
                &mut gb.norm2.shift,
            );
            add_into(&mut dx_mid, &dx); // the shortcut
            // attention: x_mid = x_in + att(ln1(x_in))
            let dctx = linear_backward(
                &c.ctx,
                &a.out_weight,
                &dx_mid,
                &mut gb.att.out_weight,
                Some(&mut gb.att.out_bias),
            );
            let (mut dq, mut dk, mut dv) = (
                Matrix::new(c.q.rows, c.q.cols),
                Matrix::new(c.q.rows, c.q.cols),
                Matrix::new(c.q.rows, c.q.cols),
            );
            for h in 0..a.num_heads {
                let (lo, hi) = (h * a.head_dim, (h + 1) * a.head_dim);
                let (qh, kh, vh) = (
                    c.q.slice_cols(lo, hi),
                    c.k.slice_cols(lo, hi),
                    c.v.slice_cols(lo, hi),
                );
                let p = &c.probs[h];
                let dctx_h = dctx.slice_cols(lo, hi);
                let dp = dctx_h.matmul(&vh.transpose());
                let dvh = p.transpose().matmul(&dctx_h);
                // softmax backward, then the 1/sqrt(d) scale (masked entries have p = 0, so no gradient)
                let mut ds = Matrix::new(p.rows, p.cols);
                for i in 0..p.rows {
                    let mut dot = 0.0f32;
                    for j in 0..p.cols {
                        dot += dp.at(i, j) * p.at(i, j);
                    }
                    for j in 0..p.cols {
                        ds.data[i * p.cols + j] = p.at(i, j) * (dp.at(i, j) - dot);
                    }
                }
                let ds = ds.scale(attn_scale(a.head_dim));
                let dqh = ds.matmul(&kh);
                let dkh = ds.transpose().matmul(&qh);
                for i in 0..dqh.rows {
                    for j in 0..a.head_dim {
                        dq.data[i * dq.cols + lo + j] = dqh.at(i, j);
                        dk.data[i * dk.cols + lo + j] = dkh.at(i, j);
                        dv.data[i * dv.cols + lo + j] = dvh.at(i, j);
                    }
                }
            }
            let mut dn1 = linear_backward(&c.n1, &a.w_query, &dq, &mut gb.att.w_query, None);
            add_into(
                &mut dn1,
                &linear_backward(&c.n1, &a.w_key, &dk, &mut gb.att.w_key, None),
            );
            add_into(
                &mut dn1,
                &linear_backward(&c.n1, &a.w_value, &dv, &mut gb.att.w_value, None),
            );
            dx = ln_backward(
                &b.norm1,
                &c.ln1,
                &dn1,
                &mut gb.norm1.scale,
                &mut gb.norm1.shift,
            );
            add_into(&mut dx, &dx_mid); // the shortcut
            let _ = &c.x_in;
        }

        // embeddings: every token's row of dx goes back to its row of the token table
        for (t, &id) in ids.iter().enumerate() {
            for j in 0..dx.cols {
                grads.tok_emb.data[id as usize * dx.cols + j] += dx.at(t, j);
                grads.pos_emb.data[t * dx.cols + j] += dx.at(t, j);
            }
        }
    }
    total / n_tokens as f32
}

pub fn zero_grads(grads: &mut GptModel) {
    grads.tensors_mut().into_iter().for_each(|t| t.fill(0.0));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::{CONFIG_TINY, Config};
    use crate::tokenization::{dataloader::DataLoader, dataset::GptDataset};

    /// The numbers come from torch's autograd on the same model and batch (python-version):
    /// the loss, and the sum of |gradient| of a few tensors.
    #[test]
    fn gradients_match_torch_autograd() {
        let cfg = Config {
            drop_rate: 0.0,
            ..CONFIG_TINY
        };
        let model = GptModel::new_initialized(cfg, 123);
        let mut grads = model.zeros_like();
        let bpe = tiktoken_rs::r50k_base().unwrap();
        let text = std::fs::read_to_string("../assets/the-verdict.txt").unwrap();
        let ids = bpe.encode_ordinary(&text[..(0.9 * text.len() as f64) as usize]);
        let ds = GptDataset::new(&ids, 16, 16);
        let batches = DataLoader::new(&ds, 4, false, true, 1).epoch();
        let loss = loss_and_grads(&model, &mut grads, &batches[0]);
        assert!((loss - 10.843978).abs() < 1e-4, "loss {loss}");
        let sums: Vec<f32> = grads
            .tensors()
            .iter()
            .map(|t| t.iter().map(|v| v.abs()).sum())
            .collect();
        // token embedding, position embedding, block 0 W_query, block 0 feed forward 1 weight, last block W_value
        for (i, expected) in [
            (0, 16.59029),
            (1, 9.77129),
            (2, 0.03454),
            (11, 16.02808),
            (21, 0.03660),
        ] {
            assert!(
                (sums[i] - expected).abs() < 1e-3 * expected.max(1.0),
                "tensor {i}: {} vs {expected}",
                sums[i]
            );
        }
    }
}
