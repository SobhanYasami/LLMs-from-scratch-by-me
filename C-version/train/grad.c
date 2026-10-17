#include "grad.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LN_EPS 1e-5f

static float attn_scale(size_t head_dim) { return 1.0f / (float)sqrt((double)head_dim); }

static void add_into(Matrix dst, Matrix src) {
    for (size_t i = 0; i < dst.rows * dst.cols; i++) dst.data[i] += src.data[i];
}

static void add_bias(Matrix m, const float *bias) {
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++) m.data[i * m.cols + j] += bias[j];
}

/* d gelu(x) / dx for the tanh approximation. */
static float gelu_grad(float x) {
    double v = x, c = sqrt(2.0 / M_PI);
    double t = tanh(c * (v + 0.044715 * v * v * v));
    return (float)(0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * v * v));
}

/* What layer norm remembers for the backward pass. */
typedef struct {
    Matrix xhat; /* the normalized values, before scale and shift */
    float *inv;  /* 1 / sqrt(variance + eps) per row */
} LnCache;

static Matrix ln_forward(Arena *a, const LayerNorm *ln, Matrix x, LnCache *c) {
    Matrix out = matrix_new(a, x.rows, x.cols);
    c->xhat = matrix_new(a, x.rows, x.cols);
    c->inv = arena_alloc(a, x.rows * sizeof(float));
    float n = (float)x.cols;
    for (size_t i = 0; i < x.rows; i++) {
        const float *row = matrix_row(x, i);
        float mean = 0, variance = 0;
        for (size_t j = 0; j < x.cols; j++) mean += row[j];
        mean /= n;
        for (size_t j = 0; j < x.cols; j++) variance += (row[j] - mean) * (row[j] - mean);
        variance /= n;
        c->inv[i] = (float)(1.0 / sqrt((double)(variance + LN_EPS)));
        for (size_t j = 0; j < x.cols; j++) {
            float h = (row[j] - mean) * c->inv[i];
            c->xhat.data[i * x.cols + j] = h;
            out.data[i * x.cols + j] = ln->scale.data[j] * h + ln->shift.data[j];
        }
    }
    return out;
}

/* With dxhat = dy * scale: dx = inv * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat));
 * dscale += dy * xhat; dshift += dy. */
static Matrix ln_backward(Arena *a, const LayerNorm *ln, const LnCache *c, Matrix dy, LayerNorm *g) {
    Matrix dx = matrix_new(a, dy.rows, dy.cols);
    float n = (float)dy.cols;
    for (size_t i = 0; i < dy.rows; i++) {
        float s1 = 0, s2 = 0;
        for (size_t j = 0; j < dy.cols; j++) {
            float dxhat = matrix_at(dy, i, j) * ln->scale.data[j];
            s1 += dxhat;
            s2 += dxhat * matrix_at(c->xhat, i, j);
            g->scale.data[j] += matrix_at(dy, i, j) * matrix_at(c->xhat, i, j);
            g->shift.data[j] += matrix_at(dy, i, j);
        }
        for (size_t j = 0; j < dy.cols; j++) {
            float dxhat = matrix_at(dy, i, j) * ln->scale.data[j];
            dx.data[i * dy.cols + j] = c->inv[i] * (dxhat - s1 / n - matrix_at(c->xhat, i, j) * (s2 / n));
        }
    }
    return dx;
}

/* y = x W (+ b): returns dx and adds to dW and db (db may be NULL). */
static Matrix linear_backward(Arena *a, Matrix x, Matrix w, Matrix dy, Matrix dw, float *db) {
    add_into(dw, matmul(a, transpose(a, x), dy));
    if (db)
        for (size_t i = 0; i < dy.rows; i++)
            for (size_t j = 0; j < dy.cols; j++) db[j] += matrix_at(dy, i, j);
    return matmul(a, dy, transpose(a, w));
}

/* Everything one transformer block saw in the forward pass. */
typedef struct {
    LnCache ln1, ln2;
    Matrix n1, q, k, v, ctx, n2, h1, g;
    Matrix *probs; /* attention weights of each head [T, T] */
} BlockCache;

float loss_and_grads(Arena *a, const GptModel *m, GptModel *grads, const Batch *batch) {
    size_t n_layers = m->cfg.n_layers, n_tokens = batch->rows * batch->len;
    for (size_t i = 0; i < n_layers; i++)
        if (m->blocks[i].att.has_qkv_bias) {
            fputs("training without the attention biases only\n", stderr);
            exit(1);
        }
    float total = 0;
    for (size_t r = 0; r < batch->rows; r++) {
        size_t mark = arena_mark(a);
        const int *ids = batch->inputs + r * batch->len, *targets = batch->targets + r * batch->len;

        /* forward pass, keeping the cache */
        Matrix x = gpt_embed(a, m, ids, batch->len);
        BlockCache *caches = arena_alloc(a, n_layers * sizeof(BlockCache));
        for (size_t bi = 0; bi < n_layers; bi++) {
            const TransformerBlock *b = &m->blocks[bi];
            const MultiHeadAttention *att = &b->att;
            BlockCache *c = &caches[bi];
            Matrix x_in = x;
            c->n1 = ln_forward(a, &b->norm1, x_in, &c->ln1);
            c->q = matmul(a, c->n1, att->w_query);
            c->k = matmul(a, c->n1, att->w_key);
            c->v = matmul(a, c->n1, att->w_value);
            c->ctx = matrix_new(a, x.rows, att->num_heads * att->head_dim);
            c->probs = arena_alloc(a, att->num_heads * sizeof(Matrix));
            for (size_t h = 0; h < att->num_heads; h++) {
                size_t lo = h * att->head_dim, hi = lo + att->head_dim;
                Matrix qh = slice_cols(a, c->q, lo, hi), kh = slice_cols(a, c->k, lo, hi), vh = slice_cols(a, c->v, lo, hi);
                Matrix scores = mask_future(a, matmul(a, qh, transpose(a, kh)));
                c->probs[h] = softmax_rows(a, scale(a, scores, attn_scale(att->head_dim)));
                Matrix head = matmul(a, c->probs[h], vh);
                for (size_t i = 0; i < head.rows; i++) memcpy(matrix_row(c->ctx, i) + lo, matrix_row(head, i), att->head_dim * sizeof(float));
            }
            Matrix x_mid = matmul(a, c->ctx, att->out_weight);
            add_bias(x_mid, att->out_bias.data);
            add_into(x_mid, x_in);
            c->n2 = ln_forward(a, &b->norm2, x_mid, &c->ln2);
            c->h1 = matmul(a, c->n2, b->ff.expand.w);
            add_bias(c->h1, b->ff.expand.b.data);
            c->g = matrix_new(a, c->h1.rows, c->h1.cols);
            for (size_t i = 0; i < c->h1.rows * c->h1.cols; i++) c->g.data[i] = gelu(c->h1.data[i]);
            Matrix f = matmul(a, c->g, b->ff.shrink.w);
            add_bias(f, b->ff.shrink.b.data);
            add_into(f, x_mid);
            x = f;
        }
        LnCache lnf;
        Matrix nf = ln_forward(a, &m->final_norm, x, &lnf);
        Matrix logits = matmul(a, nf, m->out_head);

        /* loss and its gradient: d loss / d logits = (softmax - one_hot) / N */
        Matrix dlogits = matrix_new(a, logits.rows, logits.cols);
        for (size_t i = 0; i < logits.rows; i++) {
            const float *row = matrix_row(logits, i);
            total += log_sum_exp(row, logits.cols) - row[targets[i]];
            float max = -INFINITY, sum = 0;
            for (size_t j = 0; j < logits.cols; j++)
                if (row[j] > max) max = row[j];
            for (size_t j = 0; j < logits.cols; j++) sum += (float)exp((double)(row[j] - max));
            for (size_t j = 0; j < logits.cols; j++) {
                float p = (float)exp((double)(row[j] - max)) / sum;
                float one_hot = (int)j == targets[i] ? 1.0f : 0.0f;
                dlogits.data[i * logits.cols + j] = (p - one_hot) / (float)n_tokens;
            }
        }

        /* output head and final layer norm */
        Matrix dnf = linear_backward(a, nf, m->out_head, dlogits, grads->out_head, NULL);
        Matrix dx = ln_backward(a, &m->final_norm, &lnf, dnf, &grads->final_norm);

        /* blocks, last to first */
        for (size_t bi = n_layers; bi-- > 0;) {
            const TransformerBlock *b = &m->blocks[bi];
            const MultiHeadAttention *att = &b->att;
            const BlockCache *c = &caches[bi];
            TransformerBlock *gb = &grads->blocks[bi];
            /* feed forward: x_out = x_mid + ff(ln2(x_mid)) */
            Matrix dh1 = linear_backward(a, c->g, b->ff.shrink.w, dx, gb->ff.shrink.w, gb->ff.shrink.b.data);
            for (size_t i = 0; i < dh1.rows * dh1.cols; i++) dh1.data[i] *= gelu_grad(c->h1.data[i]);
            Matrix dn2 = linear_backward(a, c->n2, b->ff.expand.w, dh1, gb->ff.expand.w, gb->ff.expand.b.data);
            Matrix dx_mid = ln_backward(a, &b->norm2, &c->ln2, dn2, &gb->norm2);
            add_into(dx_mid, dx); /* the shortcut */
            /* attention: x_mid = x_in + att(ln1(x_in)) */
            Matrix dctx = linear_backward(a, c->ctx, att->out_weight, dx_mid, gb->att.out_weight, gb->att.out_bias.data);
            Matrix dq = matrix_new(a, c->q.rows, c->q.cols), dk = matrix_new(a, c->q.rows, c->q.cols), dv = matrix_new(a, c->q.rows, c->q.cols);
            for (size_t h = 0; h < att->num_heads; h++) {
                size_t lo = h * att->head_dim, hi = lo + att->head_dim;
                Matrix qh = slice_cols(a, c->q, lo, hi), kh = slice_cols(a, c->k, lo, hi), vh = slice_cols(a, c->v, lo, hi);
                Matrix p = c->probs[h], dctx_h = slice_cols(a, dctx, lo, hi);
                Matrix dp = matmul(a, dctx_h, transpose(a, vh));
                Matrix dvh = matmul(a, transpose(a, p), dctx_h);
                /* softmax backward, then the 1/sqrt(d) scale (masked entries have p = 0, so no gradient) */
                Matrix ds = matrix_new(a, p.rows, p.cols);
                for (size_t i = 0; i < p.rows; i++) {
                    float dot = 0;
                    for (size_t j = 0; j < p.cols; j++) dot += matrix_at(dp, i, j) * matrix_at(p, i, j);
                    for (size_t j = 0; j < p.cols; j++) ds.data[i * p.cols + j] = matrix_at(p, i, j) * (matrix_at(dp, i, j) - dot);
                }
                ds = scale(a, ds, attn_scale(att->head_dim));
                Matrix dqh = matmul(a, ds, kh), dkh = matmul(a, transpose(a, ds), qh);
                for (size_t i = 0; i < dqh.rows; i++)
                    for (size_t j = 0; j < att->head_dim; j++) {
                        dq.data[i * dq.cols + lo + j] = matrix_at(dqh, i, j);
                        dk.data[i * dk.cols + lo + j] = matrix_at(dkh, i, j);
                        dv.data[i * dv.cols + lo + j] = matrix_at(dvh, i, j);
                    }
            }
            Matrix dn1 = linear_backward(a, c->n1, att->w_query, dq, gb->att.w_query, NULL);
            add_into(dn1, linear_backward(a, c->n1, att->w_key, dk, gb->att.w_key, NULL));
            add_into(dn1, linear_backward(a, c->n1, att->w_value, dv, gb->att.w_value, NULL));
            dx = ln_backward(a, &b->norm1, &c->ln1, dn1, &gb->norm1);
            add_into(dx, dx_mid); /* the shortcut */
        }

        /* embeddings: every token's row of dx goes back to its row of the token table */
        for (size_t t = 0; t < batch->len; t++)
            for (size_t j = 0; j < dx.cols; j++) {
                grads->tok_emb.data[(size_t)ids[t] * dx.cols + j] += matrix_at(dx, t, j);
                grads->pos_emb.data[t * dx.cols + j] += matrix_at(dx, t, j);
            }
        arena_reset(a, mark);
    }
    return total / (float)n_tokens;
}

void zero_grads(GptModel *grads) {
    float *d[GPT_MAX_TENSORS];
    size_t len[GPT_MAX_TENSORS], n = gpt_tensors(grads, d, len);
    for (size_t i = 0; i < n; i++) memset(d[i], 0, len[i] * sizeof(float));
}
