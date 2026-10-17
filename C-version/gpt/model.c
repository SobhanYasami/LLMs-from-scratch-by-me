#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tokenization/util.h"

const GptConfig GPT_CONFIG_124M = {50257, 1024, 768, 12, 12, 0.1f, 0};
const GptConfig GPT_CONFIG_TINY = {50257, 16, 32, 4, 2, 0.1f, 0};

const GptSize GPT_SIZES[GPT_N_SIZES] = {
    {"gpt2-small", 768, 12, 12},
    {"gpt2-medium", 1024, 24, 16},
    {"gpt2-large", 1280, 36, 20},
    {"gpt2-xl", 1600, 48, 25},
};

size_t gpt_parameter_count(const GptConfig *cfg) {
    size_t v = cfg->vocab_size, c = cfg->context_length, d = cfg->emb_dim, n = cfg->n_layers;
    size_t attention = 4 * d * d + d; /* wq, wk, wv, wo and the output bias */
    size_t feed_forward = d * 4 * d + 4 * d + 4 * d * d + d;
    size_t norms = 2 * 2 * d; /* two layer norms per block, scale + shift each */
    return v * d + c * d + n * (attention + feed_forward + norms) + 2 * d + d * v;
}

size_t gpt_arena_bytes(const GptConfig *cfg) {
    /* weights, plus room for one forward pass over a full context (generous) */
    size_t t = cfg->context_length < 64 ? cfg->context_length : 64;
    return gpt_parameter_count(cfg) * sizeof(float) + (size_t)64 * 1024 * 1024 + t * cfg->vocab_size * sizeof(float) * 3;
}

GptModel gpt_model_new(Arena *a, GptConfig cfg) {
    if (cfg.qkv_bias) {
        fputs("this implementation has no query/key/value bias (chapter 6 adds it)\n", stderr);
        exit(1);
    }
    GptModel m;
    m.cfg = cfg;
    m.tok_emb = matrix_new(a, cfg.vocab_size, cfg.emb_dim);
    m.pos_emb = matrix_new(a, cfg.context_length, cfg.emb_dim);
    m.blocks = arena_alloc(a, cfg.n_layers * sizeof(TransformerBlock));
    for (size_t i = 0; i < cfg.n_layers; i++) m.blocks[i] = transformer_block_new(a, cfg.emb_dim, cfg.n_heads);
    m.final_norm = layernorm_new(a, cfg.emb_dim);
    m.out_head = matrix_new(a, cfg.emb_dim, cfg.vocab_size);
    return m;
}

static void fill_normal(Matrix w, Rng *rng) {
    for (size_t i = 0; i < w.rows * w.cols; i++) w.data[i] = rng_normal(rng) * 0.02f;
}

void gpt_model_init(GptModel *m, unsigned long long seed) {
    Rng rng = rng_new(seed);
    fill_normal(m->tok_emb, &rng);
    fill_normal(m->pos_emb, &rng);
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        TransformerBlock *b = &m->blocks[i];
        fill_normal(b->att.w_query, &rng);
        fill_normal(b->att.w_key, &rng);
        fill_normal(b->att.w_value, &rng);
        fill_normal(b->att.out_weight, &rng);
        fill_normal(b->ff.expand.w, &rng);
        fill_normal(b->ff.shrink.w, &rng);
    }
    fill_normal(m->out_head, &rng);
}

static size_t size_of(Matrix x) { return x.rows * x.cols; }

size_t gpt_count_parameters(const GptModel *m) {
    size_t n = size_of(m->tok_emb) + size_of(m->pos_emb) + size_of(m->out_head) + 2 * size_of(m->final_norm.scale);
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        const TransformerBlock *b = &m->blocks[i];
        const MultiHeadAttention *a = &b->att;
        n += size_of(a->w_query) + size_of(a->w_key) + size_of(a->w_value) + size_of(a->out_weight) + size_of(a->out_bias);
        n += size_of(b->ff.expand.w) + size_of(b->ff.expand.b) + size_of(b->ff.shrink.w) + size_of(b->ff.shrink.b);
        n += 2 * (size_of(b->norm1.scale) + size_of(b->norm2.scale));
    }
    return n;
}

Matrix gpt_embed(Arena *a, const GptModel *m, const int *ids, size_t n) {
    size_t d = m->cfg.emb_dim;
    Matrix x = matrix_new(a, n, d);
    for (size_t t = 0; t < n; t++) {
        const float *tok = matrix_row(m->tok_emb, (size_t)ids[t]), *pos = matrix_row(m->pos_emb, t);
        for (size_t j = 0; j < d; j++) x.data[t * d + j] = tok[j] + pos[j];
    }
    return x;
}

Matrix gpt_forward(Arena *a, const GptModel *m, const int *ids, size_t n) {
    Matrix x = gpt_embed(a, m, ids, n);
    for (size_t i = 0; i < m->cfg.n_layers; i++) x = transformer_block_forward(a, &m->blocks[i], x);
    return matmul(a, layernorm_forward(a, &m->final_norm, x), m->out_head);
}

int *generate_text_simple(Arena *a, const GptModel *m, const int *ids, size_t n, size_t max_new_tokens, size_t context_size) {
    int *out = xmalloc((n + max_new_tokens) * sizeof(int));
    memcpy(out, ids, n * sizeof(int));
    for (size_t step = 0; step < max_new_tokens; step++, n++) {
        size_t start = n > context_size ? n - context_size : 0; /* keep only what fits in the context */
        size_t mark = arena_mark(a);
        Matrix logits = gpt_forward(a, m, out + start, n - start);
        const float *last = matrix_row(logits, logits.rows - 1); /* scores for the NEXT token only */
        /* Softmax would turn these into probabilities, but it keeps the order,
         * so the biggest score is also the most likely token. */
        size_t best = 0;
        for (size_t i = 1; i < logits.cols; i++)
            if (last[i] > last[best]) best = i;
        out[n] = (int)best;
        arena_reset(a, mark);
    }
    return out;
}
