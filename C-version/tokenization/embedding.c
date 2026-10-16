#include "embedding.h"

#include <stdlib.h>
#include <string.h>

#include "util.h"

Embedding embedding_new(size_t rows, size_t dim, Rng *rng) {
    Embedding e = {rows, dim, xmalloc(rows * dim * sizeof(float))};
    for (size_t i = 0; i < rows * dim; i++) e.weight[i] = rng_normal(rng);
    return e;
}

void embedding_free(Embedding *e) {
    free(e->weight);
    *e = (Embedding){0};
}

void embedding_forward(const Embedding *e, const int *ids, size_t n, float *out) {
    for (size_t i = 0; i < n; i++) /* a lookup: copy row `ids[i]` */
        memcpy(out + i * e->dim, e->weight + (size_t)ids[i] * e->dim, e->dim * sizeof(float));
}

InputEmbedding input_embedding_new(size_t vocab_size, size_t context_length, size_t dim, uint64_t seed) {
    Rng rng = rng_new(seed);
    InputEmbedding m;
    m.token = embedding_new(vocab_size, dim, &rng);
    m.pos = embedding_new(context_length, dim, &rng);
    return m;
}

void input_embedding_free(InputEmbedding *m) {
    embedding_free(&m->token);
    embedding_free(&m->pos);
}

void input_embedding_forward(const InputEmbedding *m, const int *ids, size_t batch, size_t seq_len, float *out) {
    size_t dim = m->token.dim;
    for (size_t b = 0; b < batch; b++) {
        float *row = out + b * seq_len * dim;
        embedding_forward(&m->token, ids + b * seq_len, seq_len, row);
        for (size_t t = 0; t < seq_len; t++)         /* token + position */
            for (size_t d = 0; d < dim; d++) row[t * dim + d] += m->pos.weight[t * dim + d];
    }
}
