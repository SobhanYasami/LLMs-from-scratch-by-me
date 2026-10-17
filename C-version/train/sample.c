#include "sample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../tokenization/util.h"

void softmax_row(const float *logits, size_t n, float *out) {
    float max = -INFINITY, sum = 0;
    for (size_t i = 0; i < n; i++)
        if (logits[i] > max) max = logits[i];
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)exp((double)(logits[i] - max));
        sum += out[i];
    }
    for (size_t i = 0; i < n; i++) out[i] /= sum;
}

void softmax_with_temperature(const float *logits, size_t n, float temperature, float *out) {
    float *scaled = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) scaled[i] = logits[i] / temperature;
    softmax_row(scaled, n, out);
    free(scaled);
}

static int descending(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

void top_k_logits(const float *logits, size_t n, size_t k, float *out) {
    float *sorted = xmalloc(n * sizeof(float));
    memcpy(sorted, logits, n * sizeof(float));
    qsort(sorted, n, sizeof(float), descending);
    float threshold = sorted[k - 1];
    free(sorted);
    for (size_t i = 0; i < n; i++) out[i] = logits[i] < threshold ? -INFINITY : logits[i];
}

size_t sample_index(const float *probas, size_t n, Rng *rng) {
    double u = rng_f64(rng), cumulative = 0;
    size_t last = 0;
    for (size_t i = 0; i < n; i++) {
        if (probas[i] > 0) last = i;
        cumulative += (double)probas[i];
        if (u < cumulative) return i;
    }
    return last; /* rounding left the total a hair below u */
}

int *generate(Arena *a, const GptModel *m, const int *ids, size_t n, size_t max_new_tokens, size_t context_size, Rng *rng, float temperature, size_t top_k) {
    size_t vocab = m->cfg.vocab_size;
    int *out = xmalloc((n + max_new_tokens) * sizeof(int));
    memcpy(out, ids, n * sizeof(int));
    float *last = xmalloc(vocab * sizeof(float)), *probas = xmalloc(vocab * sizeof(float));
    for (size_t step = 0; step < max_new_tokens; step++, n++) {
        size_t start = n > context_size ? n - context_size : 0;
        size_t mark = arena_mark(a);
        Matrix logits = gpt_forward(a, m, out + start, n - start);
        memcpy(last, matrix_row(logits, logits.rows - 1), vocab * sizeof(float)); /* scores for the next token */
        arena_reset(a, mark);
        if (top_k > 0) top_k_logits(last, vocab, top_k, last);
        size_t next = 0;
        if (temperature > 0) {
            softmax_with_temperature(last, vocab, temperature, probas);
            next = sample_index(probas, vocab, rng);
        } else {
            for (size_t i = 1; i < vocab; i++)
                if (last[i] > last[next]) next = i;
        }
        out[n] = (int)next;
    }
    free(last);
    free(probas);
    return out;
}
