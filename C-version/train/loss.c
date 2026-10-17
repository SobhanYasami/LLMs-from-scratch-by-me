#include "loss.h"

#include <math.h>

float log_sum_exp(const float *row, size_t n) {
    float max = -INFINITY, sum = 0;
    for (size_t i = 0; i < n; i++)
        if (row[i] > max) max = row[i];
    for (size_t i = 0; i < n; i++) sum += (float)exp((double)(row[i] - max));
    return (float)log((double)sum) + max;
}

float cross_entropy_sum(Matrix logits, const int *targets) {
    float total = 0;
    for (size_t i = 0; i < logits.rows; i++) {
        const float *row = matrix_row(logits, i);
        total += log_sum_exp(row, logits.cols) - row[targets[i]]; /* -log p(target) */
    }
    return total;
}

float perplexity(float loss) { return (float)exp((double)loss); }

float calc_loss_batch(Arena *a, const GptModel *m, const Batch *b) {
    float total = 0;
    for (size_t r = 0; r < b->rows; r++) {
        size_t mark = arena_mark(a);
        Matrix logits = gpt_forward(a, m, b->inputs + r * b->len, b->len);
        total += cross_entropy_sum(logits, b->targets + r * b->len);
        arena_reset(a, mark);
    }
    return total / (float)(b->rows * b->len);
}

float calc_loss_loader(Arena *a, const GptModel *m, const Batch *batches, size_t n_batches, size_t num_batches) {
    if (n_batches == 0) return NAN;
    size_t n = (num_batches > 0 && num_batches < n_batches) ? num_batches : n_batches;
    float total = 0;
    for (size_t i = 0; i < n; i++) total += calc_loss_batch(a, m, &batches[i]);
    return total / (float)n;
}
