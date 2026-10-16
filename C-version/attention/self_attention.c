#include "self_attention.h"

#include <math.h>

SelfAttention self_attention_random(Arena *a, size_t d_in, size_t d_out, Rng *rng) {
    SelfAttention sa;
    sa.w_query = matrix_rand(a, d_in, d_out, rng);
    sa.w_key = matrix_rand(a, d_in, d_out, rng);
    sa.w_value = matrix_rand(a, d_in, d_out, rng);
    return sa;
}

Matrix scale_by_sqrt(Arena *a, Matrix scores, size_t d_k) { return scale(a, scores, 1.0f / (float)sqrt((double)d_k)); }

Matrix self_attention_weights(Arena *a, const SelfAttention *sa, Matrix x) {
    Matrix queries = matmul(a, x, sa->w_query), keys = matmul(a, x, sa->w_key);
    Matrix scores = matmul(a, queries, transpose(a, keys));
    return softmax_rows(a, scale_by_sqrt(a, scores, keys.cols));
}

Matrix self_attention_forward(Arena *a, const SelfAttention *sa, Matrix x) {
    return matmul(a, self_attention_weights(a, sa, x), matmul(a, x, sa->w_value));
}
