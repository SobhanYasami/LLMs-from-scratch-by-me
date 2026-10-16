#include "causal.h"

Matrix causal_weights(Arena *a, const CausalAttention *ca, Matrix x, Rng *rng) {
    Matrix queries = matmul(a, x, ca->base.w_query), keys = matmul(a, x, ca->base.w_key);
    Matrix scores = mask_future(a, matmul(a, queries, transpose(a, keys)));
    Matrix weights = softmax_rows(a, scale_by_sqrt(a, scores, keys.cols));
    if (rng && ca->dropout_p > 0) weights = dropout(a, weights, ca->dropout_p, rng);
    return weights;
}

Matrix causal_forward(Arena *a, const CausalAttention *ca, Matrix x, Rng *rng) {
    return matmul(a, causal_weights(a, ca, x, rng), matmul(a, x, ca->base.w_value));
}
