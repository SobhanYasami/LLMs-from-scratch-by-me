#include "simple.h"

SimpleAttentionResult simple_attention(Arena *a, Matrix inputs) {
    SimpleAttentionResult r;
    r.scores = matmul(a, inputs, transpose(a, inputs));
    r.weights = softmax_rows(a, r.scores);
    r.context = matmul(a, r.weights, inputs);
    return r;
}
