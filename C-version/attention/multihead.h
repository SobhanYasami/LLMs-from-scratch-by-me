/* Step 9 (book 3.6) - multi-head attention.
 *
 * One attention head learns one way of relating tokens. Several heads, each
 * with its own smaller query/key/value, can look for different patterns in
 * parallel (one for grammar, one for topic, ...). Their outputs are joined
 * and mixed by a final linear layer.
 *
 *   d_out = 4, num_heads = 2  ->  each head works on head_dim = 2 columns
 *
 *   W_query columns: [ head 0 | head 1 ]     (same for W_key and W_value)
 *   per head: causal attention on its own slice -> [T, head_dim]
 *   concatenate heads -> [T, d_out] -> out_proj
 *
 * Tip: one big weight matrix cut into column slices is the same as separate
 * small matrices per head, just faster (one matmul instead of many).
 */
#ifndef MULTIHEAD_H
#define MULTIHEAD_H

#include "matrix.h"

typedef struct {
    Matrix w_query, w_key, w_value, out_weight;
    Matrix out_bias; /* 1 x d_out */
    size_t num_heads, head_dim;
    /* Biases of the query, key and value projections (1 x d_out each): unused (has_qkv_bias = 0)
     * in chapters 3 and 4, set for OpenAI's GPT-2 weights (5.5). */
    int has_qkv_bias;
    Matrix qkv_bias[3];
} MultiHeadAttention;

/* Uniform [0, 1) weights (query, key, value, out_proj) and a zero bias.
 * d_out must divide evenly by num_heads. */
MultiHeadAttention multihead_random(Arena *a, size_t d_in, size_t d_out, size_t num_heads, Rng *rng);
/* x is [T, d_in]; returns [T, d_out]. */
Matrix multihead_forward(Arena *a, const MultiHeadAttention *m, Matrix x);

#endif
