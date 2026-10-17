#include "block.h"

TransformerBlock transformer_block_new(Arena *a, size_t d, size_t n_heads, int qkv_bias) {
    TransformerBlock b;
    b.att = (MultiHeadAttention){matrix_new(a, d, d), matrix_new(a, d, d), matrix_new(a, d, d), matrix_new(a, d, d),
                                 matrix_new(a, 1, d), n_heads, d / n_heads, 0, {{0}}};
    if (qkv_bias) {
        b.att.has_qkv_bias = 1;
        for (int i = 0; i < 3; i++) b.att.qkv_bias[i] = matrix_new(a, 1, d);
    }
    b.ff = feedforward_new(a, d);
    b.norm1 = layernorm_new(a, d);
    b.norm2 = layernorm_new(a, d);
    return b;
}

static Matrix add_in_place(Matrix a, Matrix b) {
    for (size_t i = 0; i < a.rows * a.cols; i++) a.data[i] += b.data[i];
    return a;
}

Matrix transformer_block_forward(Arena *a, const TransformerBlock *b, Matrix x) {
    /* add the original input back after each sub-layer */
    x = add_in_place(multihead_forward(a, &b->att, layernorm_forward(a, &b->norm1, x)), x);
    return add_in_place(feedforward_forward(a, &b->ff, layernorm_forward(a, &b->norm2, x)), x);
}
