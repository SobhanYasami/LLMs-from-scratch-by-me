#include "multihead.h"

#include <string.h>

#include "self_attention.h"

MultiHeadAttention multihead_random(Arena *a, size_t d_in, size_t d_out, size_t num_heads, Rng *rng) {
    MultiHeadAttention m;
    m.w_query = matrix_rand(a, d_in, d_out, rng);
    m.w_key = matrix_rand(a, d_in, d_out, rng);
    m.w_value = matrix_rand(a, d_in, d_out, rng);
    m.out_weight = matrix_rand(a, d_out, d_out, rng);
    m.out_bias = matrix_new(a, 1, d_out);
    m.num_heads = num_heads;
    m.head_dim = d_out / num_heads;
    return m;
}

Matrix multihead_forward(Arena *a, const MultiHeadAttention *m, Matrix x) {
    Matrix q = matmul(a, x, m->w_query), k = matmul(a, x, m->w_key), v = matmul(a, x, m->w_value);
    Matrix context = matrix_new(a, x.rows, m->num_heads * m->head_dim);
    for (size_t h = 0; h < m->num_heads; h++) {
        size_t lo = h * m->head_dim, hi = lo + m->head_dim; /* this head's column slice */
        Matrix qh = slice_cols(a, q, lo, hi), kh = slice_cols(a, k, lo, hi), vh = slice_cols(a, v, lo, hi);
        Matrix scores = mask_future(a, matmul(a, qh, transpose(a, kh)));
        Matrix head = matmul(a, softmax_rows(a, scale_by_sqrt(a, scores, m->head_dim)), vh);
        for (size_t i = 0; i < head.rows; i++) /* glue the heads back together */
            memcpy(matrix_row(context, i) + lo, matrix_row(head, i), m->head_dim * sizeof(float));
    }
    Matrix out = matmul(a, context, m->out_weight);
    for (size_t i = 0; i < out.rows; i++)
        for (size_t j = 0; j < out.cols; j++) out.data[i * out.cols + j] += m->out_bias.data[j];
    return out;
}
