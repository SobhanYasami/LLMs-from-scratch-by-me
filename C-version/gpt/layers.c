#include "layers.h"

#include <math.h>

#define EPS 1e-5f

LayerNorm layernorm_new(Arena *a, size_t dim) {
    LayerNorm ln = {matrix_new(a, 1, dim), matrix_new(a, 1, dim)};
    for (size_t i = 0; i < dim; i++) ln.scale.data[i] = 1;
    return ln;
}

Matrix layernorm_forward(Arena *a, const LayerNorm *ln, Matrix x) {
    Matrix out = matrix_new(a, x.rows, x.cols);
    float n = (float)x.cols;
    for (size_t i = 0; i < x.rows; i++) {
        const float *row = matrix_row(x, i);
        float mean = 0;
        for (size_t j = 0; j < x.cols; j++) mean += row[j];
        mean /= n;
        float variance = 0; /* divide by n (biased), like GPT-2 */
        for (size_t j = 0; j < x.cols; j++) variance += (row[j] - mean) * (row[j] - mean);
        variance /= n;
        float inv = (float)(1.0 / sqrt((double)(variance + EPS)));
        for (size_t j = 0; j < x.cols; j++) out.data[i * x.cols + j] = ln->scale.data[j] * ((row[j] - mean) * inv) + ln->shift.data[j];
    }
    return out;
}

float gelu(float x) {
    double v = x;
    return (float)(0.5 * v * (1.0 + tanh(sqrt(2.0 / M_PI) * (v + 0.044715 * v * v * v))));
}

Linear linear_new(Arena *a, size_t in, size_t out) { return (Linear){matrix_new(a, in, out), matrix_new(a, 1, out)}; }

Matrix linear_forward(Arena *a, const Linear *l, Matrix x) {
    Matrix out = matmul(a, x, l->w);
    for (size_t i = 0; i < out.rows; i++)
        for (size_t j = 0; j < out.cols; j++) out.data[i * out.cols + j] += l->b.data[j];
    return out;
}

FeedForward feedforward_new(Arena *a, size_t dim) { return (FeedForward){linear_new(a, dim, 4 * dim), linear_new(a, 4 * dim, dim)}; }

Matrix feedforward_forward(Arena *a, const FeedForward *f, Matrix x) {
    Matrix h = linear_forward(a, &f->expand, x);
    for (size_t i = 0; i < h.rows * h.cols; i++) h.data[i] = gelu(h.data[i]);
    return linear_forward(a, &f->shrink, h);
}
