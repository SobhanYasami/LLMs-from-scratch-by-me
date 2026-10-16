#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tokenization/util.h"

Arena arena_new(size_t capacity) { return (Arena){xmalloc(capacity), 0, capacity}; }
void arena_free(Arena *a) {
    free(a->base);
    *a = (Arena){0};
}

static void *arena_alloc(Arena *a, size_t bytes) {
    bytes = (bytes + 15) & ~(size_t)15; /* keep every block 16-byte aligned */
    if (a->used + bytes > a->capacity) {
        fputs("arena out of memory: raise the capacity passed to arena_new\n", stderr);
        exit(1);
    }
    void *p = a->base + a->used;
    a->used += bytes;
    return p;
}

Matrix matrix_new(Arena *a, size_t rows, size_t cols) {
    Matrix m = {rows, cols, arena_alloc(a, rows * cols * sizeof(float))};
    memset(m.data, 0, rows * cols * sizeof(float));
    return m;
}

Matrix matrix_from(Arena *a, size_t rows, size_t cols, const float *values) {
    Matrix m = matrix_new(a, rows, cols);
    memcpy(m.data, values, rows * cols * sizeof(float));
    return m;
}

Matrix matrix_rand(Arena *a, size_t rows, size_t cols, Rng *rng) {
    Matrix m = matrix_new(a, rows, cols);
    for (size_t i = 0; i < rows * cols; i++) m.data[i] = (float)rng_f64(rng);
    return m;
}

Matrix matmul(Arena *a, Matrix x, Matrix y) {
    Matrix out = matrix_new(a, x.rows, y.cols);
    for (size_t i = 0; i < x.rows; i++)
        for (size_t j = 0; j < y.cols; j++) {
            float sum = 0;
            for (size_t k = 0; k < x.cols; k++) sum += matrix_at(x, i, k) * matrix_at(y, k, j);
            out.data[i * out.cols + j] = sum;
        }
    return out;
}

Matrix transpose(Arena *a, Matrix m) {
    Matrix out = matrix_new(a, m.cols, m.rows);
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++) out.data[j * out.cols + i] = matrix_at(m, i, j);
    return out;
}

/* Tip: subtracting the row maximum first does not change the result but keeps
 * exp() from overflowing. A -inf entry becomes exactly 0. */
Matrix softmax_rows(Arena *a, Matrix m) {
    Matrix out = matrix_new(a, m.rows, m.cols);
    for (size_t i = 0; i < m.rows; i++) {
        const float *row = matrix_row(m, i);
        float max = row[0], sum = 0;
        for (size_t j = 1; j < m.cols; j++)
            if (row[j] > max) max = row[j];
        for (size_t j = 0; j < m.cols; j++) {
            float e = (float)exp((double)(row[j] - max));
            out.data[i * m.cols + j] = e;
            sum += e;
        }
        for (size_t j = 0; j < m.cols; j++) out.data[i * m.cols + j] /= sum;
    }
    return out;
}

Matrix scale(Arena *a, Matrix m, float s) {
    Matrix out = matrix_new(a, m.rows, m.cols);
    for (size_t i = 0; i < m.rows * m.cols; i++) out.data[i] = m.data[i] * s;
    return out;
}

Matrix mask_future(Arena *a, Matrix m) {
    Matrix out = matrix_from(a, m.rows, m.cols, m.data);
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = i + 1; j < m.cols; j++) out.data[i * m.cols + j] = -INFINITY;
    return out;
}

Matrix slice_cols(Arena *a, Matrix m, size_t start, size_t end) {
    Matrix out = matrix_new(a, m.rows, end - start);
    for (size_t i = 0; i < m.rows; i++) memcpy(matrix_row(out, i), matrix_row(m, i) + start, (end - start) * sizeof(float));
    return out;
}

Matrix dropout(Arena *a, Matrix m, float p, Rng *rng) {
    Matrix out = matrix_new(a, m.rows, m.cols);
    for (size_t i = 0; i < m.rows * m.cols; i++)
        if (rng_f64(rng) >= (double)p) out.data[i] = m.data[i] / (1.0f - p);
    return out;
}
