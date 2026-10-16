/* The few matrix operations attention needs (book chapter 3).
 *
 * Files in attention/, in the order you learn them:
 *   matrix.[ch]          this file: matmul, transpose, softmax, ...
 *   simple.[ch]          simplified self-attention, no trainable weights (3.3)
 *   self_attention.[ch]  scaled dot-product attention with W_query/key/value (3.4)
 *   causal.[ch]          hide future tokens with a mask, optional dropout (3.5)
 *   multihead.[ch]       several heads in parallel plus an output projection (3.6)
 * The seeded random generator is shared with chapter 2: tokenization/rng.h.
 *
 * Tip: there is no tensor library, so a Matrix is a row-major float array.
 * Memory: every operation returns a NEW matrix allocated from an Arena, a
 * bump allocator. You free the whole arena once; no free() per matrix.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

#include "../tokenization/rng.h"

typedef struct {
    unsigned char *base;
    size_t used;
    size_t capacity;
} Arena;

Arena arena_new(size_t capacity);
void arena_free(Arena *a);

typedef struct {
    size_t rows, cols;
    float *data;
} Matrix;

Matrix matrix_new(Arena *a, size_t rows, size_t cols);                      /* zero-filled */
Matrix matrix_from(Arena *a, size_t rows, size_t cols, const float *values); /* copy of values */
Matrix matrix_rand(Arena *a, size_t rows, size_t cols, Rng *rng);           /* uniform [0, 1) */

static inline float *matrix_row(Matrix m, size_t i) { return m.data + i * m.cols; }
static inline float matrix_at(Matrix m, size_t i, size_t j) { return m.data[i * m.cols + j]; }

Matrix matmul(Arena *a, Matrix x, Matrix y);       /* [n,k] x [k,m] -> [n,m] */
Matrix transpose(Arena *a, Matrix m);
Matrix softmax_rows(Arena *a, Matrix m);           /* each row: positive, sums to 1 */
Matrix scale(Arena *a, Matrix m, float s);
Matrix mask_future(Arena *a, Matrix m);            /* above the diagonal -> -inf */
Matrix slice_cols(Arena *a, Matrix m, size_t start, size_t end); /* columns [start, end) */
Matrix dropout(Arena *a, Matrix m, float p, Rng *rng);           /* zero w.p. p, scale rest by 1/(1-p) */

#endif
