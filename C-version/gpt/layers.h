/* Steps 10-12 (book 4.2 and 4.3) - layer normalization, GELU and the feed forward block.
 *
 * Layer normalization rescales each token's vector to mean 0 and variance 1,
 * which keeps activations in a range where training is stable:
 *
 *   x [T, D]  ->  (x - mean) / sqrt(var + eps)  ->  scale * that + shift
 *
 * The mean and variance are taken over the D numbers of ONE token (one row),
 * never across tokens. scale and shift are learned, so the model can undo the
 * normalization where that helps.
 *
 * GELU is a smooth cousin of ReLU, and the feed forward block expands each
 * token's vector 4x, applies GELU, and shrinks it back:
 *
 *   [T, D] -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> [T, D]
 *
 * Files in gpt/, in the order you learn them:
 *   layers.[ch]    this file
 *   shortcut.[ch]  a tiny deep network that shows what shortcut connections do (4.4)
 *   block.[ch]     the transformer block (4.5)
 *   model.[ch]     the GPT model, its configuration and text generation (4.6, 4.7)
 *
 * Tip: like chapter 3 there is no tensor library. A sequence of T tokens is a
 * Matrix [T, D]; the C code handles one sequence at a time (the demo loops over
 * the batch), and the math per sequence is the same as the batched Python.
 */
#ifndef LAYERS_H
#define LAYERS_H

#include "../attention/matrix.h"

typedef struct {
    Matrix scale, shift; /* each 1 x D */
} LayerNorm;

LayerNorm layernorm_new(Arena *a, size_t dim); /* scale 1, shift 0 */
Matrix layernorm_forward(Arena *a, const LayerNorm *ln, Matrix x);

float gelu(float x); /* the tanh approximation GPT-2 was trained with */

/* x @ W + b with W stored [in, out] (PyTorch's nn.Linear stores it transposed). */
typedef struct {
    Matrix w; /* [in, out] */
    Matrix b; /* 1 x out */
} Linear;

Linear linear_new(Arena *a, size_t in, size_t out);
Matrix linear_forward(Arena *a, const Linear *l, Matrix x);

typedef struct {
    Linear expand, shrink;
} FeedForward;

FeedForward feedforward_new(Arena *a, size_t dim);
Matrix feedforward_forward(Arena *a, const FeedForward *f, Matrix x);

#endif
