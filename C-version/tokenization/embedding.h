/* Step 5 (book 2.7-2.8) - token and positional embeddings.
 *
 * A token ID is just an integer. An embedding layer is a lookup table with
 * one vector per ID, so every token becomes a vector the model can learn from:
 *
 *   ids [B][T]  --token embedding-->  [B][T][D]
 *   positions 0..T-1 --pos embedding-> [T][D]  (added to every batch row)
 *   input embeddings = token + position  -> [B][T][D]
 *
 * Why add positions? Attention (later chapters) treats its input as a bag of
 * vectors, so word order must be put into the vectors themselves.
 *
 * No tensor library: a tensor is one flat float array, indexed [b][t][d] as
 * (b * T + t) * D + d. Weights are random (normal, mean 0, std 1) and untrained.
 */
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>
#include <stdint.h>

#include "rng.h"

typedef struct {
    size_t rows; /* vocabulary size (or context length) */
    size_t dim;
    float *weight; /* rows x dim */
} Embedding;

Embedding embedding_new(size_t rows, size_t dim, Rng *rng);
void embedding_free(Embedding *e);
/* Look up one vector per ID; `out` holds n * dim floats. */
void embedding_forward(const Embedding *e, const int *ids, size_t n, float *out);

typedef struct {
    Embedding token;
    Embedding pos;
} InputEmbedding;

InputEmbedding input_embedding_new(size_t vocab_size, size_t context_length, size_t dim, uint64_t seed);
void input_embedding_free(InputEmbedding *m);
/* ids is [batch][seq_len] flat; out is [batch][seq_len][dim] flat. */
void input_embedding_forward(const InputEmbedding *m, const int *ids, size_t batch, size_t seq_len, float *out);

#endif
