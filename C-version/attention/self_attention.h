/* Step 7 (book 3.4) - self-attention with trainable weights (scaled dot-product).
 *
 * Three weight matrices turn each input vector into three roles:
 *
 *   query = x x W_query    "what am I looking for?"
 *   key   = x x W_key      "what do I offer to a search?"
 *   value = x x W_value    "what do I hand over if I am chosen?"
 *
 *   scores  = query x key^T                 how well does each query match each key?
 *   weights = softmax(scores / sqrt(d_k))   scale, then normalise each row
 *   context = weights x value               mix the values
 *
 * Why divide by sqrt(d_k)? Big dot products push softmax towards a hard 0/1
 * choice, which makes gradients tiny. Scaling keeps training healthy.
 *
 * Tip: do not confuse weight parameters (W_*, learned) with attention weights
 * (softmax output, computed fresh for every input).
 */
#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

#include "matrix.h"

typedef struct {
    Matrix w_query, w_key, w_value;
} SelfAttention;

/* Weights drawn uniform in [0, 1), in query, key, value order. */
SelfAttention self_attention_random(Arena *a, size_t d_in, size_t d_out, Rng *rng);
/* x is [T, d_in]; returns [T, T] weights. */
Matrix self_attention_weights(Arena *a, const SelfAttention *sa, Matrix x);
/* Context vectors [T, d_out]. */
Matrix self_attention_forward(Arena *a, const SelfAttention *sa, Matrix x);

/* scores / sqrt(d_k), shared by the attention variants. */
Matrix scale_by_sqrt(Arena *a, Matrix scores, size_t d_k);

#endif
