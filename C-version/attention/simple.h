/* Step 6 (book 3.3) - simplified self-attention, no trainable weights.
 *
 * For every token, build a context vector: an enriched embedding that mixes in
 * information from all the other tokens, weighted by how relevant each is.
 *
 *   scores  = X x X^T                  how similar is token i to token j?   [T, T]
 *   weights = softmax(scores, rows)    turn each row into weights summing to 1
 *   context = weights x X              weighted sum of all token vectors   [T, d]
 *
 * Tip: a dot product measures similarity. Similar vectors -> big score -> big weight.
 * There is nothing to learn here yet; trainable weights arrive in 3.4.
 */
#ifndef SIMPLE_ATTENTION_H
#define SIMPLE_ATTENTION_H

#include "matrix.h"

typedef struct {
    Matrix scores, weights, context;
} SimpleAttentionResult;

/* inputs is [T, d]. */
SimpleAttentionResult simple_attention(Arena *a, Matrix inputs);

#endif
