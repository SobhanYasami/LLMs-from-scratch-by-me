/* Step 14 (book 4.5) - the transformer block.
 *
 * One block is two sub-layers, each wrapped in the same recipe
 * (normalize first, run the layer, add the input back):
 *
 *   x = x + attention(norm1(x))      <- tokens look at each other
 *   x = x + feed_forward(norm2(x))   <- each token is processed alone
 *
 * The output has the same shape as the input, so blocks can be stacked as many
 * times as you like. Normalizing BEFORE each layer is called pre-LayerNorm; it
 * trains better than the original transformer's normalize-after. Dropout, which
 * the book puts after each sub-layer, only runs in training, so it is left out.
 *
 * Tip: the attention here is the chapter 3 MultiHeadAttention unchanged.
 */
#ifndef BLOCK_H
#define BLOCK_H

#include "../attention/multihead.h"
#include "layers.h"

typedef struct {
    MultiHeadAttention att;
    FeedForward ff;
    LayerNorm norm1, norm2;
} TransformerBlock;

TransformerBlock transformer_block_new(Arena *a, size_t emb_dim, size_t n_heads, int qkv_bias);
Matrix transformer_block_forward(Arena *a, const TransformerBlock *b, Matrix x);

#endif
