/* Step 8 (book 3.5) - causal attention: hide the future.
 *
 * A GPT writes text left to right, so token i may only look at tokens 0..i.
 * We hide the later tokens with a mask. Set their scores to -inf before
 * softmax: exp(-inf) = 0, and the remaining weights in each row still sum to 1.
 *
 *   scores (T x T)                       after the mask
 *   [ s00 s01 s02 ]                      [ s00 -inf -inf ]
 *   [ s10 s11 s12 ]      -------->       [ s10  s11 -inf ]
 *   [ s20 s21 s22 ]                      [ s20  s21  s22 ]
 *
 * Dropout (training only) randomly zeroes some attention weights so the model
 * does not lean on any single connection.
 */
#ifndef CAUSAL_H
#define CAUSAL_H

#include "self_attention.h"

typedef struct {
    SelfAttention base;
    float dropout_p;
} CausalAttention;

/* x is [T, d_in]. Pass an rng to apply dropout (training), or NULL (inference). */
Matrix causal_weights(Arena *a, const CausalAttention *ca, Matrix x, Rng *rng);
Matrix causal_forward(Arena *a, const CausalAttention *ca, Matrix x, Rng *rng);

#endif
