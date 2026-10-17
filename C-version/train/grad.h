/* Step 18 (book 5.2) - the backward pass, by hand.
 *
 * Training needs to know, for EVERY weight, "if I make this weight a little bigger, does the loss
 * go up or down, and how fast?" That number is the weight's gradient. torch's autograd finds all
 * of them automatically; here we do the same with the chain rule:
 *
 *  1. run the forward pass and REMEMBER what each layer saw and produced (the "cache");
 *  2. start at the loss and walk the layers backwards, turning "how the loss depends on this
 *     layer's output" into "how it depends on its weights and on its input".
 *
 * Every layer has a small backward rule:
 *
 *   y = x W + b       dW = x^T dy     db = sum of dy over tokens     dx = dy W^T
 *   y = gelu(x)       dx = dy * gelu'(x)
 *   y = layer_norm(x) see ln_backward in grad.c
 *   y = x + f(x)      dx = dy + (what flows back through f)        <- the shortcut, chapter 4.4
 *   attention         softmax backward: dS = P * (dP - sum(dP * P) per row)
 *
 * The gradients are stored in a GptModel of the same shape (a handy trick: the optimizer and the
 * checkpoint code then treat weights, gradients and optimizer state alike).
 *
 * Memory: every matrix made here comes from the arena `a`; the caller takes an arena_mark before
 * and arena_resets afterwards.
 */
#ifndef GRAD_H
#define GRAD_H

#include "loss.h"

/* Mean cross entropy of the batch, and the gradient of that mean for every weight, ADDED into
 * `grads` (zero it first with zero_grads). */
float loss_and_grads(Arena *a, const GptModel *m, GptModel *grads, const Batch *batch);
void zero_grads(GptModel *grads);

#endif
