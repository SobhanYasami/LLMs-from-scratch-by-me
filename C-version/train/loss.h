/* Step 17 (book 5.1) - how wrong is the model? Cross entropy loss and perplexity.
 *
 * For every position the model gives a score (logit) for each of the 50,257 tokens. Softmax turns
 * the scores into probabilities. The loss looks only at the probability given to the token that
 * REALLY came next:
 *
 *   loss = - mean( log p(target) )        (cross entropy)
 *   perplexity = exp(loss)                "among how many tokens is the model torn?"
 *
 * A model that guesses uniformly has loss ln(50257) = 10.82. A perfect model has loss 0.
 *
 * Files in train/, in the order you learn them:
 *   loss.[ch]        this file (5.1)
 *   grad.[ch]        the backward pass, written by hand, and adamw.[ch], the optimizer (5.2)
 *   sample.[ch]      temperature and top-k decoding (5.3)
 *   checkpoint.[ch]  saving and loading weights and optimizer state (5.4)
 *   openai.[ch]      loading OpenAI's pretrained GPT-2 weights (5.5)
 * The Python version lets torch's autograd do the backward pass. Here there is no autograd, so
 * grad.c applies the chain rule layer by layer.
 */
#ifndef LOSS_H
#define LOSS_H

#include <stddef.h>

#include "../gpt/model.h"

/* One batch: `rows` sequences of `len` token IDs each, row after row; targets are the inputs shifted by one. */
typedef struct {
    int *inputs, *targets;
    size_t rows, len;
} Batch;

float log_sum_exp(const float *row, size_t n); /* log of softmax's denominator */
/* Cross entropy of logits [T, vocab] against `targets` (one token ID per row), summed over rows. */
float cross_entropy_sum(Matrix logits, const int *targets);
float perplexity(float loss);
/* Mean cross entropy of one batch (all its tokens count equally). */
float calc_loss_batch(Arena *a, const GptModel *m, const Batch *b);
/* Average loss over (at most num_batches; 0 means all) batches. */
float calc_loss_loader(Arena *a, const GptModel *m, const Batch *batches, size_t n_batches, size_t num_batches);

#endif
