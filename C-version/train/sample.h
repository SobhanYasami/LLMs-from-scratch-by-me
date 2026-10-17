/* Step 19 (book 5.3) - decoding strategies: temperature and top-k sampling.
 *
 * Greedy decoding always takes the most likely token, so the same prompt gives the same text,
 * often copied straight from the training data. Two knobs add variety:
 *
 *  - temperature: divide the logits by T before softmax. T < 1 sharpens the distribution (more
 *    confident), T > 1 flattens it (more surprising).
 *  - top-k: keep only the k best logits, set the others to -inf (probability 0).
 *
 * Then draw the next token IN PROPORTION to its probability. The draw uses the repo's shared
 * generator (not torch.multinomial), so every language agrees.
 */
#ifndef SAMPLE_H
#define SAMPLE_H

#include "../gpt/model.h"

/* Softmax of one row of scores, like torch.softmax: subtract the max, exp, divide by the sum. */
void softmax_row(const float *logits, size_t n, float *out);
void softmax_with_temperature(const float *logits, size_t n, float temperature, float *out);
/* Keep the k largest logits (ties at the threshold are kept), -inf for the rest. */
void top_k_logits(const float *logits, size_t n, size_t k, float *out);
/* Draw one index in proportion to `probas`: inverse-CDF sampling. One uniform number u in [0, 1);
 * walk the cumulative sum and stop at the first entry where it passes u. Tokens with probability 0
 * can never be picked. */
size_t sample_index(const float *probas, size_t n, Rng *rng);
/* Like generate_text_simple but with temperature and top-k (top_k 0 = off). temperature 0 means
 * greedy. Returns n + max_new_tokens IDs (caller frees). */
int *generate(Arena *a, const GptModel *m, const int *ids, size_t n, size_t max_new_tokens, size_t context_size, Rng *rng, float temperature, size_t top_k);

#endif
