/* Steps 15-16 (book 4.6 and 4.7) - the GPT model and text generation.
 *
 *   token IDs [T]
 *     -> token embedding + position embedding        [T, D]
 *     -> N x transformer block                       [T, D]
 *     -> final layer norm                            [T, D]
 *     -> output head (Linear D -> vocab, no bias)    [T, vocab]   = "logits"
 *
 * The logits of the LAST position are the model's scores for the next token.
 * generate_text_simple picks the best one, appends it, and repeats.
 *
 * Weights start as random numbers from the repo's shared generator
 * (Normal(0, 0.02), biases 0, layer norm scale 1 / shift 0), in a fixed order,
 * so every language builds exactly the same untrained model. Untrained means
 * the generated text is gibberish; training is chapter 5.
 *
 * Memory: the model's weights live in one Arena. For each forward pass take an
 * arena_mark, compute, then arena_reset: the temporary matrices vanish at once.
 */
#ifndef MODEL_H
#define MODEL_H

#include "block.h"

typedef struct {
    size_t vocab_size;     /* BPE vocabulary of GPT-2 */
    size_t context_length; /* most tokens the model can look at */
    size_t emb_dim;        /* size of every token vector */
    size_t n_heads;        /* attention heads per block */
    size_t n_layers;       /* transformer blocks */
    float drop_rate;       /* dropout (training only) */
    int qkv_bias;          /* bias in the query/key/value projections */
} GptConfig;

extern const GptConfig GPT_CONFIG_124M;
extern const GptConfig GPT_CONFIG_TINY; /* same architecture, small enough to run in a blink */

/* Book exercise 4.2: bigger GPT-2 models differ only in the configuration. */
typedef struct {
    const char *name;
    size_t emb_dim, n_layers, n_heads;
} GptSize;

#define GPT_N_SIZES 4
extern const GptSize GPT_SIZES[GPT_N_SIZES];

typedef struct {
    GptConfig cfg;
    Matrix tok_emb, pos_emb;
    TransformerBlock *blocks;
    LayerNorm final_norm;
    Matrix out_head; /* [D, vocab] */
} GptModel;

size_t gpt_parameter_count(const GptConfig *cfg); /* by arithmetic only, no model is built */
size_t gpt_arena_bytes(const GptConfig *cfg);     /* enough for the weights plus temporaries */

GptModel gpt_model_new(Arena *a, GptConfig cfg);
/* Fill the model from the shared generator. The ORDER is part of the recipe: token emb,
 * position emb, then per block wq, wk, wv, wo, ff1, ff2, then the output head. */
void gpt_model_init(GptModel *m, unsigned long long seed);
size_t gpt_count_parameters(const GptModel *m);

Matrix gpt_embed(Arena *a, const GptModel *m, const int *ids, size_t n);   /* token + position */
Matrix gpt_forward(Arena *a, const GptModel *m, const int *ids, size_t n); /* logits [n, vocab] */

/* Greedy decoding. Returns n + max_new_tokens IDs (caller frees). */
int *generate_text_simple(Arena *a, const GptModel *m, const int *ids, size_t n, size_t max_new_tokens, size_t context_size);

#endif
