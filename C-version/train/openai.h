/* Step 21 (book 5.5) - loading OpenAI's pretrained GPT-2 weights.
 *
 * Training a GPT-2 from scratch costs a fortune; OpenAI published the result. The weights come as
 * a .safetensors file, a very plain format: 8 bytes with the header length, a JSON header that
 * says where each tensor lives, then raw float32 numbers. No library needed.
 *
 * Download once (about 500 MB, the file is ignored by git):
 *
 *   curl -L -o assets/gpt2/model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
 *
 * OpenAI's layout differs from ours in a few ways, which load_gpt2_weights fixes: one big c_attn
 * matrix holds query, key and value side by side (we keep three); the attention projections carry
 * biases (qkv_bias = 1); and the output head re-uses the token embedding (weight tying), so we
 * copy wte into it. Matrices are stored [in, out], like ours.
 */
#ifndef OPENAI_H
#define OPENAI_H

#include "../gpt/model.h"
#include "../tokenization/util.h"

extern const GptConfig GPT2_SMALL_CONFIG;

/* Fills `m`, which must have been built from GPT2_SMALL_CONFIG. Returns 0, or -1 with `err` set. */
int load_gpt2_weights(const char *path, GptModel *m, char err[ERR_LEN]);

#endif
