/* Step 20 (book 5.4) - saving and loading a model (and the optimizer).
 *
 * Training is expensive, so save the weights and reload them later. The book uses
 * torch.save(model.state_dict()). To keep the file readable by every language in this repo, we
 * write one plain format:
 *
 *   "LLMCKPT1" | 7 x uint32: vocab, context, emb, heads, layers, qkv_bias, has_optimizer
 *   then raw little-endian float32 numbers, tensor after tensor, in the order of gpt_tensors
 *   (the weights), then the AdamW state if present: step (as float32), then exp_avg and
 *   exp_avg_sq of every tensor.
 *
 * A checkpoint written by C can be loaded by Python, Go, Rust or Zig, and the other way round.
 * (The numbers are written as this machine stores them, which is little-endian on x86 and ARM.)
 */
#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "../tokenization/util.h"
#include "adamw.h"

/* Writes the model and, if opt is not NULL, the optimizer state. Returns 0 on success. */
int checkpoint_save(const char *path, const GptModel *m, const AdamW *opt, char err[ERR_LEN]);
/* Step 1 of loading: read the settings, so the caller can build a model of the right size. */
int checkpoint_read_config(const char *path, GptConfig *cfg, int *has_optimizer, char err[ERR_LEN]);
/* Step 2: fill `m` (built from that config) and, if opt is not NULL and the file has it, the optimizer. */
int checkpoint_load(const char *path, GptModel *m, AdamW *opt, char err[ERR_LEN]);

#endif
