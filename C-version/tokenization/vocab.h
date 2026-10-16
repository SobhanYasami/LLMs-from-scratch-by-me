/* Step 2 (book 2.3) - build a vocabulary: token -> unique integer ID.
 *
 *   {"!": 0, "\"": 1, ..., "<|endoftext|>": 1159, "<|unk|>": 1160}
 *
 * Models only understand numbers, so this table is the bridge between text
 * and IDs. Tip: sorting before numbering makes IDs identical on every run.
 */
#ifndef VOCAB_H
#define VOCAB_H

#include <stddef.h>

#include "preprocess.h"

typedef struct {
    char **tokens;   /* ID -> token, used by decode */
    size_t size;
    int *slots;      /* hash table of IDs (-1 = empty), used by encode */
    size_t n_slots;
} Vocab;

/* Unique tokens in sorted order, then <|endoftext|> and <|unk|>. */
Vocab build_vocab(const Tokens *tokens);
void vocab_free(Vocab *v);

/* token -> ID, or -1 if the token is not in the vocabulary. */
int vocab_id(const Vocab *v, const char *token);
/* ID -> token, or NULL if out of range. */
const char *vocab_token(const Vocab *v, int id);

#endif
