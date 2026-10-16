/* Step 3 (book 2.3-2.4) - simple tokenizers: split text, look tokens up.
 *
 *   encode: "Hello, world" -> ["Hello", ",", "world"] -> [1131, 5, 987]
 *   decode: [1131, 5, 987] -> ["Hello", ",", "world"] -> "Hello, world"
 *
 * Tip: compare V1 and V2. V1 fails on a word it never saw; V2 swaps it for
 * <|unk|> and keeps going. That one difference is why real models use BPE.
 */
#ifndef SIMPLE_H
#define SIMPLE_H

#include <stdbool.h>
#include <stddef.h>

#include "util.h"
#include "vocab.h"

typedef struct {
    const Vocab *vocab;
    bool unknown_to_unk; /* false = V1 (error), true = V2 (<|unk|>) */
} SimpleTokenizer;

SimpleTokenizer simple_tokenizer_v1(const Vocab *vocab);
SimpleTokenizer simple_tokenizer_v2(const Vocab *vocab);

/* text -> IDs. Returns NULL and fills `err` on an unknown token (V1). Caller frees. */
int *simple_encode(const SimpleTokenizer *t, const char *text, size_t *n, char err[ERR_LEN]);
/* IDs -> text. Returns NULL and fills `err` on an unknown ID. Caller frees. */
char *simple_decode(const SimpleTokenizer *t, const int *ids, size_t n, char err[ERR_LEN]);

#endif
