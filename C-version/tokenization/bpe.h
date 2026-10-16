/* Bonus (book 2.5) - byte pair encoding, the GPT-2 tokenizer.
 *
 * C has no tiktoken, so this is a small from-scratch encoder that reads the
 * GPT-2 vocabulary from assets/gpt2-ranks.txt (one line per token: the token's
 * bytes in hex, then its rank). It gives the same IDs as tiktoken's "gpt2".
 *
 * How it works, for one piece of text:
 *   1. start with one part per byte
 *   2. find the neighbouring pair whose joined bytes have the LOWEST rank
 *   3. merge it, repeat until no pair is a known token
 * Low rank = merged early in training = most frequent. Any byte string can
 * be encoded, so no <|unk|> is needed.
 */
#ifndef BPE_H
#define BPE_H

#include <stddef.h>

#include "util.h"

#define BPE_ENDOFTEXT_ID 50256

typedef struct {
    unsigned char **bytes; /* rank -> token bytes */
    size_t *lens;
    size_t size; /* regular tokens, ranks 0..size-1 */
    int *slots;  /* hash table: bytes -> rank (-1 = empty) */
    size_t n_slots;
} Bpe;

/* Load the ranks file. Returns 0 on success, -1 and fills `err` on failure. */
int bpe_load(Bpe *b, const char *path, char err[ERR_LEN]);
void bpe_free(Bpe *b);

/* Text -> token IDs. "<|endoftext|>" in the text becomes ID 50256. Caller frees. */
int *bpe_encode(const Bpe *b, const char *text, size_t *n);
/* IDs -> text. Caller frees. */
char *bpe_decode(const Bpe *b, const int *ids, size_t n);

#endif
