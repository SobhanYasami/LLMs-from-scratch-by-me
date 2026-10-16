/* Step 1 (book 2.2) - split raw text into tokens, and join them back.
 *
 *   "Hello, world!"  ->  ["Hello", ",", "world", "!"]
 *
 * Tip: start reading here, then vocab -> simple -> dataset -> dataloader ->
 * embedding. Each file uses the one before it. bpe is the bonus (book 2.5).
 */
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stddef.h>

#define END_OF_TEXT "<|endoftext|>"
#define UNK_TOKEN "<|unk|>"

/* A growable list of owned strings. */
typedef struct {
    char **items;
    size_t len;
    size_t cap;
} Tokens;

void tokens_push(Tokens *t, char *owned);
void tokens_free(Tokens *t);

/* Return the tokens of `text`; whitespace-only pieces are dropped. */
Tokens split_text(const char *text);

/* Join tokens with spaces, then remove the space before punctuation.
 * Not a perfect inverse: the original spacing is lost. Caller frees. */
char *detokenize(const char *const *tokens, size_t n);

#endif
