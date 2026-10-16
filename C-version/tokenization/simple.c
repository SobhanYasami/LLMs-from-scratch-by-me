#include "simple.h"

#include <stdio.h>
#include <stdlib.h>

#include "preprocess.h"

SimpleTokenizer simple_tokenizer_v1(const Vocab *vocab) { return (SimpleTokenizer){vocab, false}; }
SimpleTokenizer simple_tokenizer_v2(const Vocab *vocab) { return (SimpleTokenizer){vocab, true}; }

/* V1 and V2 share this code. The only difference is what happens to a token
 * that is not in the vocab: error (V1) or <|unk|> (V2). */
int *simple_encode(const SimpleTokenizer *t, const char *text, size_t *n, char err[ERR_LEN]) {
    Tokens tokens = split_text(text);
    int *ids = xmalloc(tokens.len * sizeof(int));
    for (size_t i = 0; i < tokens.len; i++) {
        int id = vocab_id(t->vocab, tokens.items[i]);
        if (id < 0 && t->unknown_to_unk) id = vocab_id(t->vocab, UNK_TOKEN);
        if (id < 0) {
            snprintf(err, ERR_LEN, "token not in vocab: \"%s\"", tokens.items[i]);
            free(ids);
            tokens_free(&tokens);
            return NULL;
        }
        ids[i] = id;
    }
    *n = tokens.len;
    tokens_free(&tokens);
    return ids;
}

char *simple_decode(const SimpleTokenizer *t, const int *ids, size_t n, char err[ERR_LEN]) {
    const char **tokens = xmalloc(n * sizeof(char *));
    for (size_t i = 0; i < n; i++) {
        tokens[i] = vocab_token(t->vocab, ids[i]);
        if (!tokens[i]) {
            snprintf(err, ERR_LEN, "id not in vocab: %d", ids[i]);
            free(tokens);
            return NULL;
        }
    }
    char *text = detokenize(tokens, n);
    free(tokens);
    return text;
}
