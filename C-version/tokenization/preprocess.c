#include "preprocess.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

/* Delimiters that become tokens of their own. "--" and whitespace are handled
 * in split_text. Same set as the Python regex: , . ? _ ! " ( ) ' */
#define SPLIT_PUNCT ",.?_!\"()'"
/* Punctuation that detokenize glues to the previous word. */
#define DETOKENIZE_PUNCT ",.?!\"()'"

void tokens_push(Tokens *t, char *owned) {
    if (t->len == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 64;
        t->items = xrealloc(t->items, t->cap * sizeof(char *));
    }
    t->items[t->len++] = owned;
}

void tokens_free(Tokens *t) {
    for (size_t i = 0; i < t->len; i++) free(t->items[i]);
    free(t->items);
    *t = (Tokens){0};
}

/* There is no regex library in C, so we scan by hand: walk the characters and
 * cut a token whenever we meet whitespace, "--" or a punctuation mark. */
Tokens split_text(const char *text) {
    Tokens out = {0};
    size_t n = strlen(text), word_start = 0;

#define FLUSH(end)                                                    \
    do {                                                              \
        if ((end) > word_start)                                       \
            tokens_push(&out, xstrndup(text + word_start, (end) - word_start)); \
    } while (0)

    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)text[i];
        if (isspace(c)) {
            FLUSH(i);
            word_start = i + 1;
        } else if (c == '-' && text[i + 1] == '-') {
            FLUSH(i);
            tokens_push(&out, xstrdup("--"));
            i++;
            word_start = i + 1;
        } else if (strchr(SPLIT_PUNCT, c)) {
            FLUSH(i);
            tokens_push(&out, xstrndup(text + i, 1));
            word_start = i + 1;
        }
    }
    FLUSH(n);
#undef FLUSH
    return out;
}

char *detokenize(const char *const *tokens, size_t n) {
    size_t total = 1;
    for (size_t i = 0; i < n; i++) total += strlen(tokens[i]) + 1;
    char *out = xmalloc(total), *p = out;
    for (size_t i = 0; i < n; i++) {
        size_t len = strlen(tokens[i]);
        bool glue = len == 1 && strchr(DETOKENIZE_PUNCT, tokens[i][0]);
        if (i > 0 && !glue) *p++ = ' ';
        memcpy(p, tokens[i], len);
        p += len;
    }
    *p = '\0';
    return out;
}
