#include "vocab.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

static uint64_t hash_str(const char *s) { /* FNV-1a */
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) h = (h ^ (unsigned char)*s) * 1099511628211ULL;
    return h;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

Vocab build_vocab(const Tokens *tokens) {
    /* sort a copy of the pointers, then skip duplicates */
    char **sorted = xmalloc(tokens->len * sizeof(char *));
    memcpy(sorted, tokens->items, tokens->len * sizeof(char *));
    qsort(sorted, tokens->len, sizeof(char *), cmp_str);

    Vocab v = {0};
    v.tokens = xmalloc((tokens->len + 2) * sizeof(char *));
    for (size_t i = 0; i < tokens->len; i++)
        if (i == 0 || strcmp(sorted[i], sorted[i - 1]) != 0)
            v.tokens[v.size++] = xstrdup(sorted[i]);
    free(sorted);
    /* special tokens go last so regular word IDs stay stable */
    v.tokens[v.size++] = xstrdup(END_OF_TEXT);
    v.tokens[v.size++] = xstrdup(UNK_TOKEN);

    v.n_slots = 1;
    while (v.n_slots < v.size * 2) v.n_slots <<= 1;
    v.slots = xmalloc(v.n_slots * sizeof(int));
    for (size_t i = 0; i < v.n_slots; i++) v.slots[i] = -1;
    for (size_t id = 0; id < v.size; id++) {
        size_t s = hash_str(v.tokens[id]) & (v.n_slots - 1);
        while (v.slots[s] != -1) s = (s + 1) & (v.n_slots - 1);
        v.slots[s] = (int)id;
    }
    return v;
}

void vocab_free(Vocab *v) {
    for (size_t i = 0; i < v->size; i++) free(v->tokens[i]);
    free(v->tokens);
    free(v->slots);
    *v = (Vocab){0};
}

int vocab_id(const Vocab *v, const char *token) {
    size_t s = hash_str(token) & (v->n_slots - 1);
    while (v->slots[s] != -1) {
        if (strcmp(v->tokens[v->slots[s]], token) == 0) return v->slots[s];
        s = (s + 1) & (v->n_slots - 1);
    }
    return -1;
}

const char *vocab_token(const Vocab *v, int id) {
    return (id >= 0 && (size_t)id < v->size) ? v->tokens[id] : NULL;
}
