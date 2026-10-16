#include "bpe.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "preprocess.h"

/* ---- loading -------------------------------------------------------- */

static uint64_t hash_bytes(const unsigned char *s, size_t n) { /* FNV-1a */
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) h = (h ^ s[i]) * 1099511628211ULL;
    return h;
}

static int hex_val(int c) { return isdigit(c) ? c - '0' : c - 'a' + 10; }

static int lookup(const Bpe *b, const unsigned char *s, size_t n) {
    size_t slot = hash_bytes(s, n) & (b->n_slots - 1);
    while (b->slots[slot] != -1) {
        int r = b->slots[slot];
        if (b->lens[r] == n && memcmp(b->bytes[r], s, n) == 0) return r;
        slot = (slot + 1) & (b->n_slots - 1);
    }
    return -1;
}

int bpe_load(Bpe *b, const char *path, char err[ERR_LEN]) {
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(err, ERR_LEN, "cannot open %s", path);
        return -1;
    }
    size_t cap = 51000;
    *b = (Bpe){0};
    b->bytes = xmalloc(cap * sizeof(char *));
    b->lens = xmalloc(cap * sizeof(size_t));
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *space = strchr(line, ' ');
        if (!space) continue;
        size_t hex_len = (size_t)(space - line);
        unsigned char *bytes = xmalloc(hex_len / 2 + 1);
        for (size_t i = 0; i < hex_len / 2; i++)
            bytes[i] = (unsigned char)(hex_val(line[2 * i]) * 16 + hex_val(line[2 * i + 1]));
        size_t rank = (size_t)atol(space + 1); /* ranks are 0,1,2,... in file order */
        if (rank >= cap) break;
        b->bytes[rank] = bytes;
        b->lens[rank] = hex_len / 2;
        if (rank + 1 > b->size) b->size = rank + 1;
    }
    fclose(f);

    b->n_slots = 1;
    while (b->n_slots < b->size * 2) b->n_slots <<= 1;
    b->slots = xmalloc(b->n_slots * sizeof(int));
    for (size_t i = 0; i < b->n_slots; i++) b->slots[i] = -1;
    for (size_t r = 0; r < b->size; r++) {
        size_t slot = hash_bytes(b->bytes[r], b->lens[r]) & (b->n_slots - 1);
        while (b->slots[slot] != -1) slot = (slot + 1) & (b->n_slots - 1);
        b->slots[slot] = (int)r;
    }
    return 0;
}

void bpe_free(Bpe *b) {
    for (size_t i = 0; i < b->size; i++) free(b->bytes[i]);
    free(b->bytes);
    free(b->lens);
    free(b->slots);
    *b = (Bpe){0};
}

/* ---- the merge loop ------------------------------------------------- */

/* Encode one piece (a few bytes) by repeatedly merging the best neighbour pair. */
static void encode_piece(const Bpe *b, const unsigned char *s, size_t n, int **out, size_t *len, size_t *cap) {
    /* bounds[i]..bounds[i+1] is part i; begin with one part per byte */
    size_t *bounds = xmalloc((n + 1) * sizeof(size_t));
    size_t parts = n;
    for (size_t i = 0; i <= n; i++) bounds[i] = i;

    for (;;) {
        int best_rank = -1;
        size_t best = 0;
        for (size_t i = 0; i + 1 < parts; i++) { /* rank of parts i and i+1 joined */
            int r = lookup(b, s + bounds[i], bounds[i + 2] - bounds[i]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) best_rank = r, best = i;
        }
        if (best_rank < 0) break;
        for (size_t i = best + 1; i < parts; i++) bounds[i] = bounds[i + 1]; /* merge */
        parts--;
    }
    for (size_t i = 0; i < parts; i++) {
        if (*len == *cap) {
            *cap *= 2;
            *out = xrealloc(*out, *cap * sizeof(int));
        }
        (*out)[(*len)++] = lookup(b, s + bounds[i], bounds[i + 1] - bounds[i]);
    }
    free(bounds);
}

/* ---- splitting text into pieces (GPT-2's regex, written by hand) ----- */

/* The GPT-2 pattern, in words:
 *   'm 'd 's 't 're 've 'll | optional space + letters | optional space + digits
 *   | optional space + other symbols | whitespace
 * Bytes >= 0x80 (UTF-8) are treated as letters, which is right for most text. */
static bool is_letter(unsigned char c) { return isalpha(c) || c >= 0x80; }
static bool is_digit(unsigned char c) { return isdigit(c); }
static bool is_space(unsigned char c) { return c < 0x80 && isspace(c); }
static bool is_other(unsigned char c) { return !is_space(c) && !is_letter(c) && !is_digit(c); }

static size_t consume(const unsigned char *s, size_t n, size_t i, bool (*pred)(unsigned char)) {
    while (i < n && pred(s[i])) i++;
    return i;
}

/* End of the piece starting at i. */
static size_t next_piece(const unsigned char *s, size_t n, size_t i) {
    if (s[i] == '\'' && i + 1 < n) {
        if (strchr("sdmt", s[i + 1])) return i + 2;
        if (i + 2 < n && ((s[i + 1] == 'l' && s[i + 2] == 'l') || (s[i + 1] == 'v' && s[i + 2] == 'e') ||
                          (s[i + 1] == 'r' && s[i + 2] == 'e')))
            return i + 3;
    }
    size_t k = (s[i] == ' ') ? i + 1 : i; /* optional single leading space */
    if (k < n) {
        if (is_letter(s[k])) return consume(s, n, k, is_letter);
        if (is_digit(s[k])) return consume(s, n, k, is_digit);
        if (is_other(s[k])) return consume(s, n, k, is_other);
    }
    /* whitespace run: leave its last char for the next piece, unless it is the end */
    size_t e = consume(s, n, i, is_space);
    return (e < n && e - i > 1) ? e - 1 : e;
}

int *bpe_encode(const Bpe *b, const char *text, size_t *n_out) {
    const unsigned char *s = (const unsigned char *)text;
    size_t n = strlen(text), len = 0, cap = 1024;
    int *out = xmalloc(cap * sizeof(int));
    const size_t special_len = strlen(END_OF_TEXT);

    size_t i = 0;
    while (i < n) {
        if (strncmp(text + i, END_OF_TEXT, special_len) == 0) { /* special token */
            if (len == cap) out = xrealloc(out, (cap *= 2) * sizeof(int));
            out[len++] = BPE_ENDOFTEXT_ID;
            i += special_len;
            continue;
        }
        size_t end = next_piece(s, n, i);
        /* never let a piece swallow the start of a special token */
        for (size_t j = i + 1; j < end; j++)
            if (strncmp(text + j, END_OF_TEXT, special_len) == 0) {
                end = j;
                break;
            }
        encode_piece(b, s + i, end - i, &out, &len, &cap);
        i = end;
    }
    *n_out = len;
    return out;
}

char *bpe_decode(const Bpe *b, const int *ids, size_t n) {
    size_t total = 1;
    for (size_t i = 0; i < n; i++)
        total += ids[i] == BPE_ENDOFTEXT_ID ? strlen(END_OF_TEXT) : b->lens[ids[i]];
    char *out = xmalloc(total), *p = out;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] == BPE_ENDOFTEXT_ID) {
            memcpy(p, END_OF_TEXT, strlen(END_OF_TEXT));
            p += strlen(END_OF_TEXT);
        } else {
            memcpy(p, b->bytes[ids[i]], b->lens[ids[i]]);
            p += b->lens[ids[i]];
        }
    }
    *p = '\0';
    return out;
}
