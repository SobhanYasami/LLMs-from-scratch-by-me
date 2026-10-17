#include "openai.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const GptConfig GPT2_SMALL_CONFIG = {50257, 1024, 768, 12, 12, 0.0f, 1};

typedef struct {
    FILE *f;
    char *header;
    long base;
} SafeTensors;

static int fail(char err[ERR_LEN], const char *what, const char *name) {
    snprintf(err, ERR_LEN, "%s: %s", name, what);
    return -1;
}

/* Reads `"key":[1,2]` from an entry into out[], returns how many numbers. */
static size_t read_list(const char *entry, const char *key, long long *out) {
    char pattern[64];
    snprintf(pattern, sizeof pattern, "\"%s\":[", key);
    const char *at = strstr(entry, pattern);
    if (!at) return 0;
    at += strlen(pattern);
    size_t n = 0;
    while (*at && *at != ']') {
        char *end;
        out[n++] = strtoll(at, &end, 10);
        at = *end == ',' ? end + 1 : end;
    }
    return n;
}

/* Reads the tensor `name` into dst (which holds `expected` numbers). The header looks like
 * "name":{"dtype":"F32","shape":[768,2304],"data_offsets":[0,7077888]}; we find the entry by name. */
static int get(SafeTensors *st, const char *name, float *dst, size_t expected, char err[ERR_LEN]) {
    char key[160];
    snprintf(key, sizeof key, "\"%s\":", name);
    const char *at = strstr(st->header, key);
    if (!at) return fail(err, "not in the file", name);
    const char *close = strchr(at, '}');
    char *entry = xstrndup(at, (size_t)(close - at));
    long long off[2];
    size_t n = read_list(entry, "data_offsets", off);
    int is_f32 = strstr(entry, "F32") != NULL;
    free(entry);
    if (n != 2 || !is_f32) return fail(err, "unsupported entry (only float32)", name);
    if ((size_t)(off[1] - off[0]) != expected * sizeof(float)) return fail(err, "shape mismatch", name);
    if (fseek(st->f, st->base + (long)off[0], SEEK_SET) != 0 || fread(dst, sizeof(float), expected, st->f) != expected) return fail(err, "read error", name);
    return 0;
}

int load_gpt2_weights(const char *path, GptModel *m, char err[ERR_LEN]) {
    SafeTensors st = {fopen(path, "rb"), NULL, 0};
    if (!st.f) return fail(err, "cannot open the weights file", path);
    uint64_t n;
    if (fread(&n, sizeof n, 1, st.f) != 1) return fail(err, "bad file", path);
    st.header = xmalloc(n + 1);
    if (fread(st.header, 1, n, st.f) != n) return fail(err, "bad header", path);
    st.header[n] = '\0';
    st.base = 8 + (long)n;
    size_t d = m->cfg.emb_dim, v = m->cfg.vocab_size;
    int rc = 0;
    float *c_attn = xmalloc(d * 3 * d * sizeof(float)), *wte = xmalloc(v * d * sizeof(float));
#define TRY(call) do { if ((rc = (call)) != 0) goto done; } while (0)
    TRY(get(&st, "wte.weight", m->tok_emb.data, v * d, err));
    TRY(get(&st, "wpe.weight", m->pos_emb.data, m->cfg.context_length * d, err));
    for (size_t i = 0; i < m->cfg.n_layers; i++) {
        TransformerBlock *b = &m->blocks[i];
        char name[96];
        /* c_attn is [d, 3d]: query, key and value side by side */
        snprintf(name, sizeof name, "h.%zu.attn.c_attn.weight", i);
        TRY(get(&st, name, c_attn, d * 3 * d, err));
        Matrix *proj[3] = {&b->att.w_query, &b->att.w_key, &b->att.w_value};
        for (int which = 0; which < 3; which++)
            for (size_t r = 0; r < d; r++) memcpy(proj[which]->data + r * d, c_attn + r * 3 * d + (size_t)which * d, d * sizeof(float));
        snprintf(name, sizeof name, "h.%zu.attn.c_attn.bias", i);
        TRY(get(&st, name, c_attn, 3 * d, err));
        for (int which = 0; which < 3; which++) memcpy(b->att.qkv_bias[which].data, c_attn + (size_t)which * d, d * sizeof(float));
        struct { const char *suffix; float *dst; size_t n; } items[] = {
            {"attn.c_proj.weight", b->att.out_weight.data, d * d}, {"attn.c_proj.bias", b->att.out_bias.data, d},
            {"mlp.c_fc.weight", b->ff.expand.w.data, d * 4 * d}, {"mlp.c_fc.bias", b->ff.expand.b.data, 4 * d},
            {"mlp.c_proj.weight", b->ff.shrink.w.data, 4 * d * d}, {"mlp.c_proj.bias", b->ff.shrink.b.data, d},
            {"ln_1.weight", b->norm1.scale.data, d}, {"ln_1.bias", b->norm1.shift.data, d},
            {"ln_2.weight", b->norm2.scale.data, d}, {"ln_2.bias", b->norm2.shift.data, d},
        };
        for (size_t k = 0; k < sizeof items / sizeof items[0]; k++) {
            snprintf(name, sizeof name, "h.%zu.%s", i, items[k].suffix);
            TRY(get(&st, name, items[k].dst, items[k].n, err));
        }
    }
    TRY(get(&st, "ln_f.weight", m->final_norm.scale.data, d, err));
    TRY(get(&st, "ln_f.bias", m->final_norm.shift.data, d, err));
    TRY(get(&st, "wte.weight", wte, v * d, err)); /* weight tying: the head is the embedding again, [vocab, d] -> [d, vocab] */
    for (size_t tok = 0; tok < v; tok++)
        for (size_t r = 0; r < d; r++) m->out_head.data[r * v + tok] = wte[tok * d + r];
done:
    free(c_attn);
    free(wte);
    free(st.header);
    fclose(st.f);
    return rc;
}
