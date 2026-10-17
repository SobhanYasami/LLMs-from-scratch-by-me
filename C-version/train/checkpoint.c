#include "checkpoint.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define MAGIC "LLMCKPT1"

static int fail(char err[ERR_LEN], const char *msg) {
    snprintf(err, ERR_LEN, "%s", msg);
    return -1;
}

static int write_tensors(FILE *f, const GptModel *m) {
    float *d[GPT_MAX_TENSORS];
    size_t len[GPT_MAX_TENSORS], n = gpt_tensors(m, d, len);
    for (size_t i = 0; i < n; i++)
        if (fwrite(d[i], sizeof(float), len[i], f) != len[i]) return -1;
    return 0;
}

static int read_tensors(FILE *f, const GptModel *m) {
    float *d[GPT_MAX_TENSORS];
    size_t len[GPT_MAX_TENSORS], n = gpt_tensors(m, d, len);
    for (size_t i = 0; i < n; i++)
        if (fread(d[i], sizeof(float), len[i], f) != len[i]) return -1;
    return 0;
}

int checkpoint_save(const char *path, const GptModel *m, const AdamW *opt, char err[ERR_LEN]) {
    char dir[512];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        mkdir(dir, 0755); /* fine if it already exists */
    }
    FILE *f = fopen(path, "wb");
    if (!f) return fail(err, "cannot create the checkpoint file");
    const GptConfig *c = &m->cfg;
    uint32_t h[7] = {(uint32_t)c->vocab_size, (uint32_t)c->context_length, (uint32_t)c->emb_dim, (uint32_t)c->n_heads,
                     (uint32_t)c->n_layers, (uint32_t)c->qkv_bias, opt != NULL};
    int bad = fwrite(MAGIC, 1, 8, f) != 8 || fwrite(h, sizeof(uint32_t), 7, f) != 7 || write_tensors(f, m) != 0;
    if (!bad && opt) {
        float step = (float)opt->step;
        bad = fwrite(&step, sizeof(float), 1, f) != 1 || write_tensors(f, &opt->exp_avg) != 0 || write_tensors(f, &opt->exp_avg_sq) != 0;
    }
    fclose(f);
    return bad ? fail(err, "write error") : 0;
}

int checkpoint_read_config(const char *path, GptConfig *cfg, int *has_optimizer, char err[ERR_LEN]) {
    FILE *f = fopen(path, "rb");
    if (!f) return fail(err, "cannot open the checkpoint file");
    char magic[8];
    uint32_t h[7];
    int ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, MAGIC, 8) == 0 && fread(h, sizeof(uint32_t), 7, f) == 7;
    fclose(f);
    if (!ok) return fail(err, "not a checkpoint of this repo");
    *cfg = (GptConfig){h[0], h[1], h[2], h[3], h[4], 0.0f, (int)h[5]};
    *has_optimizer = (int)h[6];
    return 0;
}

int checkpoint_load(const char *path, GptModel *m, AdamW *opt, char err[ERR_LEN]) {
    FILE *f = fopen(path, "rb");
    if (!f) return fail(err, "cannot open the checkpoint file");
    uint32_t h[7];
    int bad = fseek(f, 8, SEEK_SET) != 0 || fread(h, sizeof(uint32_t), 7, f) != 7 || read_tensors(f, m) != 0;
    if (!bad && opt && h[6]) {
        float step;
        bad = fread(&step, sizeof(float), 1, f) != 1 || read_tensors(f, &opt->exp_avg) != 0 || read_tensors(f, &opt->exp_avg_sq) != 0;
        opt->step = (int)step;
    }
    fclose(f);
    return bad ? fail(err, "read error: the file is shorter than the model") : 0;
}
