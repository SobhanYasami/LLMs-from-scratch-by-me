/* Chapter 5 demos: pretraining, one step at a time (book 5.1 to 5.5).
 *
 * Run from C-version/:  make run ARGS=ch05
 *                       make run ARGS=ch05-gpt2   (OpenAI's weights, 5.5)
 *
 * Tip: read the demos top to bottom. The model is the tiny GPT of chapter 4 (emb 32, 2 blocks,
 * context 16) so that every language can TRAIN it in seconds; the book trains the 124M model on a
 * GPU. All five languages use the same weights, data order (no shuffling) and sampling generator,
 * so they print the same numbers. They differ from the book's torch output.
 */
#include "ch05.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attention/matrix.h"
#include "gpt/model.h"
#include "tokenization/bpe.h"
#include "tokenization/dataloader.h"
#include "tokenization/dataset.h"
#include "tokenization/util.h"
#include "train/adamw.h"
#include "train/checkpoint.h"
#include "train/grad.h"
#include "train/loss.h"
#include "train/openai.h"
#include "train/sample.h"

#define DATA_PATH "../assets/the-verdict.txt"
#define RANKS_PATH "../assets/gpt2-ranks.txt"
#define GPT2_PATH "../assets/gpt2/model.safetensors"
#define CKPT_PATH "../assets/checkpoints/c.bin"
#define BATCH 4
#define STRIDE 16
#define EPOCHS 2
#define EVAL_FREQ 24
#define EVAL_ITER 2
#define LR 3e-3
#define WEIGHT_DECAY 0.1

static GptConfig cfg5(void) {
    GptConfig c = GPT_CONFIG_TINY;
    c.drop_rate = 0.0f; /* dropout stays off: our loop has no random masks */
    return c;
}

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

/* 4 decimals etc.; a tiny negative number is just zero. */
static void print_num(double v, int digits) {
    char buf[64], zero[64];
    snprintf(buf, sizeof buf, "%.*f", digits, v);
    snprintf(zero, sizeof zero, "-%.*f", digits, 0.0);
    printf("%s", strcmp(buf, zero) == 0 ? buf + 1 : buf);
}

static void print_row(const float *values, size_t n, int digits) {
    printf("[");
    for (size_t i = 0; i < n; i++) {
        if (i) printf(", ");
        print_num((double)values[i], digits);
    }
    printf("]");
}

static char *commas(size_t n, char *out) {
    char digits[32];
    snprintf(digits, sizeof digits, "%zu", n);
    size_t len = strlen(digits), o = 0;
    for (size_t i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0) out[o++] = ',';
        out[o++] = digits[i];
    }
    out[o] = '\0';
    return out;
}

/* Text in single quotes with newlines flattened (same in every language). */
static void print_quoted(const char *text) {
    printf("'");
    for (const char *p = text; *p; p++) putchar(*p == '\n' ? ' ' : *p);
    printf("'");
}

/* A model with its gradients and optimizer, in arenas big enough for them. */
typedef struct {
    Arena weights, aux, scratch;
    GptModel model, grads;
    AdamW opt;
} Trainer;

static Trainer trainer_new(GptConfig cfg, int random_init) {
    Trainer t;
    t.weights = arena_new(gpt_arena_bytes(&cfg));
    t.aux = arena_new(gpt_parameter_count(&cfg) * sizeof(float) * 3 + (size_t)16 * 1024 * 1024);
    t.scratch = arena_new((size_t)256 * 1024 * 1024);
    t.model = gpt_model_new(&t.weights, cfg);
    if (random_init) gpt_model_init(&t.model, 123);
    t.grads = gpt_model_zeros_like(&t.aux, &t.model);
    t.opt = adamw_new(&t.aux, &t.model, LR, WEIGHT_DECAY);
    return t;
}

static void trainer_free(Trainer *t) {
    arena_free(&t->weights);
    arena_free(&t->aux);
    arena_free(&t->scratch);
}

/* The batches of one text, in book order: windows cut and grouped like chapter 2's data loader (no shuffling). */
static Batch *make_batches(const Bpe *bpe, const char *text, size_t n_text, size_t ctx, int drop_last, size_t *count) {
    char *copy = xstrndup(text, n_text);
    size_t n_ids;
    int *ids = bpe_encode(bpe, copy, &n_ids);
    free(copy);
    GPTDataset ds = gpt_dataset_new(ids, n_ids, ctx, STRIDE);
    DataLoader dl = dataloader_new(&ds, BATCH, false, drop_last, 123);
    size_t n = dataloader_start_epoch(&dl);
    Batch *batches = xmalloc(n * sizeof(Batch));
    for (size_t b = 0; b < n; b++) {
        batches[b].inputs = xmalloc(BATCH * ctx * sizeof(int));
        batches[b].targets = xmalloc(BATCH * ctx * sizeof(int));
        batches[b].rows = dataloader_batch(&dl, b, batches[b].inputs, batches[b].targets);
        batches[b].len = ctx;
    }
    dataloader_free(&dl);
    free(ids);
    *count = n;
    return batches;
}

static void free_batches(Batch *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        free(b[i].inputs);
        free(b[i].targets);
    }
    free(b);
}

typedef struct {
    Bpe bpe;
    char *text;
    size_t n_text, split;
} Data;

static Data load_data(void) {
    Data d;
    char err[ERR_LEN];
    if (bpe_load(&d.bpe, RANKS_PATH, err) != 0) die(err);
    FILE *f = fopen(DATA_PATH, "rb");
    if (!f) die("cannot open " DATA_PATH " (run from C-version/)");
    fseek(f, 0, SEEK_END);
    d.n_text = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    d.text = xmalloc(d.n_text + 1);
    if (fread(d.text, 1, d.n_text, f) != d.n_text) die("read error");
    d.text[d.n_text] = '\0';
    fclose(f);
    d.split = (size_t)(0.9 * (double)d.n_text); /* 90% training, 10% validation */
    return d;
}

static void free_data(Data *d) {
    bpe_free(&d->bpe);
    free(d->text);
}

/* 5.1: from probabilities to a single number. */
static void demo_loss(Data *d) {
    printf("== 5.1 evaluating generative text models ==\n");
    GptConfig cfg = cfg5();
    Trainer t = trainer_new(cfg, 1);
    const int inputs[2][3] = {{16833, 3626, 6100}, {40, 1107, 588}};  /* ["every effort moves", "I really like"] */
    const int targets[2][3] = {{3626, 6100, 345}, {588, 428, 11311}}; /* [" effort moves you", " really like chocolate"] */
    printf("inputs : [[16833, 3626, 6100], [40, 1107, 588]]\n");
    printf("targets: [[3626, 6100, 345], [588, 428, 11311]]\n");
    printf("probas shape: (2, 3, %zu)  <- [batch, tokens, vocab_size]\n", cfg.vocab_size);
    float log_probas[6], ce_total = 0, *probas = xmalloc(cfg.vocab_size * sizeof(float));
    for (int i = 0; i < 2; i++) {
        size_t mark = arena_mark(&t.scratch);
        Matrix logits = gpt_forward(&t.scratch, &t.model, inputs[i], 3);
        ce_total += cross_entropy_sum(logits, targets[i]);
        float picked[3];
        for (int k = 0; k < 3; k++) {
            softmax_row(matrix_row(logits, (size_t)k), cfg.vocab_size, probas);
            picked[k] = probas[targets[i][k]];
            log_probas[i * 3 + k] = (float)log((double)picked[k]);
        }
        printf("probability of the target, text %d: ", i + 1);
        print_row(picked, 3, 6);
        printf("\n");
        arena_reset(&t.scratch, mark);
    }
    free(probas);
    printf("log probabilities: ");
    print_row(log_probas, 6, 4);
    printf("\n");
    float sum = 0;
    for (int i = 0; i < 6; i++) sum += log_probas[i];
    float avg = sum / 6;
    printf("average log probability: ");
    print_num((double)avg, 4);
    printf("\nnegative average (the loss): ");
    print_num((double)-avg, 4);
    float loss = ce_total / 6;
    printf("\ncross_entropy of the whole batch at once: ");
    print_num((double)loss, 4);
    printf("\nperplexity: ");
    print_num((double)perplexity(loss), 0);
    printf("  (a model that guesses uniformly is torn between 50,257 tokens)\n");
    size_t n_train, n_val;
    Batch *train = make_batches(&d->bpe, d->text, d->split, cfg.context_length, 1, &n_train);
    Batch *val = make_batches(&d->bpe, d->text + d->split, d->n_text - d->split, cfg.context_length, 0, &n_val);
    printf("training batches: %zu of %d x %zu tokens, validation batches: %zu\n", n_train, BATCH, cfg.context_length, n_val);
    float tl = calc_loss_loader(&t.scratch, &t.model, train, n_train, 0), vl = calc_loss_loader(&t.scratch, &t.model, val, n_val, 0);
    printf("untrained: training loss ");
    print_num((double)tl, 3);
    printf(", validation loss ");
    print_num((double)vl, 3);
    printf("  (a uniform guess gives ln(50257) = ");
    print_num(log(50257.0), 3);
    printf(")\n");
    free_batches(train, n_train);
    free_batches(val, n_val);
    trainer_free(&t);
}

static void generate_and_print_sample(Trainer *t, const Bpe *bpe, const char *start) {
    size_t n;
    int *ids = bpe_encode(bpe, start, &n);
    int *out = generate_text_simple(&t->scratch, &t->model, ids, n, 20, t->model.cfg.context_length);
    char *text = bpe_decode(bpe, out, n + 20);
    for (const char *p = text; *p; p++) putchar(*p == '\n' ? ' ' : *p); /* the book's compact print format */
    printf("\n");
    free(text);
    free(out);
    free(ids);
}

/* One training step: zero the old gradients, forward and backward, update the weights. */
static float train_step(Trainer *t, const Batch *b) {
    zero_grads(&t->grads);
    size_t mark = arena_mark(&t->scratch);
    float loss = loss_and_grads(&t->scratch, &t->model, &t->grads, b);
    arena_reset(&t->scratch, mark);
    adamw_step(&t->opt, &t->model, &t->grads);
    return loss;
}

/* 5.2: train for a few epochs. */
static Trainer train_demo(Data *d) {
    printf("== 5.2 training an LLM ==\n");
    Trainer t = trainer_new(cfg5(), 1);
    size_t n_train, n_val;
    Batch *train = make_batches(&d->bpe, d->text, d->split, t.model.cfg.context_length, 1, &n_train);
    Batch *val = make_batches(&d->bpe, d->text + d->split, d->n_text - d->split, t.model.cfg.context_length, 0, &n_val);
    int step = 0;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (size_t b = 0; b < n_train; b++) {
            train_step(&t, &train[b]);
            if (step % EVAL_FREQ == 0) {
                float tl = calc_loss_loader(&t.scratch, &t.model, train, n_train, EVAL_ITER);
                float vl = calc_loss_loader(&t.scratch, &t.model, val, n_val, EVAL_ITER);
                printf("Ep %d (Step %06d): Train loss ", epoch + 1, step);
                print_num((double)tl, 3);
                printf(", Val loss ");
                print_num((double)vl, 3);
                printf("\n");
            }
            step++;
        }
        generate_and_print_sample(&t, &d->bpe, "Every effort moves you");
    }
    free_batches(train, n_train);
    free_batches(val, n_val);
    return t;
}

static const char *VOCAB[9] = {"closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you"};
static const float NEXT_TOKEN_LOGITS[9] = {4.51f, 0.89f, -1.90f, 6.75f, 1.63f, -1.62f, -1.89f, 6.28f, 1.79f};

static void print_ids(const int *ids, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) printf(i ? ", %d" : "%d", ids[i]);
    printf("]");
}

/* 5.3: temperature and top-k. */
static void demo_decoding(Trainer *t, Data *d) {
    printf("== 5.3 decoding strategies ==\n");
    float probas[9];
    softmax_row(NEXT_TOKEN_LOGITS, 9, probas);
    printf("next-token probabilities: ");
    print_row(probas, 9, 4);
    size_t greedy_pick = 0;
    for (size_t i = 1; i < 9; i++)
        if (probas[i] > probas[greedy_pick]) greedy_pick = i;
    printf("\ngreedy pick: %s\n", VOCAB[greedy_pick]);
    const char *labels[3] = {"1", "0.1", "5"};
    const float temps[3] = {1.0f, 0.1f, 5.0f};
    for (int k = 0; k < 3; k++) {
        float p[9];
        softmax_with_temperature(NEXT_TOKEN_LOGITS, 9, temps[k], p);
        Rng rng = rng_new(123);
        int counts[9] = {0};
        for (int i = 0; i < 1000; i++) counts[sample_index(p, 9, &rng)]++;
        printf("temperature %s: P(pizza) = ", labels[k]);
        print_num((double)p[6], 4);
        printf("  1000 draws: ");
        int first = 1;
        for (int i = 0; i < 9; i++)
            if (counts[i]) {
                printf("%s%d x %s", first ? "" : ", ", counts[i], VOCAB[i]);
                first = 0;
            }
        printf("\n");
    }
    int order[9];
    for (int i = 0; i < 9; i++) order[i] = i;
    for (int i = 0; i < 9; i++) /* a tiny selection sort, biggest logit first */
        for (int j = i + 1; j < 9; j++)
            if (NEXT_TOKEN_LOGITS[order[j]] > NEXT_TOKEN_LOGITS[order[i]]) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
    float top[3] = {NEXT_TOKEN_LOGITS[order[0]], NEXT_TOKEN_LOGITS[order[1]], NEXT_TOKEN_LOGITS[order[2]]};
    printf("top-3 logits: ");
    print_row(top, 3, 2);
    printf(" at positions ");
    print_ids(order, 3);
    printf("\n");
    float masked[9], masked_probas[9];
    top_k_logits(NEXT_TOKEN_LOGITS, 9, 3, masked);
    printf("after masking with -inf: [");
    for (int i = 0; i < 9; i++) {
        printf(i ? ", '" : "'");
        if (isinf(masked[i])) printf("-inf");
        else print_num((double)masked[i], 2);
        printf("'");
    }
    printf("]\n");
    softmax_row(masked, 9, masked_probas);
    printf("top-3 probabilities: ");
    print_row(masked_probas, 9, 4);
    printf("\n");
    size_t n;
    int *start = bpe_encode(&d->bpe, "Every effort moves you", &n);
    size_t ctx = t->model.cfg.context_length;
    int *greedy = generate_text_simple(&t->scratch, &t->model, start, n, 15, ctx);
    char *text = bpe_decode(&d->bpe, greedy, n + 15);
    printf("greedy   : ");
    print_quoted(text);
    free(text);
    Rng rng = rng_new(123);
    int *sampled = generate(&t->scratch, &t->model, start, n, 15, ctx, &rng, 1.4f, 25);
    text = bpe_decode(&d->bpe, sampled, n + 15);
    printf("\ntop-k 25, temperature 1.4: ");
    print_quoted(text);
    free(text);
    Rng rng0 = rng_new(123);
    int *same = generate(&t->scratch, &t->model, start, n, 15, ctx, &rng0, 0.0f, 0);
    printf("\ntemperature 0 equals greedy: %s\n", memcmp(same, greedy, (n + 15) * sizeof(int)) == 0 ? "true" : "false");
    free(same);
    free(sampled);
    free(greedy);
    free(start);
}

/* 5.4: save, load, and carry on exactly where we stopped. */
static void demo_checkpoint(Trainer *t, Data *d) {
    printf("== 5.4 saving and loading ==\n");
    char err[ERR_LEN], buf[32];
    if (checkpoint_save(CKPT_PATH, &t->model, &t->opt, err) != 0) die(err);
    FILE *f = fopen(CKPT_PATH, "rb");
    fseek(f, 0, SEEK_END);
    printf("saved model and optimizer state: %s bytes\n", commas((size_t)ftell(f), buf));
    fclose(f);
    GptConfig cfg;
    int has_opt;
    if (checkpoint_read_config(CKPT_PATH, &cfg, &has_opt, err) != 0) die(err);
    Trainer r = trainer_new(cfg, 0); /* built from the file's settings, weights still zero */
    if (checkpoint_load(CKPT_PATH, &r.model, &r.opt, err) != 0) die(err);
    const int x[4] = {6109, 3626, 6100, 345};
    Matrix a = gpt_forward(&t->scratch, &t->model, x, 4), b = gpt_forward(&r.scratch, &r.model, x, 4);
    float diff = 0;
    for (size_t i = 0; i < a.rows * a.cols; i++) diff = fmaxf(diff, fabsf(a.data[i] - b.data[i]));
    printf("largest logit difference between the two models: ");
    print_num((double)diff, 6);
    printf("\n");
    arena_reset(&t->scratch, 0);
    arena_reset(&r.scratch, 0);
    size_t n_train;
    Batch *train = make_batches(&d->bpe, d->text, d->split, cfg.context_length, 1, &n_train);
    float continued[3], resumed[3];
    for (int i = 0; i < 3; i++) continued[i] = train_step(t, &train[i]);
    for (int i = 0; i < 3; i++) resumed[i] = train_step(&r, &train[i]);
    printf("3 more steps, continued: losses ");
    print_row(continued, 3, 3);
    printf("\n3 more steps, restored : losses ");
    print_row(resumed, 3, 3);
    printf("\nsame losses: %s\n", memcmp(continued, resumed, sizeof continued) == 0 ? "true" : "false");
    free_batches(train, n_train);
    trainer_free(&r);
}

void run_chapter5(void) {
    Data d = load_data();
    demo_loss(&d);
    Trainer t = train_demo(&d);
    demo_decoding(&t, &d);
    demo_checkpoint(&t, &d);
    trainer_free(&t);
    free_data(&d);
}

void run_chapter5_gpt2(void) {
    Data d = load_data();
    char err[ERR_LEN], buf[32];
    GptConfig cfg = GPT2_SMALL_CONFIG;
    Arena weights = arena_new(gpt_arena_bytes(&cfg)), scratch = arena_new((size_t)512 * 1024 * 1024);
    GptModel m = gpt_model_new(&weights, cfg);
    printf("== 5.5 loading pretrained weights from OpenAI ==\n");
    if (load_gpt2_weights(GPT2_PATH, &m, err) != 0) {
        fprintf(stderr, "%s (download the weights first: command in train/openai.h)\n", err);
        return;
    }
    printf("loaded GPT-2 small: %s parameters (with the attention biases)\n", commas(gpt_count_parameters(&m), buf));
    size_t n;
    int *start = bpe_encode(&d.bpe, "Every effort moves you", &n);
    Rng rng = rng_new(123);
    int *out = generate(&scratch, &m, start, n, 25, cfg.context_length, &rng, 1.5f, 50);
    char *text = bpe_decode(&d.bpe, out, n + 25);
    printf("generated: ");
    print_quoted(text);
    free(text);
    int *greedy = generate_text_simple(&scratch, &m, start, n, 25, cfg.context_length);
    text = bpe_decode(&d.bpe, greedy, n + 25);
    printf("\ngreedy   : ");
    print_quoted(text);
    free(text);
    /* exercise 5.5: the loss of the pretrained model on the validation part of The Verdict */
    char *copy = xstrndup(d.text + d.split, d.n_text - d.split);
    size_t n_ids;
    int *ids = bpe_encode(&d.bpe, copy, &n_ids);
    free(copy);
    GPTDataset ds = gpt_dataset_new(ids, n_ids, 64, 64);
    DataLoader dl = dataloader_new(&ds, 2, false, true, 123);
    size_t n_batches = dataloader_start_epoch(&dl);
    Batch *val = xmalloc(n_batches * sizeof(Batch));
    for (size_t b = 0; b < n_batches; b++) {
        val[b].inputs = xmalloc(2 * 64 * sizeof(int));
        val[b].targets = xmalloc(2 * 64 * sizeof(int));
        val[b].rows = dataloader_batch(&dl, b, val[b].inputs, val[b].targets);
        val[b].len = 64;
    }
    float loss = calc_loss_loader(&scratch, &m, val, n_batches, 2);
    printf("\nexercise 5.5: loss of the pretrained model on The Verdict (validation part, 2 batches of 2 x 64 tokens): ");
    print_num((double)loss, 3);
    printf("\n");
    free_batches(val, n_batches);
    dataloader_free(&dl);
    free(ids);
    free(greedy);
    free(out);
    free(start);
    arena_free(&weights);
    arena_free(&scratch);
    free_data(&d);
}
