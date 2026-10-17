/* Chapter 4 demos: the GPT model, one building block at a time (book 4.1 to 4.7).
 *
 * Run from C-version/:  make run ARGS=ch04
 *                       make run ARGS=ch04-124m   (the real 124M model)
 *
 * Tip: read the demos top to bottom. Numbers in 4.2, 4.3 and 4.4 match the book
 * (the 4.4 weights are the book's torch.manual_seed(123) ones, written out so
 * every language shares them). The model demos use random weights from the repo's
 * shared generator, so all five languages print the same numbers; they differ from
 * the book's torch output, and since the model is untrained its text is gibberish.
 */
#include "ch04.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attention/matrix.h"
#include "gpt/block.h"
#include "gpt/layers.h"
#include "gpt/model.h"
#include "gpt/shortcut.h"
#include "tokenization/bpe.h"
#include "tokenization/util.h"

#define RANKS_PATH "../assets/gpt2-ranks.txt"

/* Book 4.1: two short texts, already turned into token IDs by the GPT-2 tokenizer. */
static const int BATCH[2][4] = {{6109, 3626, 6100, 345}, {6109, 1110, 6622, 257}}; /* "Every effort moves you" / "Every day holds a" */

/* Book 4.2: the outputs of a Linear(5, 6) + ReLU layer for two inputs. */
static const float LAYER_OUT[2][6] = {
    {0.2260f, 0.3470f, 0.0000f, 0.2216f, 0.0000f, 0.0000f},
    {0.2133f, 0.2394f, 0.0000f, 0.5198f, 0.3297f, 0.0000f},
};

/* Book 4.4: weights and biases of five Linear layers (3->3 four times, then 3->1), torch.manual_seed(123). */
static const ExampleLayer SHORTCUT_LAYERS[5] = {
    {3, 3, {-0.23542964, 0.01912448, -0.28674594, 0.21772662, -0.49193421, 0.42322308, -0.41964141, -0.45901766, -0.36482018}, {0.26147819, -0.21332639, 0.21605217}},
    {3, 3, {-0.49001414, -0.35029206, -0.21198919, -0.11346072, -0.44043937, 0.37804362, -0.13615717, 0.18532233, 0.40826949}, {0.10756382, 0.15787685, 0.55729234}},
    {3, 3, {-0.26039040, 0.18287641, -0.25687245, 0.41260317, 0.46110451, -0.53230095, 0.49285263, 0.27569306, 0.25159022}, {0.23768058, 0.47995073, -0.07623307}},
    {3, 3, {-0.48826423, -0.16567004, -0.40661314, 0.03816671, -0.10779562, -0.30968198, -0.05249349, 0.54698384, -0.04553875}, {0.01833092, -0.09004387, 0.09076351}},
    {1, 3, {0.51442641, 0.35304558, 0.20493712}, {0.12546431}},
};

/* 4 decimals; a tiny negative number is just zero. */
static void print4(double v) {
    char buf[32];
    snprintf(buf, sizeof buf, "%.4f", v);
    printf("%s", strcmp(buf, "-0.0000") == 0 ? "0.0000" : buf);
}

static void print_row(const float *values, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) {
        if (i) printf(", ");
        print4((double)values[i]);
    }
    printf("]");
}

static void show4(const char *label, const float *values, size_t n) {
    printf("%s: ", label);
    print_row(values, n);
    printf("\n");
}

/* 1234567 -> "1,234,567" */
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

/* a float with two decimals and thousands separators */
static char *commas_f(double v, char *out) {
    char s[48], head[32];
    snprintf(s, sizeof s, "%.2f", v);
    char *dot = strchr(s, '.');
    *dot = '\0';
    snprintf(out, 48, "%s%s", commas((size_t)atoll(s), head), dot == s ? "" : ".");
    strcat(out, dot + 1);
    return out;
}

static void print_ids(const int *ids, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) printf(i ? ", %d" : "%d", ids[i]);
    printf("]");
}

/* A model with random weights, built in an arena big enough for it. */
static GptModel build(Arena *a, GptConfig cfg) {
    *a = arena_new(gpt_arena_bytes(&cfg));
    GptModel m = gpt_model_new(a, cfg);
    gpt_model_init(&m, 123);
    return m;
}

/* 4.1: token IDs in, one score per vocabulary word out, for every position. */
static void demo_big_picture(void) {
    printf("== 4.1 the big picture (tiny GPT: emb 32, 4 heads, 2 blocks) ==\n");
    printf("batch of token IDs, shape (2, 4): [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]\n");
    Arena a;
    GptModel m = build(&a, GPT_CONFIG_TINY);
    Matrix logits = gpt_forward(&a, &m, BATCH[0], 4); /* the batch is processed one sequence at a time */
    printf("logits shape: (2, %zu, %zu)  <- [batch, tokens, vocab_size]\n", logits.rows, logits.cols);
    show4("logits[0][0], first 4 of 50257", matrix_row(logits, 0), 4);
    arena_free(&a);
}

/* 4.2: make each token's numbers mean 0, variance 1. */
static void demo_layer_norm(void) {
    printf("== 4.2 layer normalization ==\n");
    Arena a = arena_new(1 << 20);
    Matrix x = matrix_from(&a, 2, 6, &LAYER_OUT[0][0]);
    float means[2], vars[2], norm0[6];
    for (size_t i = 0; i < 2; i++) {
        const float *row = matrix_row(x, i);
        float sum = 0;
        for (size_t j = 0; j < 6; j++) sum += row[j];
        means[i] = sum / 6;
        float sq = 0;
        for (size_t j = 0; j < 6; j++) sq += (row[j] - means[i]) * (row[j] - means[i]);
        vars[i] = sq / 5; /* the book first uses the unbiased variance */
    }
    for (size_t j = 0; j < 6; j++) norm0[j] = (matrix_row(x, 0)[j] - means[0]) / (float)sqrt((double)vars[0]);
    show4("mean per row", means, 2);
    show4("variance per row", vars, 2);
    show4("normalized, row 0", norm0, 6);
    LayerNorm ln = layernorm_new(&a, 6);
    Matrix out = layernorm_forward(&a, &ln, x);
    float out_mean[2], out_var[2];
    for (size_t i = 0; i < 2; i++) {
        const float *row = matrix_row(out, i);
        float sum = 0;
        for (size_t j = 0; j < 6; j++) sum += row[j];
        out_mean[i] = sum / 6;
        float sq = 0;
        for (size_t j = 0; j < 6; j++) sq += (row[j] - out_mean[i]) * (row[j] - out_mean[i]);
        out_var[i] = sq / 6;
    }
    show4("LayerNorm output mean", out_mean, 2);
    show4("LayerNorm output variance (biased)", out_var, 2);
    arena_free(&a);
}

/* 4.3: a smooth ReLU, and the expand-then-shrink feed forward block. */
static void demo_gelu_and_feed_forward(void) {
    printf("== 4.3 GELU and the feed forward network ==\n");
    float x[7], g[7], r[7];
    for (int i = 0; i < 7; i++) {
        x[i] = (float)(i - 3);
        g[i] = gelu(x[i]);
        r[i] = x[i] > 0 ? x[i] : 0;
    }
    show4("x   ", x, 7);
    show4("GELU", g, 7);
    show4("ReLU", r, 7);
    Arena a = arena_new(8 << 20);
    Rng rng = rng_new(123);
    FeedForward ffn = feedforward_new(&a, 32);
    ffn.expand.w = matrix_randn(&a, 32, 128, 0.02f, &rng);
    ffn.shrink.w = matrix_randn(&a, 128, 32, 0.02f, &rng);
    Matrix out = feedforward_forward(&a, &ffn, matrix_rand(&a, 3, 32, &rng));
    printf("feed forward: input (2, 3, 32) -> output (2, %zu, %zu) (hidden layer is 4 x 32 = 128 wide)\n", out.rows, out.cols);
    arena_free(&a);
}

/* 4.4: shortcut connections keep the gradient alive. */
static void demo_shortcut(void) {
    printf("== 4.4 shortcut connections (mean |gradient| of each layer's weights) ==\n");
    const double x[3] = {1, 0, -1};
    for (int use_shortcut = 0; use_shortcut <= 1; use_shortcut++) {
        ExampleDeepNeuralNetwork net = {SHORTCUT_LAYERS, 5, use_shortcut};
        double grads[5];
        example_gradient_means(&net, x, 0, grads);
        printf("%s shortcut: ", use_shortcut ? "with   " : "without");
        for (int i = 0; i < 5; i++) printf(i ? "  %.6f" : "%.6f", grads[i]);
        printf("   (layer 0 ... layer 4)\n");
    }
}

/* 4.5: attention + feed forward, each wrapped in norm and shortcut. Shape in = shape out. */
static void demo_transformer_block(void) {
    printf("== 4.5 the transformer block ==\n");
    Arena a;
    GptModel m = build(&a, GPT_CONFIG_TINY);
    Matrix x = gpt_embed(&a, &m, BATCH[0], 4);
    Matrix out = transformer_block_forward(&a, &m.blocks[0], x);
    printf("input shape: (2, %zu, %zu)  output shape: (2, %zu, %zu)\n", x.rows, x.cols, out.rows, out.cols);
    show4("input[0][0], first 4 numbers", matrix_row(x, 0), 4);
    show4("output[0][0], first 4 numbers", matrix_row(out, 0), 4);
    arena_free(&a);
}

/* 4.6: how big is the model? */
static void demo_parameters(void) {
    printf("== 4.6 model size ==\n");
    Arena a;
    GptModel tiny = build(&a, GPT_CONFIG_TINY);
    char b1[32], b2[32], b3[32];
    printf("tiny model: %s parameters (arithmetic says %s)\n", commas(gpt_count_parameters(&tiny), b1),
           commas(gpt_parameter_count(&GPT_CONFIG_TINY), b2));
    arena_free(&a);
    GptConfig cfg = GPT_CONFIG_124M;
    size_t total = gpt_parameter_count(&cfg), head = cfg.vocab_size * cfg.emb_dim;
    printf("GPT-2 small, no model built: %s parameters\n", commas(total, b1));
    printf("  minus the output head (weight tying in the original GPT-2): %s\n", commas(total - head, b1));
    printf("  float32 size: %.2f MB\n", (double)total * 4 / (1024.0 * 1024.0));
    size_t d = cfg.emb_dim, ff = d * 4 * d + 4 * d + 4 * d * d + d, att = 4 * d * d + d;
    printf("exercise 4.1: feed forward %s vs attention %s parameters per block\n", commas(ff, b1), commas(att, b2));
    printf("exercise 4.2: bigger models, same code, different configuration\n");
    printf("  %-20s %14s %14s %11s\n", "", "parameters", "weight-tied", "float32 MB");
    for (int i = 0; i < GPT_N_SIZES; i++) {
        GptConfig c = cfg;
        c.emb_dim = GPT_SIZES[i].emb_dim;
        c.n_layers = GPT_SIZES[i].n_layers;
        c.n_heads = GPT_SIZES[i].n_heads;
        size_t n = gpt_parameter_count(&c);
        char mb[48];
        printf("  %-20s %14s %14s %11s\n", GPT_SIZES[i].name, commas(n, b1), commas(n - c.vocab_size * c.emb_dim, b2),
               commas_f((double)n * 4 / (1024.0 * 1024.0), mb));
        (void)b3;
    }
}

/* 4.7: predict one token, append it, repeat. */
static void demo_generate(GptConfig cfg, const char *label) {
    printf("== 4.7 generating text (%s) ==\n", label);
    Arena a;
    GptModel m = build(&a, cfg);
    Bpe bpe;
    char err[ERR_LEN];
    if (bpe_load(&bpe, RANKS_PATH, err) != 0) {
        fprintf(stderr, "%s\n", err);
        exit(1);
    }
    size_t n;
    int *encoded = bpe_encode(&bpe, "Hello, I am", &n);
    printf("encoded: ");
    print_ids(encoded, n);
    printf("\n");
    int *out = generate_text_simple(&a, &m, encoded, n, 6, cfg.context_length);
    printf("output IDs: ");
    print_ids(out, n + 6);
    printf("\noutput length: %zu\n", n + 6);
    char *decoded = bpe_decode(&bpe, out, n + 6);
    printf("decoded: %s\n", decoded);
    printf("(gibberish: the weights are random, nothing has been trained yet)\n");
    free(decoded);
    free(out);
    free(encoded);
    bpe_free(&bpe);
    arena_free(&a);
}

void run_chapter4(void) {
    demo_big_picture();
    demo_layer_norm();
    demo_gelu_and_feed_forward();
    demo_shortcut();
    demo_transformer_block();
    demo_parameters();
    demo_generate(GPT_CONFIG_TINY, "tiny model");
}

void run_chapter4_124m(void) {
    Arena a;
    GptModel m = build(&a, GPT_CONFIG_124M);
    char buf[32];
    printf("== GPT-2 small (124M) ==\n");
    printf("parameters built: %s\n", commas(gpt_count_parameters(&m), buf));
    Matrix logits = gpt_forward(&a, &m, BATCH[0], 4);
    printf("logits shape: (2, %zu, %zu)\n", logits.rows, logits.cols);
    show4("logits[0][0], first 4 of 50257", matrix_row(logits, 0), 4);
    arena_free(&a);
    demo_generate(GPT_CONFIG_124M, "124M model");
}
