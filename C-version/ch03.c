/* Chapter 3 demos: attention, one idea at a time (book 3.3 to 3.6).
 *
 * Run from C-version/:  make run ARGS=ch03   (or ./build/main ch03)
 *
 * Tip: read the demos top to bottom; each builds on the one before. The numbers
 * in 3.3 and 3.4 match the book exactly (3.4 uses the book's seed-123 weights,
 * hard-coded so every language shares them).
 */
#include "ch03.h"

#include <math.h>
#include <stdio.h>

#include "attention/causal.h"
#include "attention/matrix.h"
#include "attention/multihead.h"
#include "attention/self_attention.h"
#include "attention/simple.h"

/* "Your journey starts with one step." as 3-dimensional embeddings (book 3.3.1). */
static const float INPUTS[6][3] = {
    {0.43f, 0.15f, 0.89f}, /* Your */
    {0.55f, 0.87f, 0.66f}, /* journey */
    {0.57f, 0.85f, 0.64f}, /* starts */
    {0.22f, 0.58f, 0.33f}, /* with */
    {0.77f, 0.25f, 0.10f}, /* one */
    {0.05f, 0.80f, 0.55f}, /* step */
};

/* The book's torch.manual_seed(123) weights (d_in=3, d_out=2), written out so no language needs torch. */
static const float W_QUERY[3][2] = {{0.29611194f, 0.51656228f}, {0.25167072f, 0.68855679f}, {0.07397246f, 0.86652195f}};
static const float W_KEY[3][2] = {{0.13657987f, 0.10247904f}, {0.18405646f, 0.72644675f}, {0.31525391f, 0.68710667f}};
static const float W_VALUE[3][2] = {{0.07563531f, 0.19663817f}, {0.31641197f, 0.40174013f}, {0.11856830f, 0.82739538f}};

static void print_row(const float *values, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) printf(i ? ", %.4f" : "%.4f", values[i]);
    printf("]");
}

static void show_vec(const char *label, const float *values, size_t n) {
    printf("%s: ", label);
    print_row(values, n);
    printf("\n");
}

static void show_matrix(const char *label, Matrix m) {
    printf("%s:\n", label);
    for (size_t i = 0; i < m.rows; i++) {
        printf("  ");
        print_row(matrix_row(m, i), m.cols);
        printf("\n");
    }
}

/* 3.3: one query by hand, then all tokens at once. */
static void demo_simple_attention(void) {
    Arena a = arena_new(1 << 20);
    Matrix inputs = matrix_from(&a, 6, 3, &INPUTS[0][0]);
    printf("== 3.3.1 simple attention for one query (x2, 'journey') ==\n");
    Matrix query = matrix_from(&a, 1, 3, INPUTS[1]);
    Matrix scores_2 = matmul(&a, query, transpose(&a, inputs)); /* dot product of every token with the query */
    Matrix weights_2 = softmax_rows(&a, scores_2);
    Matrix context_2 = matmul(&a, weights_2, inputs);
    show_vec("scores    ", scores_2.data, 6);
    show_vec("weights   ", weights_2.data, 6);
    show_vec("context z2", context_2.data, 3);

    printf("== 3.3.2 all tokens at once ==\n");
    SimpleAttentionResult r = simple_attention(&a, inputs);
    show_matrix("scores (T x T)", r.scores);
    show_matrix("weights (rows sum to 1)", r.weights);
    float sums[6] = {0};
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 6; j++) sums[i] += matrix_at(r.weights, i, j);
    show_vec("row sums", sums, 6);
    show_matrix("context vectors (T x d)", r.context);
    arena_free(&a);
}

/* 3.4: query/key/value with trainable (here: fixed) weights. */
static void demo_self_attention(void) {
    Arena a = arena_new(1 << 20);
    Matrix inputs = matrix_from(&a, 6, 3, &INPUTS[0][0]);
    printf("== 3.4 self-attention with trainable weights (d_in=3, d_out=2) ==\n");
    SelfAttention sa = {matrix_from(&a, 3, 2, &W_QUERY[0][0]), matrix_from(&a, 3, 2, &W_KEY[0][0]),
                        matrix_from(&a, 3, 2, &W_VALUE[0][0])};
    Matrix x2 = matrix_from(&a, 1, 3, INPUTS[1]);
    Matrix query_2 = matmul(&a, x2, sa.w_query);
    Matrix keys = matmul(&a, inputs, sa.w_key), values = matmul(&a, inputs, sa.w_value);
    show_vec("query_2", query_2.data, 2);
    printf("keys shape: (%zu, %zu), values shape: (%zu, %zu)\n", keys.rows, keys.cols, values.rows, values.cols);
    Matrix scores_2 = matmul(&a, query_2, transpose(&a, keys));
    show_vec("scores_2 (q2 . keys)", scores_2.data, 6);
    Matrix weights_2 = softmax_rows(&a, scale_by_sqrt(&a, scores_2, keys.cols));
    show_vec("weights_2 (scaled)", weights_2.data, 6);
    show_vec("context_2", matmul(&a, weights_2, values).data, 2);
    show_matrix("all context vectors (T x d_out)", self_attention_forward(&a, &sa, inputs));
    arena_free(&a);
}

/* 3.5: mask the future, then dropout. */
static void demo_causal_attention(void) {
    Arena a = arena_new(1 << 20);
    Matrix inputs = matrix_from(&a, 6, 3, &INPUTS[0][0]);
    printf("== 3.5 causal attention ==\n");
    CausalAttention ca = {{matrix_from(&a, 3, 2, &W_QUERY[0][0]), matrix_from(&a, 3, 2, &W_KEY[0][0]),
                           matrix_from(&a, 3, 2, &W_VALUE[0][0])},
                          0.0f};
    show_matrix("causal weights (upper triangle is 0)", causal_weights(&a, &ca, inputs, NULL));
    show_matrix("causal context vectors", causal_forward(&a, &ca, inputs, NULL));
    Matrix ones = matrix_new(&a, 6, 6);
    for (size_t i = 0; i < 36; i++) ones.data[i] = 1.0f;
    Rng rng = rng_new(123);
    show_matrix("dropout 0.5 on ones(6, 6), seed 123 (kept values are doubled)", dropout(&a, ones, 0.5f, &rng));
    arena_free(&a);
}

/* 3.6: split the work across heads. */
static void demo_multihead(void) {
    Arena a = arena_new(1 << 20);
    Matrix inputs = matrix_from(&a, 6, 3, &INPUTS[0][0]);
    printf("== 3.6 multi-head attention (d_in=3, d_out=2, heads=2) ==\n");
    Rng rng = rng_new(123);
    MultiHeadAttention mha = multihead_random(&a, 3, 2, 2, &rng);
    /* A batch is just the same computation repeated per example; two copies = a batch of 2. */
    Matrix out = multihead_forward(&a, &mha, inputs);
    printf("input shape: (2, %zu, %zu)  output shape: (2, %zu, %zu)\n", inputs.rows, inputs.cols, out.rows, out.cols);
    show_matrix("output[0]", out);
    arena_free(&a);
}

void run_chapter3(void) {
    demo_simple_attention();
    demo_self_attention();
    demo_causal_attention();
    demo_multihead();
}
