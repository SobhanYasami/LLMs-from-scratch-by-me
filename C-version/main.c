/* Demo runner for the educational tokenizers and data sampling.
 *
 * Run from C-version/:
 *
 *     make run              # all chapters
 *     make run ARGS=ch03    # one chapter
 *
 * Tip: read main() first, then jump into tokenization/ following each demo.
 *
 * Pipeline mirrors python-version/main.py:
 *     raw text -> split_text -> build_vocab -> SimpleTokenizer V1/V2 -> BPE (GPT-2)
 *     -> sliding-window dataset -> data loader -> embeddings
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ch03.h"
#include "tokenization/bpe.h"
#include "tokenization/dataloader.h"
#include "tokenization/dataset.h"
#include "tokenization/embedding.h"
#include "tokenization/preprocess.h"
#include "tokenization/simple.h"
#include "tokenization/vocab.h"

/* Run from C-version/, like go-version resolves ../assets. */
#define DATA_PATH "../assets/the-verdict.txt"
#define RANKS_PATH "../assets/gpt2-ranks.txt"

#define V1_TEXT "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
#define V2_TEXT "Hello, do you like tea? " END_OF_TEXT " In the sunlit terraces of the palace."

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static char *load_text(void) {
    FILE *f = fopen(DATA_PATH, "rb");
    if (!f) die("cannot open " DATA_PATH " (run from C-version/)");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *text = xmalloc((size_t)size + 1);
    size_t got = fread(text, 1, (size_t)size, f);
    text[got] = '\0';
    fclose(f);
    return text;
}

static void print_ids_inline(const int *ids, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) printf(i ? ", %d" : "%d", ids[i]);
    printf("]");
}

static void print_ids(const int *ids, size_t n) {
    print_ids_inline(ids, n);
    printf("\n");
}

static Vocab load_vocab(const char *text) {
    Tokens tokens = split_text(text);
    Vocab v = build_vocab(&tokens);
    tokens_free(&tokens);
    return v;
}

/* Print the last `count` vocabulary entries (the highest IDs). */
static void show_vocab_tail(const Vocab *v, size_t count) {
    for (size_t id = v->size - count; id < v->size; id++) printf("('%s', %zu)\n", v->tokens[id], id);
}

/* encode, print IDs, decode, print text. */
static void roundtrip(const SimpleTokenizer *t, const char *text) {
    char err[ERR_LEN];
    size_t n;
    int *ids = simple_encode(t, text, &n, err);
    if (!ids) die(err);
    print_ids(ids, n);
    char *decoded = simple_decode(t, ids, n, err);
    if (!decoded) die(err);
    printf("%s\n", decoded);
    free(decoded);
    free(ids);
}

/* V1 works only on words it has seen in the corpus. */
static void demo_v1(const Vocab *vocab) {
    SimpleTokenizer t = simple_tokenizer_v1(vocab);
    printf("version 1 tokenizer :\n");
    roundtrip(&t, V1_TEXT);
    printf("-------------------------\n");
}

/* V2 maps unknown words to <|unk|> and handles the <|endoftext|> separator. */
static void demo_v2(const Vocab *vocab) {
    SimpleTokenizer t = simple_tokenizer_v2(vocab);
    printf("version 2 tokenizer :\n");
    roundtrip(&t, V2_TEXT);
    printf("---------------------\n");
}

/* Byte pair encoding: sub-word pieces, so no <|unk|> is ever needed. */
static void demo_bpe(const Bpe *bpe) {
    printf("bpe tokenizer :\n");
    size_t n;
    int *ids = bpe_encode(bpe, V2_TEXT, &n);
    print_ids(ids, n);
    char *decoded = bpe_decode(bpe, ids, n);
    printf("%s\n", decoded);
    free(decoded);
    free(ids);
    printf("---------------------\n");
}

/* Next-word prediction pairs: the target is the input shifted by one. */
static void demo_sampling(const Bpe *bpe, const char *text) {
    size_t n;
    int *all = bpe_encode(bpe, text, &n);
    /* Drop the first 50 tokens: gives a slightly more interesting passage. */
    const int *sample = all + 50;
    const size_t context_size = 4;

    printf("data sampling :\n");
    printf("x: ");
    print_ids(sample, context_size);
    printf("y:      ");
    print_ids(sample + 1, context_size);
    for (size_t i = 1; i <= context_size; i++) {
        char *context = bpe_decode(bpe, sample, i);
        char *desired = bpe_decode(bpe, sample + i, 1);
        printf("%s ----> %s\n", context, desired);
        free(context);
        free(desired);
    }

    /* The same idea, packaged: every window of the whole text. */
    GPTDataset ds = gpt_dataset_new(all, n, 4, 4);
    const int *in, *tg;
    gpt_dataset_get(&ds, 0, &in, &tg);
    printf("windows: %zu first pair: ", ds.count);
    print_ids_inline(in, 4);
    printf(" ");
    print_ids(tg, 4);
    /* the same windows, batched */
    DataLoader first = dataloader_new(&ds, 8, false, true, 123);
    dataloader_start_epoch(&first);
    int bi[8 * 4], bt[8 * 4];
    dataloader_batch(&first, 0, bi, bt);
    printf("first batch inputs:\n");
    for (size_t r = 0; r < 8; r++) {
        printf("  ");
        print_ids(bi + r * 4, 4);
    }
    printf("first batch targets:\n");
    for (size_t r = 0; r < 8; r++) {
        printf("  ");
        print_ids(bt + r * 4, 4);
    }
    dataloader_free(&first);
    printf("---------------------\n");
    free(all);
}

/* Batches of windows. stride < max_length overlaps windows. */
static void demo_dataloader(const Bpe *bpe, const char *text) {
    size_t n;
    int *ids = bpe_encode(bpe, text, &n);
    int in[8 * 4], tg[8 * 4];
    printf("data loader :\n");

    GPTDataset overlap = gpt_dataset_new(ids, n, 4, 1);
    DataLoader dl = dataloader_new(&overlap, 1, false, true, 123);
    dataloader_start_epoch(&dl);
    for (size_t b = 0; b < 2; b++) { /* first window, then second: shifted by one */
        dataloader_batch(&dl, b, in, tg);
        print_ids_inline(in, 4);
        printf(" ");
        print_ids(tg, 4);
    }
    dataloader_free(&dl);

    GPTDataset ds = gpt_dataset_new(ids, n, 4, 4);
    dl = dataloader_new(&ds, 8, false, true, 123);
    dataloader_start_epoch(&dl);
    size_t rows = dataloader_batch(&dl, 0, in, tg);
    printf("inputs shape: (%zu, 4)\n", rows);
    dataloader_free(&dl);
    printf("---------------------\n");
    free(ids);
}

/* Token IDs -> vectors, then add position vectors. */
static void demo_embeddings(const Bpe *bpe, const char *text) {
    enum { MAX_LENGTH = 4, OUTPUT_DIM = 256, BATCH = 8 };
    size_t n;
    int *ids = bpe_encode(bpe, text, &n);
    GPTDataset ds = gpt_dataset_new(ids, n, MAX_LENGTH, MAX_LENGTH);
    DataLoader dl = dataloader_new(&ds, BATCH, false, true, 123);
    dataloader_start_epoch(&dl);
    int inputs[BATCH * MAX_LENGTH], targets[BATCH * MAX_LENGTH];
    dataloader_batch(&dl, 0, inputs, targets);

    InputEmbedding emb = input_embedding_new(BPE_ENDOFTEXT_ID + 1, MAX_LENGTH, OUTPUT_DIM, 123);
    float *token_emb = xmalloc(BATCH * MAX_LENGTH * OUTPUT_DIM * sizeof(float));
    float *input_emb = xmalloc(BATCH * MAX_LENGTH * OUTPUT_DIM * sizeof(float));
    embedding_forward(&emb.token, inputs, BATCH * MAX_LENGTH, token_emb); /* token vectors only */
    input_embedding_forward(&emb, inputs, BATCH, MAX_LENGTH, input_emb);   /* token + position */

    /* Shapes are the thing to notice: [batch, tokens] grows a new axis of size D. */
    printf("embeddings :\n");
    printf("token ids shape:      (%d, %d)\n", BATCH, MAX_LENGTH);
    printf("token embeddings:     (%d, %d, %d)\n", BATCH, MAX_LENGTH, OUTPUT_DIM);
    printf("position embeddings:  (%zu, %zu)  <- one vector per position, shared by all rows\n", emb.pos.rows,
           emb.pos.dim);
    printf("input embeddings:     (%d, %d, %d)  <- token + position\n", BATCH, MAX_LENGTH, OUTPUT_DIM);
    printf("---------------------\n");

    free(token_emb);
    free(input_emb);
    input_embedding_free(&emb);
    dataloader_free(&dl);
    free(ids);
}

static void run_chapter2(void) {
    char *text = load_text();
    Vocab vocab = load_vocab(text);
    Bpe bpe;
    char err[ERR_LEN];
    if (bpe_load(&bpe, RANKS_PATH, err) != 0) die(err);

    show_vocab_tail(&vocab, 5);
    demo_v1(&vocab);
    demo_v2(&vocab);
    demo_bpe(&bpe);
    demo_sampling(&bpe, text);
    demo_dataloader(&bpe, text);
    demo_embeddings(&bpe, text);

    bpe_free(&bpe);
    vocab_free(&vocab);
    free(text);
}

/* `./build/main ch03` runs one chapter's demos; with no argument, all of them. */
int main(int argc, char **argv) {
    if (argc == 1) {
        run_chapter2();
        run_chapter3();
        return 0;
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "ch02") == 0) run_chapter2();
        else if (strcmp(argv[i], "ch03") == 0) run_chapter3();
        else die("unknown chapter; use ch02 or ch03");
    }
    return 0;
}
