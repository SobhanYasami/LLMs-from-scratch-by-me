/* Step 4 (book 2.6) - sliding-window data sampling for next-token prediction.
 *
 *   ids      = [10 11 12 13 14 15]
 *   window 0: input [10 11 12 13]  target [11 12 13 14]
 *
 * Why shifted by one? The model's job is: given the tokens so far, guess the
 * next one. Every position in a window is a free training example.
 * `stride` is how far the window moves; stride == max_length means no overlap.
 */
#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>

typedef struct {
    const int *ids; /* borrowed: the caller keeps the token IDs alive */
    size_t n_ids;
    size_t max_length;
    size_t stride;
    size_t count; /* number of windows */
} GPTDataset;

GPTDataset gpt_dataset_new(const int *ids, size_t n_ids, size_t max_length, size_t stride);
/* Window `idx` as two pointers into the ID array (each max_length long). */
void gpt_dataset_get(const GPTDataset *d, size_t idx, const int **input, const int **target);

#endif
