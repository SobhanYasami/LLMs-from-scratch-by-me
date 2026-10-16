#include "dataset.h"

GPTDataset gpt_dataset_new(const int *ids, size_t n_ids, size_t max_length, size_t stride) {
    GPTDataset d = {ids, n_ids, max_length, stride, 0};
    /* Windows start at 0, stride, 2*stride, ... and stop before n_ids - max_length:
     * the target reaches one token past the input. */
    if (n_ids > max_length) d.count = (n_ids - max_length + stride - 1) / stride;
    return d;
}

void gpt_dataset_get(const GPTDataset *d, size_t idx, const int **input, const int **target) {
    *input = d->ids + idx * d->stride;
    *target = *input + 1; /* the same window, shifted by one */
}
