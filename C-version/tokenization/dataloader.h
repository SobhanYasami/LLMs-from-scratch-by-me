/* Step 4b (book 2.6) - batching. Mirrors torch's DataLoader.
 *
 * Why batch? The GPU processes many windows at once, which is much faster.
 * Why shuffle? Windows in book order make training swing; random order smooths it.
 * Why drop the last short batch? A different size can upset training.
 */
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdbool.h>
#include <stddef.h>

#include "dataset.h"
#include "rng.h"

typedef struct {
    const GPTDataset *dataset;
    size_t batch_size;
    bool shuffle;
    bool drop_last;
    Rng rng;
    size_t *order; /* window indices in the order this epoch serves them */
} DataLoader;

DataLoader dataloader_new(const GPTDataset *ds, size_t batch_size, bool shuffle, bool drop_last, uint64_t seed);
void dataloader_free(DataLoader *dl);

/* Start an epoch (reshuffles if shuffle is on). Returns the number of batches. */
size_t dataloader_start_epoch(DataLoader *dl);
/* Copy batch `b` into flat arrays of batch_size * max_length IDs each
 * (row r is inputs[r*max_length .. (r+1)*max_length)). Returns rows filled. */
size_t dataloader_batch(const DataLoader *dl, size_t b, int *inputs, int *targets);

#endif
