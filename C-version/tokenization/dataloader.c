#include "dataloader.h"

#include <stdlib.h>
#include <string.h>

#include "util.h"

DataLoader dataloader_new(const GPTDataset *ds, size_t batch_size, bool shuffle, bool drop_last, uint64_t seed) {
    DataLoader dl = {ds, batch_size, shuffle, drop_last, rng_new(seed), NULL};
    dl.order = xmalloc(ds->count * sizeof(size_t));
    for (size_t i = 0; i < ds->count; i++) dl.order[i] = i;
    return dl;
}

void dataloader_free(DataLoader *dl) {
    free(dl->order);
    dl->order = NULL;
}

size_t dataloader_start_epoch(DataLoader *dl) {
    if (dl->shuffle) rng_shuffle(&dl->rng, dl->order, dl->dataset->count);
    size_t n = dl->dataset->count;
    return dl->drop_last ? n / dl->batch_size : (n + dl->batch_size - 1) / dl->batch_size;
}

size_t dataloader_batch(const DataLoader *dl, size_t b, int *inputs, int *targets) {
    size_t first = b * dl->batch_size, rows = 0, len = dl->dataset->max_length;
    for (size_t i = first; i < first + dl->batch_size && i < dl->dataset->count; i++, rows++) {
        const int *in, *tg;
        gpt_dataset_get(dl->dataset, dl->order[i], &in, &tg);
        memcpy(inputs + rows * len, in, len * sizeof(int));
        memcpy(targets + rows * len, tg, len * sizeof(int));
    }
    return rows;
}
