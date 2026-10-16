/* Tiny seeded random generator (splitmix64), no library needed.
 *
 * Used for shuffling batches and for random embedding weights. Same seed,
 * same numbers, on every run. (Same algorithm as rust-version/rng.rs.)
 */
#ifndef RNG_H
#define RNG_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t state;
} Rng;

Rng rng_new(uint64_t seed);
double rng_f64(Rng *r);          /* uniform in [0, 1) */
size_t rng_below(Rng *r, size_t n); /* uniform integer in [0, n) */
float rng_normal(Rng *r);        /* Normal(0, 1), Box-Muller */
void rng_shuffle(Rng *r, size_t *items, size_t n); /* Fisher-Yates */

#endif
