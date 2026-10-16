#include "rng.h"

#include <math.h>

Rng rng_new(uint64_t seed) { return (Rng){seed}; }

static uint64_t next_u64(Rng *r) {
    r->state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = r->state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double rng_f64(Rng *r) { return (double)(next_u64(r) >> 11) / 9007199254740992.0; /* 2^53 */ }

size_t rng_below(Rng *r, size_t n) { return (size_t)(next_u64(r) % n); }

float rng_normal(Rng *r) {
    double u1 = 1.0 - rng_f64(r); /* avoid log(0) */
    double u2 = rng_f64(r);
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

void rng_shuffle(Rng *r, size_t *items, size_t n) {
    for (size_t i = n; i-- > 1;) {
        size_t j = rng_below(r, i + 1);
        size_t tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }
}
