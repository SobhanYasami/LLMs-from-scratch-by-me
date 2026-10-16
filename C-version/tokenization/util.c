#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *check(void *p) {
    if (!p) {
        fputs("out of memory\n", stderr);
        exit(1);
    }
    return p;
}

void *xmalloc(size_t n) { return check(malloc(n ? n : 1)); }
void *xcalloc(size_t n, size_t size) { return check(calloc(n ? n : 1, size ? size : 1)); }
void *xrealloc(void *p, size_t n) { return check(realloc(p, n ? n : 1)); }

char *xstrndup(const char *s, size_t n) {
    char *out = xmalloc(n + 1);
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

char *xstrdup(const char *s) { return xstrndup(s, strlen(s)); }
