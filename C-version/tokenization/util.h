/* Small shared helpers: checked allocation and the error-message size.
 *
 * Tip: C has no exceptions or Result type. Functions that can fail return NULL
 * and write a message into a caller-supplied `char err[ERR_LEN]`.
 */
#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

#define ERR_LEN 128

void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t size);
void *xrealloc(void *p, size_t n);
char *xstrndup(const char *s, size_t n);
char *xstrdup(const char *s);

#endif
