# Build and run the C demo. Run `make run` from C-version/.
CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -D_DEFAULT_SOURCE
SRC      = main.c $(wildcard tokenization/*.c)

build/main: $(SRC) $(wildcard tokenization/*.h)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

run: build/main
	./build/main

clean:
	rm -rf build

.PHONY: run clean
