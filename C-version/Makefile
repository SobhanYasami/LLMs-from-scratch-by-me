# Build and run the C demo. Run `make run` from C-version/.
CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -D_DEFAULT_SOURCE
SRC      = main.c ch03.c ch04.c $(wildcard tokenization/*.c) $(wildcard attention/*.c) $(wildcard gpt/*.c)

build/main: $(SRC) $(wildcard tokenization/*.h) $(wildcard attention/*.h) $(wildcard gpt/*.h) ch03.h ch04.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

run: build/main
	./build/main $(ARGS)

clean:
	rm -rf build

.PHONY: run clean
