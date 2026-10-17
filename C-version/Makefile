# Build and run the C demo. Run `make run` from C-version/.
CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -D_DEFAULT_SOURCE
SRC      = main.c ch03.c ch04.c ch05.c $(wildcard tokenization/*.c) $(wildcard attention/*.c) $(wildcard gpt/*.c) $(wildcard train/*.c)

build/main: $(SRC) $(wildcard tokenization/*.h) $(wildcard attention/*.h) $(wildcard gpt/*.h) $(wildcard train/*.h) ch03.h ch04.h ch05.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

run: build/main
	./build/main $(ARGS)

clean:
	rm -rf build

.PHONY: run clean
