CC = gcc
CFLAGS = -Wall -Wextra -O3 -ffast-math

all: run

run: run.c
	$(CC) $(CFLAGS) -o run run.c -lm

clean:
	rm -f run model.bin vocab.bin
