CC = gcc
CFLAGS = -Wall -Wextra -O3 -ffast-math

# To use BLAS, run: make run USE_BLAS=1
ifeq ($(USE_BLAS), 1)
	CFLAGS += -DUSE_BLAS -lopenblas
endif

all: run run_cu

run: run.c
	$(CC) $(CFLAGS) -o run run.c -lm

run_cu: run.cu
	nvcc -O3 --use_fast_math -o run_cu run.cu -lcublas

clean:
	rm -f run run_cu model.bin vocab.bin
