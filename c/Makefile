CC = gcc
CFLAGS = -Wall -Wextra -O3 -ffast-math
LDFLAGS = -lm

# To use BLAS, run: make run USE_BLAS=1
ifeq ($(USE_BLAS), 1)
	CFLAGS += -DUSE_BLAS
	LDFLAGS += -lopenblas
endif

all: run run_cu runq

run: run.c
	$(CC) $(CFLAGS) -o run run.c $(LDFLAGS)

runq: runq.c
	$(CC) $(CFLAGS) -o runq runq.c $(LDFLAGS)

run_cu: run.cu
	nvcc -O3 --use_fast_math -o run_cu run.cu -lcublas

clean:
	rm -f run run_cu runq model.bin model_q8.bin vocab.bin
