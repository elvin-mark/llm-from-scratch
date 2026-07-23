CC = gcc
CFLAGS = -Wall -Wextra -O3
LDFLAGS = -lm

EMCC = emcc
EMCFLAGS = -O3 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1

# To use BLAS, run: make run USE_BLAS=1
ifeq ($(USE_BLAS), 1)
	CFLAGS += -DUSE_BLAS
	LDFLAGS += -lopenblas
endif

all: run run_cu runq train train_cu

run: run.c
	$(CC) $(CFLAGS) -o run run.c $(LDFLAGS)

runq: runq.c
	$(CC) $(CFLAGS) -o runq runq.c $(LDFLAGS)

run_cu: run.cu
	nvcc -O3 --use_fast_math -o run_cu run.cu -lcublas

train: train.c
	$(CC) $(CFLAGS) -o train train.c $(LDFLAGS)

train_cu: train.cu
	nvcc -O3 --use_fast_math -o train_cu train.cu -lcublas

# --- Emscripten (WebAssembly / Web Edge) Targets ---
# Build standalone WebAssembly module & HTML harness:
wasm: run.c
	$(EMCC) $(EMCFLAGS) run.c -o run.html

# Build quantized INT8 WebAssembly module & HTML harness:
wasmq: runq.c
	$(EMCC) $(EMCFLAGS) runq.c -o runq.html

# Build WebAssembly HTML app preloading model.bin and vocab.bin into VFS:
wasm_bundle: run.c
	$(EMCC) $(EMCFLAGS) --preload-file model.bin --preload-file vocab.bin run.c -o run_bundle.html

# Build WebAssembly HTML app preloading model_q8.bin and vocab.bin into VFS:
wasmq_bundle: runq.c
	$(EMCC) $(EMCFLAGS) --preload-file model_q8.bin --preload-file vocab.bin runq.c -o runq_bundle.html

clean:
	rm -f run run_cu runq train train_cu *.html *.js *.wasm *.data model.bin model_q8.bin vocab.bin
