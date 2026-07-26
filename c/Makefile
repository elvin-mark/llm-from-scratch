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

all: run run_cu runq run_bitnet train train_cu

run: src/run.c
	$(CC) $(CFLAGS) -o run src/run.c $(LDFLAGS)

runq: src/runq.c
	$(CC) $(CFLAGS) -o runq src/runq.c $(LDFLAGS)

run_bitnet: src/run_bitnet.c
	$(CC) $(CFLAGS) -o run_bitnet src/run_bitnet.c $(LDFLAGS)

run_metal: metal/run_metal.m
	clang -O3 -framework Metal -framework Foundation -o run_metal metal/run_metal.m

run_metal_bitnet: metal/run_bitnet_metal.m
	clang -O3 -framework Metal -framework Foundation -o run_metal_bitnet metal/run_bitnet_metal.m

run_cu: cuda/run.cu
	nvcc -O3 --use_fast_math -o run_cu cuda/run.cu -lcublas

run_bitnet_cu: cuda/run_bitnet.cu
	nvcc -O3 --use_fast_math -o run_bitnet_cu cuda/run_bitnet.cu

train: src/train.c
	$(CC) $(CFLAGS) -o train src/train.c $(LDFLAGS)

train_metal: metal/train_metal.m
	clang -O3 -framework Metal -framework Foundation -o train_metal metal/train_metal.m

train_cu: cuda/train.cu
	nvcc -O3 --use_fast_math -o train_cu cuda/train.cu -lcublas

# --- Emscripten (WebAssembly / Web Edge) Targets ---
# Build standalone WebAssembly module & HTML harness:
wasm: src/run.c
	$(EMCC) $(EMCFLAGS) src/run.c -o run.html

# Build quantized INT8 WebAssembly module & HTML harness:
wasmq: src/runq.c
	$(EMCC) $(EMCFLAGS) src/runq.c -o runq.html

# Build WebAssembly HTML app preloading model.bin and vocab.bin into VFS:
wasm_bundle: src/run.c
	$(EMCC) $(EMCFLAGS) --preload-file model.bin --preload-file vocab.bin src/run.c -o run_bundle.html

# Build WebAssembly HTML app preloading model_q8.bin and vocab.bin into VFS:
wasmq_bundle: src/runq.c
	$(EMCC) $(EMCFLAGS) --preload-file model_q8.bin --preload-file vocab.bin src/runq.c -o runq_bundle.html

clean:
	rm -f run run_cu runq train train_cu run_metal run_metal_bitnet train_metal run_bitnet run_bitnet_cu *.html *.js *.wasm *.data model.bin model_q8.bin vocab.bin
