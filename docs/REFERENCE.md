# FastAIModel API Reference Manual

`FastAIModel` provides native local LLM (GGUF) and embedding (ONNX) inference for Java with Vulkan & Apple Metal GPU acceleration.

---

## Class: `fastaimodel.FastAIModel`

Implements `AutoCloseable` for in-process GGUF LLM execution via JNI bindings to `llama.cpp`.

### Constructors

- `public FastAIModel(String modelPath)`  
  Loads a local GGUF model or Ollama model name with default context size (4096) and CPU execution.

- `public FastAIModel(String modelPath, int ctxSize, int gpuLayers)`  
  Loads a local GGUF model with specified context window (`ctxSize`) and GPU layer offloading (`gpuLayers`).

### Methods

- `public void predict(String prompt, int maxTokens, TokenCallback cb)`  
  Executes autoregressive LLM inference and streams generated text tokens to the `TokenCallback` interface in real time.

- `public void predictFromMemoryAddress(long memoryAddress, int maxTokens, TokenCallback cb)`  
  Executes zero-copy autoregressive LLM inference reading prompt UTF-8 bytes directly from a **[FastSharedMemory](https://github.com/andrestubbe/FastSharedMemory)** native memory address, bypassing Java String allocations and reducing IPC transfer latency to < 800 nanoseconds.

- `public void close()`  
  Frees native `llama.cpp` model context, KV cache buffers, and GPU Vulkan/Metal resources.

---

## Class: `fastaimodel.FastAIModelOnnx`

In-process ONNX Runtime embedding engine for vector embeddings.

### Methods

- `public float[] embed(String text)`  
  Generates high-dimensional vector embeddings for text inputs.

---

## Class: `fastaimodel.streaming.FastAIStreamingModel`

High-throughput AIR-style chunked layer and expert streaming engine implementing `AutoCloseable`. Executes large language models (7B, 14B, 70B, MoE) under strict memory budgets (512 MB – 2 GB) by dynamically streaming layers from disk into recycled off-heap double buffers.

### Constructors

- `public FastAIStreamingModel(File modelFile)`  
  Initializes streaming pipeline with the default 512 MB chunk budget and dual-slot off-heap buffer ring.

- `public FastAIStreamingModel(File modelFile, long chunkBudgetBytes)`  
  Initializes streaming pipeline with a custom chunk budget (e.g. `1024L * 1024 * 1024` for 1 GB chunks).

### Key Features & Internal Architecture

- **`DoubleBufferRing`**: Two page-locked (`Memory.lockPages()`), 32-byte SIMD-aligned off-heap slots allocated via `FastMemory` and referenced by 64-bit `FastPointer`.
- **`NativeChunkMmap`**: Zero-copy Win32 memory-mapped layer slices with explicit cleaner invocations (`sun.misc.Unsafe`) to eliminate Windows file locks.
- **`ChunkPipelineScheduler`**: Overlaps chunk prefetching and compute using Java 21 Virtual Threads and AVX2 vector memory copies (`FastSIMD.copy()`).
- **`PersistentKVCache`**: Retains multi-layer attention key/value states permanently in RAM (~100–300 MB) across layer streaming cycles.
- **`FastGPU` Integration**: Dispatches layer GEMV operations to Vulkan Compute with automatic fallback to `FastSIMD` (AVX2/AVX-512 CPU).

