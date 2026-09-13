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

High-throughput chunked layer streaming engine implementing `AutoCloseable`. Executes large language models (1.7B, 7B, 14B, 70B, MoE) under configurable memory budgets (512 MB – 2 GB) by dynamically streaming layers from disk into recycled off-heap double buffers using native AVX2 and F16C kernels.

### Constructors

- `public FastAIStreamingModel(File modelFile)`  
  Initializes streaming pipeline with default configuration (512 MB chunk budget, overlapped I/O enabled).

- `public FastAIStreamingModel(File modelFile, StreamingConfig config)`  
  Initializes streaming pipeline with full customization (`StreamingConfig.builder().chunkBudgetMB(512).overlapIO(true).temperature(0.7f).build()`).

- `public FastAIStreamingModel(File modelFile, long chunkBudgetBytes)`  
  Initializes streaming pipeline with a custom byte budget.

### Methods

- `public void stream(String prompt, Consumer<String> tokenCallback)`  
  Generates text with streaming callback for each emitted token.

- `public void stream(String prompt, int maxTokens, Consumer<String> tokenCallback)`  
  Generates up to `maxTokens` tokens with streaming callback.

### Key Features & Internal Architecture

- **`NativeGemvBackend`**: Java 21 Foreign Function & Memory (FFM) downcall bindings to `fastai_streaming_kernels.dll` executing AVX2 GEMV/GEMM for `Q4_0`, `Q8_0`, `Q4_K` and fused multi-head Attention (`compute_attention_avx2`).
- **`DoubleBufferRing`**: Two page-locked (`Memory.lockPages()`), 32-byte SIMD-aligned off-heap slots allocated via `FastMemory` and referenced by 64-bit `FastPointer`.
- **`NativeChunkMmap`**: Zero-copy Win32 memory-mapped layer slices with explicit cleaner invocations (`sun.misc.Unsafe`) to eliminate Windows file locks.
- **`ChunkPipelineScheduler`**: Overlaps chunk prefetching and compute using Java 21 Virtual Threads and AVX2 vector memory copies (`FastSIMD.copy()`).
- **`PersistentKVCache`**: Retains multi-layer attention key/value states permanently in FP16 precision across layer streaming cycles.
- **`FastGPU` Integration**: Dispatches layer GEMV operations to Vulkan Compute with automatic fallback to native AVX2 CPU kernels.

