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

- `public void close()`  
  Frees native `llama.cpp` model context, KV cache buffers, and GPU Vulkan/Metal resources.

---

## Class: `fastaimodel.FastAIModelOnnx`

In-process ONNX Runtime embedding engine for vector embeddings.

### Methods

- `public float[] embed(String text)`  
  Generates high-dimensional vector embeddings for text inputs.
