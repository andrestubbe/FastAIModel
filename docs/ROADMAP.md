# FastAIModel Roadmap 🗺️

**Vision:** To provide the fastest, zero-dependency, local in-process AI inference runtime for Java developers, wrapping `llama.cpp` and ONNX Runtime directly.

---

## 🟢 v0.1.0: Initial JNI Scaffold (Current)
- [x] **JNI Native Interface**: Standard wrappers for loading, inference, and memory release
- [x] **DLL Loader**: Integration with `FastCore` JNI dynamic linker
- [x] **Token Callback**: Streaming callback architecture directly into Java consumer
- [x] **Quantization Support**: Basic support for GGUF format model inference

## 🟡 v0.2.0: Core Inference Engine
- [ ] **Direct llama.cpp Binding**: Compile native C++ bindings for CPU AVX2/AVX512
- [ ] **Context Window Management**: Support for token context slicing and sliding window
- [ ] **GPU Acceleration Support**: Build profiles for Vulkan, OpenCL, and CUDA
- [ ] **Direct Tokenization**: Local tokenizer exposed to Java to count tokens without full inference

## 🟠 v0.5.0: Advanced Optimization
- [ ] **Zero-Copy Arrays**: Direct ByteBuffer mappings for token arrays, minimizing GC pressure
- [ ] **Embedding Generation**: Local embeddings via GGUF embedding models
- [ ] **Vision Support**: Multi-modal local inference (passing image files directly to visual LLMs)
- [ ] **Grammar Constraints**: Grammar-based structured outputs (JSON schema constraints) at the engine level

## 🔴 v1.0.0: Production Hardening
- [ ] **Performance Benchmarks**: Native latency tracking vs Ollama and llama-server
- [ ] **Model Pooling**: Dynamic context switching and multi-model in-memory caching
- [ ] **Cross-Platform Compilation**: Unified build scripts for Linux and macOS (.so / .dylib)
