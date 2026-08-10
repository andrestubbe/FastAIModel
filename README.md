# FastAIModel 0.1.2 — Native Local Inference Runtime for Java

[![Status](https://img.shields.io/badge/status-0.1.2-brightgreen.svg)](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java](https://img.shields.io/badge/Java-17+-blue.svg)](https://www.java.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010+-lightgrey.svg)]()
[![JitPack](https://img.shields.io/badge/JitPack-0.1.2-green.svg)](https://jitpack.io/#andrestubbe/FastAIModel)

---
**💡 Ultra-fast local LLM and embedding inference directly inside your JVM process — Multi-module architecture for GGUF and ONNX.**

FastAIModel is a **modular local inference engine** for Java that provides separate lightweight modules for `llama.cpp` (GGUF) and `ONNX Runtime` (ONNX). It allows Java applications to run in-process LLM inference and ONNX embeddings with zero HTTP/network overhead.

---

## Modular Architecture (New in 0.1.2)

FastAIModel is split into independent modules so you only import what you need:

| Module | Description | Dependencies |
|---|---|---|
| **`fastaimodel-onnx`** | Ultra-lightweight ONNX Runtime wrapper for embeddings | `onnxruntime` (No C++ Llama DLLs needed) |
| **`fastaimodel-llama`** | High-performance C++ `llama.cpp` wrapper for GGUF models | `FastCore`, Native C++ DLLs |

---

## Installation

### Option 1: ONNX Embeddings Only (`fastaimodel-onnx`)

If you only need ONNX models (e.g. for vector search embeddings):

```xml
<dependencies>
    <dependency>
        <groupId>com.github.andrestubbe.FastAIModel</groupId>
        <artifactId>fastaimodel-onnx</artifactId>
        <version>0.1.2</version>
    </dependency>
</dependencies>
```

### Option 2: GGUF LLM Local Inference (`fastaimodel-llama`)

If you want in-process C++ GGUF LLM execution via `llama.cpp`:

```xml
<dependencies>
    <dependency>
        <groupId>com.github.andrestubbe.FastAIModel</groupId>
        <artifactId>fastaimodel-llama</artifactId>
        <version>0.1.2</version>
    </dependency>
</dependencies>
```

---

## Quick Start

### ONNX Model Usage (`fastaimodel-onnx`)

```java
import fastaimodel.FastAIOnnxModel;

try (FastAIOnnxModel onnx = new FastAIOnnxModel("models/model.onnx")) {
    // Run ONNX inference
    var result = onnx.run(inputs);
}
```

### GGUF Model Usage (`fastaimodel-llama`)

```java
import fastaimodel.FastAIModel;

try (FastAIModel model = new FastAIModel("models/qwen2.5-coder-1.5b.gguf")) {
    model.predict("Write a quicksort in Java:", 128, token -> {
        System.out.print(token);
    });
}
```

---

## Documentation

* **[REFERENCE.md](docs/REFERENCE.md)**: JNI contracts and module specifications.
* **[PHILOSOPHY.md](docs/PHILOSOPHY.md)**: In-process design decisions.
* **[CHANGELOG.md](docs/CHANGELOG.md)**: Releases history.

---

## License

MIT License — See [LICENSE](LICENSE) file for details.

---

**Part of the FastJava Ecosystem** — *Making the JVM faster. Small package. Maximum speed. Zero bloat. 🚀📋*
