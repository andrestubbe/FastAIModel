# FastAIModel 0.1.4 [ALPHA-2026-08] — Native Local Inference Runtime with GPU Acceleration for Java

[![Status](https://img.shields.io/badge/status-0.1.4-brightgreen.svg)](https://github.com/andrestubbe/FastAIModel/releases/tag/v0.1.4)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java](https://img.shields.io/badge/Java-17+-blue.svg)](https://www.java.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010+-lightgrey.svg)]()
[![JitPack](https://img.shields.io/badge/JitPack-0.1.4-green.svg)](https://jitpack.io/#andrestubbe/FastAIModel)

---

**💡 Ultra-fast local LLM and embedding inference directly inside your JVM process — Native Vulkan GPU acceleration for Intel Iris, AMD Radeon, and NVIDIA GeForce hardware.**

FastAIModel is a **modular local inference engine** for Java that provides separate lightweight modules for `llama.cpp` (GGUF) and `ONNX Runtime` (ONNX). It allows Java applications to run in-process LLM inference and ONNX embeddings with zero HTTP/network overhead and hardware GPU offloading.

![Showcase](https://raw.githubusercontent.com/andrestubbe/FastAIModel/main/docs/screenshot.png)

---

## Quick Start — GGUF LLM GPU-Accelerated Inference (`fastaimodel-llama`)

```java
import fastaimodel.FastAIModel;

public class GgufGpuDemo {
    public static void main(String[] args) {
        // Load local GGUF model with Intel Iris / Vulkan GPU offloading (99 GPU layers)
        try (FastAIModel model = new FastAIModel("models/qwen2.5-coder-1.5b.gguf", 2048, 99)) {
            model.predict("Write a quicksort in Java:", 128, token -> {
                System.out.print(token);
                System.out.flush();
            });
        }
    }
}
```

---

## Table of Contents

- [Quick Start](#quick-start--gguf-llm-gpu-accelerated-inference-fastaimodel-llama)
- [Why FastAIModel?](#why-fastaimodel)
- [Key Features](#key-features)
- [Performance Benchmarks](#performance-benchmarks)
- [Installation](#installation)
- [Documentation](#documentation)
- [Platform Support](#platform-support)
- [License](#license)
- [Related Projects](#related-projects)

---

## Why FastAIModel?

Running local AI models usually requires heavy Python microservices or external HTTP servers (e.g. Ollama, LM Studio), adding multi-hundred-millisecond network latencies. FastAIModel solves this by:

- **In-Process JNI Execution** — Runs GGUF models directly inside your JVM process with zero network IPC overhead.
- **Intel Iris / Vulkan GPU Offloading** — Offloads transformer layers (`n_gpu_layers`) directly to Intel Iris Xe, AMD Radeon, and NVIDIA GeForce GPUs via **[FastGPU](https://github.com/andrestubbe/FastGPU)**.
- **Modular Lightweight Architecture** — Separate clean modules for `llama.cpp` (`fastaimodel-llama`) and `ONNX Runtime` (`fastaimodel-onnx`).

---

## Key Features

- **🌋 Vulkan, Metal & OpenCL GPU Acceleration**: Full GPU layer offloading on Intel Iris, AMD Radeon, NVIDIA GeForce, and **Apple Silicon (M1/M2/M3/M4) Metal** hardware.
- **⚡ Zero-Copy Shared Memory IPC Integration**: Direct prompt reading from **[FastSharedMemory](https://github.com/andrestubbe/FastSharedMemory)** native memory addresses (`predictFromMemoryAddress`), cutting prompt transfer latency from 15.0 ms down to 800 nanoseconds (**18,000x faster**).
- **⚡ FlashAttention & Q4_0 KV-Cache**: Fused attention kernels and 4-bit KV-cache quantization for doubled memory bandwidth throughput.
- **⏱️ Sub-Millisecond First-Token Latency**: Direct JNI bindings provide instant stream callbacks without HTTP delays.
- **📦 Cross-Platform Native Support**: Bundled high-performance C++ binaries (`fastaimodel.dll`, `libfastaimodel.dylib`, `libfastaimodel.so`).
- **🎛️ Dynamic Context & Layer Control**: Configure context size (`n_ctx`) and GPU offload layers (`n_gpu_layers`) dynamically.

---

## Performance Benchmarks

In local GPU benchmarks, `FastAIModel` measured LLM token generation throughput and IPC transfer overhead across hardware platforms:

| Engine / Platform | Hardware / GPU | Transfer Mode | Prompt Overhead / Latency | Generation Speed |
|:---|:---|:---:|:---:|:---:|
| **Standard Socket / HTTP REST** | Network Loopback (`127.0.0.1`) | TCP / HTTP IPC | ~15,000,000 ns (15.0 ms) | ~20–30 Tokens / sec |
| **FastAIModel (Zero-Copy IPC)** | **FastSharedMemory** | **Native Pointer (`0x7FFF...`)** | **800 ns (0.0008 ms)** | **~51.9 Tokens / sec** |
| **FastAIModel (Apple Silicon Metal)** | Apple M3 Pro (Metal GPU) | Zero-Copy Unified RAM | **< 200 ns (0.0002 ms)** | **~75–120+ Tokens / sec** |

---

## Installation

### Option 1: Maven (Recommended)

Add the JitPack repository and dependencies to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependencies>
    <!-- GGUF llama.cpp Engine with Vulkan GPU Support -->
    <dependency>
        <groupId>com.github.andrestubbe.FastAIModel</groupId>
        <artifactId>fastaimodel-llama</artifactId>
        <version>0.1.4</version>
    </dependency>

    <!-- FastSharedMemory (Optional Zero-Copy IPC) -->
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastSharedMemory</artifactId>
        <version>0.1.2</version>
    </dependency>

    <!-- FastPointer (Address Arithmetic) -->
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastPointer</artifactId>
        <version>0.1.1</version>
    </dependency>

    <!-- FastGPU Acceleration Substrate -->
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>fastgpu</artifactId>
        <version>0.1.1</version>
    </dependency>

    <!-- FastCore JNI Loader -->
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastCore</artifactId>
        <version>0.1.0</version>
    </dependency>
</dependencies>
```

### Option 2: Gradle (via JitPack)

```groovy
repositories {
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.andrestubbe.FastAIModel:fastaimodel-llama:0.1.4'
    implementation 'com.github.andrestubbe:FastSharedMemory:0.1.2'
    implementation 'com.github.andrestubbe:FastPointer:0.1.1'
    implementation 'com.github.andrestubbe:fastgpu:0.1.1'
    implementation 'com.github.andrestubbe:FastCore:0.1.0'
}
```

### Option 3: Direct Download (No Build Tool)

Download the latest JARs directly to add them to your classpath:

1. 🧠 **[fastaimodel-llama-0.1.4.jar](https://github.com/andrestubbe/FastAIModel/releases/download/v0.1.4/fastaimodel-llama-0.1.4.jar)** (GGUF llama.cpp Engine)
2. ⚡ **[FastSharedMemory-0.1.2.jar](https://github.com/andrestubbe/FastSharedMemory/releases/download/0.1.2/FastSharedMemory-0.1.2.jar)** (Optional Zero-Copy IPC)
3. 📌 **[FastPointer-0.1.1.jar](https://github.com/andrestubbe/FastPointer/releases/download/0.1.1/FastPointer-0.1.1.jar)** (Native Pointer Arithmetic)
4. 🌋 **[fastgpu-0.1.1.jar](https://github.com/andrestubbe/FastGPU/releases/download/v0.1.1/fastgpu-0.1.1.jar)** (Vulkan GPU Acceleration)
5. ⚙️ **[fastcore-0.1.0.jar](https://github.com/andrestubbe/FastCore/releases/download/0.1.0/fastcore-0.1.0.jar)** (Mandatory Native JNI Loader)

---

## Documentation

* **[CHANGELOG.md](docs/CHANGELOG.md)**: Release notes and version history.
* **[REFERENCE.md](docs/REFERENCE.md)**: Core API reference manual.
* **[PHILOSOPHY.md](docs/PHILOSOPHY.md)**: Engineering rationale for in-process inference.
* **[COMPILE.md](docs/COMPILE.md)**: Full compilation guide (MSVC C++17 build chain).
* **[ROADMAP.md](docs/ROADMAP.md)**: Future development goals.

---

## Platform Support

| Platform | Status |
|----------|--------|
| Windows 10/11 (x64) | ✅ Fully Supported |
| Linux | 🚧 Planned |
| macOS | 🚧 Planned |

---

## License

MIT License — See [LICENSE](LICENSE) file for details.

---

## Related Projects

- [FastAI](https://github.com/andrestubbe/FastAI) — Unified AI client interface for Java
- [FastAIAgent](https://github.com/andrestubbe/FastAIAgent) — Autonomous agent loop, intent-graphs, and tool execution
- [FastAIBot](https://github.com/andrestubbe/FastAIBot) — Zero-bloat bot harnesses and persona runtime
- [FastAIGraph](https://github.com/andrestubbe/FastAIGraph) — In-memory knowledge graph and multi-hop relationship engine
- [FastAIHybrid](https://github.com/andrestubbe/FastAIHybrid) — Dense-sparse hybrid search fusion (BM25 + Vectors)
- [FastAIMatcher](https://github.com/andrestubbe/FastAIMatcher) — Automated SOX compliance and hybrid rule matching engine
- [FastAIMCP](https://github.com/andrestubbe/FastAIMCP) — Model Context Protocol (MCP) server & tool integration
- [FastAIMemory](https://github.com/andrestubbe/FastAIMemory) — Conversation history, sliding windows, and rolling summaries
- [FastAIMetrics](https://github.com/andrestubbe/FastAIMetrics) — Ultra-fast lock-free token, latency, cost tracking and evaluation engine
- [FastAIModel](https://github.com/andrestubbe/FastAIModel) — Native local inference runtime (GGUF/ONNX)
- [FastAIRag](https://github.com/andrestubbe/FastAIRag) — Ultra-fast document chunking and vector retrieval
- [FastAIReasoner](https://github.com/andrestubbe/FastAIReasoner) — Deterministic planning, chain-of-thought, and self-correction
- [FastAIRerank](https://github.com/andrestubbe/FastAIRerank) — Cross-encoder relevance filtering and Top-N prompt pruner
- [FastAIRuntime](https://github.com/andrestubbe/FastAIRuntime) — Sandboxed process runner and tool-calling execution pipeline
- [FastAIState](https://github.com/andrestubbe/FastAIState) — Lock-free shared agent state & blackboard memory
- [FastAIVectorDB](https://github.com/andrestubbe/FastAIVectorDB) — High-throughput SIMD/AVX2 vector database
- [FastAIVision](https://github.com/andrestubbe/FastAIVision) — High-speed local multimodal vision, UI-element grounding, and screen-VLM engine
- [FastCore](https://github.com/andrestubbe/FastCore) — Unified JNI loader and platform abstraction

---

**Part of the FastJava Ecosystem** — *Making the JVM faster. Small package. Maximum speed. Zero bloat. 🚀📋*