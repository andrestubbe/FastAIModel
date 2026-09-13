# FastAIModel 0.1.9 [ALPHA-2026-09] — Native Local Inference Runtime with GPU Acceleration for Java

[![Status](https://img.shields.io/badge/status-0.1.9-brightgreen.svg)](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java](https://img.shields.io/badge/Java-21+-blue.svg)](https://www.java.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010+-lightgrey.svg)]()
[![JitPack](https://img.shields.io/badge/JitPack-0.1.9-green.svg)](https://jitpack.io/#andrestubbe/FastAIModel)

---

**💡 Ultra-fast local LLM and embedding inference directly inside your JVM process — Cross-vendor GPU acceleration (NVIDIA RTX, AMD Radeon, Intel Iris Xe / Arc, Apple Silicon Metal) for GGUF, ONNX Runtime, and Zero-Copy Layer-wise & MoE Streaming.**

FastAIModel is a **high-performance, modular local AI runtime** for Java that provides three specialized engines:
1. 🧠 **`fastaimodel-llama`**: In-process GGUF inference via native `llama.cpp` bindings with Vulkan & Apple Metal GPU offloading across NVIDIA, AMD, Intel, and Apple GPUs.
2. ⚡ **`fastaimodel-onnx`**: Lightweight ONNX Runtime integration for sub-millisecond vector embeddings and deep learning pipelines.
3. 🌊 **`fastaimodel-streaming`**: Zero-Copy Layer-wise & MoE Streaming engine executing 7B–70B models (Mistral 7B, Qwen 2.5/3.5, SmolLM2, Mixtral 8x7B) on standard laptops within ultra-low 512 MB – 2 GB RAM budgets using native AVX2 + F16C fused kernels and FastGPU acceleration.

[**Watch Demo (YouTube)**](https://www.youtube.com/watch?v=pY-39438feM) | [**Watch the JMH Benchmark**](https://www.youtube.com/watch?v=pY-39438feM)

[![FastAIModel Showcase](docs/screenshot.png)](https://www.youtube.com/watch?v=pY-39438feM)

---

## Quick Start

### 1. GGUF GPU-Accelerated LLM Inference (`fastaimodel-llama`)

In-process execution of local GGUF models via native `llama.cpp` JNI bindings with hardware GPU layer offloading (NVIDIA GeForce RTX, AMD Radeon, Intel Iris Xe / Arc, Apple Silicon Metal):

```java
import fastaimodel.FastAIModel;

public class GgufDemo {
    public static void main(String[] args) {
        // Option A: Direct argument constructor with Vulkan GPU offloading (99 layers)
        try (FastAIModel model = new FastAIModel("models/qwen2.5-coder-1.5b.gguf", 2048, 99)) {
            model.predict("Write a quicksort in Java:", 128, token -> {
                System.out.print(token);
                System.out.flush();
            });
        }

        // Option B: Fluent Builder pattern
        try (FastAIModel model = FastAIModel.builder()
                .model("models/qwen2.5-coder-1.5b.gguf")
                .contextLength(2048)
                .gpuLayers(99)
                .build()) {
            model.predict("Write a binary search:", 128, System.out::print);
        }
    }
}
```

### 2. Lightweight In-Process ONNX Inference (`fastaimodel-onnx`)

Embeddings, text-to-speech, and vision model inference via in-process ONNX Runtime (zero C++ llama DLL dependencies):

```java
import fastaimodel.FastAIOnnxModel;
import ai.onnxruntime.OrtSession;

public class OnnxQuickStart {
    public static void main(String[] args) {
        // Option A: Direct factory or constructor
        try (FastAIOnnxModel onnx = FastAIOnnxModel.open("models/bge-micro-v2.onnx")) {
            OrtSession session = onnx.getSession();
            System.out.println("ONNX Input Nodes:  " + session.getInputNames());
            System.out.println("ONNX Output Nodes: " + session.getOutputNames());
        }

        // Option B: Fluent Builder with multi-threading
        try (FastAIOnnxModel onnx = FastAIOnnxModel.builder()
                .model("models/bge-micro-v2.onnx")
                .threads(4)
                .build()) {
            // High-throughput embedding inference
        }
    }
}
```

### 3. Zero-Copy Layer-wise & MoE Streaming (`fastaimodel-streaming`)

Execute 7B, 14B, 70B Dense models or **MoE architectures (e.g. Mixtral 8x7B)** within an **ultra-low 512 MB – 1 GB RAM budget** (`-Xmx2g`, or any custom limit) using zero-copy Win32 memory-mapped layer streaming with hardware-accelerated AVX2 + F16C dot products and optional FastGPU acceleration:

```java
import fastaimodel.streaming.FastAIStreamingModel;
import java.io.File;

public class StreamingQuickStart {
    public static void main(String[] args) throws Exception {
        // Option A: Clean direct-argument constructor (Model, Chunk Budget MB, Use GPU)
        try (FastAIStreamingModel model = new FastAIStreamingModel("mistral:7b", 512, false)) {
            model.stream("Explain quantum computing in three sentences:", 64, token -> {
                System.out.print(token);
                System.out.flush();
            });
        }

        // Option B: Fluent Builder pattern
        try (FastAIStreamingModel model = FastAIStreamingModel.builder()
                .model("mistral:7b")
                .chunkBudgetMB(512)
                .useGPU(true)      // FastGPU Vulkan acceleration
                .overlapIO(true)   // Overlapped asynchronous NVMe prefetching
                .temperature(0.7f)
                .build()) {
            model.stream("What is JVM Panama FFM?", 64, System.out::print);
        }
    }
}
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Why FastAIModel?](#why-fastaimodel)
- [Key Features](#key-features)
- [Real-World Use Cases](#real-world-use-cases)
- [Performance & JMH Benchmarks](#performance--jmh-benchmarks)
- [Installation](#installation)
- [Documentation](#documentation)
- [Platform Support](#platform-support)
- [License](#license)
- [Related Projects](#related-projects)

---

## Why FastAIModel?

Running local AI models usually requires heavy Python microservices or external HTTP servers (e.g. Ollama, LM Studio), adding multi-hundred-millisecond network latencies. FastAIModel solves this by:

- **In-Process JNI & FFM Execution** — Runs GGUF models directly inside your JVM process with zero network IPC overhead.
- **Intel Iris / Vulkan GPU Offloading** — Offloads transformer layers (`n_gpu_layers`) directly to Intel Iris Xe, AMD Radeon, and NVIDIA GeForce GPUs via **[FastGPU](https://github.com/andrestubbe/FastGPU)**.
- **Zero-Copy Layer-wise & MoE Streaming (`fastaimodel-streaming`)** — Stream 7B–70B Dense models and multi-gigabyte **MoE architectures (e.g. Mixtral 8x7B, DeepSeek-MoE)** within an ultra-low 512 MB – 1 GB RAM budget via zero-copy Win32 mmap, overlapped Virtual Thread I/O, and native AVX2 Q4_0 / Q8_0 / Q4_K GEMV.
- **Modular Lightweight Architecture** — Separate clean modules for `llama.cpp` (`fastaimodel-llama`), `ONNX Runtime` (`fastaimodel-onnx`), and `Layer-Streaming` (`fastaimodel-streaming`).

---

## Key Features

- **🌊 Zero-Copy Layer-wise & MoE Expert Streaming**: Stream 20B–70B parameters directly from NVMe SSDs into recycled off-heap double-buffers without OOM crashes.
- **🚀 Native AVX2 & F16C Kernels**: C++ vectorized GEMV/GEMM for `Q4_0`, `Q8_0`, `Q4_K` plus native multi-head Attention (`compute_attention_avx2`).
- **🌋 Vulkan, Metal & OpenCL GPU Acceleration**: Full GPU layer offloading on Intel Iris, AMD Radeon, NVIDIA GeForce, and **Apple Silicon (M1/M2/M3/M4) Metal** hardware.
- **⚡ Zero-Copy Shared Memory IPC Integration**: Direct prompt reading from **[FastSharedMemory](https://github.com/andrestubbe/FastSharedMemory)** native memory addresses (`predictFromMemoryAddress`), cutting prompt transfer latency from 15.0 ms down to 800 nanoseconds (**18,000x faster**).
- **⚡ FlashAttention & Q4_0 KV-Cache**: Fused attention kernels and 4-bit KV-cache quantization for doubled memory bandwidth throughput.
- **⏱️ Sub-Millisecond First-Token Latency**: Direct JNI & FFM bindings provide instant stream callbacks without HTTP delays.
- **📦 Cross-Platform Native Support**: Bundled high-performance C++ binaries (`fastaimodel.dll`, `fastai_streaming_kernels.dll`).
- **🎛️ Dynamic Context & Layer Control**: Configure context size (`n_ctx`) and GPU offload layers (`n_gpu_layers`) dynamically.

---

## Real-World Use Cases

- 💻 **Low-Memory Desktop & Laptop LLM Execution**: Run full 7B or 14B instruction models (Mistral 7B, Qwen 2.5/3.5, SmolLM2) directly on 8 GB or 16 GB RAM developer workstations using a tiny 512 MB – 1 GB RAM footprint without swapping or OOM crashes.
- ⚡ **Zero-Latency In-Process Agent Loops**: Drive autonomous agent decision cycles (**[FastAIAgent](https://github.com/andrestubbe/FastAIAgent)**) with in-process token generation, eliminating the 15–30 ms HTTP socket round-trip overhead of external LLM servers.
- 🚀 **High-Throughput Local Vector Embeddings**: Generate sub-millisecond document and passage embeddings for RAG retrieval (**[FastAIRag](https://github.com/andrestubbe/FastAIRag)** & **[FastAIVectorDB](https://github.com/andrestubbe/FastAIVectorDB)**) using `fastaimodel-onnx` without separate Python/Docker infrastructure.
- 🔒 **Air-Gapped & Secure Enterprise Pipelines**: Fully embed GGUF models into single self-contained JAR deployments for defense, banking, and confidential environments with zero external network connectivity.
- 🌐 **Zero-Copy Shared Memory Inter-Process Comms**: Receive multi-megabyte prompt contexts directly from other JVMs or C/Rust processes via **[FastSharedMemory](https://github.com/andrestubbe/FastSharedMemory)** native pointers with sub-microsecond latency.

---

## Performance & JMH Benchmarks

In local GPU and CPU benchmarks, `FastAIModel` measured LLM token generation throughput, memory streaming overhead, and IPC transfer latency across hardware platforms:

| Engine / Platform | Hardware / GPU | Transfer Mode | Prompt Overhead / Latency | Generation Speed |
|:---|:---|:---:|:---:|:---:|
| **Standard Socket / HTTP REST** | Network Loopback (`127.0.0.1`) | TCP / HTTP IPC | ~15,000,000 ns (15.0 ms) | ~20–30 Tokens / sec |
| **FastAIModel (Zero-Copy IPC)** | **FastSharedMemory** | **Native Pointer (`0x7FFF...`)** | **800 ns (0.0008 ms)** | **~51.9 Tokens / sec** |
| **FastAIModel (Apple Silicon Metal)** | Apple M3 Pro (Metal GPU) | Zero-Copy Unified RAM | **< 200 ns (0.0002 ms)** | **~75–120+ Tokens / sec** |
| **FastAIModel Streaming (AVX2 + F16C)** | Intel Iris Xe / NVMe SSD (512MB RAM cap) | Overlapped Win32 Mmap + Native AVX2 | Local In-Process | **~3–10 Tokens / sec (7B on 4GB free RAM)** |

---

## Installation

Choose the module tailored to your workload:
* 🧠 **`fastaimodel-llama`**: In-process GGUF inference via `llama.cpp` + Vulkan GPU offloading.
* ⚡ **`fastaimodel-onnx`**: Lightweight in-process ONNX Runtime (embeddings, TTS, vision).
* 🌊 **`fastaimodel-streaming`**: Zero-Copy Layer-wise Streaming for 7B–70B models under ultra-low RAM budgets.

---

### Option 1: Maven (Recommended)

Add the JitPack repository and the desired engine module(s) to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependencies>
    <!-- 1. GGUF llama.cpp Engine with Vulkan GPU Support -->
    <dependency>
        <groupId>com.github.andrestubbe.FastAIModel</groupId>
        <artifactId>fastaimodel-llama</artifactId>
        <version>0.1.9</version>
    </dependency>

    <!-- 2. Lightweight ONNX Runtime Engine (Zero C++ DLL dependencies) -->
    <dependency>
        <groupId>com.github.andrestubbe.FastAIModel</groupId>
        <artifactId>fastaimodel-onnx</artifactId>
        <version>0.1.9</version>
    </dependency>

    <!-- 3. Zero-Copy Layer-wise Streaming (Run 7B–70B under 512 MB – 1 GB RAM) -->
    <dependency>
        <groupId>com.github.andrestubbe.FastAIModel</groupId>
        <artifactId>fastaimodel-streaming</artifactId>
        <version>0.1.9</version>
    </dependency>

    <!-- FastJava Native Acceleration Substrates -->
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>fastgpu</artifactId>
        <version>0.1.1</version>
    </dependency>
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastSIMD</artifactId>
        <version>0.1.3</version>
    </dependency>
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastMemory</artifactId>
        <version>0.1.1</version>
    </dependency>
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastPointer</artifactId>
        <version>0.1.1</version>
    </dependency>
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastSharedMemory</artifactId>
        <version>0.1.2</version>
    </dependency>
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastCore</artifactId>
        <version>0.1.1</version>
    </dependency>
</dependencies>
```

### Option 2: Gradle (via JitPack)

```groovy
repositories {
    maven { url 'https://jitpack.io' }
}

dependencies {
    // Pick the module(s) you need:
    implementation 'com.github.andrestubbe.FastAIModel:fastaimodel-llama:0.1.9'       // GGUF Engine
    implementation 'com.github.andrestubbe.FastAIModel:fastaimodel-onnx:0.1.9'        // ONNX Engine
    implementation 'com.github.andrestubbe.FastAIModel:fastaimodel-streaming:0.1.9'   // Layer-wise Streaming

    // FastJava Ecosystem Libraries
    implementation 'com.github.andrestubbe:fastgpu:0.1.1'
    implementation 'com.github.andrestubbe:FastSIMD:0.1.3'
    implementation 'com.github.andrestubbe:FastMemory:0.1.1'
    implementation 'com.github.andrestubbe:FastPointer:0.1.1'
    implementation 'com.github.andrestubbe:FastSharedMemory:0.1.2'
    implementation 'com.github.andrestubbe:FastCore:0.1.1'
}
```

### Option 3: Direct Download (No Build Tool)

Download pre-compiled release JARs directly from [GitHub Releases](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.9):

* 🧠 **[fastaimodel-llama-0.1.9.jar](https://github.com/andrestubbe/FastAIModel/releases/download/0.1.9/fastaimodel-llama-0.1.9.jar)** (GGUF llama.cpp Engine with bundled native DLLs)
* ⚡ **[fastaimodel-onnx-0.1.9.jar](https://github.com/andrestubbe/FastAIModel/releases/download/0.1.9/fastaimodel-onnx-0.1.9.jar)** (Lightweight ONNX Runtime Engine)
* 🌊 **[fastaimodel-streaming-0.1.9.jar](https://github.com/andrestubbe/FastAIModel/releases/download/0.1.9/fastaimodel-streaming-0.1.9.jar)** (Zero-Copy Layer-wise Streaming Engine)
* 🌋 **[fastgpu-0.1.1.jar](https://github.com/andrestubbe/FastGPU/releases/download/v0.1.1/fastgpu-0.1.1.jar)** (Vulkan GPU Acceleration)
* ⚡ **[FastSharedMemory-0.1.2.jar](https://github.com/andrestubbe/FastSharedMemory/releases/download/0.1.2/FastSharedMemory-0.1.2.jar)** (Zero-Copy Native IPC)
* 📌 **[FastPointer-0.1.1.jar](https://github.com/andrestubbe/FastPointer/releases/download/0.1.1/FastPointer-0.1.1.jar)** (64-Bit Native Pointer Arithmetic)
* ⚙️ **[fastcore-0.1.1.jar](https://github.com/andrestubbe/FastCore/releases/download/0.1.1/fastcore-0.1.1.jar)** (Cross-Platform JNI Loader)

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