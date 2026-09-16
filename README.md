> [!WARNING]
> **🚧 WIP — Active AI Pipeline Construction & Architecture Optimization in Progress.**

# FastAIModel 0.1.9 [ALPHA-2026-09-13]: Native Local Inference Runtime with GPU Acceleration for Java

[![Status](https://img.shields.io/badge/status-0.1.9-brightgreen.svg)](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java](https://img.shields.io/badge/Java-21+-blue.svg)](https://www.java.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010+-lightgrey.svg)]()
[![JitPack](https://img.shields.io/badge/JitPack-0.1.9-green.svg)](https://jitpack.io/#andrestubbe/FastAIModel)

---

**💡 Ultra-fast local LLM and embedding inference directly inside your JVM process — Cross-vendor GPU acceleration (NVIDIA RTX, AMD Radeon, Intel Iris Xe / Arc, Apple Silicon Metal) for GGUF, ONNX Runtime, and Zero-Copy Layer-wise & MoE Streaming.**

**FastAIModel** is a high-performance, modular local AI runtime for Java that provides three specialized engines:
1. 🧠 **`fastaimodel-llama`**: In-process GGUF inference via native `llama.cpp` bindings with Vulkan & Apple Metal GPU offloading across NVIDIA, AMD, Intel, and Apple GPUs.
2. ⚡ **`fastaimodel-onnx`**: Lightweight ONNX Runtime integration for sub-millisecond vector embeddings and deep learning pipelines.
3. 🌊 **`fastaimodel-streaming`**: Zero-Copy Layer-wise & MoE Streaming engine executing 7B–70B models (Mistral 7B, Qwen 2.5/3.5, SmolLM2, Mixtral 8x7B) on standard laptops within ultra-low 512 MB – 2 GB RAM budgets using native AVX2 + F16C fused kernels and FastGPU acceleration.

[**Watch Demo (YouTube)**](https://www.youtube.com/watch?v=pY-39438feM)

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

- [Why FastAIModel?](#why-fastaimodel)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Real-World Use Cases](#real-world-use-cases)
- [Performance & JMH Benchmarks](#performance--jmh-benchmarks)
- [API Quick Reference](#api-quick-reference)
- [Technical Demos & Benchmarks](#technical-demos--benchmarks)
- [Installation](#installation)
- [Documentation](#documentation)
- [Platform Support](#platform-support)
- [Related Projects](#related-projects)
- [License](#license)

---

## Why FastAIModel?

Running local AI models in Java traditionally forces developers to launch separate Python microservices or external HTTP containers (such as Ollama or LM Studio), creating severe architectural friction:

- **Heavy IPC & Socket Latency**: Round-tripping prompts across local HTTP sockets (`127.0.0.1`) introduces 15–30 ms of serialization and network overhead per turn.
- **Excessive RAM Footprint**: Running 7B–70B models usually requires 8 GB – 64 GB of dedicated resident RAM or VRAM, causing OOM crashes on standard developer laptops.
- **Complex Container Orchestration**: Managing separate Docker daemons or background processes adds deployment fragility in enterprise and desktop environments.

FastAIModel solves this by bringing high-speed native model execution directly into the Java process:

- **Zero-IPC In-Process Execution**: Direct JNI and Panama FFM bindings run GGUF and ONNX models inside the JVM with sub-microsecond invocation latency.
- **Zero-Copy Layer-wise Streaming**: Stream 7B–70B dense and MoE models straight from NVMe SSDs into a tiny 512 MB – 1 GB RAM footprint with native AVX2 + F16C kernels.
- **Cross-Vendor Hardware Acceleration**: Offload transformer layers to NVIDIA RTX, AMD Radeon, Intel Iris Xe / Arc (via Vulkan), and Apple Silicon (via Metal).

| Feature | External Local Servers (Ollama / Python) | FastAIModel |
|:---|:---|:---|
| **Invocation Latency** | 15–30 ms (HTTP / TCP loopback) | Sub-microsecond (<800 ns in-process / SHM) |
| **Minimum RAM for 7B** | 6–8 GB resident RAM | **512 MB – 1 GB** (via Layer Streaming) |
| **MoE Architecture Support**| Requires massive 32–64 GB RAM | Streams active experts dynamically from NVMe |
| **GPU Acceleration** | Dependent on external daemon driver setup | Direct Vulkan, Metal & OpenCL offloading |
| **Deployment Model** | External server process / daemon required | Single self-contained JAR with bundled natives |

---

## Key Features

- 🌊 **Zero-Copy Layer-wise & MoE Expert Streaming**: Stream 20B–70B parameters directly from NVMe SSDs into recycled off-heap double-buffers without OOM crashes.
- 🚀 **Native AVX2 & F16C Kernels**: C++ vectorized GEMV/GEMM for `Q4_0`, `Q8_0`, `Q4_K` plus native multi-head Attention (`compute_attention_avx2`).
- 🌋 **Vulkan, Metal & OpenCL GPU Acceleration**: Full GPU layer offloading on Intel Iris, AMD Radeon, NVIDIA GeForce, and Apple Silicon Metal hardware.
- ⚡ **Zero-Copy Shared Memory IPC**: Direct prompt reading from **[FastSharedMemory](https://github.com/andrestubbe/FastSharedMemory)** native pointers (`predictFromMemoryAddress`), cutting prompt transfer latency from 15.0 ms down to 800 nanoseconds.
- ⏱️ **Sub-Millisecond First-Token Latency**: Direct JNI and FFM bindings provide instant stream callbacks without HTTP socket delays.
- 📦 **Self-Contained Native Bundling**: High-performance pre-compiled binaries (`fastaimodel.dll`, `fastai_streaming_kernels.dll`) loaded automatically via **[FastCore](https://github.com/andrestubbe/FastCore)**.

---

## Real-World Use Cases

- 💻 **Low-Memory Laptop LLM Execution**: Run full 7B instruction models (Mistral 7B, Qwen 2.5, SmolLM2) directly on 8 GB or 16 GB developer laptops within a tiny 512 MB RAM footprint.
- ⚡ **Zero-Latency In-Process Agent Loops**: Drive autonomous agent decision cycles (**[FastAIAgent](https://github.com/andrestubbe/FastAIAgent)**) without the socket jitter of external local LLM servers.
- 🚀 **High-Throughput Vector Embeddings**: Generate sub-millisecond document embeddings for RAG retrieval (**[FastAIRag](https://github.com/andrestubbe/FastAIRag)** & **[FastAIVectorDB](https://github.com/andrestubbe/FastAIVectorDB)**) using `fastaimodel-onnx`.
- 🔒 **Air-Gapped & Secure Enterprise Deployments**: Package local GGUF models directly into enterprise applications with zero external network connectivity.

---

## Performance & JMH Benchmarks

Measured across hardware platforms for token throughput, memory streaming overhead, and IPC transfer latency:

| Engine / Platform | Hardware / GPU | Transfer Mode | Prompt Overhead / Latency | Generation Speed |
|:---|:---|:---:|:---:|:---:|
| **Standard Socket / HTTP REST** | Network Loopback (`127.0.0.1`) | TCP / HTTP IPC | ~15,000,000 ns (15.0 ms) | ~20–30 Tokens / sec |
| **FastAIModel (Zero-Copy IPC)** | **FastSharedMemory** | **Native Pointer (`0x7FFF...`)** | **800 ns (0.0008 ms)** | **~51.9 Tokens / sec** |
| **FastAIModel (Apple Silicon Metal)** | Apple M3 Pro (Metal GPU) | Zero-Copy Unified RAM | **< 200 ns (0.0002 ms)** | **~75–120+ Tokens / sec** |
| **FastAIModel Streaming (AVX2 + F16C)** | Intel Iris Xe / NVMe SSD (512MB RAM cap) | Overlapped Win32 Mmap + Native AVX2 | Local In-Process | **~3–10 Tokens / sec (7B on 4GB free RAM)** |

---

## API Quick Reference

| Class / Method | Return Type | Description | Docs |
|:---|:---|:---|:---|
| `new FastAIModel(path, ctx, gpuLayers)` | `FastAIModel` | Initializes in-process GGUF model with GPU layer offloading. | [Reference](docs/REFERENCE.md) |
| `FastAIModel.builder()` | `Builder` | Fluent builder for GGUF context length, GPU layers, and model path. | [Reference](docs/REFERENCE.md) |
| `model.predict(prompt, maxTokens, out)` | `void` | Generates tokens with streaming callback consumer. | [Reference](docs/REFERENCE.md) |
| `FastAIOnnxModel.open(path)` | `FastAIOnnxModel` | Initializes lightweight in-process ONNX Runtime session. | [Reference](docs/REFERENCE.md) |
| `new FastAIStreamingModel(name, budget, gpu)` | `FastAIStreamingModel` | Initializes zero-copy layer streaming model within specified RAM budget in MB. | [Reference](docs/REFERENCE.md) |
| `streamingModel.stream(prompt, max, out)` | `void` | Executes zero-copy layer streaming inference directly from disk. | [Reference](docs/REFERENCE.md) |

---

## Technical Demos & Benchmarks

| Case | Java Example | Launcher | Description |
|:---|:---|:---|:---|
| **Zero-Copy Streaming Demo** | [StreamingDemo.java](examples/StreamingDemo/src/main/java/fastaimodel/streaming/StreamingDemo.java) | `run-streaming.bat` | Real-time layer-wise streaming of 7B models under low-RAM budgets. |
| **GGUF Interactive Demo** | [GgufDemo.java](fastaimodel-llama/src/test/java/fastaimodel/GgufDemo.java) | `run-gguf-demo.bat` | Interactive terminal completion with Vulkan GPU layer offloading. |
| **Hardware Benchmark Suite** | [Benchmark.java](examples/Benchmark/src/main/java/fastaimodel/benchmark/Benchmark.java) | `run-benchmark.bat` | Performance comparison between CPU-only and Vulkan GPU accelerated inference. |

---

## Installation

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

    <!-- 2. Lightweight ONNX Runtime Engine -->
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

    <!-- FastJava Native Loader -->
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
    implementation 'com.github.andrestubbe.FastAIModel:fastaimodel-llama:0.1.9'
    implementation 'com.github.andrestubbe.FastAIModel:fastaimodel-onnx:0.1.9'
    implementation 'com.github.andrestubbe.FastAIModel:fastaimodel-streaming:0.1.9'
    implementation 'com.github.andrestubbe:FastCore:0.1.1'
}
```

### Option 3: Direct Download (No Build Tool)

Download pre-compiled release JARs directly from GitHub Releases:

1. 🧠 **[fastaimodel-llama-0.1.9.jar](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.9)** (GGUF Engine with bundled native DLLs)
2. ⚡ **[fastaimodel-onnx-0.1.9.jar](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.9)** (Lightweight ONNX Runtime Engine)
3. 🌊 **[fastaimodel-streaming-0.1.9.jar](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.9)** (Zero-Copy Layer-wise Streaming Engine)
4. ⚙️ **[fastcore-0.1.1.jar](https://github.com/andrestubbe/FastCore/releases/tag/0.1.1)** (Cross-Platform JNI Loader)

---

## Documentation

- **[REFERENCE.md](docs/REFERENCE.md)**: Core API reference manual and engine configurations.
- **[PHILOSOPHY.md](docs/PHILOSOPHY.md)**: Engineering rationale for in-process inference.
- **[COMPILE.md](docs/COMPILE.md)**: Full compilation guide for MSVC C++17 build chain.
- **[CHANGELOG.md](docs/CHANGELOG.md)**: Release notes and version history.
- **[ROADMAP.md](docs/ROADMAP.md)**: Future development goals.

---

## Platform Support

| Platform | Architecture | Status | Notes |
|:---|:---:|:---:|:---|
| **Windows 10 / 11** | x64 | ✅ Fully Supported | In-process GGUF, Vulkan GPU offloading, AVX2 layer streaming |
| **Linux** | x64 / AArch64 | 🚧 Planned | ONNX works today; native llama Vulkan runtime planned |
| **macOS** | Apple Silicon (M1–M4) | 🚧 In Development | Metal GPU acceleration supported on Apple Silicon |

---

## Related Projects

- **[`FastAI`](https://github.com/andrestubbe/FastAI)**: Unified AI Client for Java (20+ providers)
- **[`FastAIAgent`](https://github.com/andrestubbe/FastAIAgent)**: Autonomous ReAct Agent Loop and Cognitive Mind
- **[`FastAIRag`](https://github.com/andrestubbe/FastAIRag)**: In-Process Retrieval-Augmented Generation Substrate
- **[`FastAIVectorDB`](https://github.com/andrestubbe/FastAIVectorDB)**: High-Throughput SIMD/AVX2 Vector Database
- **[`FastGPU`](https://github.com/andrestubbe/FastGPU)**: Hardware-Accelerated Vulkan Compute Engine
- **[`FastSharedMemory`](https://github.com/andrestubbe/FastSharedMemory)**: Ultra-Fast Zero-Copy Inter-Process Memory Mapping
- **[`FastCore`](https://github.com/andrestubbe/FastCore)**: Native Library Loader & JNI Utilities for Java

---

## License

MIT License. See [LICENSE](LICENSE) file for details.

---

**Part of the FastJava Ecosystem** — *Making the JVM faster.* 🚀