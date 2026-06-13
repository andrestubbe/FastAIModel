# FastAIModel 0.1.0 [ALPHA] — Native Local Inference Runtime for Java

[![Status](https://img.shields.io/badge/status-0.1.0-brightgreen.svg)](https://github.com/andrestubbe/FastAIModel/releases/tag/0.1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java](https://img.shields.io/badge/Java-17+-blue.svg)](https://www.java.com)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010+-lightgrey.svg)]()
[![JitPack](https://img.shields.io/badge/JitPack-ready-green.svg)](https://jitpack.io/#andrestubbe)

---

**💡 Ultra-fast local LLM and embedding inference directly inside your JVM process — Zero-copy, Zero HTTP overhead, C++ native speed.**

FastAIModel is a **retained-memory local inference engine** for Java that wraps `llama.cpp` (for GGUF) and `ONNX Runtime` (for ONNX) using direct JNI bindings. It is the engine that drives offline execution in the **FastJava Ecosystem**, giving Java developers native LLM and embedding capabilities without keeping heavy external apps (like LM Studio or Ollama) open.

---

## Table of Contents
- [Why FastAIModel?](#why-fastaimodel)
- [Key Features](#key-features)
- [Installation](#installation)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [License](#license)

---

## Quick Start

```java
import fastaimodel.FastAIModel;

public class Demo {
    public static void main(String[] args) {
        // Load local GGUF model directly into memory
        try (FastAIModel model = new FastAIModel("models/qwen2.5-coder-1.5b.gguf", 2048, 0.7f)) {
            model.generate("Write a quicksort in Java:", token -> {
                System.out.print(token);
                System.out.flush();
            });
        }
    }
}
```

---

## Why FastAIModel?

Running LLMs locally in Java typically requires invoking external subprocesses or running local HTTP servers. FastAIModel eliminates this bloat by running the model directly inside your Java process:

- **True In-Process Execution** — Runs the model in the same process space, bypassing system context-switches and network sockets.
- **Zero HTTP/JSON Overhead** — Text and tokens flow directly between Java and C++ memory.
- **Low Memory Overhead** — Eliminates the footprint of keeping GUI-based desktop inference servers running in the background.

---

## Key Features

- **🚀 Native llama.cpp Performance** — Direct integration with CPU AVX2/AVX512 instruction sets and GPU computation (Vulkan/CUDA).
- **🌊 Direct Token Streaming** — Direct native callbacks stream tokens back to your Java consumer in real-time.
- **📦 GGUF Support** — Native compatibility with any GGUF quantized models (Llama, Qwen, Mistral, Gemma).
- **🧠 Zero-Copy Memory** — Shared token handling minimizing garbage collection strain on the JVM.

---

## Installation

### Option 1: Maven (via JitPack)

Add the JitPack repository and the dependencies to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependencies>
    <!-- FastAIModel Engine -->
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastAIModel</artifactId>
        <version>0.1.0</version>
    </dependency>

    <!-- FastCore (Mandatory Native DLL Loader) -->
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
    implementation 'com.github.andrestubbe:FastAIModel:0.1.0'
    implementation 'com.github.andrestubbe:FastCore:0.1.0'
}
```

---

## Documentation

* **[ROADMAP.md](docs/ROADMAP.md)**: Planned milestone features and performance extensions.
* **[REFERENCE.md](docs/REFERENCE.md)**: JNI contracts and configuration options.
* **[PHILOSOPHY.md](docs/PHILOSOPHY.md)**: In-process design decisions.
* **[CHANGELOG.md](docs/CHANGELOG.md)**: Releases history.

---

## Platform Support

| Platform | Status |
|---|---|
| Windows 10/11 (x64) | ✅ Fully Supported |
| Linux | 🚧 Planned |
| macOS | 🚧 Planned |

---

## License

MIT License — See [LICENSE](LICENSE) file for details.

---

**Part of the FastJava Ecosystem** — *Making the JVM faster. Small package. Maximum speed. Zero bloat. 🚀📋*
