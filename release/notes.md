## What's New in 0.1.2

### Multi-Module Architecture
- **`fastaimodel-onnx`**: Ultra-lightweight ONNX Runtime module (0 C++ Llama DLLs needed).
- **`fastaimodel-llama`**: In-process `llama.cpp` GGUF local inference module.

### Installation

```xml
<!-- For ONNX Embeddings -->
<dependency>
    <groupId>com.github.andrestubbe.FastAIModel</groupId>
    <artifactId>fastaimodel-onnx</artifactId>
    <version>0.1.2</version>
</dependency>

<!-- For GGUF LLM Execution -->
<dependency>
    <groupId>com.github.andrestubbe.FastAIModel</groupId>
    <artifactId>fastaimodel-llama</artifactId>
    <version>0.1.2</version>
</dependency>
```
